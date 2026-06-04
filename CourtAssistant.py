import os
import fitz
from dotenv import load_dotenv
from typing import TypedDict, Optional

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END

load_dotenv()

# -------------------------
# LLM
# -------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# -------------------------
# Embeddings
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Supreme Court FAISS DB
# -------------------------
supreme_db = FAISS.load_local(
    "supreme_court_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# PDF DB
vector_db = None


# -------------------------
# State
# -------------------------
class ChatState(TypedDict):
    query: str
    pdf_uploaded: bool

    case_docs: str
    pdf_docs: str

    response: str


# -------------------------
# Upload PDF
# -------------------------
def load_pdf_to_db(pdf_path: str):
    global vector_db

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db


# -------------------------
# Retrieve Previous Cases
# -------------------------
def retrieve_previous_cases(query):

    docs = supreme_db.similarity_search(
        query,
        k=5
    )

    return docs


# -------------------------
# Previous Case Agent
# -------------------------
def PreviousCaseAgent(state: ChatState):

    docs = retrieve_previous_cases(
        state["query"]
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "case_docs": context
    }


# -------------------------
# PDF RAG Agent
# -------------------------
def RagAgent(state: ChatState):

    docs = vector_db.similarity_search(
        state["query"],
        k=5
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "pdf_docs": context
    }


# -------------------------
# Router
# -------------------------
def route_after_case_search(state: ChatState):

    if state["pdf_uploaded"] and vector_db is not None:
        return "rag"

    return "response"


# -------------------------
# Response Generation
# -------------------------
def ResponseAgent(state: ChatState):

    query = state["query"]

    case_context = state.get(
        "case_docs",
        ""
    )

    pdf_context = state.get(
        "pdf_docs",
        ""
    )

    prompt = f"""
    You are a legal assistant helping answer questions based on previous court cases and an uploaded PDF document.If pdf is uploaded, use it as additional context along with the retrieved cases to answer the user's legal question. Always prioritize information from the retrieved cases, but supplement with PDF findings when relevant.
    If user ask general question then give the answer using llm's knowledge without relying on the retrieved cases or PDF context.
- Answer naturally like a normal AI assistant.
- Use your general knowledge.
- Do not use the legal response format.
- Do not mention case law, retrieved cases, or PDF context unless the user specifically asks about them.

Retrieved Previous Cases:
-------------------------
{case_context}

Uploaded PDF Context:
-------------------------
{pdf_context}

User Question:
-------------------------
{query}

If the question is LEGAL, respond using exactly this structure:

### Answer
<detailed answer>

### Relevant Case Insights
<important points from retrieved cases, if available>

### PDF Findings
<important points from uploaded PDF, if available>

### Notes
<Mention limitations, missing information, or whether general legal knowledge was used>

If the question is NOT LEGAL, answer normally as a helpful AI assistant without using the above sections.
"""

    response = ""

    for chunk in llm.stream(prompt):
        response += chunk.content

    return {
        "response": response
    }


# -------------------------
# Graph
# -------------------------
graph = StateGraph(ChatState)
graph.add_node("previous_cases",PreviousCaseAgent)
graph.add_node("rag", RagAgent)
graph.add_node("response",ResponseAgent)
graph.add_edge(START,"previous_cases")

graph.add_conditional_edges(
    "previous_cases",
    route_after_case_search,
    {
        "rag": "rag",
        "response": "response"
    }
)
graph.add_edge("rag", "response")
graph.add_edge("response", END)
chatbot = graph.compile()