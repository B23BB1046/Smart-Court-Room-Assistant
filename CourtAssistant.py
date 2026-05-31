import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import pickle
from langchain_community.vectorstores import FAISS

# Load faiss Index
# LOAD FAISS INDEX
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    "supreme_court_db",
    model,
    allow_dangerous_deserialization=True
)
load_dotenv()
llm = ChatGroq(
    model = 'llama-3.1-8b-instant',
    api_key = os.getenv('GROQ_API_KEY'),    
    temperature = 0.7
)


def response_generation(query):
    
    # search for similar documents
    Docs = db.similarity_search(query, k=5)
    # retrieve the documents
    Releavant_chnks = [doc.page_content for doc in Docs]
    # Create context for the LLM
    context = "\n".join(Releavant_chnks)
    # Generate response using LLM
    prompt = f"""
    You are a legal assistant. Give the answer based on the context and answer should
    be very clear and must be humanize so that normal people can also understand the langugage and use proper scientific terminology also 
    by which answer should be look real and also give summary of that answer. If you really don't know the answer then use llm to give the answer
    Context: {context}
    Question: {query}
    Answer:
    """
    for chunk in llm.stream(prompt):
        yield chunk
if __name__ == "__main__":
    
    while True:
        
        user_query = input("Enter your legal question (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        answer = response_generation(user_query)
        print(f"Answer: {answer}\n")
