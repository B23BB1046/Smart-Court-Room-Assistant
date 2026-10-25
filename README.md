# Smart Court Assistant

A Streamlit-based legal research prototype that answers questions using retrieved Supreme Court judgments and optional user-uploaded PDFs. It combines vector search, document chunking, embeddings, and an LLM response workflow.

> **Disclaimer:** This is an educational research tool, not legal advice. Verify all legal claims and cited judgments against authoritative sources.

## Features
- Retrieve relevant passages from a local Supreme Court judgment index.
- Upload a PDF and query its contents using a document-specific retrieval flow.
- Keep separate conversations in the Streamlit interface.
- Generate answers from retrieved case and document context.

## Architecture
```text
Streamlit UI (Streamlit.py)
        ↓
Chat workflow (CourtAssistant.py)
        ├── Supreme Court FAISS index
        ├── Uploaded-PDF chunking and FAISS index
        ├── MiniLM sentence embeddings
        └── Groq-hosted LLM
```

## Repository files
| File | Purpose |
|---|---|
| `Streamlit.py` | User interface, conversation state, PDF upload |
| `CourtAssistant.py` | Retrieval and answer-generation workflow |
| `Judgments_Embedding.py` | Embedding/index preparation workflow |

## Requirements
Python 3.10+ is recommended. The code uses Streamlit, PyMuPDF, python-dotenv, LangChain integrations, Hugging Face sentence embeddings, FAISS, LangGraph, and Groq.

## Getting started
1. Clone the repository:
   `git clone https://github.com/B23BB1046/Smart-Court-Room-Assistant.git`
2. Install the dependencies imported by the scripts. Add a pinned `requirements.txt` to make setup reproducible.
3. Set `GROQ_API_KEY` in your environment or a local `.env` file excluded from version control.
4. Ensure the local `supreme_court_db` FAISS index has been built and is available at the expected path.
5. Launch the interface with `streamlit run Streamlit.py`.

## Important setup note
The assistant loads a prebuilt local FAISS index from `supreme_court_db`. That index is not listed among the tracked files in this repository, so you must create or provide it before the app can start successfully.

## Limitations and next steps
- LLM output can be incomplete or incorrect; retrieved evidence and citations need verification.
- Add a dependency file, dataset/index preparation instructions, and a small smoke test.
- Add robust handling for missing indexes, API failures, and malformed PDFs.
- Avoid logging or committing sensitive uploaded documents.

## Responsible use
Do not use this prototype as the sole basis for legal decisions. Confirm all findings with qualified legal professionals and primary sources.