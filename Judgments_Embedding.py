from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import pickle

# Root folder containing Supreme Court PDFs
ROOT = "/home/yogendra/Documents/SmartCourtRoomAssistant/SmartCourtAssistant/supreme_court_judgments"

# Store all loaded documents
all_documents = []


# =========================
# LOAD PDF DOCUMENTS
# =========================
def load_documents(root_path):

    pdf_count = 0

    for root, dirs, files in os.walk(root_path):

        for file in files:

            if file.lower().endswith(".pdf"):

                file_path = os.path.join(root, file)

                print(f"\nProcessing: {file_path}")

                try:

                    loader = PyPDFLoader(file_path)

                    documents = loader.load()

                    all_documents.extend(documents)

                    pdf_count += 1

                    print(f"Loaded successfully: {file}")

                except Exception as e:

                    print(f"Error loading {file_path}")
                    print(e)

    print("\n==============================")
    print(f"Total PDFs processed: {pdf_count}")
    print(f"Total pages loaded: {len(all_documents)}")
    print("==============================\n")


# =========================
# GENERATE VECTOR DATABASE
# =========================
def create_vector_store(documents):

    print("Loading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Creating FAISS vector store...")

    db = FAISS.from_documents(documents, embedding_model)

    # Save FAISS database locally
    db.save_local("supreme_court_db")

    print("FAISS vector database saved successfully.")

    return db


# =========================
# SAVE DOCUMENTS AS PICKLE
# =========================
def save_documents(documents, output_file):

    with open(output_file, 'wb') as f:

        pickle.dump(documents, f)

    print(f"Documents saved to {output_file}")


# =========================
# MAIN FUNCTION
# =========================
if __name__ == "__main__":

    # Step 1: Load PDFs
    load_documents(ROOT)

    # Step 2: Create FAISS vector store
    db = create_vector_store(all_documents)

    # Step 3: Save documents separately
    save_documents(all_documents, "judgments.pkl")

    print("\nAll tasks completed successfully.")