from pathlib import Path
import sys

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


# -------------------------------------------------
# Project paths
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"

# Allow Python to find src/config.py
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EMBEDDING_MODEL


# -------------------------------------------------
# Load FAISS vector store
# -------------------------------------------------

def load_vectorstore():

    print("Loading FAISS vector store...")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS vector store loaded successfully.")

    return vectorstore


# -------------------------------------------------
# Retrieve relevant documents
# -------------------------------------------------

def retrieve_documents(question, k=4):

    vectorstore = load_vectorstore()

    documents = vectorstore.similarity_search(
        question,
        k=k
    )

    return documents


# -------------------------------------------------
# Test retrieval
# -------------------------------------------------

if __name__ == "__main__":

    question = input("Enter your question: ")

    documents = retrieve_documents(question)

    print("\n========== RETRIEVED DOCUMENTS ==========\n")

    for i, document in enumerate(documents, start=1):

        print(f"--- Document {i} ---")

        print(document.page_content)

        print()