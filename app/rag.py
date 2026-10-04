from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# FAISS vector database location
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"


def load_vectorstore():
    print("Loading FAISS vector store...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS vector store loaded successfully.")

    return vectorstore


def retrieve_documents(question, k=3):
    vectorstore = load_vectorstore()

    documents = vectorstore.similarity_search(
        question,
        k=k
    )

    return documents


if __name__ == "__main__":

    question = input("Enter your question: ")

    documents = retrieve_documents(question)

    print("\n========== RETRIEVED DOCUMENTS ==========\n")

    for i, document in enumerate(documents, start=1):

        print(f"--- Document {i} ---")
        print(document.page_content)
        print()