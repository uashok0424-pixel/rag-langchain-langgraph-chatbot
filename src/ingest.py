from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents():
    """Load all PDF files from the documents folder."""

    documents = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return documents

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} pages.")

    return documents


def split_documents(documents):
    """Split documents into smaller chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_vectorstore(chunks):
    """Create embeddings and save them in FAISS."""

    print("Creating embeddings...")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTORSTORE_DIR)
    )

    print("FAISS vector store saved successfully.")


def main():

    print("=" * 50)
    print("RAG DOCUMENT INGESTION")
    print("=" * 50)

    documents = load_documents()

    if not documents:
        return

    chunks = split_documents(documents)

    create_vectorstore(chunks)

    print("=" * 50)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    main()