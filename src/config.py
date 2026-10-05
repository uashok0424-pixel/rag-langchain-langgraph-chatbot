from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Documents directory
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

# FAISS vector database directory
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"


# Models
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
# Text splitting settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Number of documents to retrieve
TOP_K = 4