from pathlib import Path


# ==========================================
# PROJECT ROOT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================
# DIRECTORIES
# ==========================================

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"


# ==========================================
# LOCAL OLLAMA MODELS
# ==========================================

LLM_MODEL = "llama3.2"

EMBEDDING_MODEL = "nomic-embed-text"


# ==========================================
# RAG SETTINGS
# ==========================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

TOP_K = 3