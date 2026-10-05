from pathlib import Path
import sys

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


# ==========================================
# PROJECT ROOT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==========================================
# VECTOR STORE
# ==========================================

VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"


# ==========================================
# CONFIG
# ==========================================

SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EMBEDDING_MODEL


# ==========================================
# LOAD VECTOR STORE
# ==========================================

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