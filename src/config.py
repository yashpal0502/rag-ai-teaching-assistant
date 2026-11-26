import os
from pathlib import Path

# Project Directories

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"

EMBEDDINGS_FILE = EMBEDDINGS_DIR / "chunk_embeddings.joblib"


# Ollama Configuration

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# Model Configuration

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")


# API Configuration

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
