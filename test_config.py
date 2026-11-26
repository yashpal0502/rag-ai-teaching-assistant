import sys

sys.path.append("..")

from src.config import *

print("Project root:", PROJECT_ROOT)
print("Data directory:", DATA_DIR)
print("Embeddings file:", EMBEDDINGS_FILE)

print("Embedding model:", EMBEDDING_MODEL)
print("LLM model:", LLM_MODEL)

print("Embeddings exist:", EMBEDDINGS_FILE.exists())
