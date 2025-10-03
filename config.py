import os
from dotenv import load_dotenv
load_dotenv()

# API keys and model names
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")


# Chunking defaults
DEFAULT_TARGET_CHUNKS = int(os.getenv("DEFAULT_TARGET_CHUNKS", 50))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 100))

# FAISS / retrieval defaults
DEFAULT_K = int(os.getenv("DEFAULT_K", 3))

# Local storage for temporary files or indexes
TMP_DIR = os.getenv("TMP_DIR", "/tmp/pdf_qa")

# export GEMINI_API_KEY="AIzaSyDiekJUeSHryzix7kFoFMTLxhiq9kmU884"
# export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"