"""
Configuration loader for the Hybrid RAG application.
Loads variables from .env and exposes them as typed constants.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Provider Configuration
USE_GROQ = os.getenv("USE_GROQ", "true").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Fallback Offline LLM Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embeddings Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval Settings
TOP_K = int(os.getenv("TOP_K", "5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
ENSEMBLE_DENSE_WEIGHT = float(os.getenv("ENSEMBLE_DENSE_WEIGHT", "0.6"))
ENSEMBLE_SPARSE_WEIGHT = float(os.getenv("ENSEMBLE_SPARSE_WEIGHT", "0.4"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))

# Chunking Settings for Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Storage Locations
VECTORDB_PATH = os.getenv("VECTORDB_PATH", "vectorstore/faiss_index")
MEMORY_K = int(os.getenv("MEMORY_K", "5"))
