"""
Embeddings generation module.
Uses HuggingFace local models for 100% free embeddings on CPU.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_huggingface import HuggingFaceEmbeddings
from utils.config import EMBEDDING_MODEL
from utils.logger import logger

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace local embedding model.
    Configured to run on CPU and normalize embeddings.
    """
    logger.info(f"Loading local HuggingFace embedding model: {EMBEDDING_MODEL}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        raise e
