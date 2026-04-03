"""
Hybrid Retriever Module.
Combines FAISS (dense) and BM25 (sparse) into an EnsembleRetriever,
then applies FlashrankRerank cross-encoder for final sorting.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import pickle
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.vectorstores import FAISS

from processing.embeddings import get_embedding_model
from utils.config import (
    VECTORDB_PATH, 
    TOP_K, 
    RERANK_TOP_N, 
    ENSEMBLE_DENSE_WEIGHT, 
    ENSEMBLE_SPARSE_WEIGHT
)
from utils.logger import logger

def get_hybrid_retriever():
    """Loads FAISS & BM25 and returns the configured hybrid retriever."""
    if not os.path.exists(VECTORDB_PATH):
        logger.error(f"FAISS index not found at {VECTORDB_PATH}. Run vectordb.py first.")
        return None
        
    bm25_path = os.path.join(os.path.dirname(VECTORDB_PATH), "bm25_index.pkl")
    if not os.path.exists(bm25_path):
        logger.error(f"BM25 index not found at {bm25_path}. Run vectordb.py first.")
        return None
        
    try:
        # Load Dense Retriever
        embeddings = get_embedding_model()
        faiss_index = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": TOP_K})
        logger.info("Loaded FAISS Retriever.")
        
        # Load Sparse Retriever
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
            bm25_retriever.k = TOP_K
        logger.info("Loaded BM25 Retriever.")
        
        # Build Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[ENSEMBLE_DENSE_WEIGHT, ENSEMBLE_SPARSE_WEIGHT]
        )
        logger.info("Built Ensemble Retriever.")
        
        # Add Reranker
        compressor = FlashrankRerank(top_n=RERANK_TOP_N)
        hybrid_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        logger.info("Built Contextual Compression Retriever with FlashRank.")
        
        return hybrid_retriever
        
    except Exception as e:
        logger.error(f"Failed to initialize hybrid retriever: {str(e)}")
        return None
