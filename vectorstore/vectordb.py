"""
Builds and saves the FAISS dense index and BM25 sparse index.
Should be run as a script after generating PDFs.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain_core.documents import Document

# Custom modules
from ingestion.pdf_loader import load_all_pdfs
from ingestion.web_loader import load_web_urls
from processing.preprocessing import preprocess_documents
from processing.chunking import chunk_documents
from processing.embeddings import get_embedding_model

from utils.config import VECTORDB_PATH
from utils.logger import logger

def build_vector_store():
    """Ingests all docs, chunks them, builds FAISS & BM25 indices, and saves them."""
    logger.info("Starting Vector Store build process...")
    
    # 1. Ingest Data
    base_dir = os.path.dirname(os.path.dirname(__file__))
    pdfs_dir = os.path.join(base_dir, "data", "pdfs")
    
    pdf_docs = load_all_pdfs(pdfs_dir)
    web_docs = load_web_urls()
    
    all_docs = pdf_docs + web_docs
    if not all_docs:
        logger.error("No documents found to index.")
        return
        
    logger.info(f"Loaded {len(all_docs)} total raw documents.")
    
    # 2. Preprocessing & Chunking
    cleaned_docs = preprocess_documents(all_docs)
    chunks = chunk_documents(cleaned_docs)
    
    # 3. Embedding Model
    embeddings = get_embedding_model()
    
    # 4. Build FAISS (Dense)
    logger.info("Building FAISS index...")
    faiss_index = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(VECTORDB_PATH, exist_ok=True)
    faiss_index.save_local(VECTORDB_PATH)
    logger.info(f"Saved FAISS index to {VECTORDB_PATH}")
    
    # 5. Build BM25 (Sparse)
    logger.info("Building BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    
    # Save BM25 retriever via pickle
    bm25_path = os.path.join(os.path.dirname(VECTORDB_PATH), "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    logger.info(f"Saved BM25 retriever to {bm25_path}")
    
    logger.info("Vector Store build process completed successfully.")

if __name__ == "__main__":
    build_vector_store()
