"""
Loader for PDF documents.
Uses PyPDFLoader from LangChain to process all generated PDFs.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from utils.logger import logger

def load_all_pdfs(pdfs_dir: str) -> List[Document]:
    """Loads all PDF files from the specified directory."""
    documents = []
    
    if not os.path.exists(pdfs_dir):
        logger.warning(f"PDF directory does not exist: {pdfs_dir}")
        return documents
        
    pdf_files = glob.glob(os.path.join(pdfs_dir, "*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdfs_dir}")
    
    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}")
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            
    return documents
