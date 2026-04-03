"""
Document chunking module.
Uses RecursiveCharacterTextSplitter for optimal NLP chunking.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import logger

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks using RecursiveCharacterTextSplitter.
    Injects metadata such as chunk index and filename.
    """
    logger.info(f"Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Enhance metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        source_val = chunk.metadata.get("source", "Unknown")
        # Ensure we only show filename
        filename = os.path.basename(source_val) if "\\" in source_val or "/" in source_val else source_val
        chunk.metadata["document"] = filename

    logger.info(f"Generated {len(chunks)} chunks from {len(documents)} documents.")
    return chunks
