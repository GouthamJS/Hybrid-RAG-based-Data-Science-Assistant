"""
Text preprocessing module.
Cleans documents loaded from various sources before chunking.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List
from langchain_core.documents import Document
from utils.helpers import clean_text
from utils.logger import logger

def preprocess_documents(documents: List[Document]) -> List[Document]:
    """
    Cleans text content of Langchain Documents:
    - Strips HTML tags
    - Removes extra whitespace
    - Normalizes unicode
    - Removes empty documents
    """
    cleaned_docs = []
    
    for doc in documents:
        original_text = doc.page_content
        cleaned_content = clean_text(original_text)
        
        # Only keep non-empty documents
        if cleaned_content and len(cleaned_content.strip()) > 5:
            doc.page_content = cleaned_content
            cleaned_docs.append(doc)
            
    logger.info(f"Preprocessed {len(documents)} documents. Retained {len(cleaned_docs)}.")
    return cleaned_docs
