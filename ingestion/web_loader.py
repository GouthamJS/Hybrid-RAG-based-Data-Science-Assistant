"""
Loader for Web content.
Uses WebBaseLoader from LangChain to fetch Wikipedia articles.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from utils.logger import logger

URLS_TO_LOAD = [
    "https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff",
    "https://en.wikipedia.org/wiki/Regularization_(mathematics)",
    "https://en.wikipedia.org/wiki/Overfitting"
]

def load_web_urls() -> List[Document]:
    """Loads content from predefined URLs."""
    documents = []
    
    for url in URLS_TO_LOAD:
        try:
            logger.info(f"Loading web content from: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            # Add context metadata
            for doc in docs:
                doc.metadata["source"] = url
                
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error loading web URL {url}: {str(e)}")
            
    logger.info(f"Total web documents loaded: {len(documents)}")
    return documents
