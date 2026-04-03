"""
Centralized logger for the Hybrid RAG application.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import os
from dotenv import load_dotenv

load_dotenv()

def setup_logger(name: str) -> logging.Logger:
    """Sets up a centralized logger."""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler("logs/app.log")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

logger = setup_logger("hybrid_rag")
