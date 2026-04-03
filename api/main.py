"""
FastAPI Backend Module.
Provides REST endpoints for querying the hybrid RAG system and managing sessions.
"""
import os
import sys

# Add project root to sys.path to allow importing from the 'rag' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from rag.chain import HybridRAGChain
from memory.conversation_memory import delete_session
from utils.logger import logger

app = FastAPI(title="Hybrid RAG Assistant API")

# Lazy load the chain to avoid massive initialization on import
_rag_chain = None

def get_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = HybridRAGChain()
    return _rag_chain

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: str
    suggested_questions: list[str]
    intent: Optional[str] = "concept_explanation"

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    """Processes a user query through the RAG pipeline."""
    try:
        chain = get_chain()
        result = chain.process_query(req.query, req.session_id)
        return result
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while processing query.")

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clears the conversational memory for a given session."""
    success = delete_session(session_id)
    if success:
        return {"message": f"Session {session_id} memory cleared successfully."}
    return {"message": f"Session {session_id} not found."}
