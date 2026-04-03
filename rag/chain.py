"""
Core RAG Chain Module.
Implements the full true hybrid RAG pipeline:
Retrieval -> FAISS Confidence Score Check -> JSON Prompting -> Validation
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import json
from typing import Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from rag.router import QueryRouter
from rag.retriever import get_hybrid_retriever
from rag.prompt_template import get_rag_prompt
from memory.conversation_memory import get_formatted_chat_history, add_to_history
from processing.embeddings import get_embedding_model
from utils.helpers import validate_json_response, format_sources
from utils.logger import logger
from utils.config import (
    USE_GROQ, 
    GROQ_API_KEY, 
    GROQ_MODEL, 
    OLLAMA_MODEL, 
    OLLAMA_BASE_URL,
    VECTORDB_PATH,
    CONFIDENCE_THRESHOLD
)

class HybridRAGChain:
    """Main execution chain for answering RAG queries."""

    def __init__(self):
        self.router = QueryRouter()
        self.retriever = get_hybrid_retriever()
        self.llm = self._init_llm()
        self.prompt = get_rag_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # We need a raw FAISS index just to compute confidence score
        try:
            self.faiss_index = FAISS.load_local(
                VECTORDB_PATH, 
                get_embedding_model(), 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS for confidence scoring: {e}")
            self.faiss_index = None

    def _init_llm(self):
        """Initializes the text generation LLM."""
        if USE_GROQ and GROQ_API_KEY:
            from langchain_groq import ChatGroq
            return ChatGroq(
                temperature=0.1,
                groq_api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
        else:
            from langchain_community.llms import Ollama
            return Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
                format="json"
            )

    def _get_confidence_score(self, query: str) -> float:
        """Calculates confidence score using FAISS top match."""
        if not self.faiss_index:
            return 0.0
            
        # FAISS returns L2 distance by default (lower is better, but with normalize_embeddings it equals cosine distance)
        # However similarity_search_with_relevance_scores returns 0 to 1 scale relevance.
        results = self.faiss_index.similarity_search_with_relevance_scores(query, k=1)
        if not results:
            return 0.0
            
        doc, score = results[0]
        # In case the score is negative (sometimes FAISS behavior), clip to 0
        confidence = max(0.0, score)
        return round(confidence * 100, 1)

    def generate_fallback(self) -> Dict[str, Any]:
        """Generates the static fallback response for low confidence."""
        return {
            "answer": "I don't have enough information based on the available data.",
            "sources": [],
            "confidence": "0.0%",
            "suggested_questions": ["What is a neural network?", "Can you explain overfitting?"]
        }

    def process_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """End-to-end processing of a user query."""
        logger.info(f"Processing query for session {session_id}: '{query}'")
        
        # 1. Routing Intent
        intent = self.router.classify(query)
        
        # 2. Check if CSV
        if intent == "csv_query":
            return self._handle_csv_query(query, session_id, intent)
            
        # 3. Retrieve Context from Hybrid Retriever
        retrieved_docs = self.retriever.invoke(query) if self.retriever else []
        
        # 4. Check confidence directly from FAISS
        confidence = self._get_confidence_score(query)
        logger.info(f"FAISS Retrieval Confidence Score: {confidence}%")
        
        if confidence < (CONFIDENCE_THRESHOLD * 100):
            logger.warning(f"Low confidence ({confidence}%) - triggering fallback limit.")
            fallback = self.generate_fallback()
            # Still set the actual confidence and intent calculated
            fallback["confidence"] = f"{confidence}%"
            fallback["intent"] = intent
            add_to_history(session_id, query, fallback["answer"])
            return fallback
            
        # 5. Build context & chat history
        context = "\n\n".join([d.page_content for d in retrieved_docs])
        chat_history = get_formatted_chat_history(session_id)
        
        # 6. Call LLM
        try:
            raw_response = self.chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": query
            })
            
            # 7. Validate JSON Output
            structured_response = validate_json_response(raw_response)
            
            # 8. Augment response with our own calculated metadata
            structured_response["sources"] = format_sources(retrieved_docs)
            structured_response["confidence"] = f"{confidence}%"
            structured_response["intent"] = intent
            
            # Save memory
            add_to_history(session_id, query, structured_response["answer"])
            
            return structured_response
            
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            fallback = self.generate_fallback()
            fallback["intent"] = intent
            return fallback

    def _handle_csv_query(self, query: str, session_id: str, intent: str) -> Dict[str, Any]:
        """Handles queries classified as csv_query using Pandas Agent."""
        from ingestion.csv_loader import get_csv_agent
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(base_dir, "data", "csv", "sales_data.csv")
        
        agent = get_csv_agent(csv_path)
        if not agent:
            resp = self.generate_fallback()
            resp["answer"] = "CSV data not available."
            resp["intent"] = intent
            return resp
            
        try:
            result = agent.invoke({"input": query})
            answer = result["output"]
            
            structured_response = {
                "answer": answer,
                "sources": [{"document": "sales_data.csv", "page": 1}],
                "confidence": "95.0%",  # High confidence as it directly runs dataframe logic
                "suggested_questions": ["What is the total revenue?", "Show me sales by region."],
                "intent": intent
            }
            add_to_history(session_id, query, answer)
            return structured_response
        except Exception as e:
            logger.error(f"CSV Agent failed: {e}")
             # We let it pass but send fallback instead of raw error
            fallback = self.generate_fallback()
            fallback["intent"] = intent
            return fallback

