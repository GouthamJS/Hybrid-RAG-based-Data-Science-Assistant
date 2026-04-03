"""
Query Router Module.
Classifies user intents using an LLM to decide whether to route to the true Hybrid RAG
or the Pandas DataFrame CSV Agent.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Literal
from langchain_core.prompts import PromptTemplate
from utils.logger import logger
from utils.config import USE_GROQ, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL

def get_router_llm():
    """Initializes the LLM specifically for routing."""
    if USE_GROQ and GROQ_API_KEY:
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0,  # Intent classification needs 0 temperature
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL
        )
    else:
        from langchain_community.llms import Ollama
        return Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )

class QueryRouter:
    """Routes a query to one of the predefined categories."""
    
    def __init__(self):
        self.llm = get_router_llm()
        self.prompt = PromptTemplate(
            template='''
Classify this query into exactly one category:
[concept_explanation, comparison, summarization, csv_query]

Rules:
- If it asks about structured data, sales, revenue, region aggregates, or amounts, output "csv_query".
- If it asks to compare two things, output "comparison".
- If it asks to summarize a topic, output "summarization".
- If it asks what something is or how it works, output "concept_explanation".

Query: {query}
Category:''',
            input_variables=["query"]
        )
        self.chain = self.prompt | self.llm
        
    def classify(self, query: str) -> str:
        """Classifies the string query into an intent category."""
        target_intents = ["concept_explanation", "comparison", "summarization", "csv_query"]
        logger.info(f"Classifying intent for query: {query}")
        
        try:
            # We enforce string result and clean it
            result = self.chain.invoke({"query": query})
            
            # Ollama might return AIMessage or str depending on exactly how it is configured, so we handle both
            intent = result.content.strip().lower() if hasattr(result, "content") else str(result).strip().lower()
            
            # Clean up the output in case LLM is chatty
            for target in target_intents:
                if target in intent:
                    logger.info(f"Routed query as: {target}")
                    return target
            
            # Fallback
            logger.warning(f"Could not cleanly classify intent '{intent}', falling back to 'concept_explanation'")
            return "concept_explanation"
            
        except Exception as e:
            logger.error(f"Routing failed due to error: {str(e)}")
            return "concept_explanation"
