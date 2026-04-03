"""
Loads the Sales CSV Data and provides the Pandas DataFrame Agent for natural language queries.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import pandas as pd
from typing import Optional
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# LLM imports for the DF Agent
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama

from utils.logger import logger
from utils.config import USE_GROQ, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL

def get_llm():
    """Returns the configured LLM based on environment variables."""
    if USE_GROQ and GROQ_API_KEY:
        from langchain_groq import ChatGroq
        logger.info(f"Initializing Groq Model: {GROQ_MODEL} for CSV Agent")
        return ChatGroq(
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL
        )
    else:
        logger.info(f"Initializing Fallback Local Ollama Model: {OLLAMA_MODEL} for CSV Agent")
        return Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )

def get_csv_agent(csv_path: str):
    """Creates and returns a Pandas DataFrame Agent."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at: {csv_path}")
        return None
        
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded DataFrame with shape {df.shape}")
        
        llm = get_llm()
        
        # We allow dangerous code (it is running locally on pandas) 
        # required by new version of langchain_experimental
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        return agent
    except Exception as e:
        logger.error(f"Error creating CSV agent: {str(e)}")
        return None
