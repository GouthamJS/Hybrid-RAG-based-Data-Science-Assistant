"""
Helper functions for text cleaning and JSON validation.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
import json
import os
from typing import Dict, Any, List
from bs4 import BeautifulSoup
from langchain_core.documents import Document

def clean_text(text: str) -> str:
    """Strips HTML, removes extra whitespace and normalizes text."""
    if not text:
        return ""
    
    # Strip HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def validate_json_response(response: str) -> Dict[str, Any]:
    """Ensures that the LLM response is valid JSON and contains required keys."""
    try:
        # Try direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        # Clean markdown codeblocks
        clean_resp = re.sub(r'```json', '', response)
        clean_resp = re.sub(r'```', '', clean_resp).strip()
        try:
            return json.loads(clean_resp)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {response}") from e

def format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """Converts Langchain Documents to source dictionaries."""
    sources = []
    seen = set()
    for doc in docs:
        source_val = doc.metadata.get("source", "Unknown")
        # Ensure we only show filename if it's a file
        source = os.path.basename(source_val) if "\\" in source_val or "/" in source_val else source_val
        page = doc.metadata.get("page", 1)
        
        key = f"{source}_{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"document": source, "page": page})
    return sources
