"""
Memory Management Module.
Maintains conversational history per session using a ConversationBufferWindowMemory.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List
from langchain.memory import ConversationBufferWindowMemory
from utils.config import MEMORY_K
from utils.logger import logger

# In-memory store for session configurations.
# In a true production environment, this would be a Redis store or database.
SESSION_MEMORY_STORE: Dict[str, ConversationBufferWindowMemory] = {}

def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Retrieves or creates a memory object for the given session ID."""
    if session_id not in SESSION_MEMORY_STORE:
        logger.info(f"Creating new memory buffer for session: {session_id}")
        SESSION_MEMORY_STORE[session_id] = ConversationBufferWindowMemory(
            k=MEMORY_K,
            return_messages=True,
            memory_key="chat_history"
        )
    return SESSION_MEMORY_STORE[session_id]

def delete_session(session_id: str) -> bool:
    """Deletes a session from memory."""
    if session_id in SESSION_MEMORY_STORE:
        del SESSION_MEMORY_STORE[session_id]
        logger.info(f"Deleted memory for session: {session_id}")
        return True
    return False

def get_formatted_chat_history(session_id: str) -> str:
    """Gets formatted chat history from the session memory."""
    memory = get_session_memory(session_id)
    messages = memory.load_memory_variables({}).get("chat_history", [])
    
    formatted_history = ""
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        formatted_history += f"{role}: {msg.content}\n"
        
    return formatted_history

def add_to_history(session_id: str, question: str, answer: str):
    """Adds a human and AI message to the memory."""
    memory = get_session_memory(session_id)
    memory.save_context({"input": question}, {"output": answer})
