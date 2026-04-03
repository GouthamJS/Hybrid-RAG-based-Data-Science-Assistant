"""
Streamlit Frontend Module.
A UI to interact with the true hybrid RAG API.
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import requests
import uuid

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Data Science Assistant", page_icon="🤖", layout="wide")

# Initialize session state configuration
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and Layout
st.title("🤖 True Hybrid RAG Data Science Assistant")
st.markdown("Ask anything regarding Machine Learning, Data Science, SQL, or local CSV sales data!")

# Sidebar commands
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat Memory"):
        requests.delete(f"{API_BASE_URL}/session/{st.session_state.session_id}")
        st.session_state.messages = []
        st.success("Memory cleared!")
    st.markdown("---")
    st.caption(f"Session ID: {st.session_state.session_id}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display Sources, Intent, and Confidence if present
        if message["role"] == "assistant":
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                conf = message.get("confidence", "0%")
                conf_val = float(conf.strip("%").strip()) if "%" in conf else 0.0
                color = "green" if conf_val >= 70 else "orange" if conf_val >= 40 else "red"
                st.markdown(f"**Confidence:** :{color}[{conf}]")
                
            with col2:
                intent = message.get("intent", "N/A")
                st.markdown(f"**Intent:** `{intent}`")
                
            with col3:
                sources = message.get("sources", [])
                if sources:
                    with st.expander("View Sources"):
                        for s in sources:
                            st.caption(f"- {s.get('document', 'Unknown')} (Page: {s.get('page', 1)})")
                            
            # Render Suggested questions
            suggs = message.get("suggested_questions", [])
            if suggs:
                st.markdown("**Suggested Follow-ups:**")
                for sq in suggs:
                    st.button(sq, key=f"btn_{sq}_{uuid.uuid4()}", on_click=lambda q=sq: st.chat_input(q)) # Auto-fill input won't work perfectly in native Streamlit, but gives a visual representation

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # API Request
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_BASE_URL}/query", json={
                    "query": prompt,
                    "session_id": st.session_state.session_id
                })
                response.raise_for_status()
                data = response.json()
                
                answer = data.get("answer", "Error retrieving answer.")
                st.markdown(answer)
                
                conf = data.get("confidence", "0%")
                conf_val = float(conf.strip("%").strip())
                color = "green" if conf_val >= 70 else "orange" if conf_val >= 40 else "red"
                st.markdown(f"**Confidence:** :{color}[{conf}]")
                intent = data.get("intent", "Unknown")
                st.markdown(f"**Intent:** `{intent}`")
                
                sources = data.get("sources", [])
                if sources:
                    with st.expander("View Sources"):
                        for s in sources:
                            st.caption(f"- {s.get('document', 'Unknown')} (Page: {s.get('page', 1)})")
                            
                suggs = data.get("suggested_questions", [])
                if suggs:
                    st.markdown("**Suggested Follow-ups:**")
                    for sq in suggs:
                        st.caption(f"👉 {sq}")
                
                # Save assistant msg
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": conf,
                    "suggested_questions": suggs,
                    "intent": intent
                })
                
            except Exception as e:
                st.error("Failed to connect to backend api.")
                st.exception(e)
