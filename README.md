# Hybrid RAG Assistant

A robust, production-ready "Hybrid RAG-based Data Science & Machine Learning Assistant" using LangChain.

This project uses a TRUE HYBRID RAG architecture (dense + sparse retrieval) optimized for a DOMAIN-SPECIFIC knowledge base regarding Data Science, Machine Learning, SQL, and Python.

## Features

- True Hybrid Retreival (FAISS + BM25 + FlashrankRerank)
- Intent Query Routing (Concept Explanation, Summarization, Comparison, CSV Queries)
- 100% Free Stack (Groq, Local Sentence Embeddings, local FAISS)
- Conversational Memory handling via session IDs.
- Local fallback model using Ollama.

## Setup Instructions

1. **Clone the repo**
   ```bash
   git clone <repository-url>
   cd rag_project
   ```

2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   - Copy `.env.example` to `.env`
   - Sign up for a free Groq API key at https://console.groq.com
   - Set the `GROQ_API_KEY` in `.env`

4. **Generate Sample Knowledge Base Data**
   ```bash
   python scripts/generate_pdfs.py
   python scripts/generate_csv.py
   ```

5. **Build FAISS + BM25 Indexes**
   ```bash
   python vectorstore/vectordb.py
   ```

6. **Start the FastAPI Backend**
   ```bash
   uvicorn api.main:app --reload
   ```

7. **Start the Streamlit Frontend**
   ```bash
   streamlit run ui/app.py
   ```

## Alternative: Run via Docker

You can use Docker to spin up both the FastAPI backend and Streamlit UI after creating your `.env` file and running steps `4` and `5` above.

```bash
docker-compose up --build
```
