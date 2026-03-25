# Quantum RAG Assistant

> An AI-powered tutoring system for Quantum Computing — built with Retrieval-Augmented Generation (RAG), FAISS vector search, and a FastAPI + Streamlit interface. 

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

**Quantum RAG Assistant** is a production-ready, document-grounded AI tutor for Quantum Computing. It retrieves relevant content from trusted sources using semantic search (FAISS), then feeds them to an LLM to generate accurate, explainable answers — minimizing hallucinations.

Built for learners, researchers, and developers who want a transparent, explainable AI system backed by real documents.

---

## Architecture

```
User Query
    │
    ▼
[Streamlit UI / FastAPI]
    │
    ▼
[Query Encoder] ──► FAISS Vector Index
    │                      │
    │              Top-K Chunks Retrieved
    │                      │
    └──────────────────────┘
                   │
                   ▼
        [Groq LLM — llama-3.3-70b-versatile]
                   │
                   ▼
        Grounded Answer + Source Citations
```

---

## Project Structure

```
quantum-rag-assistant/
│
├── data/
│   ├── raw/                   # Source PDFs and documents
│   ├── processed/             # Cleaned text chunks (JSON)
│   └── embeddings/            # Saved FAISS indexes
│
├── src/
│   ├── ingestion/
│   │   ├── load_documents.py  # PDF/text loader
│   │   └── chunker.py         # Text chunking strategies
│   ├── embeddings/
│   │   ├── embed_documents.py # Embedding pipeline (local)
│   │   └── vector_store.py    # FAISS index management
│   ├── retrieval/
│   │   └── retriever.py       # Semantic search
│   ├── llm/
│   │   ├── qa_chain.py        # RAG chain
│   │   └── prompt_templates.py# Prompt engineering
│   └── utils/
│       ├── config.py          # Pydantic Settings config
│       └── logger.py          # Rich structured logging
│
├── app/
│   ├── main.py                # FastAPI app (lifespan startup)
│   ├── state.py               # Shared app state (avoids circular imports)
│   ├── routes/
│   │   ├── chat.py            # POST /api/v1/chat
│   │   └── health.py          # GET  /api/v1/health
│   └── streamlit_app.py       # Streamlit UI
│
├── tests/
│   ├── unit/                  # Unit tests
│   └── integration/           # API integration tests
│
├── configs/
│   └── settings.yaml          # App configuration defaults
│
├── scripts/
│   ├── ingest.py              # CLI: load & chunk documents
│   └── build_index.py         # CLI: build FAISS index
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml
│
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/quantum-rag-assistant.git
cd quantum-rag-assistant

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key (free at [console.groq.com](https://console.groq.com)):

```env
GROQ_API_KEY=your_key_here
LLM_MODEL=llama-3.3-70b-versatile
```

### 3. Ingest Documents

```bash
# Add PDFs/text files to data/raw/ (a sample knowledge base is included)
python scripts/ingest.py

# Build FAISS vector index
python scripts/build_index.py
```

### 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
# Swagger UI available at http://localhost:8000/docs
```

### 5. Run the UI

```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

### 6. Docker (Full Stack)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---


## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Ask a question |
| `GET`  | `/api/v1/health` | Health check |
| `GET`  | `/docs` | Swagger UI |

**Example Request:**
```json
POST /api/v1/chat
{
  "question": "What is quantum entanglement?",
  "top_k": 5
}
```

**Example Response:**
```json
{
  "answer": "Quantum entanglement is...",
  "sources": ["quantum_computing_fundamentals.txt"],
  "confidence": 0.94,
  "processing_time_ms": 342
}
```

---

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Coverage report
pytest --cov=src --cov-report=html
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| Vector DB | FAISS (`faiss-cpu`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| API | FastAPI |
| UI | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Config | Pydantic Settings v2 + YAML |
| Logging | Rich structured logging |

---


## Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

