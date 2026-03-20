"""
FastAPI application entrypoint.
Provides REST API for the Quantum RAG Assistant.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import app.state as state
from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from src.llm.qa_chain import QAChain
from src.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources at startup, clean up at shutdown."""
    log.info("🚀 Starting Quantum RAG Assistant API...")
    try:
        state.qa_chain = QAChain()
        state.qa_chain.retriever.vector_store.load()
        log.info("✓ Vector store loaded and ready")
    except FileNotFoundError:
        log.warning(
            "⚠️  FAISS index not found. Run: python scripts/build_index.py\n"
            "   The API will start, but /chat will return errors until index is built."
        )
    yield
    log.info("🛑 Shutting down Quantum RAG Assistant API")


app = FastAPI(
    title="Quantum RAG Assistant",
    description=(
        "AI-powered Quantum Computing tutor using Retrieval-Augmented Generation. "
        "Answers are grounded in trusted quantum computing documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# CORS — allow all for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "⚛️ Quantum RAG Assistant API",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health",
    }