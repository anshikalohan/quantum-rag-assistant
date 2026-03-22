"""
Chat endpoint — the core RAG question-answering API.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import app.state as state
from src.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The quantum computing question to answer",
        examples=["What is quantum superposition?"],
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve",
    )
    threshold: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval",
    )


class RetrievalInfo(BaseModel):
    source: str
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    retrieval_info: List[RetrievalInfo]
    processing_time_ms: float
    model_used: str
    has_context: bool


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about Quantum Computing.

    The system will:
    1. Retrieve relevant passages from the knowledge base
    2. Use an LLM to generate a grounded, accurate answer
    3. Return the answer with source citations
    """
    if state.qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="QA chain not initialized. Check server logs.",
        )

    try:
        response = state.qa_chain.answer(
            question=request.question,
            top_k=request.top_k,
            threshold=request.threshold,
        )

        return ChatResponse(
            question=response.question,
            answer=response.answer,
            sources=response.sources,
            retrieval_info=[
                RetrievalInfo(source=r.source, score=round(r.score, 4))
                for r in response.retrieval_results
            ],
            processing_time_ms=round(response.processing_time_ms, 2),
            model_used=response.model_used,
            has_context=response.has_context,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge base not ready: {e}. Run: python scripts/build_index.py",
        )
    except Exception as e:
        log.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))