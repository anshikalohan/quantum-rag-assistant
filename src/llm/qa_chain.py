"""
RAG Question-Answering chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from src.llm.prompt_templates import (
    FALLBACK_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
)
from src.retrieval.retriever import RetrievalResult, Retriever
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class QAResponse:

    question: str
    answer: str
    sources: List[str]
    retrieval_results: List[RetrievalResult]
    processing_time_ms: float
    model_used: str
    has_context: bool = True

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_scores": [
                {"source": r.source, "score": round(r.score, 4)}
                for r in self.retrieval_results
            ],
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_used": self.model_used,
            "has_context": self.has_context,
        }


class QAChain:

    def __init__(self, retriever: Retriever = None):
        self.retriever = retriever or Retriever()
        self._llm_client = None

    def _get_llm_client(self):
        """Lazy-initialize the Groq client on first use."""
        if self._llm_client is not None:
            return self._llm_client

        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "  1. Get a free key at https://console.groq.com\n"
                "  2. Add GROQ_API_KEY=your_key to your .env file"
            )

        try:
            from groq import Groq
            self._llm_client = Groq(api_key=settings.groq_api_key)
            log.info(f"✓ Groq client initialized (model: {settings.llm_model})")
        except ImportError:
            raise ImportError(
                "Groq package not found. Install it with:\n"
                "  pip install groq"
            )

        return self._llm_client

    def answer(
        self,
        question: str,
        top_k: int = None,
        threshold: float = None,
    ) -> QAResponse:
        start_time = time.time()

        # Step 1: Retrieve relevant context
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k or settings.top_k_results,
            threshold=threshold or settings.similarity_threshold,
        )

        # Step 2: Build prompt
        has_context = len(results) > 0

        if has_context:
            context = self.retriever.format_context(results)
            user_message = RAG_USER_TEMPLATE.format(
                context=context,
                question=question,
            )
            system_prompt = RAG_SYSTEM_PROMPT
        else:
            user_message = FALLBACK_PROMPT.format(question=question)
            system_prompt = "You are a helpful Quantum Computing tutor."

        # Step 3: Generate answer via Groq
        answer_text = self._generate(system_prompt, user_message)

        # Step 4: Build response
        sources = list({r.source for r in results}) if results else []
        elapsed_ms = (time.time() - start_time) * 1000

        return QAResponse(
            question=question,
            answer=answer_text,
            sources=sources,
            retrieval_results=results,
            processing_time_ms=elapsed_ms,
            model_used=settings.llm_model,
            has_context=has_context,
        )

    def _generate(self, system_prompt: str, user_message: str) -> str:
        client = self._get_llm_client()

        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content

        except Exception as e:
            log.error(f"Groq generation failed: {e}")
            raise RuntimeError(f"LLM generation error: {e}") from e