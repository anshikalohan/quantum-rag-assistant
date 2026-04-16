"""
Central configuration using Pydantic Settings.
All values can be overridden via environment variables or .env file.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: Literal["groq"] = "groq"
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    llm_model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 1024
    temperature: float = 0.1

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Retrieval ---
    top_k_results: int = 5
    similarity_threshold: float = 0.4
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Paths ---
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    embeddings_dir: Path = Path("data/embeddings")
    faiss_index_path: Path = Path("data/embeddings/faiss_index")

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    log_level: str = "INFO"

    # --- Streamlit ---
    streamlit_port: int = 8501

    @field_validator("llm_model", mode="before")
    @classmethod
    def map_deprecated_models(cls, v: str) -> str:
        if v == "llama3-70b-8192":
            return "llama-3.3-70b-versatile"
        if v == "llama3-8b-8192":
            return "llama-3.1-8b-instant"
        return v

    @field_validator("data_raw_dir", "data_processed_dir", "embeddings_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def faiss_index_file(self) -> Path:
        return Path(f"{self.faiss_index_path}.index")

    @property
    def chunks_file(self) -> Path:
        return self.data_processed_dir / "chunks.json"


# Singleton instance
settings = Settings()