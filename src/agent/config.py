"""Application settings loaded from environment variables / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Required: no defaults ---
    google_api_key: str
    tavily_api_key: str

    # --- Model selection ---
    planner_model_name: str = "gemini-2.5-flash"
    synthesize_model_name: str = "gemini-2.5-flash"
    compiler_model_name: str = "gemini-2.5-flash"

    # --- Memory ---
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "dossier_memory"

    # Confidence thresholds - tune via .env
    memory_hit_confidence_threshold: float = 0.7  # min cosine similarity to count as a memory hit
    memory_write_confidence_threshold: float = 0.7  # min SubResult.confidence to upsert into ChromaDB

    # --- Parallelism ---
    max_parallel_researchers: int = 5

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "text"  # "text" | "json"
