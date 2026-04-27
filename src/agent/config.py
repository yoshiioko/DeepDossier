"""Application settings loaded from environment variables / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Required: no defaults ---
    google_api_key: str
    tavily_api_key: str

    # --- Optional: LangSmith ---
    langsmith_api_key: str = ""
    langsmith_project: str = "deep-dossier"
    langsmith_tracing: bool = True

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
    rate_limit_per_minute: int = 10

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "text"  # "text" | "json"

    # --- Server ---
    port: int = 8000

    # --- Cost constants (Gemini 2.5 Flash, USD per 1K tokens) ---
    cost_per_1k_input_tokens: float = 0.000075
    cost_per_1k_output_tokens: float = 0.0003
