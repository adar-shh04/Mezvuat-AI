"""Centralised configuration. Every setting comes from the environment or `.env`; secrets are never logged."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Both locations so that commands work from the repo root and from backend/. Empty values in .env are ignored,
    # so a blank `POSTGRES_PORT=` keeps the default instead of failing validation.
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore", env_ignore_empty=True)

    mezvuat_env: Literal["dev", "test", "prod"] = "dev"
    mezvuat_log_level: str = "INFO"
    mezvuat_api_key: SecretStr | None = None
    # Server secret for HMAC-derived identifiers sent to ModelMesh (the cache namespace). Required in ModelMesh modes.
    mezvuat_secret_key: SecretStr | None = None
    mezvuat_data_dir: Path = Path("data/runtime")
    mezvuat_corpus_manifest: Path = Path("../data/manifests/demo_corpus.json")

    postgres_host: str = "127.0.0.1"  # not "localhost": on Windows psycopg stalls trying ::1 first
    postgres_port: int = 15433
    postgres_db: str = "mezvuat"
    postgres_user: str = "mezvuat"
    postgres_password: SecretStr = SecretStr("mezvuat-dev-only")

    qdrant_url: str = "http://127.0.0.1:16333"
    qdrant_api_key: SecretStr | None = None

    # Inference (ADR-0011). `modelmesh` has no fallback; `modelmesh_with_fallback` uses the direct backends below.
    inference_mode: Literal["direct", "modelmesh", "modelmesh_with_fallback"] = "direct"
    # Unset by default on purpose: nothing talks to ModelMesh unless this is configured. Without the `/v1` suffix.
    modelmesh_base_url: str | None = None
    modelmesh_api_key: SecretStr | None = None
    # Per-request socket timeout ceiling; each call also gets at most its remaining deadline.
    modelmesh_timeout_s: float = Field(default=60.0, gt=0)
    modelmesh_contract_ttl_s: float = Field(default=600.0, gt=0)
    modelmesh_calibration_ttl_s: float = Field(default=3600.0, gt=0)
    modelmesh_breaker_failures: int = Field(default=5, gt=0)
    modelmesh_breaker_open_s: float = Field(default=30.0, gt=0)
    # `min_quality` per task type, as JSON. Sent only where `GET /v1/routing/calibration` says `calibrated: true`.
    modelmesh_min_quality: dict[str, float] = Field(default_factory=dict)
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_small: str = "openai/gpt-oss-20b"
    groq_model_large: str = "openai/gpt-oss-120b"
    # Decision D1: private workspaces only ever use a local backend. Unset means private workspaces cannot run.
    ollama_base_url: str | None = None
    ollama_model: str = "llama3.2:3b"
    ollama_model_large: str | None = None
    # Which direct backend serves public-corpus workspaces. `ollama` keeps development off the Groq quota.
    direct_public_backend: Literal["groq", "ollama"] = "groq"

    # Research runs (ADR-0008 bounds).
    run_deadline_s: float = Field(default=120.0, gt=0)
    run_max_llm_calls: int = Field(default=12, gt=0)
    run_max_tokens: int = Field(default=150_000, gt=0)
    run_concurrency: int = Field(default=2, gt=0)
    run_checkpointer: Literal["postgres", "memory"] = "postgres"

    # Langfuse (ADR-0012, decision D3). Unset keys mean no tracing. Content is masked unless capture is turned on.
    langfuse_host: str | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_capture_content: bool = False

    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    reranker_model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ml_device: Literal["auto", "cpu", "cuda"] = "auto"
    # Claim verification (ADR-0009 amendment, 2026-11-16). The NLI model runs on CPU unless told otherwise.
    nli_enabled: bool = True
    nli_model_id: str = "cross-encoder/nli-deberta-v3-small"
    nli_device: Literal["auto", "cpu", "cuda"] = "cpu"
    # Written by `mezvuat-dataset nli-calibrate`. Without it the thresholds are the provisional, uncalibrated rule.
    nli_thresholds_file: Path = Path("../data/domain/nli/v1/thresholds.json")
    # The LLM judge for claims NLI leaves undecided: off until it has been checked against the owner's labels.
    verify_llm_judge: bool = False
    verify_max_judge_calls: int = Field(default=3, ge=0)
    index_version: str = "v1"

    upload_max_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    pdf_max_pages: int = Field(default=1000, gt=0)
    url_fetch_timeout_s: float = Field(default=15.0, gt=0)
    url_fetch_max_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    worker_poll_interval_s: float = Field(default=1.0, gt=0)
    job_max_attempts: int = Field(default=3, gt=0)

    @property
    def database_url(self) -> str:
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def psycopg_dsn(self) -> str:
        """libpq connection string for the LangGraph checkpointer, which uses psycopg directly."""
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"postgresql://{quote(self.postgres_user, safe='')}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def modelmesh_url(self) -> str | None:
        if not self.modelmesh_base_url:
            return None
        return self.modelmesh_base_url.rstrip("/").removesuffix("/v1")

    @property
    def blob_dir(self) -> Path:
        return self.mezvuat_data_dir / "blobs"

    @property
    def lexical_dir(self) -> Path:
        return self.mezvuat_data_dir / "bm25"


@lru_cache
def get_settings() -> Settings:
    return Settings()
