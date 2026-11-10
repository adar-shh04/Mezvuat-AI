"""Centralised configuration. Every setting comes from the environment or `.env`; secrets are never logged."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Both locations so that commands work from the repo root and from backend/. Empty values in .env are ignored,
    # so a blank `POSTGRES_PORT=` keeps the default instead of failing validation.
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore", env_ignore_empty=True)

    mezvuat_env: Literal["dev", "test", "prod"] = "dev"
    mezvuat_log_level: str = "INFO"
    mezvuat_api_key: SecretStr | None = None
    mezvuat_data_dir: Path = Path("data/runtime")
    mezvuat_corpus_manifest: Path = Path("../data/manifests/demo_corpus.json")

    postgres_host: str = "127.0.0.1"  # not "localhost": on Windows psycopg stalls trying ::1 first
    postgres_port: int = 15433
    postgres_db: str = "mezvuat"
    postgres_user: str = "mezvuat"
    postgres_password: SecretStr = SecretStr("mezvuat-dev-only")

    qdrant_url: str = "http://127.0.0.1:16333"
    qdrant_api_key: SecretStr | None = None

    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_small: str = "openai/gpt-oss-20b"

    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    reranker_model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ml_device: Literal["auto", "cpu", "cuda"] = "auto"
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
    def blob_dir(self) -> Path:
        return self.mezvuat_data_dir / "blobs"

    @property
    def lexical_dir(self) -> Path:
        return self.mezvuat_data_dir / "bm25"


@lru_cache
def get_settings() -> Settings:
    return Settings()
