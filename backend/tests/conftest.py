"""Test configuration. Integration tests use a dedicated database, data directory and vector collection."""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

_TEST_DATA = Path(tempfile.mkdtemp(prefix="mezvuat-test-"))
# Applied before application settings are first read.
os.environ.update(
    {
        "MEZVUAT_ENV": "test",
        "POSTGRES_DB": "mezvuat_test",
        "POSTGRES_HOST": os.environ.get("TEST_POSTGRES_HOST", "127.0.0.1"),
        "POSTGRES_PORT": os.environ.get("TEST_POSTGRES_PORT", "15433"),
        "QDRANT_URL": os.environ.get("TEST_QDRANT_URL", "http://127.0.0.1:16333"),
        "MEZVUAT_DATA_DIR": str(_TEST_DATA),
        "MEZVUAT_CORPUS_MANIFEST": str(_TEST_DATA / "allowlist.json"),
        "INDEX_VERSION": f"test{os.getpid()}",
        "MEZVUAT_API_KEY": "",
        # Tests never reach a real LLM provider: a placeholder key (respx mocks the endpoint) and no local backend.
        "GROQ_API_KEY": "test-placeholder-not-a-key",
        "INFERENCE_MODE": "direct",
        "DIRECT_PUBLIC_BACKEND": "groq",
        "RUN_CHECKPOINTER": "memory",
    }
)


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False
