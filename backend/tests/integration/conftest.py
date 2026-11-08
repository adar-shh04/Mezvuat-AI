from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psycopg
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from mezvuat.core.config import get_settings
from tests.conftest import port_open

BACKEND = Path(__file__).resolve().parents[2]


def _services_available() -> bool:
    qdrant = urlsplit(os.environ["QDRANT_URL"])
    return port_open(os.environ["POSTGRES_HOST"], int(os.environ["POSTGRES_PORT"])) and port_open(
        qdrant.hostname or "localhost", qdrant.port or 6333
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if _services_available():
        return
    skip = pytest.mark.skip(reason="compose Postgres/Qdrant not reachable")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def migrated_database() -> Iterator[None]:
    settings = get_settings()
    admin = (
        f"host={settings.postgres_host} port={settings.postgres_port} dbname=postgres "
        f"user={settings.postgres_user} password={settings.postgres_password.get_secret_value()}"
    )
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(f'DROP DATABASE IF EXISTS "{settings.postgres_db}" WITH (FORCE)')
        connection.execute(f'CREATE DATABASE "{settings.postgres_db}"')
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "migrations"))
    command.upgrade(config, "head")
    yield
    from mezvuat.persistence.db import get_engine

    get_engine().dispose()
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(f'DROP DATABASE IF EXISTS "{settings.postgres_db}" WITH (FORCE)')


@pytest.fixture
def clean_db(migrated_database: None) -> Iterator[None]:
    from mezvuat.persistence.db import get_engine

    yield
    with get_engine().begin() as connection:
        connection.execute(text("TRUNCATE workspace CASCADE"))


@pytest.fixture
def client(clean_db: None) -> Iterator[TestClient]:
    from mezvuat.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="session", autouse=True)
def drop_test_collections() -> Iterator[None]:
    yield
    if not _services_available():
        return
    from mezvuat.api.deps import get_dense_index

    with contextlib.suppress(Exception):
        get_dense_index().drop_collection()


@pytest.fixture(scope="session")
def http_client() -> Iterator[httpx.Client]:
    with httpx.Client() as client_:
        yield client_
