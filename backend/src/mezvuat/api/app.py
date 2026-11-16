"""FastAPI application."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from mezvuat.api.deps import get_dense_index, negotiate_modelmesh, preload_nli
from mezvuat.api.routers import documents, runs, search, workspaces
from mezvuat.api.schemas import HealthOut
from mezvuat.core.config import get_settings
from mezvuat.core.errors import MezvuatError, RetrievalUnavailableError
from mezvuat.core.ids import new_id
from mezvuat.core.logging import configure_logging, request_id_var
from mezvuat.persistence.db import get_engine

log = logging.getLogger("mezvuat.api")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_:.-]{1,128}$")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.mezvuat_log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # The NLI model takes tens of seconds to load on CPU; load it before the first run needs it.
        if settings.nli_enabled and settings.mezvuat_env != "test":
            threading.Thread(target=preload_nli, name="nli-preload", daemon=True).start()
        if settings.inference_mode != "direct":
            negotiate_modelmesh()
        yield

    app = FastAPI(title="Mezvuat AI", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        incoming = request.headers.get("x-request-id", "")
        value = incoming if _SAFE_REQUEST_ID.match(incoming) else new_id("req")
        token = request_id_var.set(value)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = value
        return response

    @app.exception_handler(MezvuatError)
    async def handle_domain_error(request: Request, exc: MezvuatError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": {"code": exc.code, "message": exc.message, "request_id": request_id_var.get()}},
        )

    @app.get("/health", response_model=HealthOut, tags=["health"])
    def health(response: Response) -> HealthOut:
        try:
            with get_engine().connect() as connection:
                connection.execute(text("SELECT 1"))
            postgres = "ok"
        except OperationalError:
            postgres = "down"
        try:
            get_dense_index().ping()
            qdrant = "ok"
        except RetrievalUnavailableError:
            qdrant = "down"
        healthy = postgres == "ok" and qdrant == "ok"
        if not healthy:
            response.status_code = 503
        return HealthOut(status="ok" if healthy else "degraded", postgres=postgres, qdrant=qdrant)

    app.include_router(workspaces.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(runs.router)
    return app


app = create_app()
