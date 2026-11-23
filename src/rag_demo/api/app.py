"""FastAPI application factory.

The interface is served as static files by the same process, so there is no build
step and no second server. Nothing from the server configuration is exposed to the
browser: the page only ever talks to ``/api/*``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_demo.api.routes import router
from rag_demo.api.services import Services, build_services, initialise
from rag_demo.config import Settings, load_settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static"


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the application.

    Passing ``services`` allows tests to inject fakes without any external
    dependency being reachable.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    resolved = settings or load_settings(root=Path.cwd())
    graph = services or build_services(resolved)

    app = FastAPI(
        title="RAG Demo",
        description="Hybrid retrieval over your own documents",
        version="0.1.0",
    )
    app.state.services = graph
    app.state.settings = resolved
    app.include_router(router)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def create_and_initialise(settings: Settings | None = None) -> FastAPI:
    """Build the application and verify dependencies.

    Verification failures are logged, not raised: the process must come up so the
    interface can report what is misconfigured.
    """
    resolved = settings or load_settings(root=Path.cwd())
    graph = build_services(resolved)
    initialise(graph)
    return create_app(resolved, graph)