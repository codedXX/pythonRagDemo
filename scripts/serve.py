#!/usr/bin/env python
"""Run the API server.

    uv run python scripts/serve.py

Dependency verification runs first and its result is logged; a misconfigured provider
does not prevent the process from starting, because the interface needs to be
reachable in order to report what is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn  # noqa: E402

from rag_demo.api.app import create_and_initialise  # noqa: E402
from rag_demo.config import ConfigurationError, load_settings  # noqa: E402


def main(argv: list[str]) -> int:
    root = Path.cwd()
    try:
        settings = load_settings(root=root)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1

    if not (root / "config.yaml").exists():
        print(
            "note: config.yaml not found. Copy config.example.yaml to config.yaml "
            "and fill in the API keys.",
            file=sys.stderr,
        )

    host = settings.api.host
    port = settings.api.port
    for argument in argv:
        if argument.startswith("--port="):
            port = int(argument.split("=", 1)[1])
        elif argument.startswith("--host="):
            host = argument.split("=", 1)[1]

    app = create_and_initialise(settings)
    print(f"serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))