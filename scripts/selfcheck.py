#!/usr/bin/env python
"""Verify external dependencies before using the system.

Run this first, before ingesting anything:

    uv run python scripts/selfcheck.py

It answers three questions that cannot be settled by reading configuration:

* Does the embedding model exist, and how many dimensions does it return?
  The dimension is fixed when a collection is created, so guessing wrong means
  rebuilding the knowledge base.
* Does the reranking model exist, and does it rank a case with an obvious answer
  sensibly?
* Does the generation model exist, are the credentials accepted, and does it
  actually stream?

Exit code is 0 when every dependency is healthy and 1 otherwise, so this can gate
a deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.config import ConfigurationError, load_settings  # noqa: E402
from rag_demo.selfcheck import format_report, run_selfcheck, write_meta  # noqa: E402


def main(argv: list[str]) -> int:
    root = Path.cwd()
    skip_llm = "--skip-llm" in argv

    try:
        settings = load_settings(root=root)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1

    config_path = root / "config.yaml"
    if not config_path.exists():
        print(
            f"note: {config_path} does not exist; using defaults and environment "
            "variables only. Copy config.example.yaml to config.yaml and fill in "
            "the API keys.",
            file=sys.stderr,
        )

    print(f"checking dependencies (config: {config_path})")
    print()
    report = run_selfcheck(settings, include_llm=not skip_llm)
    print(format_report(report))

    # Record the discovered dimension so a later run can detect a model change.
    if report.embedding_dimension is not None:
        try:
            write_meta(settings, report.embedding_dimension)
            print("recorded embedding model and dimension in the meta collection")
        except Exception as exc:
            print(f"could not record metadata: {exc}", file=sys.stderr)

    return 0 if report.healthy else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))