"""Tests asserting the static interface exists and does not expose secrets.

The page is plain HTML/CSS/ES modules with no build step, so these tests check the
contract that matters: the files are served, they bind to the endpoints the API
actually exposes, and nothing sensitive is embedded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"


@pytest.fixture(scope="module")
def html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def styles() -> str:
    return (STATIC / "style.css").read_text(encoding="utf-8")


def test_static_files_exist():
    for name in ("index.html", "app.js", "style.css"):
        assert (STATIC / name).is_file(), f"{name} is missing"


def test_page_references_assets_relative_to_static(html):
    assert '/static/style.css' in html
    assert '/static/app.js' in html


def test_page_contains_the_qa_and_upload_affordances(html):
    assert 'id="question"' in html
    assert 'id="dropzone"' in html
    assert 'id="messages"' in html
    assert 'id="doc-list"' in html
    assert 'id="health-panel"' in html


def test_upload_accepts_only_pdf_and_markdown(html):
    assert 'accept=".pdf,.md,.markdown"' in html


def test_script_calls_the_endpoints_that_exist(script):
    for endpoint in ("/api/chat", "/api/docs", "/api/health"):
        assert endpoint in script


def test_script_handles_every_chat_event(script):
    """Each event the API emits must have a branch, or it is silently dropped."""
    for event in ("status", "rewritten", "sources", "token", "debug", "error", "done"):
        assert f"'{event}'" in script, f"unhandled chat event: {event}"


def test_script_handles_every_upload_event(script):
    for event in ("progress", "done", "failed", "rejected"):
        assert f"'{event}'" in script, f"unhandled upload event: {event}"


def test_no_secrets_or_upstream_endpoints_are_referenced(script, html, styles):
    blob = script + html + styles
    # The browser must never learn what the server talks to.
    assert "dashscope" not in blob.lower()
    assert "8.134.19.201" not in blob
    assert "api_key" not in blob.lower()
    assert "sk-" not in blob


def test_markdown_output_is_escaped_before_tag_insertion(script):
    """Uploaded documents are untrusted; markup must not be interpreted."""
    assert "escapeHtml" in script
    assert ".replace(/&/g, '&amp;')" in script
    assert "escapeHtml(code" in script


def test_styles_provide_code_list_and_table_rendering(styles):
    for selector in (".turn-answer pre", ".turn-answer ul", ".turn-answer table"):
        assert selector in styles


def test_styles_handle_long_conversations(styles):
    assert "overflow-y: auto" in styles


def test_empty_state_is_dismissed_when_a_turn_starts(script):
    assert "empty-state" in script
    assert "empty.remove()" in script


def test_history_is_bounded_client_side(script):
    """The server keeps no session, so the client must trim to the same window."""
    assert "while (history.length > 6)" in script


def test_debug_panel_is_opt_in(html):
    assert 'id="debug-check"' in html
    # Unchecked by default so retrieval internals are not shown unless asked for.
    assert 'id="debug-check" checked' not in html