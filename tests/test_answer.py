"""Tests for prompt assembly, history bounds and follow-up rewriting."""

from __future__ import annotations

import pytest

from rag_demo.config import Settings
from rag_demo.errors import ConfigurationIssue
from rag_demo.generation.answer import (
    NO_CONTEXT_ANSWER,
    Answerer,
    Conversation,
    Turn,
    build_messages,
    describe_generation_error,
    format_contexts,
    needs_rewrite,
)
from rag_demo.retrieval.hybrid import Context


def _context(
    text: str = "some passage",
    source: str = "paper.pdf",
    page: int = 3,
    breadcrumb: str = "Chapter One",
) -> Context:
    return Context(
        parent_id="p1",
        text=text,
        breadcrumb=breadcrumb,
        page=page,
        source_name=source,
    )


def _settings(**overrides) -> Settings:
    settings = Settings()
    for key, value in overrides.items():
        setattr(settings.llm, key, value)
    return settings


class FakeGenerator:
    """Stands in for the relay, recording what it was asked."""

    def __init__(self, answer: str = "the answer", rewrite: str | None = None, failing: bool = False) -> None:
        self.answer = answer
        self.rewrite = rewrite
        self.failing = failing
        self.calls: list[list] = []

    def stream_chat(self, messages):
        self.calls.append(list(messages))
        if self.failing:
            raise ConfigurationIssue("llm", "bad key")
        for piece in self.answer.split():
            yield piece + " "

    def complete(self, messages):
        self.calls.append(list(messages))
        if self.failing:
            raise ConfigurationIssue("llm", "bad key")
        return self.rewrite if self.rewrite is not None else self.answer


class TestContextFormatting:
    def test_contexts_are_numbered(self):
        rendered = format_contexts([_context("first"), _context("second")])

        assert "[1]" in rendered
        assert "[2]" in rendered

    def test_provenance_is_included(self):
        rendered = format_contexts([_context("body", source="report.pdf", page=7)])

        assert "report.pdf" in rendered
        assert "page 7" in rendered
        assert "Chapter One" in rendered

    def test_missing_page_is_omitted_cleanly(self):
        rendered = format_contexts([_context("body", page=0)])

        assert "page 0" not in rendered
        assert "body" in rendered

    def test_empty_breadcrumb_is_omitted(self):
        rendered = format_contexts([_context("body", breadcrumb="")])

        assert ", \n" not in rendered


class TestPromptAssembly:
    def test_system_prompt_requires_grounding(self):
        messages = build_messages("q", [_context()])

        system = messages[0]
        assert system.role == "system"
        assert "only from the numbered context" in system.content
        assert "do not cover it" in system.content.lower()

    def test_contexts_appear_before_the_question(self):
        messages = build_messages("what is X", [_context("passage about X")])

        user = messages[-1]
        assert user.role == "user"
        assert user.content.index("passage about X") < user.content.index("what is X")

    def test_history_precedes_the_new_question(self):
        conversation = Conversation(turns=[Turn(question="first q", answer="first a")])

        messages = build_messages("second q", [_context()], conversation)

        contents = [m.content for m in messages]
        assert any("first q" in c for c in contents)
        assert contents[-1].endswith("second q")

    def test_no_history_means_no_history_messages(self):
        messages = build_messages("q", [_context()])

        assert len(messages) == 2


class TestConversationBounds:
    def test_history_is_capped(self):
        conversation = Conversation(max_turns=3)

        for index in range(10):
            conversation.append(f"q{index}", f"a{index}")

        assert len(conversation.turns) == 3
        # The newest turns survive.
        assert conversation.turns[-1].question == "q9"
        assert conversation.turns[0].question == "q7"

    def test_messages_alternate_roles(self):
        conversation = Conversation()
        conversation.append("q", "a")

        roles = [m.role for m in conversation.as_messages()]

        assert roles == ["user", "assistant"]


class TestRewriteDecision:
    def test_first_turn_is_never_rewritten(self):
        assert needs_rewrite("what is retrieval", Conversation()) is False

    def test_anaphoric_follow_up_is_rewritten(self):
        conversation = Conversation(turns=[Turn("tell me about RRF", "RRF fuses ranks")])

        assert needs_rewrite("why does it work?", conversation) is True

    def test_chinese_anaphora_is_rewritten(self):
        conversation = Conversation(turns=[Turn("介绍一下混合检索", "混合检索...")])

        assert needs_rewrite("它的缺点是什么？", conversation) is True

    def test_self_contained_question_is_not_rewritten(self):
        conversation = Conversation(turns=[Turn("q", "a")])
        question = "How does reciprocal rank fusion combine two different ranked lists?"

        assert needs_rewrite(question, conversation) is False

    def test_very_short_question_is_rewritten(self):
        conversation = Conversation(turns=[Turn("q", "a")])

        assert needs_rewrite("more?", conversation) is True


class TestAnswerer:
    def test_answer_uses_contexts(self):
        generator = FakeGenerator(answer="grounded answer")
        answerer = Answerer(_settings(), generator)

        text = answerer.answer("q", [_context("the passage")])

        assert "grounded" in text
        assert any("the passage" in m.content for m in generator.calls[0])

    def test_empty_contexts_short_circuit_without_calling_model(self):
        generator = FakeGenerator()
        answerer = Answerer(_settings(), generator)

        text = answerer.answer("q", [])

        assert text == NO_CONTEXT_ANSWER
        assert generator.calls == []

    def test_rewrite_is_skipped_for_first_turn(self):
        generator = FakeGenerator()
        answerer = Answerer(_settings(), generator)

        query, rewritten = answerer.rewrite("what is retrieval", Conversation())

        assert rewritten is False
        assert query == "what is retrieval"
        assert generator.calls == []

    def test_rewrite_is_used_for_follow_up(self):
        generator = FakeGenerator(rewrite="why does reciprocal rank fusion work")
        answerer = Answerer(_settings(), generator)
        conversation = Conversation(turns=[Turn("tell me about RRF", "RRF fuses ranks")])

        query, rewritten = answerer.rewrite("why does it work?", conversation)

        assert rewritten is True
        assert query == "why does reciprocal rank fusion work"

    def test_rewrite_strips_quotes(self):
        generator = FakeGenerator(rewrite='"a quoted rewrite"')
        answerer = Answerer(_settings(), generator)
        conversation = Conversation(turns=[Turn("q", "a")])

        query, _ = answerer.rewrite("it?", conversation)

        assert query == "a quoted rewrite"

    def test_failing_rewrite_falls_back_to_original(self):
        generator = FakeGenerator(failing=True)
        answerer = Answerer(_settings(), generator)
        conversation = Conversation(turns=[Turn("q", "a")])

        query, rewritten = answerer.rewrite("why does it work?", conversation)

        assert query == "why does it work?"
        assert rewritten is False

    def test_empty_rewrite_falls_back_to_original(self):
        generator = FakeGenerator(rewrite="   ")
        answerer = Answerer(_settings(), generator)
        conversation = Conversation(turns=[Turn("q", "a")])

        query, rewritten = answerer.rewrite("it?", conversation)

        assert query == "it?"
        assert rewritten is False

    def test_overlong_rewrite_is_rejected(self):
        generator = FakeGenerator(rewrite="x" * 700)
        answerer = Answerer(_settings(), generator)
        conversation = Conversation(turns=[Turn("q", "a")])

        query, rewritten = answerer.rewrite("it?", conversation)

        assert query == "it?"
        assert rewritten is False


class TestErrorDescriptions:
    def test_configuration_error_is_actionable(self):
        message = describe_generation_error(ConfigurationIssue("llm", "bad key"))

        assert "misconfigured" in message.lower()

    def test_unreachable_error_mentions_service(self):
        from rag_demo.errors import Unavailable

        message = describe_generation_error(Unavailable("llm", "connection refused"))

        assert "unreachable" in message.lower()

    def test_unknown_error_is_still_descriptive(self):
        message = describe_generation_error(RuntimeError("boom"))

        assert "boom" in message