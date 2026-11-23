"""Answer generation: prompt assembly, follow-up rewriting, history bounds.

Keeping these three concerns here rather than in the API layer means they can be
tested without HTTP, and the interface stays a thin transport shell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_demo.config import Settings
from rag_demo.errors import ProviderError
from rag_demo.generation.llm import Generator, Message
from rag_demo.retrieval.hybrid import Context

SYSTEM_PROMPT = """You answer questions about the user's own documents.

Rules:
- Answer only from the numbered context passages provided.
- Cite passages inline using their bracket number, for example [1] or [2][3].
- If the passages do not contain the answer, say plainly that the provided
  documents do not cover it. Do not fill the gap from general knowledge.
- Reply in the same language as the question.
- Be concise. Do not restate the question or narrate what you are doing."""

NO_CONTEXT_ANSWER = (
    "There is no content available to answer this question yet. "
    "Upload a PDF or Markdown document first."
)

# Words that usually signal the question leans on an earlier turn. Used to skip the
# rewrite call when it is clearly unnecessary.
_ANAPHORA = re.compile(
    r"(它|他|她|这个|那个|这些|那些|上面|上述|前面|刚才|该|此|其|这样|那样"
    r"|it|its|they|them|this|that|these|those|above|former|latter|previous)",
    re.IGNORECASE,
)

REWRITE_PROMPT = """Rewrite the follow-up question so it can be understood on its own.

Use the conversation history to resolve references such as "it", "that document" or
"the second approach". Keep the original meaning, keep the original language, and
do not answer the question.

Reply with the rewritten question only, with no preamble and no quotation marks."""


@dataclass(slots=True)
class Turn:
    """One conversational exchange."""

    question: str
    answer: str


@dataclass(slots=True)
class Conversation:
    """Bounded conversation history.

    Retaining the newest turns only: an unbounded history would eventually exceed
    the context window and quietly degrade answer quality.
    """

    turns: list[Turn] = field(default_factory=list)
    max_turns: int = 6

    def append(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))
        if len(self.turns) > self.max_turns:
            # Drop the oldest, which are the least relevant to a follow-up.
            del self.turns[: len(self.turns) - self.max_turns]

    def is_empty(self) -> bool:
        return not self.turns

    def as_messages(self) -> list[Message]:
        messages: list[Message] = []
        for turn in self.turns:
            messages.append(Message(role="user", content=turn.question))
            messages.append(Message(role="assistant", content=turn.answer))
        return messages


def needs_rewrite(question: str, conversation: Conversation) -> bool:
    """Whether a follow-up needs to be made self-contained before retrieval.

    Rewriting costs a round trip and adds latency before the first token, so it is
    skipped when the question is already self-contained: no history, no anaphora and
    enough length to carry its own subject.
    """
    if conversation.is_empty():
        return False
    stripped = question.strip()
    if len(stripped) >= 40 and not _ANAPHORA.search(stripped):
        return False
    return bool(_ANAPHORA.search(stripped)) or len(stripped) < 12


def format_contexts(contexts: list[Context]) -> str:
    """Render contexts as numbered passages for the prompt."""
    blocks: list[str] = []
    for index, context in enumerate(contexts, start=1):
        location = f", page {context.page}" if context.page else ""
        heading = f", {context.breadcrumb}" if context.breadcrumb else ""
        blocks.append(
            f"[{index}] source: {context.source_name}{location}{heading}\n{context.text}"
        )
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    contexts: list[Context],
    conversation: Conversation | None = None,
) -> list[Message]:
    """Assemble the message list for the answering call."""
    messages: list[Message] = [Message(role="system", content=SYSTEM_PROMPT)]

    if conversation is not None:
        messages.extend(conversation.as_messages())

    messages.append(
        Message(
            role="user",
            content=(
                f"Context passages:\n\n{format_contexts(contexts)}\n\n"
                f"Question: {question}"
            ),
        )
    )
    return messages


class Answerer:
    """Produces grounded answers, rewriting follow-ups when needed."""

    def __init__(self, settings: Settings, generator: Generator | None = None) -> None:
        self._settings = settings
        self._generator = generator or Generator(settings.llm)

    @property
    def generator(self) -> Generator:
        return self._generator

    def rewrite(self, question: str, conversation: Conversation) -> tuple[str, bool]:
        """Return ``(query_for_retrieval, was_rewritten)``.

        A failing rewrite must not block the answer: the original question is used
        and retrieval proceeds, which is worse but still useful.
        """
        if not needs_rewrite(question, conversation):
            return question, False

        history = conversation.as_messages()
        messages = [
            Message(role="system", content=REWRITE_PROMPT),
            *history[-4:],
            Message(role="user", content=question),
        ]
        try:
            rewritten = self._generator.complete(messages).strip()
        except ProviderError:
            return question, False
        except Exception:
            return question, False

        # A model that answers instead of rewriting, or returns nothing, should not
        # replace a perfectly good question.
        if not rewritten or len(rewritten) > 600:
            return question, False
        rewritten = rewritten.strip().strip('"').strip("“”").strip()
        if not rewritten:
            return question, False
        return rewritten, True

    def stream_answer(
        self,
        question: str,
        contexts: list[Context],
        conversation: Conversation | None = None,
    ):
        """Yield answer text incrementally.

        An empty context set short-circuits rather than calling the model, so an
        empty knowledge base is answered honestly instead of hallucinated.
        """
        if not contexts:
            yield NO_CONTEXT_ANSWER
            return

        messages = build_messages(question, contexts, conversation)
        yield from self._generator.stream_chat(messages)

    def answer(
        self,
        question: str,
        contexts: list[Context],
        conversation: Conversation | None = None,
    ) -> str:
        return "".join(self.stream_answer(question, contexts, conversation))


def describe_generation_error(error: Exception) -> str:
    """Turn an exception into something a user can act on."""
    from rag_demo.errors import ConfigurationIssue, ContractMismatch, Unavailable

    if isinstance(error, ConfigurationIssue):
        return f"Generation is misconfigured: {error.message}"
    if isinstance(error, Unavailable):
        return f"The generation service is unreachable: {error.message}"
    if isinstance(error, ContractMismatch):
        return f"The generation service returned an unexpected response: {error.message}"
    return f"Generation failed: {type(error).__name__}: {error}"