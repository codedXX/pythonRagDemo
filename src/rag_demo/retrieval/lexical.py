"""Lexical (keyword) retrieval with Chinese word segmentation.

Why BM25 lives in Python rather than in Milvus' native sparse vectors, given that
the server does support them:

* The whole point of this project is being able to see what each channel did. A
  server-side fused search hides the two rankings, and the fusion parameter cannot
  be inspected or tuned.
* Segmentation is ours: a user dictionary and a stopword list can be edited and
  their effect observed immediately.

The trade-off is scale — this holds the corpus in memory — which is acceptable for
the tens-of-documents target and is why the interface is kept narrow, so a native
implementation can replace it later.

The corpus is rebuilt from Milvus rather than from files on disk on purpose: if the
two channels disagreed about what exists, fused results would be unexplainable.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import jieba
import bm25s

# Latin words, numbers and CJK runs. Used only for the fields where jieba is not
# appropriate; jieba handles the CJK case.
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?")

DEFAULT_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "和",
        "是",
        "在",
        "我",
        "有",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "to",
        "in",
        "is",
        "are",
        "for",
        "on",
        "with",
        "as",
        "by",
        "at",
        "be",
        "this",
        "that",
        "it",
    }
)


@dataclass(slots=True)
class LexicalHit:
    """One lexical search result."""

    id: str
    score: float
    rank: int


@dataclass(slots=True)
class LexicalIndex:
    """In-memory BM25 index over child units.

    Rebuilt wholesale rather than updated incrementally: at the target corpus size a
    rebuild takes well under a second, while incremental bookkeeping would add a
    class of consistency bugs for no measurable gain.
    """

    user_dictionary: Sequence[str] = ()
    stopwords: frozenset[str] = DEFAULT_STOPWORDS
    k1: float = 1.5
    b: float = 0.75

    _ids: list[str] = field(default_factory=list)
    _texts: list[str] = field(default_factory=list)
    _retriever: object | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._configure_segmentation()

    def _configure_segmentation(self) -> None:
        for word in self.user_dictionary:
            if word:
                jieba.add_word(word)

    # ------------------------------------------------------------------ building

    def rebuild(self, rows: Iterable[dict]) -> None:
        """Replace the index contents from ``rows`` (each with ``id`` and ``text``)."""
        ids: list[str] = []
        texts: list[str] = []
        tokens: list[list[str]] = []

        for row in rows:
            text = row.get("text") or ""
            if not text.strip():
                continue
            ids.append(str(row["id"]))
            texts.append(text)
            tokens.append(self.tokenize(text))

        with self._lock:
            self._ids = ids
            self._texts = texts
            if tokens:
                # build on the tokenized corpus; bm25s keeps its own structures
                self._retriever = bm25s.BM25(k1=self.k1, b=self.b)
                self._retriever.index(tokens, show_progress=False)
            else:
                self._retriever = None

    def add(self, rows: Sequence[dict]) -> None:
        """Append rows to the index.

        Implemented as a rebuild because BM25 statistics (document frequencies)
        change with every added document, so an append would still require
        recomputation.
        """
        combined = list(zip(self._ids, self._texts, strict=True))
        combined.extend((str(row["id"]), row.get("text") or "") for row in rows)
        self.rebuild([{"id": i, "text": t} for i, t in combined])

    # ----------------------------------------------------------------- searching

    def search(self, query: str, limit: int = 40) -> list[LexicalHit]:
        """Return the best ``limit`` matches, most relevant first."""
        with self._lock:
            retriever = self._retriever
            ids = list(self._ids)

        if retriever is None or not ids or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        k = min(max(1, limit), len(ids))
        try:
            results, scores = retriever.retrieve([query_tokens], k=k, show_progress=False)
        except Exception:
            return []

        hits: list[LexicalHit] = []
        for rank, (index, score) in enumerate(zip(results[0], scores[0], strict=True)):
            if index < 0 or index >= len(ids):
                continue
            # bm25s may return padded indices for missing results.
            if score <= 0 and rank > 0 and float(score) == 0.0:
                continue
            hits.append(
                LexicalHit(id=ids[int(index)], score=float(score), rank=rank)
            )
        return hits

    # -------------------------------------------------------------- segmentation

    def tokenize(self, text: str) -> list[str]:
        """Segment text for indexing and querying.

        Chinese is segmented with ``cut_for_search``, which also emits sub-words so
        a query for a word inside a longer compound still matches. Latin words and
        numbers are kept whole.
        """
        if not text:
            return []

        tokens: list[str] = []
        for chunk in _split_scripts(text):
            if _is_cjk(chunk):
                tokens.extend(
                    token
                    for token in jieba.cut_for_search(chunk)
                    if token.strip() and token not in self.stopwords
                )
            else:
                tokens.extend(
                    match.group(0).lower()
                    for match in _LATIN.finditer(chunk)
                    if match.group(0).lower() not in self.stopwords
                )
        return [token for token in tokens if len(token) > 1 or _is_cjk(token)]

    # -------------------------------------------------------------------- access

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._ids)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._ids)

    def text_of(self, identifier: str) -> str | None:
        with self._lock:
            try:
                return self._texts[self._ids.index(identifier)]
            except ValueError:
                return None


def _is_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _split_scripts(text: str) -> list[str]:
    """Split text into CJK and non-CJK runs.

    jieba applied to a Latin run would fragment identifiers and numbers, so the two
    are separated first.
    """
    runs: list[str] = []
    current = ""
    current_is_cjk = None
    for char in text:
        char_is_cjk = "\u4e00" <= char <= "\u9fff"
        if current_is_cjk is None or char_is_cjk == current_is_cjk:
            current += char
            current_is_cjk = char_is_cjk
        else:
            if current:
                runs.append(current)
            current = char
            current_is_cjk = char_is_cjk
    if current:
        runs.append(current)
    return runs