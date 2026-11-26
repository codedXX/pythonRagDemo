"""将 PDF 或 Markdown 导入可检索的文档库。"""

import time
from dataclasses import dataclass

from rag_demo.chunking.chunker import Chunker
from rag_demo.config import Settings
from rag_demo.embedding.provider import Embedder
from rag_demo.parsers.base import check_extension, document_id
from rag_demo.parsers.markdown import parse_markdown
from rag_demo.parsers.pdf import parse_pdf
from rag_demo.store.milvus_store import MilvusStore
from rag_demo.store.registry import DocumentRegistry


@dataclass(slots=True)
class IngestResult:
    doc_id: str
    name: str
    parent_count: int
    child_count: int
    char_count: int
    duration_seconds: float


class IngestionError(RuntimeError):
    """带有失败环节和具体原因的导入错误。"""


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        store: MilvusStore,
        registry: DocumentRegistry,
        embedder: Embedder,
        chunker: Chunker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.registry = registry
        self.embedder = embedder
        self.chunker = chunker or Chunker(settings.chunking, embedder=embedder)

    def ingest(self, name: str, content: bytes, doc_id: str | None = None) -> IngestResult:
        started = time.monotonic()
        identifier = doc_id or document_id(name, content)
        self.registry.begin(identifier, name)
        stage = "parsing"
        try:
            suffix = check_extension(name)
            if not content:
                raise ValueError(f"{name} is empty")
            if suffix == ".pdf" or content.startswith(b"%PDF-"):
                document = parse_pdf(name, content, self.settings.parsing.min_chars_per_page)
            else:
                document = parse_markdown(name, content)
            if document.is_empty():
                raise ValueError(f"{name} produced no extractable text")

            stage = "chunking"
            chunks = self.chunker.chunk(document, identifier)
            if not chunks.children:
                raise ValueError(f"{name} produced no retrievable units")

            stage = "embedding"
            texts = [child.text for child in chunks.children]
            vectors = self.embedder.embed_texts(texts)
            if len(vectors) != len(texts):
                raise ValueError(f"asked for {len(texts)} vectors but received {len(vectors)}")
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) != 1:
                raise ValueError(f"inconsistent vector dimensions: {sorted(dimensions)}")

            stage = "persisting"
            for previous in self.registry.find_by_name(name):
                if previous.doc_id != identifier:
                    self.store.delete_document(previous.doc_id)
                    self.registry.remove(previous.doc_id)

            self.store.delete_document(identifier)
            self.store.insert_units(chunks.children, chunks.parents, vectors)
            parents, children, chars = len(chunks.parents), len(chunks.children), chunks.char_count
            try:
                self.store.activate_document(identifier)
                self.registry.mark_active(
                    identifier, parent_count=parents, child_count=children, char_count=chars
                )
            except Exception:
                # 激活会分别更新两个集合，登记失败时也要撤回已经发布的内容。
                self.store.delete_document(identifier)
                raise
        except Exception as error:
            message = f"{stage}: {error}"
            self.registry.mark_failed(identifier, message)
            raise IngestionError(message) from error

        return IngestResult(identifier, name, parents, children, chars, time.monotonic() - started)


def remove_document(doc_id: str, store: MilvusStore, registry: DocumentRegistry) -> bool:
    if registry.get(doc_id) is None:
        return False
    store.delete_document(doc_id)
    registry.remove(doc_id)
    return True
