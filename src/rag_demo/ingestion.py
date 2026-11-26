"""将 PDF 或 Markdown 导入可检索的文档库。"""

import time  # 使用单调时钟测量导入耗时。
from dataclasses import dataclass  # 用数据类表示一次导入的结果。

from rag_demo.chunking.chunker import Chunker  # 引入生成父块和子块的分块器。
from rag_demo.config import Settings  # 读取解析与分块配置。
from rag_demo.embedding.provider import Embedder  # 引入批量生成向量的编码器。
from rag_demo.parsers.base import check_extension, document_id  # 校验扩展名并为文档生成稳定标识。
from rag_demo.parsers.markdown import parse_markdown  # 引入 Markdown 结构解析器。
from rag_demo.parsers.pdf import parse_pdf  # 引入保留页码的 PDF 解析器。
from rag_demo.store.milvus_store import MilvusStore  # 通过 Milvus 保存父块、子块和向量。
from rag_demo.store.registry import DocumentRegistry  # 通过登记表记录文档的导入状态。


@dataclass(slots=True)  # 按字段生成初始化方法并使用固定的属性槽。
class IngestResult:  # 汇总导入后的文档标识、规模与耗时。
    doc_id: str  # 供查询和删除使用的文档标识。
    name: str  # 保留上传时的文件名。
    parent_count: int  # 记录提供上下文的父块数量。
    child_count: int  # 记录参与召回的子块数量。
    char_count: int  # 记录分块结果统计的字符数。
    duration_seconds: float  # 以秒为单位记录整次导入耗时。


class IngestionError(RuntimeError):  # 用专门的异常将失败阶段传递给调用方。
    """带有失败环节和具体原因的导入错误。"""


class IngestionPipeline:  # 串联解析、分块、向量生成与入库。
    def __init__(  # 组装单份文档导入所需的依赖。
        self,  # 接收当前导入流程实例。
        settings: Settings,  # 传入解析阈值与分块参数。
        store: MilvusStore,  # 传入保存文档内容的向量库。
        registry: DocumentRegistry,  # 传入跟踪导入状态的登记表。
        embedder: Embedder,  # 传入子块文本使用的向量编码器。
        chunker: Chunker | None = None,  # 允许调用方指定分块器。
    ) -> None:  # 初始化流程依赖。
        self.settings = settings  # 保存导入时使用的项目配置。
        self.store = store  # 保留读写 Milvus 集合的入口。
        self.registry = registry  # 保留文档状态读写入口。
        self.embedder = embedder  # 保留批量向量编码入口。
        self.chunker = chunker or Chunker(settings.chunking, embedder=embedder)  # 优先复用传入的分块器，否则按配置创建。

    def ingest(self, name: str, content: bytes, doc_id: str | None = None) -> IngestResult:  # 导入一个具名文件并返回统计信息。
        started = time.monotonic()  # 记录起点以计算整次导入的耗时。
        identifier = doc_id or document_id(name, content)  # 优先采用指定标识，否则由文件名和内容计算。
        self.registry.begin(identifier, name)  # 先将该文档登记为导入中。
        stage = "parsing"  # 将当前环节标记为解析，以便报告错误。
        try:  # 统一记录后续各环节抛出的异常。
            suffix = check_extension(name)  # 验证文件类型并取得规范化后缀。
            if not content:  # 在交给解析器前识别空文件。
                raise ValueError(f"{name} is empty")  # 将空文件名写入错误信息。
            if suffix == ".pdf" or content.startswith(b"%PDF-"):  # 用 PDF 扩展名或文件头识别 PDF。
                document = parse_pdf(name, content, self.settings.parsing.min_chars_per_page)  # 按配置的每页最低字符数解析 PDF。
            else:  # 其余受支持的文件按 Markdown 解析。
                document = parse_markdown(name, content)  # 从 Markdown 字节内容提取结构片段。
            if document.is_empty():  # 检查解析结果中是否有可用正文。
                raise ValueError(f"{name} produced no extractable text")  # 报告该文件未能提取文本。

            stage = "chunking"  # 后续错误归入分块环节。
            chunks = self.chunker.chunk(document, identifier)  # 把结构片段拆为关联的父块和子块。
            if not chunks.children:  # 检查是否生成可召回的子块。
                raise ValueError(f"{name} produced no retrievable units")  # 报告没有可供检索的内容。

            stage = "embedding"  # 后续错误归入向量生成环节。
            texts = [child.text for child in chunks.children]  # 按子块顺序提取待编码文本。
            vectors = self.embedder.embed_texts(texts)  # 为每份子块文本生成向量。
            if len(vectors) != len(texts):  # 核对响应数量与请求文本数量。
                raise ValueError(f"asked for {len(texts)} vectors but received {len(vectors)}")  # 报告请求和返回的向量数以定位缺失。
            dimensions = {len(vector) for vector in vectors}  # 收集响应中出现的向量维度。
            if len(dimensions) != 1:  # 要求本批次所有向量维度相同。
                raise ValueError(f"inconsistent vector dimensions: {sorted(dimensions)}")  # 在错误中列出实际收到的维度。

            stage = "persisting"  # 后续错误归入持久化环节。
            for previous in self.registry.find_by_name(name):  # 查找同名文档以清理被替换的版本。
                if previous.doc_id != identifier:  # 只清理标识不同的旧版本。
                    self.store.delete_document(previous.doc_id)  # 删除旧版本的父块和子块。
                    self.registry.remove(previous.doc_id)  # 移除旧版本在登记表中的记录。

            self.store.delete_document(identifier)  # 清除当前标识可能已有的片段以便重新写入。
            self.store.insert_units(chunks.children, chunks.parents, vectors)  # 将新子块、对应向量及父块写入存储。
            parents, children, chars = len(chunks.parents), len(chunks.children), chunks.char_count  # 汇总稍后登记并返回的三项规模指标。
            try:  # 在发布内容和登记状态期间处理失败。
                self.store.activate_document(identifier)  # 将新片段设为可检索。
                self.registry.mark_active(  # 把登记状态更新为可用。
                    identifier, parent_count=parents, child_count=children, char_count=chars  # 同时记录文档标识和最终分块统计。
                )  # 完成登记表更新。
            except Exception:  # 捕获发布或登记过程中出现的错误。
                # 激活会分别更新两个集合，登记失败时也要撤回已经发布的内容。
                self.store.delete_document(identifier)  # 删除本次已写入的片段以撤回发布。
                raise  # 让外层保留原始失败原因。
        except Exception as error:  # 在共同出口记录本次导入失败。
            message = f"{stage}: {error}"  # 将环节名称与具体错误组合成提示。
            self.registry.mark_failed(identifier, message)  # 把失败原因保存到文档登记表。
            raise IngestionError(message) from error  # 抛出带阶段信息的导入异常并保留原因链。

        return IngestResult(identifier, name, parents, children, chars, time.monotonic() - started)  # 返回文档统计及单调时钟测得的总耗时。


def remove_document(doc_id: str, store: MilvusStore, registry: DocumentRegistry) -> bool:  # 根据登记标识删除文档内容及其登记记录。
    if registry.get(doc_id) is None:  # 先检查文档是否存在于登记表。
        return False  # 返回未找到文档的结果。
    store.delete_document(doc_id)  # 清理 Milvus 中属于该文档的片段。
    registry.remove(doc_id)  # 删除对应的登记记录。
    return True  # 通知调用方文档已删除。
