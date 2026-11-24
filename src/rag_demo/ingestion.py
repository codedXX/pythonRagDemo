"""将上传文件转换为可检索内容。

先解析文件并拒绝不可用的输入，再切块、分批计算向量。父子块先以未激活状态写入，整份文档写完后才开放检索。各阶段通过回调上报进度，使大文件的处理过程可见。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import time  # 引入时间工具。
from collections.abc import Callable  # 从 collections.abc 导入Callable。
from dataclasses import dataclass  # 从 dataclasses 导入数据类装饰器。
from enum import Enum  # 从 enum 导入枚举。

from rag_demo.chunking.chunker import ChunkSet, Chunker  # 从 rag_demo.chunking.chunker 引入本模块使用的类型和工具。
from rag_demo.config import Settings  # 从 rag_demo.config 导入项目配置。
from rag_demo.embedding.provider import Embedder  # 从 rag_demo.embedding.provider 导入向量编码器。
from rag_demo.parsers import ParsedDocument, parse_document  # 从 rag_demo.parsers 引入本模块使用的类型和工具。
from rag_demo.parsers.base import document_id  # 从 rag_demo.parsers.base 导入文档标识。
from rag_demo.store.milvus_store import ChildUnit, MilvusStore, ParentUnit, normalise  # 从 rag_demo.store.milvus_store 引入本模块使用的类型和工具。
from rag_demo.store.registry import DocumentRegistry  # 从 rag_demo.store.registry 导入文档登记表。


class Stage(str, Enum):  # 定义 Stage 类型。
    PARSING = "parsing"  # 声明“解析”状态的序列化值。
    CHUNKING = "chunking"  # 声明“切块”状态的序列化值。
    EMBEDDING = "embedding"  # 声明“向量服务”状态的序列化值。
    PERSISTING = "persisting"  # 声明“持久化”状态的序列化值。
    DONE = "done"  # 声明“完成”状态的序列化值。
    FAILED = "failed"  # 声明“失败”状态的序列化值。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class Progress:  # 记录一次入库进度通知。
    """记录一次入库进度通知。"""

    stage: Stage  # 声明阶段字段。
    message: str  # 声明消息字段。
    current: int = 0  # 设置当前值字段的默认值。
    total: int = 0  # 设置总数字段的默认值。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def fraction(self) -> float:  # 定义fraction的处理逻辑。
        if self.total <= 0:  # 当总数不超过0时进入此分支。
            return 0.0  # 返回0.0。
        return min(1.0, self.current / self.total)  # 计算已完成比例，并将最大值限制为一。

    def as_dict(self) -> dict:  # 将对象字段转换为可序列化的字典。
        return {  # 返回字段映射。
            "stage": self.stage.value,  # 填入阶段字段。
            "message": self.message,  # 填入消息字段。
            "current": self.current,  # 填入当前值字段。
            "total": self.total,  # 填入总数字段。
            "fraction": round(self.fraction, 4),  # 填入fraction字段。
        }  # 结束本组字段。


ProgressCallback = Callable[[Progress], None]  # 将ProgressCallback设为Callable中对应的项。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class IngestResult:  # 保存一次入库产生的标识和统计信息。
    """保存一次入库产生的标识和统计信息。"""

    doc_id: str  # 声明文档标识字段。
    name: str  # 声明名称字段。
    parent_count: int  # 声明父块数字段。
    child_count: int  # 声明子块数字段。
    char_count: int  # 声明字符数字段。
    duration_seconds: float  # 声明耗时秒数字段。


class IngestionError(RuntimeError):  # 记录入库失败的阶段和原因。
    """记录入库失败的阶段和原因。"""

    def __init__(self, stage: Stage, message: str) -> None:  # 保存实例所需的配置和依赖。
        self.stage = stage  # 保存传入的阶段。
        self.message = message  # 保存传入的消息。
        super().__init__(f"{stage.value}: {message}")  # 初始化。


class IngestionPipeline:  # 将上传的字节流转成可检索的父块和子块。
    """将上传的字节流转成可检索的父块和子块。"""

    def __init__(  # 保存实例所需的配置和依赖。
        self,  # 通过实例读取本方法所需状态。
        settings: Settings,  # 接收配置参数。
        store: MilvusStore,  # 接收向量库参数。
        registry: DocumentRegistry,  # 接收登记表参数。
        embedder: Embedder,  # 接收向量服务参数。
        chunker: Chunker | None = None,  # 接收分块器参数。
    ) -> None:  # 结束参数声明，进入函数或语句主体。
        self._settings = settings  # 将测试配置设为配置。
        self._store = store  # 保存传入的向量库。
        self._registry = registry  # 保存传入的登记表。
        self._embedder = embedder  # 保存传入的向量服务。
        self._chunker = chunker or Chunker(settings.chunking, embedder=embedder)  # 优先复用传入的分块器，否则按项目配置创建。

    def ingest(  # 执行单份文档入库。
        self,  # 通过实例读取本方法所需状态。
        name: str,  # 接收名称参数。
        content: bytes,  # 接收内容参数。
        on_progress: ProgressCallback | None = None,  # 接收进度回调参数。
        doc_id: str | None = None,  # 接收文档标识参数。
    ) -> IngestResult:  # 结束参数声明，进入函数或语句主体。
        """执行单份文档入库；失败时抛出带阶段信息的异常，并在登记表中记录原因。"""
        started = time.time()  # 将开始时间设为时间工具的结果。
        notify = on_progress or (lambda _progress: None)  # 有进度回调时调用它，否则使用空操作。
        identifier = doc_id or document_id(name, content)  # 优先采用指定标识，否则从文件名和内容生成。

        self._registry.begin(identifier, name)  # 登记文档并标记正在入库。

        try:  # 在此捕获后续操作可能产生的异常。
            parsed = self._parse(name, content, notify)  # 解析上传内容，同时上报解析进度。
            chunks = self._chunk(parsed, identifier, notify)  # 从结构化文档生成有父子关系的检索片段。
            children, parents = self._embed_and_persist(identifier, chunks, notify)  # 对子块生成向量，写入父子块并激活。
        except IngestionError as error:  # 捕获入库异常。
            self._registry.mark_failed(identifier, str(error))  # 把带有失败阶段的原因记入登记表。
            notify(Progress(stage=Stage.FAILED, message=str(error)))  # 通过回调把失败原因通知客户端。
            raise  # 向上抛出当前异常。
        except Exception as error:  # 意外异常同样需要登记，便于定位失败环节。
            message = f"{type(error).__name__}: {error}"  # 将消息设为格式化文本。
            self._registry.mark_failed(identifier, message)  # 登记失败原因。
            notify(Progress(stage=Stage.FAILED, message=message))  # 进度回调。
            raise IngestionError(Stage.PERSISTING, message) from error  # 向上抛出入库异常。

        duration = time.time() - started  # 用结束时间减去开始时间，统计整份文档的入库耗时。
        notify(  # 进度回调。
            Progress(  # 为进度通知提供输入。
                stage=Stage.DONE,  # 传入阶段参数。
                message=f"indexed {len(parents)} parents and {len(children)} children",  # 传入消息参数。
                current=1,  # 传入当前值参数。
                total=1,  # 传入总数参数。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。
        return IngestResult(  # 返回入库结果的结果。
            doc_id=identifier,  # 传入文档标识参数。
            name=name,  # 传入名称参数。
            parent_count=len(parents),  # 传入父块数参数。
            child_count=len(children),  # 传入子块数参数。
            char_count=chunks.char_count,  # 传入字符数参数。
            duration_seconds=duration,  # 传入耗时秒数参数。
        )  # 结束当前多行表达式。

    # 入库各阶段。

    def _parse(  # 定义解析响应的处理逻辑。
        self, name: str, content: bytes, notify: ProgressCallback  # 组合当前结果所需的各项数据。
    ) -> ParsedDocument:  # 结束参数声明，进入函数或语句主体。
        notify(Progress(stage=Stage.PARSING, message=f"parsing {name}"))  # 进度回调。
        try:  # 在此捕获后续操作可能产生的异常。
            document = parse_document(name, content, self._settings.parsing)  # 将文档设为解析文档的结果。
        except Exception as error:  # 捕获通用异常。
            raise IngestionError(Stage.PARSING, str(error)) from error  # 向上抛出入库异常。

        if document.is_empty():  # 提取不到正文时在解析阶段报错。
            raise IngestionError(  # 向上抛出入库异常。
                Stage.PARSING,  # 传入Stage.PARSING。
                f"{name} produced no extractable text",  # 指出具体哪个文件没有可提取正文。
            )  # 结束当前多行表达式。
        return document  # 返回文档。

    def _chunk(  # 定义片段的处理逻辑。
        self, document: ParsedDocument, identifier: str, notify: ProgressCallback  # 组合当前结果所需的各项数据。
    ) -> ChunkSet:  # 结束参数声明，进入函数或语句主体。
        notify(  # 进度回调。
            Progress(  # 为进度通知提供输入。
                stage=Stage.CHUNKING,  # 传入阶段参数。
                message=f"chunking {document.name}",  # 传入消息参数。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。
        try:  # 在此捕获后续操作可能产生的异常。
            chunks = self._chunker.chunk(document, identifier)  # 按文档标识切出父块和子块。
        except Exception as error:  # 捕获通用异常。
            raise IngestionError(Stage.CHUNKING, str(error)) from error  # 向上抛出入库异常。

        if not chunks.children:  # 没有子块意味着没有可用于召回的内容。
            raise IngestionError(  # 向上抛出入库异常。
                Stage.CHUNKING, f"{document.name} produced no retrievable units"  # 为入库异常提供输入。
            )  # 结束当前多行表达式。
        return chunks  # 返回分块列表。

    def _embed_and_persist(  # 定义_embed_and_persist的处理逻辑。
        self, identifier: str, chunks: ChunkSet, notify: ProgressCallback  # 组合当前结果所需的各项数据。
    ) -> tuple[list[ChildUnit], list[ParentUnit]]:  # 结束参数声明，进入函数或语句主体。
        texts = [child.text for child in chunks.children]  # 按子块顺序提取待生成向量的正文。
        notify(  # 进度回调。
            Progress(  # 为进度通知提供输入。
                stage=Stage.EMBEDDING,  # 传入阶段参数。
                message=f"embedding {len(texts)} units",  # 传入消息参数。
                current=0,  # 传入当前值参数。
                total=len(texts),  # 传入总数参数。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。

        try:  # 在此捕获后续操作可能产生的异常。
            vectors = self._embedder.embed_texts(texts)  # 将全部子块文本交给向量编码器。
        except Exception as error:  # 捕获通用异常。
            raise IngestionError(Stage.EMBEDDING, str(error)) from error  # 向上抛出入库异常。

        if len(vectors) != len(texts):  # 检查向量个数是否与子块数一致，以防错位写入。
            raise IngestionError(  # 向上抛出入库异常。
                Stage.EMBEDDING,  # 传入Stage.EMBEDDING。
                f"asked for {len(texts)} vectors but received {len(vectors)}",  # 列出输入文本条数和实际向量条数。
            )  # 结束当前多行表达式。

        dimensions = {len(vector) for vector in vectors}  # 收集本批所有向量的维度。
        if len(dimensions) != 1:  # 同一集合中的向量必须具有相同维度。
            raise IngestionError(  # 向上抛出入库异常。
                Stage.EMBEDDING, f"inconsistent vector dimensions: {sorted(dimensions)}"  # 为sorted提供输入。
            )  # 结束当前多行表达式。

        notify(  # 进度回调。
            Progress(  # 为进度通知提供输入。
                stage=Stage.EMBEDDING,  # 传入阶段参数。
                message=f"embedded {len(vectors)} units",  # 传入消息参数。
                current=len(vectors),  # 传入当前值参数。
                total=len(vectors),  # 传入总数参数。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。

        children = [  # 按原始顺序组装待持久化的子块记录。
            ChildUnit(  # 为子块记录提供输入。
                id=child.id,  # 传入标识参数。
                doc_id=identifier,  # 传入文档标识参数。
                parent_id=child.parent_id,  # 传入所属父块标识参数。
                child_idx=child.child_idx,  # 传入子块序号参数。
                text=child.text,  # 传入文本参数。
                breadcrumb=child.breadcrumb,  # 传入标题路径参数。
                page=child.page,  # 传入页码参数。
                source_name=child.source_name,  # 传入来源文件名参数。
                # 归一化向量，使点积也能用于比较余弦相似度。
                embedding=normalise(vector),  # 归一化该子块的真实向量。
            )  # 结束当前多行表达式。
            for child, vector in zip(chunks.children, vectors, strict=True)  # 严格逐项配对子块与向量，保证数量一致。
        ]  # 结束列表。
        parents = [  # 组装供按主键回查正文的父块记录。
            ParentUnit(  # 为父块记录提供输入。
                parent_id=parent.parent_id,  # 传入所属父块标识参数。
                doc_id=identifier,  # 传入文档标识参数。
                text=parent.text,  # 传入文本参数。
                breadcrumb=parent.breadcrumb,  # 传入标题路径参数。
                page_start=parent.page_start,  # 传入起始页参数。
                source_name=parent.source_name,  # 传入来源文件名参数。
            )  # 结束当前多行表达式。
            for parent in chunks.parents  # 逐项处理推导式中的父块。
        ]  # 结束列表。

        notify(  # 进度回调。
            Progress(  # 为进度通知提供输入。
                stage=Stage.PERSISTING,  # 传入阶段参数。
                message=f"writing {len(parents)} parents and {len(children)} children",  # 传入消息参数。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。
        try:  # 在此捕获后续操作可能产生的异常。
            # 按文件名替换旧版本；先删除旧内容，避免失败时同时留下旧版和部分新版。
            for existing in self._registry.find_by_name(chunks.source_name):  # 逐一检查登记表中与本次上传同名的文档。
                if existing.doc_id != identifier:  # 同名但标识不同的是待替换的旧版本。
                    self._store.delete_document(existing.doc_id)  # 删除文档。
                    self._registry.remove(existing.doc_id)  # 移除。

            self._store.delete_document(identifier)  # 清除当前标识的旧片段，防止重复入库。
            self._store.insert_units(children, parents)  # 先以未激活状态写入全部父块和子块。
            self._store.activate_document(identifier)  # 写入完成后才将该文档开放给检索。
        except Exception as error:  # 捕获通用异常。
            raise IngestionError(Stage.PERSISTING, str(error)) from error  # 向上抛出入库异常。

        self._registry.mark_active(  # 最后在登记表记录激活状态和切块统计。
            identifier,  # 传入标识。
            parent_count=len(parents),  # 传入父块数参数。
            child_count=len(children),  # 传入子块数参数。
            char_count=chunks.char_count,  # 传入字符数参数。
        )  # 结束当前多行表达式。
        return children, parents  # 返回子块列表、父块列表。


def remove_document(  # 删除文档登记信息及其所有可检索片段。
    doc_id: str, store: MilvusStore, registry: DocumentRegistry  # 组合当前结果所需的各项数据。
) -> bool:  # 结束参数声明，进入函数或语句主体。
    """删除文档登记信息及其所有可检索片段。"""
    record = registry.get(doc_id)  # 将登记记录设为读取的结果。
    if record is None:  # 当登记记录为空值时进入此分支。
        return False  # 返回假。
    store.delete_document(doc_id)  # 删除文档。
    registry.remove(doc_id)  # 移除。
    return True  # 返回真。