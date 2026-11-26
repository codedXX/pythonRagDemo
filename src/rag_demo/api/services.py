"""按配置组装每个进程共享的服务。各组件通过对象传递，测试可以独立注入替身。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import logging  # 引入日志模块。
import threading  # 引入线程工具。
from collections.abc import Iterator  # 从 collections.abc 导入Iterator。
from dataclasses import dataclass, field  # 从 dataclasses 引入本模块使用的类型和工具。

from rag_demo.config import Settings  # 从 rag_demo.config 导入项目配置。
from rag_demo.embedding.provider import Embedder  # 从 rag_demo.embedding.provider 导入向量编码器。
from rag_demo.embedding.rerank import Reranker  # 从 rag_demo.embedding.rerank 导入重排服务。
from rag_demo.generation.answer import Answerer  # 从 rag_demo.generation.answer 导入问答服务。
from rag_demo.generation.llm import Generator  # 从 rag_demo.generation.llm 导入生成客户端。
from rag_demo.ingestion import IngestionPipeline  # 从 rag_demo.ingestion 导入入库流程。
from rag_demo.retrieval.hybrid import HybridRetriever  # 从 rag_demo.retrieval.hybrid 导入混合检索器。
from rag_demo.retrieval.lexical import LexicalIndex  # 从 rag_demo.retrieval.lexical 导入关键词索引。
from rag_demo.selfcheck import (  # 导入启动和首次写入时使用的自检接口。
    DependencyReport, SelfCheckReport, Status, check_dependencies, check_stored_dimension,  # 引入报告类型、依赖探测和存储维度校验。
)  # 结束自检接口导入。
from rag_demo.store.milvus_store import MilvusStore  # 从 rag_demo.store.milvus_store 导入向量库客户端。
from rag_demo.store.registry import DocumentRegistry  # 从 rag_demo.store.registry 导入文档登记表。

logger = logging.getLogger(__name__)  # 将日志记录器设为获取当前模块的日志记录器的结果。


@dataclass  # 根据字段声明生成数据类的初始化和比较方法。
class Services:  # 保存接口处理请求所需的服务实例。
    """保存接口处理请求所需的服务实例。"""

    settings: Settings  # 声明配置字段。
    store: MilvusStore  # 声明向量库字段。
    registry: DocumentRegistry  # 声明登记表字段。
    embedder: Embedder  # 声明向量服务字段。
    reranker: Reranker  # 声明重排器字段。
    generator: Generator  # 声明生成器字段。
    lexical: LexicalIndex  # 声明关键词索引字段。
    retriever: HybridRetriever  # 声明检索器字段。
    answerer: Answerer  # 声明问答服务字段。
    pipeline: IngestionPipeline  # 声明入库流程字段。
    selfcheck: SelfCheckReport | None = None  # 设置自检字段的默认值。
    dimension: int | None = None  # 设置向量维度字段的默认值。
    _lexical_lock: threading.Lock = field(default_factory=threading.Lock)  # 设置关键词索引重建锁字段的默认值。

    _storage_lock: threading.Lock = field(default_factory=threading.Lock, init=False)  # 为每个服务实例创建独立的存储初始化锁。
    _storage_ready: bool = field(default=False, init=False)  # 记录本进程是否已成功初始化集合。

    def refresh_lexical_index(self) -> int:  # 按存储中的有效文档重建关键词索引。
        """按存储中的有效文档重建关键词索引；用锁串行处理上传和删除触发的重建。"""
        with self._lexical_lock:  # 让多个文档变更触发的索引重建依次执行。
            return self.retriever.rebuild_lexical_index()  # 从有效子块重建关键词索引，并返回条目数。

    def ensure_collections(self) -> None:  # 确保集合与模型兼容并可接收入库数据。
        """首次写入前校验建库信息；并发上传共享本进程的初始化结果。"""
        with self._storage_lock:  # 串行执行首次写入前的检查与建库。
            if self._storage_ready:  # 检测其他请求是否已完成初始化。
                return  # 复用本进程的初始化结果。
            report = self.selfcheck  # 取当前服务持有的依赖报告。
            previous = report.by_name("store_schema") if report else None  # 查找上次对存储结构的检查结果。
            if previous and not previous.healthy:  # 已有失败记录时阻止继续写入。
                raise RuntimeError(previous.message)  # 将之前的失败原因传递给请求方。

            if self.dimension is None:  # 检查是否还缺少模型输出维度。
                self.dimension = self.embedder.probe_dimension("ping")  # 用探测文本取得实际向量维度。
            schema = check_stored_dimension(  # 核对当前模型与库中登记的模型及维度。
                self.settings, self.dimension, store=self.store, record_if_missing=True  # 允许为首次使用的空库写入模型信息。
            )  # 获得本次存储兼容性报告。
            if schema.healthy:  # 仅在模型与存储兼容时准备集合。
                try:  # 捕获集合初始化产生的故障。
                    self.store.ensure_collections(self.dimension)  # 按实测维度创建或确认所需集合。
                except Exception as exc:  # 将初始化错误记入依赖状态。
                    schema = DependencyReport(  # 构造存储不可用的检查结果。
                        "store_schema", Status.UNREACHABLE, f"cannot initialise storage collections: {exc}",  # 保留建库异常供界面显示。
                    )  # 完成集合故障报告。

            if report is None:  # 首次检查时需要创建汇总报告。
                report = SelfCheckReport([], self.dimension, self.settings.embedding.model)  # 将当前模型与实测维度写入新报告。
                self.selfcheck = report  # 让后续健康检查读取同一份报告。
            if previous:  # 已有存储检查项时先替换旧值。
                report.dependencies.remove(previous)  # 移除上一轮的存储状态。
            report.dependencies.append(schema)  # 保存本次存储检查结果。
            if not schema.healthy:  # 检查初始化是否以错误结束。
                raise RuntimeError(schema.message)  # 将报告中的原因返回给调用方。
            self._storage_ready = True  # 标记初始化成功，供后续请求直接复用。


def build_services(settings: Settings) -> Services:  # 根据配置连接各项服务。
    """根据配置连接各项服务；创建集合后维度已固定，因此先从模型实测。"""
    store = MilvusStore(settings)  # 按连接配置建立向量存储对象。
    registry = DocumentRegistry(settings.mysql)  # 使用 MySQL 配置连接文档登记库。
    embedder = Embedder(settings.embedding)  # 构造支持缓存和批处理的向量编码器。
    reranker = Reranker(settings.rerank)  # 为候选重排创建百炼原生协议客户端。
    generator = Generator(settings.llm)  # 创建回答及问题改写使用的生成服务。
    lexical = LexicalIndex()  # 初始化存储在进程内的关键词索引。

    retriever = HybridRetriever(settings.retrieval, store, embedder, lexical, reranker)  # 把两路召回和重排服务组合成混合检索器。
    answerer = Answerer(settings, generator)  # 让问答服务复用同一生成客户端。
    pipeline = IngestionPipeline(settings, store, registry, embedder)  # 让入库流程复用同一套存储、登记和向量服务。

    return Services(  # 返回应用服务集合的结果。
        settings=settings,  # 传入配置参数。
        store=store,  # 传入向量库参数。
        registry=registry,  # 传入登记表参数。
        embedder=embedder,  # 传入向量服务参数。
        reranker=reranker,  # 传入重排器参数。
        generator=generator,  # 传入生成器参数。
        lexical=lexical,  # 传入关键词索引参数。
        retriever=retriever,  # 传入检索器参数。
        answerer=answerer,  # 传入问答服务参数。
        pipeline=pipeline,  # 传入入库流程参数。
    )  # 结束当前多行表达式。


def initialise(services: Services, verify: bool = True) -> SelfCheckReport | None:  # 准备对外服务。
    """准备对外服务；将依赖故障记录下来，供启动后的页面展示。"""
    report: SelfCheckReport | None = None  # 将报告设为空值。
    if verify:  # 当自检开关有效时进入此分支。
        try:  # 在此捕获后续操作可能产生的异常。
            report = check_dependencies(services.settings, include_llm=False)  # 执行启动自检，这里跳过单独的生成探测。
            services.selfcheck = report  # 将自检设为报告。
            for dependency in report.dependencies:  # 遍历依赖项列表，逐项取得依赖项。
                logger.info(  # 记录运行信息。
                    "dependency %s: %s (%s)",  # 日志中依次显示依赖名称、状态和说明。
                    dependency.name,  # 传入dependency.name。
                    dependency.status.value,  # 传入dependency.status.value。
                    dependency.message,  # 传入dependency.message。
                )  # 结束当前多行表达式。
        except Exception as error:  # pragma: no cover  # 记录后继续运行。
            logger.warning("self-check failed to run: %s", error)  # 记录警告。

    if report is not None:  # 检查启动探测是否返回了报告。
        services.dimension = report.embedding_dimension  # 保留探测到的向量维度供建库使用。

    try:  # 捕获启动时准备存储与索引产生的异常。
        services.ensure_collections()  # 在重建关键词索引前确认集合可用。
        count = services.refresh_lexical_index()  # 从数据库中的有效子块恢复关键词索引。
        logger.info("lexical index rebuilt with %s units", count)  # 记录运行信息。
    except Exception as error:  # pragma: no cover  # 启动期间记录后继续运行。
        logger.warning("store initialisation failed: %s", error)  # 记录警告。

    return report  # 返回报告。


def iter_sse(event: str, data: str) -> Iterator[str]:  # 编码服务端推送事件。
    """编码服务端推送事件；多行载荷分别加上 data: 前缀，接收端可重新拼合。"""
    lines = [f"event: {event}"]  # 用事件名称构造推送帧的首行。
    for line in data.split("\n"):  # 按换行符拆开载荷，以保留多行内容。
        lines.append(f"data: {line}")  # 每行分别加上服务端推送协议的数据前缀。
    return iter(["\n".join(lines) + "\n\n"])  # 拼合事件帧并用空行分隔下一帧。