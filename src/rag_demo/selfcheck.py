"""启动时验证外部依赖并实测向量维度。维度一旦用于创建集合就难以更改，自检可以及早发现模型名称、凭据或存储配置问题。各项故障汇总为报告，让服务仍可启动并显示原因。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import json  # 引入结构化数据的编解码工具。
from dataclasses import dataclass, field  # 从 dataclasses 引入本模块使用的类型和工具。
from enum import Enum  # 从 enum 导入枚举。
from pathlib import Path  # 从 pathlib 导入路径对象。
from typing import TYPE_CHECKING

from rag_demo.config import Settings, missing_secrets  # 从 rag_demo.config 引入本模块使用的类型和工具。
from rag_demo.embedding.provider import Embedder  # 从 rag_demo.embedding.provider 导入向量编码器。
from rag_demo.embedding.rerank import Reranker  # 从 rag_demo.embedding.rerank 导入重排服务。
from rag_demo.errors import (  # 从 rag_demo.errors 引入本模块使用的类型和工具。
    ConfigurationIssue,  # 导入供应商配置异常。
    ContractMismatch,  # 导入响应协议异常。
    ProviderError,  # 导入供应商异常。
    TransientFailure,  # 导入暂时性故障。
    Unavailable,  # 导入服务不可达异常。
)  # 结束当前多行表达式。
from rag_demo.generation.llm import Generator  # 从 rag_demo.generation.llm 导入生成客户端。

if TYPE_CHECKING:
    from rag_demo.store.milvus_store import MilvusStore


class Status(str, Enum):  # 定义 Status 类型。
    HEALTHY = "healthy"  # 声明“正常”状态的序列化值。
    MISCONFIGURED = "misconfigured"  # 声明“配置错误”状态的序列化值。
    UNREACHABLE = "unreachable"  # 声明“不可达”状态的序列化值。
    CONTRACT_MISMATCH = "contract_mismatch"  # 声明“协议不匹配”状态的序列化值。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class DependencyReport:  # 记录一项依赖检查的结论。
    """记录一项依赖检查的结论。"""

    name: str  # 声明名称字段。
    status: Status  # 声明状态字段。
    message: str  # 声明消息字段。
    detail: dict = field(default_factory=dict)  # 设置详情字段的默认值。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def healthy(self) -> bool:  # 判断依赖检查是否通过。
        return self.status is Status.HEALTHY  # 返回状态为Status.HEALTHY。

    def as_dict(self) -> dict:  # 将对象字段转换为可序列化的字典。
        return {  # 返回字段映射。
            "name": self.name,  # 填入名称字段。
            "status": self.status.value,  # 填入状态字段。
            "message": self.message,  # 填入消息字段。
            "detail": self.detail,  # 填入详情字段。
        }  # 结束本组字段。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class SelfCheckReport:  # 汇总全部依赖的检查结论。
    """汇总全部依赖的检查结论。"""

    dependencies: list[DependencyReport]  # 声明依赖项列表字段。
    embedding_dimension: int | None = None  # 设置实测向量维度字段的默认值。
    embedding_model: str = ""  # 设置向量模型名字段的默认值。
    stored: dict | None = None  # 设置存储值字段的默认值。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def healthy(self) -> bool:  # 判断依赖检查是否通过。
        return all(d.healthy for d in self.dependencies)  # 仅当所有检查项均通过时返回真。

    def by_name(self, name: str) -> DependencyReport | None:  # 按名称查找单项依赖检查结果。
        return next((d for d in self.dependencies if d.name == name), None)  # 返回next的结果。

    def as_dict(self) -> dict:  # 将对象字段转换为可序列化的字典。
        return {  # 返回字段映射。
            "healthy": self.healthy,  # 填入依赖健康状态字段。
            "embedding_dimension": self.embedding_dimension,  # 填入实测向量维度字段。
            "embedding_model": self.embedding_model,  # 填入向量模型名字段。
            "dependencies": [d.as_dict() for d in self.dependencies],  # 填入依赖项列表字段。
            "stored": self.stored,  # 填入存储值字段。
        }  # 结束本组字段。


def _status_for(exc: Exception) -> Status:  # 定义判定依赖状态的处理逻辑。
    if isinstance(exc, ConfigurationIssue):  # 当异常属于供应商配置异常类型时进入此分支。
        return Status.MISCONFIGURED  # 返回Status.MISCONFIGURED。
    if isinstance(exc, Unavailable):  # 当异常属于服务不可达异常类型时进入此分支。
        return Status.UNREACHABLE  # 返回Status.UNREACHABLE。
    if isinstance(exc, ContractMismatch):  # 当异常属于响应协议异常类型时进入此分支。
        return Status.CONTRACT_MISMATCH  # 返回Status.CONTRACT_MISMATCH。
    if isinstance(exc, TransientFailure):  # 当异常属于暂时性故障类型时进入此分支。
        return Status.UNREACHABLE  # 返回Status.UNREACHABLE。
    return Status.UNREACHABLE  # 返回Status.UNREACHABLE。


def check_embedding(settings: Settings) -> DependencyReport:  # 探测向量服务并读取实测维度。
    """探测向量服务并读取实测维度。"""
    embedder = Embedder(settings.embedding)  # 将向量服务设为向量编码器的结果。
    try:  # 在此捕获后续操作可能产生的异常。
        dimension = embedder.probe_dimension("ping")  # 将向量维度设为探测向量维度的结果。
    except ProviderError as exc:  # 捕获供应商异常。
        return DependencyReport(  # 返回单项依赖报告的结果。
            name="embedding",  # 传入名称参数。
            status=_status_for(exc),  # 传入状态参数。
            message=str(exc),  # 传入消息参数。
            detail={"model": settings.embedding.model},  # 传入详情参数。
        )  # 结束当前多行表达式。
    except Exception as exc:  # pragma: no cover  # 处理传输层的意外响应。
        return DependencyReport(  # 返回单项依赖报告的结果。
            name="embedding",  # 传入名称参数。
            status=Status.UNREACHABLE,  # 传入状态参数。
            message=f"unexpected failure: {exc}",  # 传入消息参数。
            detail={"model": settings.embedding.model},  # 传入详情参数。
        )  # 结束当前多行表达式。

    return DependencyReport(  # 返回单项依赖报告的结果。
        name="embedding",  # 传入名称参数。
        status=Status.HEALTHY,  # 传入状态参数。
        message=f"model available, dimension {dimension}",  # 传入消息参数。
        detail={"model": settings.embedding.model, "dimension": dimension},  # 传入详情参数。
    )  # 结束当前多行表达式。


def check_rerank(settings: Settings) -> DependencyReport:  # 用相关性明确的候选验证重排服务及其排序结果。
    """用相关性明确的候选验证重排服务及其排序结果。"""
    try:  # 在此捕获后续操作可能产生的异常。
        results = Reranker(settings.rerank).verify()  # 将结果列表设为自检开关的结果。
    except ProviderError as exc:  # 捕获供应商异常。
        return DependencyReport(  # 返回单项依赖报告的结果。
            name="rerank",  # 传入名称参数。
            status=_status_for(exc),  # 传入状态参数。
            message=str(exc),  # 传入消息参数。
            detail={"model": settings.rerank.model},  # 传入详情参数。
        )  # 结束当前多行表达式。
    except Exception as exc:  # pragma: no cover  # 处理意外故障。
        return DependencyReport(  # 返回单项依赖报告的结果。
            name="rerank",  # 传入名称参数。
            status=Status.UNREACHABLE,  # 传入状态参数。
            message=f"unexpected failure: {exc}",  # 传入消息参数。
            detail={"model": settings.rerank.model},  # 传入详情参数。
        )  # 结束当前多行表达式。

    return DependencyReport(  # 返回单项依赖报告的结果。
        name="rerank",  # 传入名称参数。
        status=Status.HEALTHY,  # 传入状态参数。
        message="model available, ranking plausible",  # 传入消息参数。
        detail={  # 传入详情参数。
            "model": settings.rerank.model,  # 填入模型字段。
            "top_index": results[0].index,  # 填入top_index字段。
            "top_score": round(results[0].score, 4),  # 填入top_score字段。
        },  # 结束本组字段。
    )  # 结束当前多行表达式。


def check_llm(settings: Settings) -> DependencyReport:  # 验证生成模型存在、凭据有效且能流式返回文本。
    """验证生成模型存在、凭据有效且能流式返回文本。"""
    try:  # 在此捕获后续操作可能产生的异常。
        text = Generator(settings.llm).verify()  # 将文本设为自检开关的结果。
    except ProviderError as exc:  # 捕获供应商异常。
        return DependencyReport(  # 返回单项依赖报告的结果。
            name="llm",  # 传入名称参数。
            status=_status_for(exc),  # 传入状态参数。
            message=str(exc),  # 传入消息参数。
            detail={"model": settings.llm.model, "base_url": settings.llm.base_url},  # 传入详情参数。
        )  # 结束当前多行表达式。
    except Exception as exc:  # pragma: no cover  # 处理意外故障。
        return DependencyReport(  # 返回单项依赖报告的结果。
            name="llm",  # 传入名称参数。
            status=Status.UNREACHABLE,  # 传入状态参数。
            message=f"unexpected failure: {exc}",  # 传入消息参数。
            detail={"model": settings.llm.model},  # 传入详情参数。
        )  # 结束当前多行表达式。

    return DependencyReport(  # 返回单项依赖报告的结果。
        name="llm",  # 传入名称参数。
        status=Status.HEALTHY,  # 传入状态参数。
        message="model available, streaming confirmed",  # 传入消息参数。
        detail={"model": settings.llm.model, "sample": text[:40]},  # 传入详情参数。
    )  # 结束当前多行表达式。


def check_milvus(settings: Settings) -> DependencyReport:  # 检查向量数据库的可达性及版本。
    """检查向量数据库的可达性及版本。"""
    try:  # 在此捕获后续操作可能产生的异常。
        from pymilvus import MilvusClient  # 从 pymilvus 导入数据库客户端。

        client = MilvusClient(uri=settings.milvus.uri, token=settings.milvus.token or None)  # 将客户端设为数据库客户端的结果。
        version = client.get_server_version()  # 将版本设为读取数据库版本的结果。
    except ImportError as exc:  # pragma: no cover  # 处理意外故障。
        return DependencyReport(  # 返回单项依赖报告的结果。
            name="milvus",  # 传入名称参数。
            status=Status.MISCONFIGURED,  # 传入状态参数。
            message=f"pymilvus is not installed: {exc}",  # 传入消息参数。
        )  # 结束当前多行表达式。
    except Exception as exc:  # 捕获通用异常。
        return DependencyReport(  # 返回单项依赖报告的结果。
            name="milvus",  # 传入名称参数。
            status=Status.UNREACHABLE,  # 传入状态参数。
            message=f"cannot reach {settings.milvus.uri}: {exc}",  # 传入消息参数。
            detail={"uri": settings.milvus.uri},  # 传入详情参数。
        )  # 结束当前多行表达式。

    return DependencyReport(  # 返回单项依赖报告的结果。
        name="milvus",  # 传入名称参数。
        status=Status.HEALTHY,  # 传入状态参数。
        message=f"server {version} reachable",  # 传入消息参数。
        detail={"uri": settings.milvus.uri, "server_version": version},  # 传入详情参数。
    )  # 结束当前多行表达式。


def check_stored_dimension(
    settings: Settings,
    dimension: int,
    *,
    store: MilvusStore | None = None,
    record_if_missing: bool = False,
) -> DependencyReport:
    """校验已有库，只在确认空库且调用方要求初始化时写入元数据。"""
    try:
        from rag_demo.store.milvus_store import InvalidStoreMetadata, MilvusStore
    except ImportError as exc:
        return DependencyReport("store_schema", Status.MISCONFIGURED, f"cannot load Milvus client: {exc}")

    if store is None:
        store = MilvusStore(settings)
    try:
        stored = store.read_meta()
        if not stored:
            if any(store.collection_stats().values()):
                raise InvalidStoreMetadata("existing passages have no schema record")
            if not record_if_missing:
                return DependencyReport("store_schema", Status.HEALTHY, "empty store ready for initialisation")
            stored = {"embedding_model": settings.embedding.model, "dimension": dimension}
            store.write_meta(stored)
    except InvalidStoreMetadata as exc:
        return DependencyReport(
            "store_schema", Status.CONTRACT_MISMATCH,
            f"{exc}; restore the original metadata or rebuild the knowledge base",
        )
    except Exception as exc:
        return DependencyReport("store_schema", Status.UNREACHABLE, f"cannot access stored metadata: {exc}")

    stored_model = stored["embedding_model"]
    stored_dim = stored["dimension"]
    if stored_model == settings.embedding.model and stored_dim == dimension:  # 当已存储的模型等于模型且已存储的维度等于向量维度时进入此分支。
        return DependencyReport(  # 返回单项依赖报告的结果。
            name="store_schema",  # 传入名称参数。
            status=Status.HEALTHY,  # 传入状态参数。
            message=f"stored schema matches configuration ({stored_model}, dim {stored_dim})",  # 传入消息参数。
            detail={"embedding_model": stored_model, "dimension": stored_dim},  # 传入详情参数。
        )  # 结束当前多行表达式。

    return DependencyReport(  # 返回单项依赖报告的结果。
        name="store_schema",  # 传入名称参数。
        status=Status.MISCONFIGURED,  # 传入状态参数。
        message=(  # 传入消息参数。
            "existing storage was built with a different embedding configuration: "  # 提示当前向量配置与已有集合不符。
            f"stored model '{stored_model}' (dim {stored_dim}) but the provider now "  # 列出建库时使用的模型和维度。
            f"reports '{settings.embedding.model}' (dim {dimension}). "  # 列出当前供应商的模型和实测维度。
            "Restore the original configuration or explicitly rebuild the knowledge base before ingesting."
        ),  # 结束当前多行表达式。
        detail={  # 传入详情参数。
            "stored_model": stored_model,  # 填入已存储的模型字段。
            "stored_dimension": stored_dim,  # 填入stored_dimension字段。
            "current_model": settings.embedding.model,  # 填入当前模型字段。
            "current_dimension": dimension,  # 填入当前向量维度字段。
        },  # 结束本组字段。
    )  # 结束当前多行表达式。


def check_dependencies(settings: Settings, include_llm: bool = True) -> SelfCheckReport:  # 分别检查依赖，并返回汇总报告。
    """分别检查依赖，并返回汇总报告。"""
    absent = missing_secrets(settings)  # 将缺失项设为列出缺失凭据的结果。

    embedding = check_embedding(settings)  # 将向量设为检查向量服务的结果。
    dimension = embedding.detail.get("dimension") if embedding.healthy else None  # 将向量维度设为按条件选取的值。

    rerank = check_rerank(settings)  # 将重排设为检查重排服务的结果。
    milvus = check_milvus(settings)  # 将向量库设为检查向量数据库的结果。
    dependencies = [embedding, rerank, milvus]  # 将依赖项列表设为向量、重排、向量库。

    if include_llm:  # 当include_llm有效时进入此分支。
        dependencies.append(check_llm(settings))  # 将检查生成服务的结果加入依赖项列表。

    # 单独列出缺失的密钥字段，便于直接定位配置问题。
    for section, keys in absent.items():  # 遍历缺失项的键值对，逐项取得section、键列表。
        dependencies.append(  # 将单项依赖报告的结果加入依赖项列表。
            DependencyReport(  # 为单项依赖报告提供输入。
                name=f"{section}_credentials",  # 传入名称参数。
                status=Status.MISCONFIGURED,  # 传入状态参数。
                message=(  # 传入消息参数。
                    f"{section} has no {', '.join(keys)} configured; "  # 列出当前配置分组中缺失的凭据字段。
                    f"fill it in config.yaml or set RAG__{section.upper()}__API_KEY"  # 给出本地配置及环境变量的填写位置。
                ),  # 结束当前多行表达式。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。

    return SelfCheckReport(
        dependencies=dependencies,  # 传入依赖项列表参数。
        embedding_dimension=dimension,  # 传入实测向量维度参数。
        embedding_model=settings.embedding.model,  # 传入向量模型名参数。
    )  # 结束当前多行表达式。


def run_selfcheck(
    settings: Settings, include_llm: bool = True, *, record_if_missing: bool = False,
) -> SelfCheckReport:
    report = check_dependencies(settings, include_llm)
    if report.embedding_dimension is not None:
        schema = check_stored_dimension(
            settings, report.embedding_dimension, record_if_missing=record_if_missing,
        )
        report.dependencies.append(schema)
    return report  # 返回报告。


def format_report(report: SelfCheckReport) -> str:  # 将检查结果格式化为终端可读的报告。
    """将检查结果格式化为终端可读的报告。"""
    lines = []  # 初始化文本行列表为空列表。
    for dependency in report.dependencies:  # 遍历依赖项列表，逐项取得依赖项。
        marker = {  # 将marker设为字段映射中对应的项。
            Status.HEALTHY: "OK  ",  # 组合当前结果所需的各项数据。
            Status.MISCONFIGURED: "FAIL",  # 组合当前结果所需的各项数据。
            Status.UNREACHABLE: "DOWN",  # 组合当前结果所需的各项数据。
            Status.CONTRACT_MISMATCH: "BAD ",  # 组合当前结果所需的各项数据。
        }[dependency.status]  # 组合当前结果所需的各项数据。
        lines.append(f"[{marker}] {dependency.name:<22} {dependency.message}")  # 将格式化文本加入文本行列表。

    lines.append("")  # 将空字符串加入文本行列表。
    if report.embedding_dimension is not None:  # 当实测向量维度不为空值时进入此分支。
        lines.append(  # 将格式化文本加入文本行列表。
            f"embedding: {report.embedding_model} -> dimension "  # 输出被检测到的向量模型名称。
            f"{report.embedding_dimension}"  # 附上模型实测返回的维度。
        )  # 结束当前多行表达式。
    lines.append("result: " + ("all dependencies healthy" if report.healthy else "problems found"))  # 将“result: ”加上按条件选取的值加入文本行列表。
    return "\n".join(lines)  # 返回拼接后的文本行列表。


def report_to_json(report: SelfCheckReport) -> str:  # 定义导出检查报告的处理逻辑。
    return json.dumps(report.as_dict(), ensure_ascii=False, indent=2)  # 返回dumps的结果。


def default_config_path(root: Path | None = None) -> Path:  # 定义默认配置路径的处理逻辑。
    return (root or Path.cwd()) / "config.yaml"  # 返回项目根目录或当前工作目录的结果下的“config.yaml”。