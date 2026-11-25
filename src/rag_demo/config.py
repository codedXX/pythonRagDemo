"""按模型默认值、本地 config.yaml、环境变量的顺序覆盖配置。

环境变量用双下划线分隔路径，例如 RAG__EMBEDDING__API_KEY 对应 embedding.api_key。这允许容器和持续集成从环境传入密钥；日志和对外状态应隐藏密钥值。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import os  # 引入操作系统接口。
from pathlib import Path  # 从 pathlib 导入路径对象。
from typing import Any  # 从 typing 导入任意类型。

import yaml  # 引入配置文件解析库。
from pydantic import BaseModel, Field  # 从 pydantic 引入本模块使用的类型和工具。

ENV_PREFIX = "RAG__"  # 将环境变量前缀设为“RAG__”。

# 需要在日志、错误提示和健康状态中隐藏的字段。
SECRET_FIELDS = frozenset({"api_key", "token", "password"})  # 将敏感字段集合设为frozenset的结果。


class ConfigurationError(RuntimeError):  # 配置缺失或各字段相互矛盾时抛出此异常。
    """配置缺失或各字段相互矛盾时抛出此异常。"""


class LLMSettings(BaseModel):  # 集中管理生成模型参数。
    base_url: str = "http://8.134.19.201:20089/v1"  # 设置服务根地址字段的默认值。
    api_key: str = ""  # 设置接口密钥字段的默认值。
    model: str = "claude-sonnet-5"  # 设置模型字段的默认值。
    temperature: float = 0.2  # 设置采样温度字段的默认值。
    max_tokens: int = 2048  # 设置最大生成词元数字段的默认值。
    timeout_seconds: float = 120.0  # 设置超时秒数字段的默认值。
    history_turns: int = 6  # 设置历史轮数上限字段的默认值。


class EmbeddingSettings(BaseModel):  # 集中管理向量参数。
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 设置服务根地址字段的默认值。
    api_key: str = ""  # 设置接口密钥字段的默认值。
    model: str = "qwen3.7-text-embedding-flash"  # 设置模型字段的默认值。
    batch_size: int = 10  # 设置单批文本数字段的默认值。
    max_retries: int = 5  # 设置最大尝试次数字段的默认值。
    retry_base_delay_seconds: float = 1.0  # 设置初始退避秒数字段的默认值。


class RerankSettings(BaseModel):  # 集中管理重排参数。
    base_url: str = "https://dashscope.aliyuncs.com"  # 设置服务根地址字段的默认值。
    path: str = "/api/v1/services/rerank/text-rerank/text-rerank"  # 设置路径字段的默认值。
    api_key: str = ""  # 设置接口密钥字段的默认值。
    model: str = "qwen3.7-text-rerank"  # 设置模型字段的默认值。
    timeout_seconds: float = 60.0  # 设置超时秒数字段的默认值。
    max_retries: int = 3  # 设置最大尝试次数字段的默认值。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def url(self) -> str:  # 拼出服务接口的完整地址。
        return self.base_url.rstrip("/") + "/" + self.path.lstrip("/")  # 将根地址和原生重排路径用一个斜杠连接。


class MilvusSettings(BaseModel):  # 集中管理向量库参数。
    uri: str = "http://127.0.0.1:19530"  # 设置连接地址字段的默认值。
    token: str = ""  # 向量库访问令牌，免认证的本地服务可以留空。
    child_collection: str = "rag_child"  # 设置子块集合名字段的默认值。
    parent_collection: str = "rag_parent"  # 设置父块集合名字段的默认值。
    meta_collection: str = "rag_meta"  # 设置元数据集合名字段的默认值。
    # 强一致性保证文档入库后可以立即检索；详见示例配置。
    consistency_level: str = "Strong"  # 设置一致性级别字段的默认值。


class ChunkingSettings(BaseModel):  # 集中管理切块参数。
    parent_target_chars: int = 1200  # 设置父块目标字符数字段的默认值。
    parent_max_chars: int = 2000  # 设置父块字符上限字段的默认值。
    child_target_chars: int = 320  # 设置子块目标字符数字段的默认值。
    child_overlap_chars: int = 60  # 设置相邻子块重叠字符数字段的默认值。
    semantic_enabled: bool = True  # 设置语义切分开关字段的默认值。
    semantic_break_percentile: int = 88  # 设置语义距离切分百分位字段的默认值。
    semantic_window_chars: int = 180  # 设置语义窗口字符数字段的默认值。


class RetrievalSettings(BaseModel):  # 集中管理检索配置参数。
    dense_top_k: int = 40  # 设置向量召回条数字段的默认值。
    sparse_top_k: int = 40  # 设置关键词召回条数字段的默认值。
    rrf_k: int = 60  # 设置倒数排名融合常数字段的默认值。
    rerank_candidates: int = 30  # 设置重排候选数量字段的默认值。
    context_count: int = 6  # 设置上下文数量字段的默认值。
    context_min_count: int = 3  # 设置最少上下文数量字段的默认值。
    context_char_budget: int = 8000  # 设置上下文总字符预算字段的默认值。
    parent_excerpt_chars: int = 1800  # 设置父块摘录字符数字段的默认值。


class ParsingSettings(BaseModel):  # 集中管理解析配置参数。
    min_chars_per_page: int = 40  # 设置最低平均字符数字段的默认值。


class MySQLSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 3306
    database: str = "rag_demo"
    user: str = "rag_demo"
    password: str = Field(default="", repr=False)


class StorageSettings(BaseModel):  # 集中管理存储参数。
    embedding_cache_path: str = "data/embedding_cache"  # 设置向量缓存路径字段的默认值。
    max_upload_mb: int = 50  # 设置单文件上传上限（兆字节）字段的默认值。


class ApiSettings(BaseModel):  # 集中管理接口配置参数。
    host: str = "127.0.0.1"  # 设置监听地址字段的默认值。
    port: int = 8000  # 设置监听端口字段的默认值。


class Settings(BaseModel):  # 集中管理参数。
    llm: LLMSettings = Field(default_factory=LLMSettings)  # 设置生成模型字段的默认值。
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)  # 设置向量字段的默认值。
    rerank: RerankSettings = Field(default_factory=RerankSettings)  # 设置重排字段的默认值。
    mysql: MySQLSettings = Field(default_factory=MySQLSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)  # 设置向量库字段的默认值。
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)  # 设置切块字段的默认值。
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)  # 设置检索配置字段的默认值。
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)  # 设置解析配置字段的默认值。
    storage: StorageSettings = Field(default_factory=StorageSettings)  # 设置存储字段的默认值。
    api: ApiSettings = Field(default_factory=ApiSettings)  # 设置接口配置字段的默认值。

    # 由加载器填写项目路径。
    root: Path = Field(default_factory=Path.cwd, exclude=True)  # 默认从当前目录启动，导出配置时排除运行目录。

    def data_dir(self) -> Path:  # 定义数据目录的处理逻辑。
        return self.root / "data"  # 将数据目录定位在项目根目录下。

    def embedding_cache_dir(self) -> Path:  # 定义向量缓存目录的处理逻辑。
        return self.root / self.storage.embedding_cache_path  # 把配置的向量缓存路径解析到项目根目录。


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:  # 递归合并覆盖项，返回新的配置字典。
    """递归合并覆盖项，返回新的配置字典。"""
    merged = dict(base)  # 复制基础配置，避免改写调用方传入的字典。
    for key, value in override.items():  # 遍历override的键值对，逐项取得键、值。
        existing = merged.get(key)  # 将existing设为合并结果中的对应值。
        if isinstance(existing, dict) and isinstance(value, dict):  # 只有原值和覆盖值都是字典时才逐层合并。
            merged[key] = _deep_merge(existing, value)  # 递归覆盖当前分组内部的字段。
        else:  # 处理其余情况。
            merged[key] = value  # 将合并结果中对应的项设为值。
    return merged  # 返回合并结果。


def _coerce(raw: str, current: Any) -> Any:  # 根据原配置项的类型转换环境变量字符串。
    """根据原配置项的类型转换环境变量字符串。"""
    if isinstance(current, bool):  # 当当前值属于布尔值类型时进入此分支。
        return raw.strip().lower() in {"1", "true", "yes", "on"}  # 将去掉空白并转成小写的约定值识别为真。
    if isinstance(current, int):  # 当当前值属于整数类型时进入此分支。
        return int(raw)  # 返回原始数据的整数形式。
    if isinstance(current, float):  # 当当前值属于浮点数类型时进入此分支。
        return float(raw)  # 返回原始数据的浮点数形式。
    return raw  # 返回原始数据。


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:  # 将双下划线分隔的环境变量应用到嵌套配置中。
    """将双下划线分隔的环境变量应用到嵌套配置中。"""
    result = dict(data)  # 将结果设为数据的字典形式。
    for env_name, env_value in os.environ.items():  # 逐个读取进程环境中的变量名和值。
        if not env_name.startswith(ENV_PREFIX):  # 忽略未使用项目专属前缀的变量。
            continue  # 跳过本项，继续处理下一项。
        parts = [p.lower() for p in env_name[len(ENV_PREFIX) :].split("__") if p]  # 去掉前缀，用双下划线分层并统一转为小写键。
        if not parts:  # 当路径片段为空或未启用时进入此分支。
            continue  # 跳过本项，继续处理下一项。

        cursor: Any = result  # 从配置根节点开始定位待覆盖项。
        for part in parts[:-1]:  # 先沿各级分组路径向下查找，留下最后一级作为字段。
            node = cursor.get(part)  # 将当前节点设为当前位置中的对应值。
            if not isinstance(node, dict):  # 遇到缺失或非字典分组时建立新节点。
                node = {}  # 初始化当前节点为空字典。
                cursor[part] = node  # 将当前位置中对应的项设为当前节点。
            cursor = node  # 继续进入下一层配置。

        leaf = parts[-1]  # 最后一个路径片段是目标字段名。
        cursor[leaf] = _coerce(env_value, cursor.get(leaf))  # 参考现有值的类型转换环境字符串，再覆盖字段。
    return result  # 返回结果。


def _read_yaml(path: Path) -> dict[str, Any]:  # 定义读取配置文件的处理逻辑。
    if not path.exists():  # 当不满足“文件存在的结果”时进入此分支。
        return {}  # 返回空字典。
    try:  # 在此捕获后续操作可能产生的异常。
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))  # 按统一字符编码读取文件并安全解析配置。
    except yaml.YAMLError as exc:  # pragma: no cover  # 防御性处理。
        raise ConfigurationError(f"{path.name} is not valid YAML: {exc}") from exc  # 向上抛出配置校验异常。
    if loaded is None:  # 当loaded为空值时进入此分支。
        return {}  # 返回空字典。
    if not isinstance(loaded, dict):  # 顶层须是可按字段名访问的映射。
        raise ConfigurationError(f"{path.name} must contain a mapping at the top level")  # 向上抛出配置校验异常。
    return loaded  # 返回loaded。


def load_settings(  # 从项目根目录加载配置。
    root: Path | str | None = None,  # 接收项目根目录参数。
    config_path: Path | str | None = None,  # 接收配置文件路径参数。
    use_env: bool = True,  # 接收是否使用环境覆盖参数。
) -> Settings:  # 结束参数声明，进入函数或语句主体。
    """从项目根目录加载配置。可指定配置文件路径，测试可关闭环境覆盖以保证结果可重复。"""
    resolved_root = Path(root) if root is not None else Path.cwd()  # 显式传入目录时使用该目录，否则取当前工作目录。
    resolved_config = (  # 将resolved_config设为按条件选取的值。
        Path(config_path)  # 采用调用方指定的配置文件路径。
        if config_path is not None  # 仅当提供了文件路径时选用上一项。
        else resolved_root / "config.yaml"  # 默认读取项目根目录下的配置文件。
    )  # 结束当前多行表达式。

    data = _read_yaml(resolved_config)  # 将数据设为读取配置文件的结果。
    if use_env:  # 当是否使用环境覆盖有效时进入此分支。
        data = _apply_env_overrides(data)  # 将数据设为应用环境覆盖的结果。
    data.setdefault("root", resolved_root)  # 补充运行根目录，保留已有设置。

    try:  # 在此捕获后续操作可能产生的异常。
        return Settings.model_validate(data)  # 校验配置类型并补齐模型定义的默认值。
    except Exception as exc:  # 统一处理模型校验等异常。
        raise ConfigurationError(f"invalid configuration: {exc}") from exc  # 向上抛出配置校验异常。


def redacted(settings: Settings) -> dict[str, Any]:  # 生成可写入日志或健康响应的配置副本，并遮蔽密钥。
    """生成可写入日志或健康响应的配置副本，并遮蔽密钥。"""
    view = settings.model_dump(exclude={"root"})  # 导出配置副本，省略内部根路径。
    for section, values in view.items():  # 遍历脱敏副本的键值对，逐项取得section、值列表。
        if not isinstance(values, dict):  # 值列表不属于字典类型时进入此分支。
            continue  # 跳过本项，继续处理下一项。
        for key in list(values):  # 遍历值列表的列表形式，逐项取得键。
            if key in SECRET_FIELDS:  # 当键包含在敏感字段集合时进入此分支。
                values[key] = "***set***" if values[key] else ""  # 已设置的密钥仅显示占位符，未设置的仍显示为空。
    return view  # 返回脱敏副本。


def missing_secrets(settings: Settings) -> dict[str, list[str]]:  # 按配置分组列出缺失凭据的字段名，供健康接口展示。
    """按配置分组列出缺失凭据的字段名，供健康接口展示。"""
    sections = {  # 将sections设为字段映射。
        "llm": settings.llm,  # 填入生成模型字段。
        "embedding": settings.embedding,  # 填入向量字段。
        "rerank": settings.rerank,  # 填入重排字段。
        "milvus": settings.milvus,  # 填入向量库字段。
    }  # 结束本组字段。
    absent: dict[str, list[str]] = {}  # 初始化缺失项为空字典。
    for name, section in sections.items():  # 遍历sections的键值对，逐项取得名称、section。
        missing = [  # 将缺失项设为从敏感字段集合中提取的各项。
            key  # 仅报告缺失的字段名。
            for key in SECRET_FIELDS  # 依次检查敏感字段集合。
            if hasattr(section, key) and not getattr(section, key)  # 只选出本分组确实声明且没有值的凭据字段。
            # 本地免认证的 Milvus 服务可以省略令牌。
            and not (name == "milvus" and key == "token")  # 排除本地可选的向量库令牌。
        ]  # 结束列表。
        if missing:  # 当缺失项有效时进入此分支。
            absent[name] = missing  # 将缺失项中对应的项设为缺失项。
    return absent  # 返回缺失项。