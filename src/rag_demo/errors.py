"""定义模型服务共用的错误类型。调用方据此区分适合重试的临时故障，以及需要用户修改配置的认证、模型或协议问题。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。


class ProviderError(RuntimeError):  # 请求模型或生成答案时使用的基础异常。
    """请求模型或生成答案时使用的基础异常。"""


class ConfigurationIssue(ProviderError):  # 处理凭据或模型配置问题。
    """处理凭据或模型配置问题。"""

    def __init__(self, provider: str, message: str) -> None:  # 保存实例所需的配置和依赖。
        self.provider = provider  # 保存传入的供应商。
        self.message = message  # 保存传入的消息。
        super().__init__(f"{provider}: {message}")  # 初始化。


class TransientFailure(ProviderError):  # 标记上游故障、限流或超时等可以重试的情况。
    """标记上游故障、限流或超时等可以重试的情况。"""

    def __init__(self, provider: str, message: str, status: int | None = None) -> None:  # 保存实例所需的配置和依赖。
        self.provider = provider  # 保存传入的供应商。
        self.message = message  # 保存传入的消息。
        self.status = status  # 保存传入的状态。
        super().__init__(f"{provider}: {message}")  # 初始化。


class ContractMismatch(ProviderError):  # 服务返回的数据结构不符合预期。
    """服务返回的数据结构不符合预期；立即报告以避免错误结果继续传播。"""

    def __init__(self, provider: str, message: str) -> None:  # 保存实例所需的配置和依赖。
        self.provider = provider  # 保存传入的供应商。
        self.message = message  # 保存传入的消息。
        super().__init__(f"{provider}: {message}")  # 初始化。


class Unavailable(ProviderError):  # 无法连接到供应商服务。
    """无法连接到供应商服务。"""

    def __init__(self, provider: str, message: str) -> None:  # 保存实例所需的配置和依赖。
        self.provider = provider  # 保存传入的供应商。
        self.message = message  # 保存传入的消息。
        super().__init__(f"{provider}: {message}")  # 初始化。