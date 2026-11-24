"""通过百炼的 OpenAI 兼容接口生成文本向量。

传输客户端负责请求和解析，Embedder 负责分批、退避重试和缓存。重排服务使用另一套原生协议，由独立适配器处理。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import hashlib  # 引入摘要计算模块。
import json  # 引入结构化数据的编解码工具。
import random  # 引入随机数工具。
import time  # 引入时间工具。
from collections.abc import Sequence  # 从 collections.abc 导入Sequence。
from pathlib import Path  # 从 pathlib 导入路径对象。

import httpx  # 引入网络客户端库。

from rag_demo.config import EmbeddingSettings  # 从 rag_demo.config 导入向量服务配置。
from rag_demo.errors import (  # 从 rag_demo.errors 引入本模块使用的类型和工具。
    ConfigurationIssue,  # 导入供应商配置异常。
    ContractMismatch,  # 导入响应协议异常。
    TransientFailure,  # 导入暂时性故障。
    Unavailable,  # 导入服务不可达异常。
)  # 结束当前多行表达式。

# 只对暂时性状态码重试；请求格式、认证、权限和路径错误需要先修正配置。
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}  # 将可重试状态码集合设为集合。


class EmbeddingClient:  # 封装兼容接口的向量请求与响应。
    """封装兼容接口的向量请求与响应。"""

    def __init__(self, settings: EmbeddingSettings, timeout: float = 60.0) -> None:  # 保存实例所需的配置和依赖。
        self._settings = settings  # 将测试配置设为配置。
        self._timeout = timeout  # 保存传入的超时。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def settings(self) -> EmbeddingSettings:  # 取得当前使用的配置。
        return self._settings  # 返回测试配置。

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # 用一次请求计算一组文本的向量，并按输入顺序返回。
        """用一次请求计算一组文本的向量，并按输入顺序返回。"""
        if not texts:  # 当待编码文本为空或未启用时进入此分支。
            return []  # 返回空列表。
        if not self._settings.api_key:  # 当接口密钥为空或未启用时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "embedding",  # 标注故障来自向量服务。
                "no API key configured; set embedding.api_key in config.yaml "  # 提示本地配置缺少向量服务密钥。
                "or RAG__EMBEDDING__API_KEY",  # 给出通过环境变量提供密钥的方式。
            )  # 结束当前多行表达式。

        url = self._settings.base_url.rstrip("/") + "/embeddings"  # 规范根地址后接上向量接口路径。
        payload = {"model": self._settings.model, "input": list(texts)}  # 按兼容协议同时传入模型名称与文本列表。
        headers = {  # 将请求头设为字段映射。
            "Authorization": f"Bearer {self._settings.api_key}",  # 填入Authorization字段。
            "Content-Type": "application/json",  # 填入Content-Type字段。
        }  # 结束本组字段。

        try:  # 在此捕获后续操作可能产生的异常。
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)  # 带认证信息提交文本，并应用本次请求的超时设置。
        except httpx.TimeoutException as exc:  # 捕获httpx.TimeoutException。
            raise TransientFailure("embedding", f"request timed out after {self._timeout}s") from exc  # 向上抛出暂时性故障。
        except httpx.HTTPError as exc:  # 捕获httpx.HTTPError。
            raise Unavailable("embedding", f"cannot reach {url}: {exc}") from exc  # 向上抛出服务不可达异常。

        if response.status_code != 200:  # 当响应状态码不等于200时进入此分支。
            self._raise_for_status(response)  # 按状态码分类错误。

        return self._parse(response, expected=len(texts))  # 解析响应，并核对是否每条输入都有对应向量。

    def _raise_for_status(self, response: httpx.Response) -> None:  # 根据响应状态将请求失败映射为对应异常。
        """根据响应状态将请求失败映射为对应异常。"""
        detail = self._extract_message(response)  # 将详情设为提取错误消息的结果。
        status = response.status_code  # 将状态设为响应状态码。

        if status in _RETRYABLE_STATUS:  # 当状态包含在可重试状态码集合时进入此分支。
            raise TransientFailure("embedding", f"HTTP {status}: {detail}", status=status)  # 向上抛出暂时性故障。

        if status in {401, 403}:  # 当状态包含在集合时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "embedding",  # 标注鉴权失败的服务。
                f"credentials rejected (HTTP {status}): {detail}. "  # 附上认证失败的状态码和原因。
                "Check embedding.api_key.",  # 提示检查向量服务密钥。
            )  # 结束当前多行表达式。
        if status == 404:  # 当状态等于404时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "embedding",  # 标注找不到资源的服务。
                f"endpoint or model not found (HTTP 404): {detail}. "  # 提示接口或模型不存在。
                f"Check embedding.base_url and embedding.model "  # 列出需要核对的根地址和模型配置。
                f"(currently '{self._settings.model}').",  # 显示当前实际选择的模型。
            )  # 结束当前多行表达式。
        if status == 400 and _mentions_model(detail):  # 当状态等于400且_mentions_model的结果时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "embedding",  # 指明是向量模型被拒绝。
                f"model rejected by the provider (HTTP 400): {detail}. "  # 给出供应商拒绝模型的错误信息。
                f"The configured model is '{self._settings.model}'; it may not exist "  # 列出所选模型以便核对名称。
                "or may not be available to this account.",  # 提示检查账号是否有模型权限。
            )  # 结束当前多行表达式。
        raise ContractMismatch("embedding", f"unexpected HTTP {status}: {detail}")  # 向上抛出响应协议异常。

    @staticmethod  # 声明无需实例状态的静态方法。
    def _extract_message(response: httpx.Response) -> str:  # 定义提取错误消息的处理逻辑。
        try:  # 在此捕获后续操作可能产生的异常。
            body = response.json()  # 将正文设为解析后的响应数据。
        except (json.JSONDecodeError, ValueError):  # 捕获(json.JSONDecodeError, ValueError)。
            return response.text[:300] or "(empty body)"  # 错误正文无法解码时只截取前三百个字符。
        if isinstance(body, dict):  # 当正文属于字典类型时进入此分支。
            error = body.get("error")  # 将错误设为正文中的错误。
            if isinstance(error, dict) and error.get("message"):  # 当错误属于字典类型且错误中的消息时进入此分支。
                return str(error["message"])  # 返回错误中的消息的字符串形式。
            if body.get("message"):  # 当正文中的消息时进入此分支。
                return str(body["message"])  # 返回正文中的消息的字符串形式。
        return json.dumps(body)[:300]  # 返回dumps的结果的指定区间。

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:  # 定义解析响应的处理逻辑。
        try:  # 在此捕获后续操作可能产生的异常。
            body = response.json()  # 将正文设为解析后的响应数据。
        except (json.JSONDecodeError, ValueError) as exc:  # 捕获(json.JSONDecodeError, ValueError)。
            raise ContractMismatch("embedding", "response was not JSON") from exc  # 向上抛出响应协议异常。

        data = body.get("data") if isinstance(body, dict) else None  # 确认顶层结构可按键访问后，读取向量数据数组。
        if not isinstance(data, list) or not data:  # 当数据属于列表类型为假或数据为假时进入此分支。
            raise ContractMismatch(  # 向上抛出响应协议异常。
                "embedding", "response has no 'data' array of embeddings"  # 指出向量响应缺少数据数组。
            )  # 结束当前多行表达式。

        # 供应商可能打乱结果顺序，按响应中的原始下标还原。
        ordered: list[tuple[int, list[float]]] = []  # 初始化排序结果为空列表。
        for position, item in enumerate(data):  # 遍历数据及各项的下标，逐项取得位置、条目。
            if not isinstance(item, dict):  # 条目不属于字典类型时进入此分支。
                raise ContractMismatch("embedding", "response item is not an object")  # 向上抛出响应协议异常。
            vector = item.get("embedding")  # 将向量设为条目中的向量。
            if not isinstance(vector, list) or not vector:  # 当向量属于列表类型为假或向量为假时进入此分支。
                raise ContractMismatch("embedding", "response item has no 'embedding' vector")  # 向上抛出响应协议异常。
            if not all(isinstance(x, (int, float)) for x in vector):  # 当不满足“是否各项均满足条件的结果”时进入此分支。
                raise ContractMismatch("embedding", "embedding vector contains non-numeric values")  # 向上抛出响应协议异常。
            index = item.get("index", position)  # 优先采用响应给出的输入下标，缺省时沿用出现位置。
            if not isinstance(index, int):  # 索引不属于整数类型时进入此分支。
                raise ContractMismatch("embedding", "response item has a non-integer 'index'")  # 向上抛出响应协议异常。
            ordered.append((index, [float(x) for x in vector]))  # 保留下标，并把各向量分量统一转为浮点数。

        ordered.sort(key=lambda pair: pair[0])  # 按原始输入下标排序，纠正服务端可能的乱序。
        vectors = [vector for _, vector in ordered]  # 去掉用于排序的下标，留下向量序列。

        if len(vectors) != expected:  # 当向量列表的长度不等于预期值时进入此分支。
            # 核对向量数量，避免缺失的向量导致后续片段错配。
            raise ContractMismatch(  # 向上抛出响应协议异常。
                "embedding",  # 标注返回数量不符的服务。
                f"asked for {expected} vectors but received {len(vectors)}",  # 同时列出期望条数和实际条数。
            )  # 结束当前多行表达式。

        dimensions = {len(v) for v in vectors}  # 统计本批向量包含哪些维度。
        if len(dimensions) != 1:  # 当维度集合的长度不等于1时进入此分支。
            raise ContractMismatch(  # 向上抛出响应协议异常。
                "embedding", f"inconsistent vector dimensions in one response: {sorted(dimensions)}"  # 指出当前批次返回了不同维度。
            )  # 结束当前多行表达式。

        return vectors  # 返回向量列表。


def _mentions_model(detail: str) -> bool:  # 定义_mentions_model的处理逻辑。
    low = detail.lower()  # 将low设为详情的小写形式。
    return any(  # 返回是否存在满足条件的项的结果。
        token in low  # 为是否存在满足条件的项提供输入。
        for token in ("model", "does not exist", "not found", "unsupported", "invalid_model")  # 逐项处理推导式中的词元。
    )  # 结束当前多行表达式。


def content_hash(text: str, model: str) -> str:  # 用文本和模型名称生成稳定缓存键，使切换模型后自动重新计算向量。
    """用文本和模型名称生成稳定缓存键，使切换模型后自动重新计算向量。"""
    digest = hashlib.sha256()  # 初始化用于生成稳定缓存键的摘要计算器。
    digest.update(model.encode("utf-8"))  # 将模型名称纳入摘要，隔离不同模型的向量空间。
    digest.update(b"\x00")  # 用零字节隔开模型名和正文，避免拼接歧义。
    digest.update(text.encode("utf-8"))  # 把完整文本加入摘要计算。
    return digest.hexdigest()  # 返回十六进制摘要作为文件缓存键。


class EmbeddingCache:  # 按内容将向量缓存在磁盘，反复调整切分参数时只需计算新出现的文本。
    """按内容将向量缓存在磁盘，反复调整切分参数时只需计算新出现的文本。"""

    def __init__(self, directory: Path) -> None:  # 保存实例所需的配置和依赖。
        self._dir = directory  # 将目录设为directory。
        self._dir.mkdir(parents=True, exist_ok=True)  # 按需创建目录。

    def _path(self, key: str) -> Path:  # 定位当前缓存键对应的磁盘文件。
        # 按键的前两个字符分目录，限制单个目录的文件数。
        return self._dir / key[:2] / f"{key}.json"  # 用缓存键前两位分目录，再用完整键命名文件。

    def get(self, key: str) -> list[float] | None:  # 定义读取的处理逻辑。
        path = self._path(key)  # 将路径设为缓存文件路径的结果。
        if not path.exists():  # 当不满足“文件存在的结果”时进入此分支。
            return None  # 返回空值。
        try:  # 在此捕获后续操作可能产生的异常。
            payload = json.loads(path.read_text(encoding="utf-8"))  # 读取并解码保存在磁盘上的缓存条目。
        except (json.JSONDecodeError, OSError):  # 捕获(json.JSONDecodeError, OSError)。
            # 缓存文件损坏时按未命中处理，重新生成向量。
            return None  # 返回空值。
        vector = payload.get("vector")  # 将向量设为载荷中的向量。
        return vector if isinstance(vector, list) else None  # 仅在缓存字段为列表时视为命中。

    def put(self, key: str, vector: list[float], text: str) -> None:  # 将本条向量结果保存到磁盘缓存。
        path = self._path(key)  # 将路径设为缓存文件路径的结果。
        path.parent.mkdir(parents=True, exist_ok=True)  # 按需创建父块。
        # 先写临时文件再原子替换，避免读取到中断写入的内容。
        temporary = path.with_suffix(".json.tmp")  # 构造同目录的临时文件路径。
        payload = json.dumps({"vector": vector, "text": text[:200]}, ensure_ascii=False)  # 序列化向量和最多两百字符的文本预览。
        temporary.write_text(payload, encoding="utf-8")  # 先将完整内容写入临时文件。
        temporary.replace(path)  # 用写完的临时文件原子替换目标缓存条目。

    def stats(self) -> dict[str, int]:  # 统计当前可用的记录数。
        count = sum(1 for _ in self._dir.rglob("*.json"))  # 将数量设为求和的结果。
        return {"entries": count}  # 返回字段映射。


class Embedder:  # 在传输客户端上实现向量缓存、批处理和退避重试。
    """在传输客户端上实现向量缓存、批处理和退避重试。"""

    def __init__(  # 保存实例所需的配置和依赖。
        self,  # 通过实例读取本方法所需状态。
        settings: EmbeddingSettings,  # 接收配置参数。
        client: EmbeddingClient | None = None,  # 接收客户端参数。
        cache: EmbeddingCache | None = None,  # 接收缓存参数。
        sleep=time.sleep,  # 接收等待函数参数。
    ) -> None:  # 结束参数声明，进入函数或语句主体。
        self._settings = settings  # 将测试配置设为配置。
        self._client = client or EmbeddingClient(settings)  # 将测试客户端设为客户端或向量请求客户端的结果。
        self._cache = cache  # 保存传入的缓存。
        self._sleep = sleep  # 保存传入的等待函数。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def client(self) -> EmbeddingClient:  # 定义客户端的处理逻辑。
        return self._client  # 返回测试客户端。

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:  # 优先读取缓存，再将未命中的文本分批提交。
        """优先读取缓存，再将未命中的文本分批提交。"""
        if not texts:  # 当待编码文本为空或未启用时进入此分支。
            return []  # 返回空列表。

        model = self._settings.model  # 取出当前使用的模型。
        keys = [content_hash(t, model) for t in texts]  # 按模型和每条输入文本生成独立缓存键。
        vectors: list[list[float] | None] = [None] * len(texts)  # 按输入条数预留位置，稍后分别填入命中值和新向量。

        if self._cache is not None:  # 当缓存不为空值时进入此分支。
            for position, key in enumerate(keys):  # 遍历键列表及各项的下标，逐项取得位置、键。
                cached = self._cache.get(key)  # 按文本对应的缓存键查找已算向量。
                if cached is not None:  # 当缓存向量不为空值时进入此分支。
                    vectors[position] = cached  # 将缓存向量写回输入列表中的原位置。

        pending = [i for i, v in enumerate(vectors) if v is None]  # 只收集仍然缺少向量的输入下标。
        batch_size = max(1, self._settings.batch_size)  # 单次至少处理一条文本，避免零步长。
        for start in range(0, len(pending), batch_size):  # 遍历指定的下标范围，逐项取得起点。
            batch_indices = pending[start : start + batch_size]  # 取出本批未命中项在原列表中的位置。
            batch_texts = [texts[i] for i in batch_indices]  # 按本批下标组装请求文本。
            produced = self._embed_with_retry(batch_texts)  # 调用带退避重试的向量请求。
            for index, vector in zip(batch_indices, produced, strict=True):  # 严格逐项对应批次中的原始下标和新向量。
                vectors[index] = vector  # 按输入位置回填，保证最终结果保持顺序。
                if self._cache is not None:  # 当缓存不为空值时进入此分支。
                    self._cache.put(keys[index], vector, texts[index])  # 将新生成的向量及对应文本写入磁盘缓存。

        missing = [i for i, v in enumerate(vectors) if v is None]  # 检查每个输入位置是否都已回填。
        if missing:  # pragma: no cover  # 防御性处理。
            raise ContractMismatch("embedding", f"no vector produced for indices {missing}")  # 向上抛出响应协议异常。

        return [v for v in vectors if v is not None]  # 按原输入顺序返回生成完成的向量。

    def _embed_with_retry(self, texts: Sequence[str]) -> list[list[float]]:  # 定义带重试的向量请求的处理逻辑。
        attempts = max(1, self._settings.max_retries)  # 总尝试次数至少为一。
        last: Exception | None = None  # 将末项设为空值。
        for attempt in range(attempts):  # 遍历指定的下标范围，逐项取得尝试序号。
            try:  # 在此捕获后续操作可能产生的异常。
                return self._client.embed(texts)  # 返回生成向量的结果。
            except TransientFailure as exc:  # 捕获暂时性故障。
                last = exc  # 将末项设为异常。
                if attempt == attempts - 1:  # 当尝试序号等于总尝试次数减去1时进入此分支。
                    break  # 结束当前循环。
                # 指数退避叠加随机抖动，避免多个请求同时再次冲击服务。
                delay = self._settings.retry_base_delay_seconds * (2**attempt)  # 将退避时长设为初始退避秒数乘以2的幂次为尝试序号。
                self._sleep(delay + random.uniform(0, delay * 0.25))  # 调用等待函数。
            # 配置错误、响应协议错误和不可达错误直接向上传递。
        assert last is not None  # 确认末项不为空值。
        raise last  # 向上抛出当前异常。

    def probe_dimension(self, text: str = "ping") -> int:  # 启动时实测所选模型的向量维数，确保创建集合时使用正确的维度。
        """启动时实测所选模型的向量维数，确保创建集合时使用正确的维度。"""
        vectors = self._client.embed([text])  # 将向量列表设为生成向量的结果。
        if not vectors:  # pragma: no cover  # 防御性处理。
            raise ContractMismatch("embedding", "probe returned no vectors")  # 向上抛出响应协议异常。
        return len(vectors[0])  # 返回向量列表中对应的项的长度。