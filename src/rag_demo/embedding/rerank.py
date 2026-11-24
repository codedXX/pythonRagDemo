"""调用百炼原生重排接口。

请求发送至 POST /api/v1/services/rerank/text-rerank/text-rerank，将查询与候选文档放入 input 字段；从 output.results 读取原始下标和相关性分数。该接口使用独立的路径和错误结构，错误码与消息通常位于响应顶层。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import time  # 引入时间工具。
from collections.abc import Sequence  # 从 collections.abc 导入Sequence。

import httpx  # 引入网络客户端库。

from rag_demo.config import RerankSettings  # 从 rag_demo.config 导入重排配置。
from rag_demo.errors import (  # 从 rag_demo.errors 引入本模块使用的类型和工具。
    ConfigurationIssue,  # 导入供应商配置异常。
    ContractMismatch,  # 导入响应协议异常。
    TransientFailure,  # 导入暂时性故障。
    Unavailable,  # 导入服务不可达异常。
)  # 结束当前多行表达式。

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}  # 将可重试状态码集合设为集合。


class RerankResult:  # 记录重排命中的原始文档位置和相关性分数。
    """记录重排命中的原始文档位置和相关性分数。"""

    __slots__ = ("index", "score")  # 将__slots__设为组合值。

    def __init__(self, index: int, score: float) -> None:  # 保存实例所需的配置和依赖。
        self.index = index  # 保存传入的索引。
        self.score = score  # 保存传入的分数。

    def __repr__(self) -> str:  # pragma: no cover  # 供调试输出使用。
        return f"RerankResult(index={self.index}, score={self.score:.4f})"  # 返回格式化文本。

    def __eq__(self, other: object) -> bool:  # 比较两个对象是否代表同一结果。
        if not isinstance(other, RerankResult):  # 另一对象不属于重排结果类型时进入此分支。
            return NotImplemented  # 返回NotImplemented。
        return self.index == other.index and self.score == other.score  # 返回索引等于索引且分数等于分数。


class Reranker:  # 封装百炼原生文本重排请求。
    """封装百炼原生文本重排请求。"""

    def __init__(self, settings: RerankSettings, sleep=time.sleep) -> None:  # 保存实例所需的配置和依赖。
        self._settings = settings  # 将测试配置设为配置。
        self._sleep = sleep  # 保存传入的等待函数。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def settings(self) -> RerankSettings:  # 取得当前使用的配置。
        return self._settings  # 返回测试配置。

    def rerank(self, query: str, documents: Sequence[str], top_n: int | None = None) -> list[RerankResult]:  # 按相关性从高到低返回候选文档。
        """按相关性从高到低返回候选文档。"""
        if not documents:  # 当文档为空或未启用时进入此分支。
            return []  # 返回空列表。
        if not self._settings.api_key:  # 当接口密钥为空或未启用时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "rerank",  # 标注故障来自重排服务。
                "no API key configured; set rerank.api_key in config.yaml "  # 提示缺少重排密钥并给出本地配置项。
                "or RAG__RERANK__API_KEY",  # 给出对应的环境变量名。
            )  # 结束当前多行表达式。

        payload: dict = {  # 将载荷设为字段映射。
            "model": self._settings.model,  # 填入模型字段。
            "input": {"query": query, "documents": list(documents)},  # 填入文档字段。
        }  # 结束本组字段。
        if top_n is not None:  # 当top_n不为空值时进入此分支。
            payload["parameters"] = {"top_n": top_n}  # 将载荷中的parameters设为字段映射。

        return self._request_with_retry(payload, len(documents))  # 返回可重试的重排请求的结果。

    def _request_with_retry(self, payload: dict, expected: int) -> list[RerankResult]:  # 定义可重试的重排请求的处理逻辑。
        attempts = max(1, self._settings.max_retries)  # 将总尝试次数设为各项的较大值。
        last: Exception | None = None  # 将末项设为空值。
        for attempt in range(attempts):  # 遍历指定的下标范围，逐项取得尝试序号。
            try:  # 在此捕获后续操作可能产生的异常。
                return self._request_once(payload, expected)  # 返回单次网络请求的结果。
            except TransientFailure as exc:  # 捕获暂时性故障。
                last = exc  # 将末项设为异常。
                if attempt == attempts - 1:  # 当尝试序号等于总尝试次数减去1时进入此分支。
                    break  # 结束当前循环。
                delay = 1.0 * (2**attempt)  # 将退避时长设为1.0乘以2的幂次为尝试序号。
                self._sleep(delay)  # 调用等待函数。
        assert last is not None  # 确认末项不为空值。
        raise last  # 向上抛出当前异常。

    def _request_once(self, payload: dict, expected: int) -> list[RerankResult]:  # 定义单次网络请求的处理逻辑。
        url = self._settings.url  # 取出当前使用的服务地址。
        headers = {  # 将请求头设为字段映射。
            "Authorization": f"Bearer {self._settings.api_key}",  # 填入Authorization字段。
            "Content-Type": "application/json",  # 填入Content-Type字段。
        }  # 结束本组字段。
        try:  # 在此捕获后续操作可能产生的异常。
            response = httpx.post(  # 向服务地址发送提交请求并保存响应。
                url, json=payload, headers=headers, timeout=self._settings.timeout_seconds  # 为提交提供输入。
            )  # 结束当前多行表达式。
        except httpx.TimeoutException as exc:  # 捕获httpx.TimeoutException。
            raise TransientFailure("rerank", "request timed out") from exc  # 向上抛出暂时性故障。
        except httpx.HTTPError as exc:  # 捕获httpx.HTTPError。
            raise Unavailable("rerank", f"cannot reach {url}: {exc}") from exc  # 向上抛出服务不可达异常。

        if response.status_code != 200:  # 当响应状态码不等于200时进入此分支。
            self._raise_for_status(response)  # 按状态码分类错误。

        return self._parse(response, expected)  # 返回解析响应的结果。

    def _raise_for_status(self, response: httpx.Response) -> None:  # 定义按状态码分类错误的处理逻辑。
        detail = self._extract_message(response)  # 将详情设为提取错误消息的结果。
        status = response.status_code  # 将状态设为响应状态码。

        if status in _RETRYABLE_STATUS:  # 当状态包含在可重试状态码集合时进入此分支。
            raise TransientFailure("rerank", f"HTTP {status}: {detail}", status=status)  # 向上抛出暂时性故障。
        if status in {401, 403}:  # 当状态包含在集合时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "rerank",  # 标注重排凭据校验失败。
                f"credentials rejected (HTTP {status}): {detail}. Check rerank.api_key.",  # 附上认证错误原因并提示检查密钥。
            )  # 结束当前多行表达式。
        if status == 404:  # 当状态等于404时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "rerank",  # 标注无法找到重排接口。
                f"endpoint not found (HTTP 404) at {self._settings.url}. "  # 显示请求失败的实际地址。
                "Bailian reranking is only available on the native path "  # 说明百炼重排需要使用原生协议。
                "/api/v1/services/rerank/text-rerank/text-rerank, never on "  # 给出原生文本重排接口路径。
                "/compatible-mode/v1/rerank.",  # 标出会导致此错误的兼容模式路径。
            )  # 结束当前多行表达式。
        if status == 400:  # 当状态等于400时进入此分支。
            # 该接口的请求错误通常与模型名、空候选列表或文档长度有关。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "rerank",  # 标注请求被重排服务拒绝。
                f"request rejected (HTTP 400): {detail}. "  # 附上供应商拒绝请求的原因。
                f"The configured model is '{self._settings.model}'; it may not exist.",  # 提示核对当前选择的模型名称。
            )  # 结束当前多行表达式。
        raise ContractMismatch("rerank", f"unexpected HTTP {status}: {detail}")  # 向上抛出响应协议异常。

    @staticmethod  # 声明无需实例状态的静态方法。
    def _extract_message(response: httpx.Response) -> str:  # 定义提取错误消息的处理逻辑。
        try:  # 在此捕获后续操作可能产生的异常。
            body = response.json()  # 将正文设为解析后的响应数据。
        except ValueError:  # 捕获值错误。
            return response.text[:300] or "(empty body)"  # 返回文本的指定区间或“(empty body)”。
        if isinstance(body, dict):  # 当正文属于字典类型时进入此分支。
            # 优先解析百炼原生错误结构。
            if body.get("message"):  # 原生接口把错误提示直接放在响应顶层。
                code = body.get("code")  # 尝试取回供应商定义的错误码。
                return f"{code}: {body['message']}" if code else str(body["message"])  # 同时保留错误码和说明，只有说明时直接返回文字。
            # 兼容网关包装在 error 对象中的错误信息。
            error = body.get("error")  # 将错误设为正文中的错误。
            if isinstance(error, dict) and error.get("message"):  # 当错误属于字典类型且错误中的消息时进入此分支。
                return str(error["message"])  # 返回错误中的消息的字符串形式。
        return str(body)[:300]  # 返回正文的字符串形式的指定区间。

    def _parse(self, response: httpx.Response, expected: int) -> list[RerankResult]:  # 定义解析响应的处理逻辑。
        try:  # 在此捕获后续操作可能产生的异常。
            body = response.json()  # 将正文设为解析后的响应数据。
        except (ValueError) as exc:  # 捕获(ValueError)。
            raise ContractMismatch("rerank", "response was not JSON") from exc  # 向上抛出响应协议异常。

        if not isinstance(body, dict):  # 正文不属于字典类型时进入此分支。
            raise ContractMismatch("rerank", "response is not a JSON object")  # 向上抛出响应协议异常。

        output = body.get("output")  # 从原生响应顶层读取输出对象。
        if not isinstance(output, dict):  # 输出不属于字典类型时进入此分支。
            raise ContractMismatch(  # 向上抛出响应协议异常。
                "rerank",  # 标明重排响应结构异常。
                "response has no 'output' object; the endpoint may not be a "  # 指出响应中缺少原生输出对象。
                "text-rerank endpoint",  # 提示检查是否连接了正确的重排服务。
            )  # 结束当前多行表达式。

        results = output.get("results")  # 在输出对象中取出重排结果数组。
        if not isinstance(results, list) or not results:  # 当结果列表属于列表类型为假或结果列表为假时进入此分支。
            raise ContractMismatch("rerank", "response has no 'output.results' array")  # 向上抛出响应协议异常。

        parsed: list[RerankResult] = []  # 初始化解析结果为空列表。
        for item in results:  # 遍历结果列表，逐项取得条目。
            if not isinstance(item, dict):  # 条目不属于字典类型时进入此分支。
                raise ContractMismatch("rerank", "result item is not an object")  # 向上抛出响应协议异常。
            index = item.get("index")  # 取得该结果在原候选列表中的位置。
            score = item.get("relevance_score")  # 取得模型评估的相关性分数。
            if not isinstance(index, int):  # 索引不属于整数类型时进入此分支。
                raise ContractMismatch("rerank", f"result has a non-integer 'index': {index!r}")  # 向上抛出响应协议异常。
            if not isinstance(score, (int, float)):  # 分数不属于整数、浮点数类型时进入此分支。
                raise ContractMismatch(  # 向上抛出响应协议异常。
                    "rerank", f"result has a non-numeric 'relevance_score': {score!r}"  # 在错误中给出无法作为分数使用的值。
                )  # 结束当前多行表达式。
            if not 0 <= index < expected:  # 确保下标能映射到实际提交的某条文档。
                raise ContractMismatch(  # 向上抛出响应协议异常。
                    "rerank", f"result index {index} is outside the {expected} submitted documents"  # 显示越界下标和提交的候选总数。
                )  # 结束当前多行表达式。
            parsed.append(RerankResult(index=index, score=float(score)))  # 将合法下标与统一为浮点数的分数保存为一条结果。

        if len({r.index for r in parsed}) != len(parsed):  # 比较去重后的下标数与总数，以识别重复结果。
            raise ContractMismatch("rerank", "response contains duplicate result indices")  # 向上抛出响应协议异常。

        parsed.sort(key=lambda r: r.score, reverse=True)  # 按相关性从高到低排列候选。
        return parsed  # 返回解析结果。

    def verify(self) -> list[RerankResult]:  # 用相关性明确的样例检查接口连通性和实际排序效果。
        """用相关性明确的样例检查接口连通性和实际排序效果。"""
        query = "猫"  # 将查询设为“猫”。
        documents = ["猫是一种小型猫科动物。", "汽车需要汽油才能行驶。", "天空是蓝色的。"]  # 将猫的描述置于首项，并加入两条明显无关的干扰文本。
        results = self.rerank(query, documents)  # 将结果列表设为重排的结果。
        if not results:  # 当结果列表为空或未启用时进入此分支。
            raise ContractMismatch("rerank", "self-check received no results")  # 向上抛出响应协议异常。
        if results[0].index != 0:  # 自检要求关于猫的段落排在首位。
            raise ContractMismatch(  # 向上抛出响应协议异常。
                "rerank",  # 标注重排自检失败。
                "self-check found the ranking implausible: expected the passage "  # 说明排序结果与已知相关性不符。
                f"about cats to rank first, got index {results[0].index}. "  # 列出实际排在第一位的候选下标。
                f"Scores: {[(r.index, r.score) for r in results]}",  # 附上各候选下标及相关性分数。
            )  # 结束当前多行表达式。
        return results  # 返回结果列表。