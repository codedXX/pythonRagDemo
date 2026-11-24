"""通过兼容 OpenAI 的中转服务请求生成模型。

使用 /v1/chat/completions 以便和后续评测工具共用协议。启动自检实际读取流式响应，以提前发现模型、凭据或流式传输配置问题。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import json  # 引入结构化数据的编解码工具。
import time  # 引入时间工具。
from collections.abc import Iterator, Sequence  # 从 collections.abc 引入本模块使用的类型和工具。
from dataclasses import dataclass  # 从 dataclasses 导入数据类装饰器。

import httpx  # 引入网络客户端库。

from rag_demo.config import LLMSettings  # 从 rag_demo.config 导入生成模型配置。
from rag_demo.errors import (  # 从 rag_demo.errors 引入本模块使用的类型和工具。
    ConfigurationIssue,  # 导入供应商配置异常。
    ContractMismatch,  # 导入响应协议异常。
    TransientFailure,  # 导入暂时性故障。
    Unavailable,  # 导入服务不可达异常。
)  # 结束当前多行表达式。

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}  # 将可重试状态码集合设为集合。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class Message:  # 表示带角色和内容的对话消息。
    """表示带角色和内容的对话消息。"""

    role: str  # 声明角色字段。
    content: str  # 声明内容字段。

    def as_payload(self) -> dict[str, str]:  # 定义组装请求载荷的处理逻辑。
        return {"role": self.role, "content": self.content}  # 将消息角色和正文转换为上游接口要求的键值结构。


class Generator:  # 通过聊天补全接口读取流式或完整回答。
    """通过聊天补全接口读取流式或完整回答。"""

    def __init__(self, settings: LLMSettings, sleep=time.sleep) -> None:  # 保存实例所需的配置和依赖。
        self._settings = settings  # 将测试配置设为配置。
        self._sleep = sleep  # 保存传入的等待函数。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def settings(self) -> LLMSettings:  # 取得当前使用的配置。
        return self._settings  # 返回测试配置。

    # 对外提供的生成方法。

    def stream_chat(self, messages: Sequence[Message]) -> Iterator[str]:  # 收到一段模型输出就交给调用方。
        """收到一段模型输出就交给调用方。"""
        for chunk in self._stream(messages):  # 遍历读取响应流的结果，逐项取得片段。
            yield chunk  # 向调用方逐项提供片段。

    def complete(self, messages: Sequence[Message]) -> str:  # 一次性读取完整回答，供问题改写使用。
        """一次性读取完整回答，供问题改写使用。"""
        return "".join(self.stream_chat(messages))  # 连接所有响应片段，得到完整回答。

    def verify(self) -> str:  # 检查模型名称、访问凭据和流式响应。
        """检查模型名称、访问凭据和流式响应。"""
        pieces = list(  # 实际保存流式响应的各段输出以供自检。
            self.stream_chat(  # 自检走与问答相同的流式入口。
                [  # 开始列出各项。
                    Message(role="user", content="Reply with the single word: ok"),  # 用只要求一个词的简短提问探测生成服务。
                ]  # 结束列表。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。
        text = "".join(pieces)  # 将文本设为拼接后的片段列表。
        if not text.strip():  # 当不满足“清理首尾空白后的文本”时进入此分支。
            raise ContractMismatch("llm", "self-check received an empty response")  # 向上抛出响应协议异常。
        if len(pieces) < 1:  # pragma: no cover  # 防御性处理。
            raise ContractMismatch("llm", "self-check observed no streamed chunks")  # 向上抛出响应协议异常。
        return text.strip()  # 返回清理首尾空白后的文本。

    # 内部传输与解析方法。

    def _endpoint(self) -> str:  # 定义获取接口地址的处理逻辑。
        return self._settings.base_url.rstrip("/") + "/chat/completions"  # 从中转服务根地址构造聊天补全路径。

    def _payload(self, messages: Sequence[Message], stream: bool) -> dict:  # 定义构造请求体的处理逻辑。
        payload = {  # 将载荷设为字段映射。
            "model": self._settings.model,  # 填入模型字段。
            "messages": [m.as_payload() for m in messages],  # 按对话原顺序序列化每条消息。
            "temperature": self._settings.temperature,  # 填入采样温度字段。
            "max_tokens": self._settings.max_tokens,  # 填入最大生成词元数字段。
            "stream": stream,  # 填入数据流字段。
        }  # 结束本组字段。
        if stream:  # 当数据流有效时进入此分支。
            # 让网关按事件发送数据，避免等到整段回答完成才输出。
            payload["stream_options"] = {"include_usage": False}  # 流式请求关闭额外的用量统计块。
        return payload  # 返回载荷。

    def _headers(self) -> dict[str, str]:  # 定义构造请求头的处理逻辑。
        return {  # 返回字段映射。
            "Authorization": f"Bearer {self._settings.api_key}",  # 填入Authorization字段。
            "Content-Type": "application/json",  # 填入Content-Type字段。
        }  # 结束本组字段。

    def _stream(self, messages: Sequence[Message]) -> Iterator[str]:  # 定义读取响应流的处理逻辑。
        if not self._settings.api_key:  # 当接口密钥为空或未启用时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "llm",  # 标注故障来自生成服务。
                "no API key configured; set llm.api_key in config.yaml "  # 提示缺少生成服务密钥并给出本地配置项。
                "or RAG__LLM__API_KEY",  # 给出对应的环境变量名。
            )  # 结束当前多行表达式。

        url = self._endpoint()  # 将服务地址设为获取接口地址的结果。
        attempts = 3  # 将总尝试次数设为3。
        for attempt in range(attempts):  # 遍历指定的下标范围，逐项取得尝试序号。
            emitted = False  # 本轮尝试尚未向调用方交付任何文字。
            try:  # 在此捕获后续操作可能产生的异常。
                with httpx.stream(  # 在数据流的结果的作用域内执行后续操作。
                    "POST",  # 以提交请求的方式建立流式连接。
                    url,  # 传入服务地址。
                    json=self._payload(messages, stream=True),  # 构造启用流式输出的消息载荷。
                    headers=self._headers(),  # 传入请求头参数。
                    timeout=self._settings.timeout_seconds,  # 传入超时参数。
                ) as response:  # 为数据流提供输入。
                    if response.status_code != 200:  # 当响应状态码不等于200时进入此分支。
                        body = response.read().decode("utf-8", "replace")  # 失败时读取错误正文，并容许有损解码以保留诊断信息。
                        self._raise_for_status(response.status_code, body)  # 按状态码分类错误。

                    for line in response.iter_lines():  # 逐行读取上游推送的事件。
                        if not line:  # 当文本行为空或未启用时进入此分支。
                            continue  # 跳过本项，继续处理下一项。
                        if not line.startswith("data:"):  # 只处理承载数据的事件行。
                            continue  # 跳过本项，继续处理下一项。
                        data = line[len("data:") :].strip()  # 去掉数据行前缀及两端空白。
                        if data == "[DONE]":  # 供应商发送结束标记时停止读取。
                            return  # 结束当前处理。
                        piece = self._parse_chunk(data)  # 从本次数据块提取新增回答文本。
                        if piece:  # 当片段有效时进入此分支。
                            emitted = True  # 记住回答已经开始输出，避免中途重试造成重复。
                            yield piece  # 立即交付增量文本。
                return  # 结束当前处理。
            except TransientFailure:  # 捕获暂时性故障。
                # 只在尚未输出文本时重试，防止用户收到重复片段。
                if emitted or attempt == attempts - 1:  # 已经输出部分回答或耗尽尝试次数时，保留原异常。
                    raise  # 向上抛出当前异常。
                self._sleep(1.0 * (2**attempt))  # 按尝试序号以二的幂次增加退避时间。

    @staticmethod  # 声明无需实例状态的静态方法。
    def _parse_chunk(data: str) -> str:  # 定义解析响应片段的处理逻辑。
        try:  # 在此捕获后续操作可能产生的异常。
            payload = json.loads(data)  # 将载荷设为loads的结果。
        except ValueError as exc:  # 捕获值错误。
            raise ContractMismatch("llm", f"stream chunk was not JSON: {data[:120]}") from exc  # 向上抛出响应协议异常。

        choices = payload.get("choices")  # 读取响应提供的候选回答。
        if not isinstance(choices, list) or not choices:  # 当choices属于列表类型为假或choices为假时进入此分支。
            return ""  # 返回空字符串。
        first = choices[0]  # 只处理排在第一的回答候选。
        if not isinstance(first, dict):  # 首项不属于字典类型时进入此分支。
            return ""  # 返回空字符串。
        delta = first.get("delta")  # 优先解析流式协议中的增量对象。
        if isinstance(delta, dict):  # 当delta属于字典类型时进入此分支。
            content = delta.get("content")  # 提取当前增量中的正文。
            if isinstance(content, str):  # 当内容属于字符串类型时进入此分支。
                return content  # 返回内容。
        # 兼容把流式文本放在 message 字段里的服务。
        message = first.get("message")  # 兼容把回答放在常规消息字段内的供应商。
        if isinstance(message, dict):  # 当消息属于字典类型时进入此分支。
            content = message.get("content")  # 从常规消息中提取正文。
            if isinstance(content, str):  # 当内容属于字符串类型时进入此分支。
                return content  # 返回内容。
        return ""  # 返回空字符串。

    def _raise_for_status(self, status: int, body: str) -> None:  # 定义按状态码分类错误的处理逻辑。
        detail = self._extract_message(body)  # 将详情设为提取错误消息的结果。
        if status in _RETRYABLE_STATUS:  # 当状态包含在可重试状态码集合时进入此分支。
            raise TransientFailure("llm", f"HTTP {status}: {detail}", status=status)  # 向上抛出暂时性故障。
        if status in {401, 403}:  # 当状态包含在集合时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "llm",  # 标注生成服务认证失败。
                f"credentials rejected (HTTP {status}): {detail}. Check llm.api_key.",  # 提示核对凭据并列出服务端原因。
            )  # 结束当前多行表达式。
        if status == 404:  # 当状态等于404时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "llm",  # 标注找不到所请求的生成接口。
                f"endpoint not found (HTTP 404): {detail}. Check llm.base_url and "  # 提示核对中转根地址及模型名称。
                f"llm.model (currently '{self._settings.model}').",  # 列出当前实际请求的模型。
            )  # 结束当前多行表达式。
        if status == 400:  # 当状态等于400时进入此分支。
            raise ConfigurationIssue(  # 向上抛出供应商配置异常。
                "llm",  # 标注生成请求被拒绝。
                f"request rejected (HTTP 400): {detail}. The configured model is "  # 附上错误原因及模型配置。
                f"'{self._settings.model}'; the relay may not offer it.",  # 提示中转服务可能未提供该模型。
            )  # 结束当前多行表达式。
        raise ContractMismatch("llm", f"unexpected HTTP {status}: {detail}")  # 向上抛出响应协议异常。

    @staticmethod  # 声明无需实例状态的静态方法。
    def _extract_message(body: str) -> str:  # 定义提取错误消息的处理逻辑。
        try:  # 在此捕获后续操作可能产生的异常。
            payload = json.loads(body)  # 将载荷设为loads的结果。
        except ValueError:  # 捕获值错误。
            return body[:300] or "(empty body)"  # 返回正文的指定区间或“(empty body)”。
        if isinstance(payload, dict):  # 当载荷属于字典类型时进入此分支。
            error = payload.get("error")  # 将错误设为载荷中的错误。
            if isinstance(error, dict) and error.get("message"):  # 当错误属于字典类型且错误中的消息时进入此分支。
                return str(error["message"])  # 返回错误中的消息的字符串形式。
            if payload.get("message"):  # 当载荷中的消息时进入此分支。
                return str(payload["message"])  # 返回载荷中的消息的字符串形式。
        return body[:300]  # 返回正文的指定区间。

    def list_models(self) -> list[str]:  # 列出中转服务公布的模型，便于排查配置中的模型名称。
        """列出中转服务公布的模型，便于排查配置中的模型名称。"""
        if not self._settings.api_key:  # 当接口密钥为空或未启用时进入此分支。
            raise ConfigurationIssue("llm", "no API key configured")  # 向上抛出供应商配置异常。
        url = self._settings.base_url.rstrip("/") + "/models"  # 拼接中转服务公开模型列表的查询地址。
        try:  # 在此捕获后续操作可能产生的异常。
            response = httpx.get(url, headers=self._headers(), timeout=30.0)  # 向服务地址发送读取请求并保存响应。
        except httpx.HTTPError as exc:  # 捕获httpx.HTTPError。
            raise Unavailable("llm", f"cannot reach {url}: {exc}") from exc  # 向上抛出服务不可达异常。
        if response.status_code != 200:  # 当响应状态码不等于200时进入此分支。
            raise ContractMismatch(  # 向上抛出响应协议异常。
                "llm", f"model listing returned HTTP {response.status_code}"  # 将模型列表查询失败的状态码写入错误说明。
            )  # 结束当前多行表达式。
        try:  # 在此捕获后续操作可能产生的异常。
            body = response.json()  # 将正文设为解析后的响应数据。
        except ValueError as exc:  # 捕获值错误。
            raise ContractMismatch("llm", "model listing was not JSON") from exc  # 向上抛出响应协议异常。
        data = body.get("data") if isinstance(body, dict) else None  # 将数据设为按条件选取的值。
        if not isinstance(data, list):  # 数据不属于列表类型时进入此分支。
            raise ContractMismatch("llm", "model listing has no 'data' array")  # 向上抛出响应协议异常。
        return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]  # 返回从数据中提取的各项。