"""组装带来源的回答提示，管理有限轮数的历史，并将依赖前文的追问改写为独立检索问题。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import re  # 引入正则表达式模块。
from dataclasses import dataclass, field  # 从 dataclasses 引入本模块使用的类型和工具。

from rag_demo.config import Settings  # 从 rag_demo.config 导入项目配置。
from rag_demo.errors import ProviderError  # 从 rag_demo.errors 导入供应商异常。
from rag_demo.generation.llm import Generator, Message  # 从 rag_demo.generation.llm 引入本模块使用的类型和工具。
from rag_demo.retrieval.hybrid import Context  # 从 rag_demo.retrieval.hybrid 导入引用上下文。

# 提示模型按编号引用文档、只依据给定片段作答，并使用与问题相同的语言。
SYSTEM_PROMPT = """You answer questions about the user's own documents.

Rules:
- Answer only from the numbered context passages provided.
- Cite passages inline using their bracket number, for example [1] or [2][3].
- If the passages do not contain the answer, say plainly that the provided
  documents do not cover it. Do not fill the gap from general knowledge.
- Reply in the same language as the question.
- Be concise. Do not restate the question or narrate what you are doing."""

NO_CONTEXT_ANSWER = (  # 定义无可用上下文时直接返回的回答。
    "There is no content available to answer this question yet. "  # 说明当前没有可支持回答的文档内容。
    "Upload a PDF or Markdown document first."  # 提示先上传受支持的文档。
)  # 结束当前多行表达式。

# 用这些指代词识别可能依赖上一轮内容的提问，减少多余的改写请求。
_ANAPHORA = re.compile(  # 编译用于识别追问中指代词的正则表达式。
    r"(它|他|她|这个|那个|这些|那些|上面|上述|前面|刚才|该|此|其|这样|那样"  # 匹配可能指向前文的中文词语。
    r"|it|its|they|them|this|that|these|those|above|former|latter|previous)",  # 补充英文中的指示词和代词。
    re.IGNORECASE,  # 匹配英文时忽略大小写。
)  # 结束当前多行表达式。

# 提示模型根据对话消解指代，只输出保留原意和语言的独立问题。
REWRITE_PROMPT = """Rewrite the follow-up question so it can be understood on its own.

Use the conversation history to resolve references such as "it", "that document" or
"the second approach". Keep the original meaning, keep the original language, and
do not answer the question.

Reply with the rewritten question only, with no preamble and no quotation marks."""


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class Turn:  # 保存同一轮的提问及回答。
    """保存同一轮的提问及回答。"""

    question: str  # 声明问题字段。
    answer: str  # 声明回答字段。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class Conversation:  # 只保留最近若干轮对话，限制历史内容占用的上下文空间。
    """只保留最近若干轮对话，限制历史内容占用的上下文空间。"""

    turns: list[Turn] = field(default_factory=list)  # 设置历史轮次列表字段的默认值。
    max_turns: int = 6  # 设置历史轮数上限字段的默认值。

    def append(self, question: str, answer: str) -> None:  # 定义追加一项的处理逻辑。
        self.turns.append(Turn(question=question, answer=answer))  # 将问题与回答作为同一轮追加。
        if len(self.turns) > self.max_turns:  # 检查历史轮数是否超过上限。
            # 优先丢弃最早的一轮，为后续追问保留近期内容。
            del self.turns[: len(self.turns) - self.max_turns]  # 按超出的轮数从开头移除最旧的记录。

    def is_empty(self) -> bool:  # 判断当前内容是否为空。
        return not self.turns  # 检查当前是否还没有对话记录。

    def as_messages(self) -> list[Message]:  # 定义生成历史消息列表的处理逻辑。
        messages: list[Message] = []  # 初始化消息列表为空列表。
        for turn in self.turns:  # 按先后顺序展开已保存的各轮对话。
            messages.append(Message(role="user", content=turn.question))  # 先加入本轮的用户提问。
            messages.append(Message(role="assistant", content=turn.answer))  # 再加入与之配对的助手回答。
        return messages  # 返回消息列表。


def needs_rewrite(question: str, conversation: Conversation) -> bool:  # 判断是否需要将追问改成独立问题。
    """判断是否需要将追问改成独立问题；已有足够信息时省去一次模型调用。"""
    if conversation.is_empty():  # 首次提问没有前文可供改写。
        return False  # 返回假。
    stripped = question.strip()  # 去掉问题首尾空白，再判断长度和指代词。
    if len(stripped) >= 40 and not _ANAPHORA.search(stripped):  # 较长且无指代词的问题直接用于检索。
        return False  # 返回假。
    return bool(_ANAPHORA.search(stripped)) or len(stripped) < 12  # 检测到指代词或问题少于十二个字符时尝试改写。


def format_contexts(contexts: list[Context]) -> str:  # 将检索上下文格式化为带编号的引用片段。
    """将检索上下文格式化为带编号的引用片段。"""
    blocks: list[str] = []  # 初始化块列表为空列表。
    for index, context in enumerate(contexts, start=1):  # 引用编号从一开始，供回答中的方括号标记使用。
        location = f", page {context.page}" if context.page else ""  # 页码已知时附上页码。
        heading = f", {context.breadcrumb}" if context.breadcrumb else ""  # 有章节路径时附在文件名与页码之后。
        blocks.append(  # 将格式化文本加入块列表。
            f"[{index}] source: {context.source_name}{location}{heading}\n{context.text}"  # 第一行标注编号和来源，下一行给出正文。
        )  # 结束当前多行表达式。
    return "\n\n".join(blocks)  # 各段来源之间保留空行。


def build_messages(  # 拼接回答请求所需的消息列表。
    question: str,  # 接收问题参数。
    contexts: list[Context],  # 接收上下文列表参数。
    conversation: Conversation | None = None,  # 接收对话历史参数。
) -> list[Message]:  # 结束参数声明，进入函数或语句主体。
    """拼接回答请求所需的消息列表。"""
    messages: list[Message] = [Message(role="system", content=SYSTEM_PROMPT)]  # 首条系统消息规定模型如何使用文档和引用。

    if conversation is not None:  # 当对话历史不为空值时进入此分支。
        messages.extend(conversation.as_messages())  # 把已有的历史问答接在系统提示之后。

    messages.append(  # 在消息列表末尾加入本轮问题和资料。
        Message(  # 创建供模型读取的用户消息。
            role="user",  # 将本条消息标记为用户输入。
            content=(  # 把引用段落与当前问题拼成消息正文。
                f"Context passages:\n\n{format_contexts(contexts)}\n\n"  # 先列出带编号的上下文。
                f"Question: {question}"  # 最后附上当前问题。
            ),  # 结束当前多行表达式。
        )  # 结束当前多行表达式。
    )  # 结束当前多行表达式。
    return messages  # 返回消息列表。


class Answerer:  # 必要时改写追问，并依据检索到的上下文生成回答。
    """必要时改写追问，并依据检索到的上下文生成回答。"""

    def __init__(self, settings: Settings, generator: Generator | None = None) -> None:  # 保存实例所需的配置和依赖。
        self._settings = settings  # 将测试配置设为配置。
        self._generator = generator or Generator(settings.llm)  # 将生成器设为生成器或生成客户端的结果。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def generator(self) -> Generator:  # 定义生成器的处理逻辑。
        return self._generator  # 返回生成器。

    def rewrite(self, question: str, conversation: Conversation) -> tuple[str, bool]:  # 返回检索用问题及是否已改写。
        """返回检索用问题及是否已改写；改写失败时继续用原问题检索。"""
        if not needs_rewrite(question, conversation):  # 问题能够独立理解时省去模型往返。
            return question, False  # 返回原问题并标记未改写。

        history = conversation.as_messages()  # 将对话轮次展开为模型消息。
        messages = [  # 将消息列表设为列表。
            Message(role="system", content=REWRITE_PROMPT),  # 系统消息限定模型只做问题改写。
            *history[-4:],  # 只带最近四条历史消息，让模型理解指代。
            Message(role="user", content=question),  # 把待改写的原问题作为最后一条用户消息。
        ]  # 结束列表。
        try:  # 在此捕获后续操作可能产生的异常。
            rewritten = self._generator.complete(messages).strip()  # 获取模型的完整改写文本并清理外侧空白。
        except ProviderError:  # 捕获供应商异常。
            return question, False  # 上游请求失败时继续用原问题检索。
        except Exception:  # 捕获通用异常。
            return question, False  # 意外异常同样回退到原提问。

        # 模型输出为空或像是在回答问题时，保留原提问。
        if not rewritten or len(rewritten) > 600:  # 丢弃空响应或超过六百字符的可疑结果。
            return question, False  # 保留原问题，告知调用方改写未采用。
        rewritten = rewritten.strip().strip('"').strip("“”").strip()  # 依次去掉首尾空白及包住问题的中英文双引号。
        if not rewritten:  # 当改写标记为空或未启用时进入此分支。
            return question, False  # 清理后没有实际内容时仍使用原问题。
        return rewritten, True  # 返回清理后的独立问题并标记改写成功。

    def stream_answer(  # 逐段输出回答。
        self,  # 通过实例读取本方法所需状态。
        question: str,  # 接收问题参数。
        contexts: list[Context],  # 接收上下文列表参数。
        conversation: Conversation | None = None,  # 接收对话历史参数。
    ):  # 结束参数声明，进入函数或语句主体。
        """逐段输出回答。没有上下文时直接告知缺少相关内容，避免模型凭空作答。"""
        if not contexts:  # 没有可引用的文本时跳过模型调用。
            yield NO_CONTEXT_ANSWER  # 直接向客户端提供空知识库说明。
            return  # 结束当前处理。

        messages = build_messages(question, contexts, conversation)  # 把本轮问题、上下文和可选历史整理成消息。
        yield from self._generator.stream_chat(messages)  # 逐段转发模型生成的回答。

    def answer(  # 定义回答的处理逻辑。
        self,  # 通过实例读取本方法所需状态。
        question: str,  # 接收问题参数。
        contexts: list[Context],  # 接收上下文列表参数。
        conversation: Conversation | None = None,  # 接收对话历史参数。
    ) -> str:  # 结束参数声明，进入函数或语句主体。
        return "".join(self.stream_answer(question, contexts, conversation))  # 为需要完整回答的调用方拼接所有输出片段。


def describe_generation_error(error: Exception) -> str:  # 将生成异常转为用户可操作的错误提示。
    """将生成异常转为用户可操作的错误提示。"""
    from rag_demo.errors import ConfigurationIssue, ContractMismatch, Unavailable  # 从 rag_demo.errors 引入本模块使用的类型和工具。

    if isinstance(error, ConfigurationIssue):  # 当错误属于供应商配置异常类型时进入此分支。
        return f"Generation is misconfigured: {error.message}"  # 返回格式化文本。
    if isinstance(error, Unavailable):  # 当错误属于服务不可达异常类型时进入此分支。
        return f"The generation service is unreachable: {error.message}"  # 返回格式化文本。
    if isinstance(error, ContractMismatch):  # 当错误属于响应协议异常类型时进入此分支。
        return f"The generation service returned an unexpected response: {error.message}"  # 返回格式化文本。
    return f"Generation failed: {type(error).__name__}: {error}"  # 返回格式化文本。