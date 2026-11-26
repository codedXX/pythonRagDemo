"""用中文分词和 BM25 实现关键词召回。

在内存中独立建立索引，方便观察各通道排名，并调整用户词典和停用词。该做法面向数十份文档的小型知识库。索引从 Milvus 中有效的子块重建，使关键词与向量检索使用同一批内容。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import re  # 引入正则表达式模块。
import threading  # 引入线程工具。
from collections.abc import Iterable, Sequence  # 从 collections.abc 引入本模块使用的类型和工具。
from dataclasses import dataclass, field  # 从 dataclasses 引入本模块使用的类型和工具。

import jieba  # 引入jieba。
import bm25s  # 引入bm25s。

# 识别拉丁文字、数字及中日韩文字片段，按文字类型选择分词方式。
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?")  # 识别拉丁单词、标识符以及带小数点的数字。

DEFAULT_STOPWORDS = frozenset(  # 默认过滤这些高频虚词，减少无区分度的匹配。
    {  # 开始构造字段映射。
        "的",  # 将此常用词列入默认停用词表。
        "了",  # 将此常用词列入默认停用词表。
        "和",  # 将此常用词列入默认停用词表。
        "是",  # 将此常用词列入默认停用词表。
        "在",  # 将此常用词列入默认停用词表。
        "我",  # 将此常用词列入默认停用词表。
        "有",  # 将此常用词列入默认停用词表。
        "就",  # 将此常用词列入默认停用词表。
        "不",  # 将此常用词列入默认停用词表。
        "人",  # 将此常用词列入默认停用词表。
        "都",  # 将此常用词列入默认停用词表。
        "一",  # 将此常用词列入默认停用词表。
        "一个",  # 将此常用词列入默认停用词表。
        "上",  # 将此常用词列入默认停用词表。
        "也",  # 将此常用词列入默认停用词表。
        "很",  # 将此常用词列入默认停用词表。
        "到",  # 将此常用词列入默认停用词表。
        "说",  # 将此常用词列入默认停用词表。
        "要",  # 将此常用词列入默认停用词表。
        "去",  # 将此常用词列入默认停用词表。
        "你",  # 将此常用词列入默认停用词表。
        "会",  # 将此常用词列入默认停用词表。
        "着",  # 将此常用词列入默认停用词表。
        "没有",  # 将此常用词列入默认停用词表。
        "看",  # 将此常用词列入默认停用词表。
        "好",  # 将此常用词列入默认停用词表。
        "自己",  # 将此常用词列入默认停用词表。
        "这",  # 将此常用词列入默认停用词表。
        "the",  # 将此常用词列入默认停用词表。
        "a",  # 将此常用词列入默认停用词表。
        "an",  # 将此常用词列入默认停用词表。
        "of",  # 将此常用词列入默认停用词表。
        "and",  # 将此常用词列入默认停用词表。
        "or",  # 将此常用词列入默认停用词表。
        "to",  # 将此常用词列入默认停用词表。
        "in",  # 将此常用词列入默认停用词表。
        "is",  # 将此常用词列入默认停用词表。
        "are",  # 将此常用词列入默认停用词表。
        "for",  # 将此常用词列入默认停用词表。
        "on",  # 将此常用词列入默认停用词表。
        "with",  # 将此常用词列入默认停用词表。
        "as",  # 将此常用词列入默认停用词表。
        "by",  # 将此常用词列入默认停用词表。
        "at",  # 将此常用词列入默认停用词表。
        "be",  # 将此常用词列入默认停用词表。
        "this",  # 将此常用词列入默认停用词表。
        "that",  # 将此常用词列入默认停用词表。
        "it",  # 将此常用词列入默认停用词表。
    }  # 结束本组字段。
)  # 结束当前多行表达式。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class LexicalHit:  # 表示一条关键词检索命中。
    """表示一条关键词检索命中。"""

    id: str  # 声明标识字段。
    score: float  # 声明分数字段。
    rank: int  # 声明排名字段。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class LexicalIndex:  # 在内存中保存子块的 BM25 索引。
    """在内存中保存子块的 BM25 索引；小型语料发生变化后整体重建，以保持统计一致。"""

    user_dictionary: Sequence[str] = ()  # 设置用户词表字段的默认值。
    stopwords: frozenset[str] = DEFAULT_STOPWORDS  # 设置停用词字段的默认值。
    k1: float = 1.5  # 控制词频增加时分数趋于饱和的速度。
    b: float = 0.75  # 控制文档长度归一化的强度。

    _ids: list[str] = field(default_factory=list)  # 设置标识列表字段的默认值。
    _texts: list[str] = field(default_factory=list)  # 设置待编码文本字段的默认值。
    _retriever: object | None = None  # 设置检索器字段的默认值。
    _lock: threading.Lock = field(default_factory=threading.Lock)  # 设置互斥锁字段的默认值。

    def __post_init__(self) -> None:  # 定义__post_init__的处理逻辑。
        self._configure_segmentation()  # 在首次分词前加载用户词典。

    def _configure_segmentation(self) -> None:  # 定义加载分词设置的处理逻辑。
        for word in self.user_dictionary:  # 遍历用户词表，逐项取得词条。
            if word:  # 当词条有效时进入此分支。
                jieba.add_word(word)  # 将非空的自定义词加入分词器。

    # 索引构建。

    def rebuild(self, rows: Iterable[dict]) -> None:  # 根据含标识和正文的记录替换当前索引。
        """根据含标识和正文的记录替换当前索引。"""
        ids: list[str] = []  # 初始化标识列表为空列表。
        texts: list[str] = []  # 初始化待编码文本为空列表。
        tokens: list[list[str]] = []  # 初始化词元列表为空列表。

        for row in rows:  # 遍历记录列表，逐项取得记录。
            text = row.get("text") or ""  # 将文本设为记录中的文本或空字符串。
            if not text.strip():  # 当不满足“清理首尾空白后的文本”时进入此分支。
                continue  # 跳过本项，继续处理下一项。
            ids.append(str(row["id"]))  # 按语料的存放顺序保留子块主键。
            texts.append(text)  # 按相同顺序保存原文以支持按标识回查。
            tokens.append(self.tokenize(text))  # 对正文分词，生成该文档的检索词项。

        with self._lock:  # 持锁切换语料和索引，使查询看到同一版本。
            self._ids = ids  # 替换语料位置对应的子块标识。
            self._texts = texts  # 替换与标识同序的正文。
            if tokens:  # 当词元列表有效时进入此分支。
                # 使用已分词的语料建索引，由底层库保存评分所需的结构。
                self._retriever = bm25s.BM25(k1=self.k1, b=self.b)  # 用配置的词频和长度参数创建评分器。
                self._retriever.index(tokens, show_progress=False)  # 按已分词的语料构建索引，并关闭建库进度显示。
            else:  # 处理其余情况。
                self._retriever = None  # 将检索器设为空值。

    def add(self, rows: Sequence[dict]) -> None:  # 追加记录后重建索引，因为新文档也会改变旧文档所依赖的词频统计。
        """追加记录后重建索引，因为新文档也会改变旧文档所依赖的词频统计。"""
        combined = list(zip(self._ids, self._texts, strict=True))  # 将现有子块标识与正文按原位置配对。
        combined.extend((str(row["id"]), row.get("text") or "") for row in rows)  # 把新行的标识和正文追加到语料末尾。
        self.rebuild([{"id": i, "text": t} for i, t in combined])  # 根据完整语料重新计算词频和索引。

    # 关键词检索。

    def search(self, query: str, limit: int = 40) -> list[LexicalHit]:  # 按相关性降序返回不超过指定数量的匹配。
        """按相关性降序返回不超过指定数量的匹配。"""
        with self._lock:  # 在同一把锁内获取检索器及与其匹配的标识快照。
            retriever = self._retriever  # 取出当前使用的检索器。
            ids = list(self._ids)  # 复制标识列表，供解读本次评分结果使用。

        if retriever is None or not ids or not query.strip():  # 索引尚未建立或查询无正文时直接返回。
            return []  # 返回空列表。

        query_tokens = self.tokenize(query)  # 用与索引相同的分词规则处理问题。
        if not query_tokens:  # 当查询词项为空或未启用时进入此分支。
            return []  # 返回空列表。

        k = min(max(1, limit), len(ids))  # 把请求条数限制在一到语料总数之间。
        results, scores = retriever.retrieve([query_tokens], k=k, show_progress=False)  # 用分词后的单条查询获取前 k 个 BM25 结果及分数。

        hits: list[LexicalHit] = []  # 初始化命中项为空列表。
        for rank, (index, score) in enumerate(zip(results[0], scores[0], strict=True)):  # 遍历逐项配对的结果及各项的下标，逐项取得组合值。
            if index < 0 or index >= len(ids):  # 过滤超出标识列表范围的填充下标。
                continue  # 跳过本项，继续处理下一项。
            # 结果不足时底层库可能补入占位下标。
            if score <= 0 and rank > 0 and float(score) == 0.0:  # 跳过非首位的零分项。
                continue  # 跳过本项，继续处理下一项。
            hits.append(  # 将关键词命中项的结果加入命中项。
                LexicalHit(id=ids[int(index)], score=float(score), rank=rank)  # 将语料下标还原为子块标识，同时保留得分和原始排名。
            )  # 结束当前多行表达式。
        return hits  # 返回命中项。

    # 文本分词。

    def tokenize(self, text: str) -> list[str]:  # 建索引和查询共用分词规则。
        """建索引和查询共用分词规则。中文使用搜索模式拆出复合词的子词，拉丁文字和数字单独处理。"""
        if not text:  # 当文本为空或未启用时进入此分支。
            return []  # 返回空列表。

        tokens: list[str] = []  # 初始化词元列表为空列表。
        for chunk in _split_scripts(text):  # 按文字种类切成连续片段，分别分词。
            if _is_cjk(chunk):  # 中日韩统一表意文字片段使用中文分词器。
                tokens.extend(  # 将从搜索模式分词的结果中提取的各项逐项加入词元列表。
                    token  # 将通过过滤的词项加入列表。
                    for token in jieba.cut_for_search(chunk)  # 使用搜索模式同时获得复合词及其子词。
                    if token.strip() and token not in self.stopwords  # 舍弃空白词和配置的停用词。
                )  # 结束当前多行表达式。
            else:  # 处理其余情况。
                tokens.extend(  # 将从finditer的结果中提取的各项逐项加入词元列表。
                    match.group(0).lower()  # 把命中的拉丁文字转为小写。
                    for match in _LATIN.finditer(chunk)  # 逐个识别单词、标识符和数字。
                    if match.group(0).lower() not in self.stopwords  # 按小写形式匹配停用词。
                )  # 结束当前多行表达式。
        return [token for token in tokens if len(token) > 1 or _is_cjk(token)]  # 保留多字符词项，也允许单字汉字参与匹配。

    # 索引信息读取。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def size(self) -> int:  # 返回索引中的条目数。
        with self._lock:  # 在互斥锁的作用域内执行后续操作。
            return len(self._ids)  # 返回标识列表的长度。

    def ids(self) -> list[str]:  # 返回当前索引中保存的标识。
        with self._lock:  # 在互斥锁的作用域内执行后续操作。
            return list(self._ids)  # 返回标识列表的列表形式。

    def text_of(self, identifier: str) -> str | None:  # 按标识读取已编入索引的正文。
        with self._lock:  # 在互斥锁的作用域内执行后续操作。
            try:  # 在此捕获后续操作可能产生的异常。
                return self._texts[self._ids.index(identifier)]  # 根据标识在索引中的位置取回原文。
            except ValueError:  # 捕获值错误。
                return None  # 返回空值。


def _is_cjk(text: str) -> bool:  # 定义_is_cjk的处理逻辑。
    return any("\u4e00" <= char <= "\u9fff" for char in text)  # 检测文本中是否有基本汉字区的字符。


def _split_scripts(text: str) -> list[str]:  # 按文字类型切开文本，让中文分词器处理中文，保留拉丁标识符与数字。
    """按文字类型切开文本，让中文分词器处理中文，保留拉丁标识符与数字。"""
    runs: list[str] = []  # 初始化连续文本段为空列表。
    current = ""  # 将当前值设为空字符串。
    current_is_cjk = None  # 将当前片段的文字类别设为空值。
    for char in text:  # 遍历文本，逐项取得字符。
        char_is_cjk = "\u4e00" <= char <= "\u9fff"  # 用字符码位判断当前字符是否属于基本汉字区。
        if current_is_cjk is None or char_is_cjk == current_is_cjk:  # 首个字符或同类字符继续当前片段。
            current += char  # 将字符累加到当前值。
            current_is_cjk = char_is_cjk  # 将当前片段的文字类别设为当前字符是否为汉字区字符。
        else:  # 处理其余情况。
            if current:  # 当当前值有效时进入此分支。
                runs.append(current)  # 将当前值加入连续文本段。
            current = char  # 文字类型切换后，以当前字符开一个新片段。
            current_is_cjk = char_is_cjk  # 将当前片段的文字类别设为当前字符是否为汉字区字符。
    if current:  # 当当前值有效时进入此分支。
        runs.append(current)  # 将当前值加入连续文本段。
    return runs  # 返回连续文本段。