"""将文档切成供生成使用的父块，以及供检索使用的子块。

子块较短，一个向量能更准确地代表匹配的内容。命中子块后通过父块标识取回完整上下文。先按标题和段落等结构切分，只对仍然过长的片段尝试逐句计算向量，寻找话题边界。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import hashlib  # 引入摘要计算模块。
import re  # 引入正则表达式模块。
from dataclasses import dataclass, field  # 从 dataclasses 引入本模块使用的类型和工具。

from rag_demo.config import ChunkingSettings  # 从 rag_demo.config 导入切块配置。
from rag_demo.parsers.base import ParsedDocument, SegmentKind, StructuralSegment  # 从 rag_demo.parsers.base 引入本模块使用的类型和工具。

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=[.])\s+(?=[A-Z\u4e00-\u9fff])")  # 编译分句规则，用句末标点及其后续文字识别句子边界。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class Parent:  # 提供给语言模型的完整上下文块。
    """提供给语言模型的完整上下文块。"""

    parent_id: str  # 声明所属父块标识字段。
    doc_id: str  # 声明文档标识字段。
    text: str  # 声明文本字段。
    breadcrumb: str  # 声明标题路径字段。
    page_start: int  # 声明起始页字段。
    source_name: str  # 声明来源文件名字段。
    index: int  # 声明索引字段。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class Child:  # 建立向量并参与检索的子块。
    """建立向量并参与检索的子块。"""

    id: str  # 声明标识字段。
    doc_id: str  # 声明文档标识字段。
    parent_id: str  # 声明所属父块标识字段。
    child_idx: int  # 声明子块序号字段。
    text: str  # 声明文本字段。
    breadcrumb: str  # 声明标题路径字段。
    page: int  # 声明页码字段。
    source_name: str  # 声明来源文件名字段。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class ChunkSet:  # 保存单份文档切分得到的父块和子块。
    """保存单份文档切分得到的父块和子块。"""

    doc_id: str  # 声明文档标识字段。
    source_name: str  # 声明来源文件名字段。
    parents: list[Parent] = field(default_factory=list)  # 设置父块列表字段的默认值。
    children: list[Child] = field(default_factory=list)  # 设置子块列表字段的默认值。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def char_count(self) -> int:  # 定义字符数的处理逻辑。
        return sum(len(parent.text) for parent in self.parents)  # 对父块正文的长度求和，避免重复计算重叠子块。


class Chunker:  # 先按文档结构生成父块，再把父块切成可检索的子块。
    """先按文档结构生成父块，再把父块切成可检索的子块。"""

    def __init__(self, settings: ChunkingSettings, embedder=None) -> None:  # 保存实例所需的配置和依赖。
        self._settings = settings  # 将测试配置设为配置。
        # 按需注入向量服务，使普通结构切分可以离线使用和测试。
        self._embedder = embedder  # 保存传入的向量服务。

    # 文档切分入口。

    def chunk(self, document: ParsedDocument, doc_id: str) -> ChunkSet:  # 生成文档的父块、子块及其对应关系。
        """生成文档的父块、子块及其对应关系。"""
        result = ChunkSet(doc_id=doc_id, source_name=document.name)  # 用文档标识和文件名初始化分块结果。
        groups = self._group_segments(document.segments)  # 按照标题及其路径将结构片段分组。

        parent_index = 0  # 将父块序号设为0。
        for group in groups:  # 遍历分组列表，逐项取得分组。
            for parent_text, breadcrumb, page in self._split_group_to_parents(group):  # 逐个取得父块的正文、标题路径和页码。
                parent_id = f"{doc_id}_p{parent_index}"  # 文档标识与父块序号共同构成稳定主键。
                result.parents.append(  # 将父块的结果加入父块列表。
                    Parent(  # 将正文和来源信息封装为父块。
                        parent_id=parent_id,  # 传入所属父块标识参数。
                        doc_id=doc_id,  # 传入文档标识参数。
                        text=parent_text,  # 传入文本参数。
                        breadcrumb=breadcrumb,  # 传入标题路径参数。
                        page_start=page,  # 传入起始页参数。
                        source_name=document.name,  # 传入来源文件名参数。
                        index=parent_index,  # 保存父块在文档中的顺序。
                    )  # 结束当前多行表达式。
                )  # 结束当前多行表达式。
                for child_idx, child_text in enumerate(  # 为当前父块的子块从零开始编号。
                    self._split_parent_to_children(parent_text)  # 按目标长度和重叠范围滑动切分父块。
                ):  # 结束参数声明，进入函数或语句主体。
                    result.children.append(  # 将子块的结果加入子块列表。
                        Child(  # 将命中文本及其所属父块封装为子块。
                            id=f"{parent_id}_c{child_idx}",  # 把父块标识和子块序号组合为子块主键。
                            doc_id=doc_id,  # 传入文档标识参数。
                            parent_id=parent_id,  # 传入所属父块标识参数。
                            child_idx=child_idx,  # 传入子块序号参数。
                            text=child_text,  # 传入文本参数。
                            # 子块携带标题路径，让重排模型能理解“如下表所示”一类依赖上下文的片段。
                            breadcrumb=breadcrumb,  # 传入标题路径参数。
                            page=page,  # 传入页码参数。
                            source_name=document.name,  # 传入来源文件名参数。
                        )  # 结束当前多行表达式。
                    )  # 结束当前多行表达式。
                parent_index += 1  # 处理完该父块后，为下一个父块递增序号。

        return result  # 返回结果。

    # 按标题分组。

    def _group_segments(  # 将同一标题下的片段归为一组，让父块保留正文所属的标题。
        self, segments: list[StructuralSegment]  # 组合当前结果所需的各项数据。
    ) -> list[list[StructuralSegment]]:  # 结束参数声明，进入函数或语句主体。
        """将同一标题下的片段归为一组，让父块保留正文所属的标题。"""
        groups: list[list[StructuralSegment]] = []  # 初始化分组列表为空列表。
        current: list[StructuralSegment] = []  # 初始化当前值为空列表。

        for segment in segments:  # 遍历结构片段列表，逐项取得结构片段。
            if segment.kind is SegmentKind.HEADING:  # 遇到标题时结束上一组片段。
                # 遇到标题时另起一组，并把标题放在新组开头。
                if current:  # 当当前值有效时进入此分支。
                    groups.append(current)  # 将当前值加入分组列表。
                current = [segment]  # 从当前标题开始收集新一组内容。
                continue  # 跳过本项，继续处理下一项。
            if not current:  # 当当前值为空或未启用时进入此分支。
                current = [segment]  # 将当前值设为结构片段。
                continue  # 跳过本项，继续处理下一项。
            # 所属标题路径改变时，从当前片段重新分组。
            previous_path = current[-1].heading_path  # 取得当前组最后一个片段的标题路径。
            if segment.heading_path != previous_path and segment.heading_path:  # 新片段具有不同的非空标题路径时另起一组。
                groups.append(current)  # 将当前值加入分组列表。
                current = [segment]  # 将当前值设为结构片段。
            else:  # 处理其余情况。
                current.append(segment)  # 将结构片段加入当前值。

        if current:  # 循环结束后补入末组片段。
            groups.append(current)  # 将当前值加入分组列表。
        return groups  # 返回分组列表。

    # 生成父块。

    def _split_group_to_parents(  # 将同组片段转成正文、标题路径和页码三元组。
        self, group: list[StructuralSegment]  # 组合当前结果所需的各项数据。
    ) -> list[tuple[str, str, int]]:  # 结束参数声明，进入函数或语句主体。
        """将同组片段转成正文、标题路径和页码三元组。合并短片段，并按边界切开过长片段。"""
        if not group:  # 当分组为空或未启用时进入此分支。
            return []  # 返回空列表。

        max_chars = self._settings.parent_max_chars  # 将字符上限设为父块字符上限。
        target = self._settings.parent_target_chars  # 将目标值设为父块目标字符数。
        breadcrumb = " > ".join(  # 用层级分隔符拼接组内的标题路径。
            next(  # 采用第一个可用路径，无标题时使用空序列。
                (s.heading_path for s in group if s.heading_path),  # 按原顺序寻找带标题路径的片段。
                (),  # 结束当前多行表达式。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。
        page = next((s.page for s in group if s.page), 0)  # 读取组内第一个有效页码，缺省为零。

        # 累积片段，直到接近父块的目标长度。
        parents: list[tuple[str, str, int]] = []  # 初始化父块列表为空列表。
        buffer: list[str] = []  # 初始化缓冲区为空列表。
        buffer_len = 0  # 将缓冲区字符数设为0。

        def flush() -> None:  # 输出当前缓冲的内容，为下一段做准备。
            nonlocal buffer, buffer_len  # 允许写出缓冲后重置外层列表和计数器。
            text = "\n\n".join(buffer).strip()  # 用空行连接暂存片段，形成父块正文。
            if text:  # 当文本有效时进入此分支。
                parents.append((text, breadcrumb, page))  # 把父块正文和它的标题、页码一起保存。
            buffer = []  # 初始化缓冲区为空列表。
            buffer_len = 0  # 将缓冲区字符数设为0。

        for segment in group:  # 遍历分组，逐项取得结构片段。
            text = segment.text.strip()  # 将文本设为清理首尾空白后的文本。
            if not text:  # 当文本为空或未启用时进入此分支。
                continue  # 跳过本项，继续处理下一项。

            if len(text) > max_chars:  # 单个片段本身已经超过父块上限时单独处理。
                # 当前片段单独就超长，先输出已累积内容，再拆分它。
                flush()  # 输出当前缓冲片段。
                for piece in self._split_oversized(text, segment):  # 按结构或语义将超长片段拆成可用的父块。
                    parents.append((piece, breadcrumb, segment.page or page))  # 将组合值加入父块列表。
                continue  # 跳过本项，继续处理下一项。

            if buffer_len + len(text) > max_chars and buffer:  # 同时容纳新片段将超出硬上限时，先写出已有缓冲。
                flush()  # 输出当前缓冲片段。
            buffer.append(text)  # 将文本加入缓冲区。
            buffer_len += len(text) + 2  # 累计正文与段间两个换行符的长度。
            if buffer_len >= target:  # 达到目标长度时输出一个父块。
                flush()  # 输出当前缓冲片段。

        flush()  # 输出当前缓冲片段。
        return parents  # 返回父块列表。

    def _split_oversized(  # 切分超长文本。
        self, text: str, segment: StructuralSegment  # 组合当前结果所需的各项数据。
    ) -> list[str]:  # 结束参数声明，进入函数或语句主体。
        """切分超长文本。分别评估段落和句子方案的边界质量，避免选择一个长度合规却从词中间截断的方案。"""
        max_chars = self._settings.parent_max_chars  # 将字符上限设为父块字符上限。

        best: list[str] | None = None  # 初始时还没有可用的结构切分方案。
        best_score = float("-inf")  # 用负无穷作为基线，使第一个可用方案总能入选。
        for splitter in (self._split_by_paragraphs, self._split_by_sentences):  # 分别尝试段落切分和句子切分。
            pieces = splitter(text, max_chars)  # 取得当前切分方法产生的片段。
            if not pieces or any(len(p) > max_chars for p in pieces):  # 忽略空结果或仍含超长片段的切法。
                continue  # 跳过本项，继续处理下一项。
            score = _boundary_quality(pieces)  # 比较有多少切口恰好位于句末。
            if score > best_score:  # 当分数大于最佳切分得分时进入此分支。
                best, best_score = pieces, score  # 用更自然的边界方案更新当前最佳结果。

        if best is not None and best_score > 0:  # 已经找到有效句末切口时直接使用结构切分结果。
            return best  # 返回最佳方案。

        if self._settings.semantic_enabled and self._embedder is not None:  # 启用语义切分且向量服务可用时尝试按话题拆分。
            semantic = self._split_semantically(text, max_chars)  # 将语义切分设为按语义边界切分的结果。
            if semantic and all(len(p) <= max_chars for p in semantic):  # 只有所有语义块都符合长度上限才采用它们。
                return semantic  # 返回语义切分。

        if best is not None:  # 当最佳方案不为空值时进入此分支。
            return best  # 返回最佳方案。
        return _hard_split(text, max_chars)  # 结构与语义方案都不可用时按长度截断。

    def _split_by_paragraphs(self, text: str, max_chars: int) -> list[str]:  # 定义按段落切分的处理逻辑。
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]  # 按空白行拆出段落并去除空段。
        return _pack(paragraphs, max_chars)  # 返回合并分段的结果。

    def _split_by_sentences(self, text: str, max_chars: int) -> list[str]:  # 定义按句子切分的处理逻辑。
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]  # 按句末标点分句并去除空句。
        return _pack(sentences, max_chars)  # 返回合并分段的结果。

    def _split_semantically(self, text: str, max_chars: int) -> list[str]:  # 计算相邻句向量的余弦距离，以距离分位数筛选话题变化处作为切点。
        """计算相邻句向量的余弦距离，以距离分位数筛选话题变化处作为切点。"""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]  # 将句子列表设为从切分的结果中提取的各项。
        if len(sentences) < 3:  # 句子过少时无法可靠比较相邻话题变化。
            return []  # 返回空列表。

        try:  # 在此捕获后续操作可能产生的异常。
            vectors = self._embedder.embed_texts(sentences)  # 将向量列表设为批量生成文本向量的结果。
        except Exception:  # 捕获通用异常。
            # 向量服务失败时退回结构切分，保证入库可以继续。
            return []  # 返回空列表。

        distances: list[float] = []  # 初始化距离列表为空列表。
        for left, right in zip(vectors, vectors[1:], strict=False):  # 把每个句向量和它后面的句向量配成一对。
            distances.append(1.0 - _cosine(left, right))  # 用一减余弦相似度表示相邻句子的语义差异。

        if not distances:  # 当距离列表为空或未启用时进入此分支。
            return []  # 返回空列表。

        threshold = _percentile(distances, self._settings.semantic_break_percentile)  # 根据文档自身的距离分布计算切分阈值。
        candidates = sorted(  # 将candidates设为排序后的从距离列表及各项的下标中提取的各项。
            (distance, index)  # 将距离大小及其所在位置配对。
            for index, distance in enumerate(distances)  # 遍历所有相邻句子的距离。
            if distance >= threshold  # 仅保留语义差异达到阈值的位置。
        )  # 结束当前多行表达式。
        # 优先切开距离最大的边界；每次切分都会改变剩余可用切点。
        candidates.reverse()  # 反转candidates的排列顺序。

        boundaries = {index + 1 for _, index in candidates}  # 把向量对的下标转换为句子组的结束位置。
        sentences_with_cuts: list[list[str]] = []  # 初始化按边界分组的句子为空列表。
        current: list[str] = []  # 初始化当前值为空列表。
        for index, sentence in enumerate(sentences):  # 遍历句子列表及各项的下标，逐项取得索引、句子。
            current.append(sentence)  # 将句子加入当前值。
            if index + 1 in boundaries:  # 读到切分位置时写出当前句子组。
                sentences_with_cuts.append(current)  # 将当前值加入按边界分组的句子。
                current = []  # 初始化当前值为空列表。
        if current:  # 当当前值有效时进入此分支。
            sentences_with_cuts.append(current)  # 将当前值加入按边界分组的句子。

        pieces: list[str] = []  # 初始化片段列表为空列表。
        for run in sentences_with_cuts:  # 遍历按边界分组的句子，逐项取得连续片段。
            merged = " ".join(run).strip()  # 将合并结果设为清理首尾空白后的拼接后的连续片段。
            if len(merged) <= max_chars:  # 当合并结果的长度不超过字符上限时进入此分支。
                pieces.append(merged)  # 将合并结果加入片段列表。
            else:  # 处理其余情况。
                pieces.extend(_hard_split(merged, max_chars))  # 话题块仍然过长时继续按长度切分。

        return _merge_short(pieces, max_chars)  # 将过短片段与相邻片段合并后返回。

    # 生成子块。

    def _split_parent_to_children(self, parent_text: str) -> list[str]:  # 将父块切为互有重叠的子块，使跨越边界的句子仍有机会被完整检索。
        """将父块切为互有重叠的子块，使跨越边界的句子仍有机会被完整检索。"""
        target = self._settings.child_target_chars  # 将目标值设为子块目标字符数。
        overlap = self._settings.child_overlap_chars  # 将重叠长度设为相邻子块重叠字符数。
        if len(parent_text) <= target:  # 当父块正文的长度不超过目标值时进入此分支。
            return [parent_text.strip()]  # 返回列表。

        children: list[str] = []  # 初始化子块列表为空列表。
        start = 0  # 将起点设为0。
        length = len(parent_text)  # 将长度设为父块正文的长度。
        while start < length:  # 在起点小于长度期间重复处理。
            end = min(start + target, length)  # 计算本块的理想结束位置，并限制在父块正文以内。
            if end < length:  # 当终点小于长度时进入此分支。
                # 在目标长度附近优先寻找自然结尾。
                boundary = _find_boundary(parent_text, start, end, target)  # 在理想结束位置附近向前寻找自然断点。
                if boundary is not None:  # 当切分边界不为空值时进入此分支。
                    end = boundary  # 将终点设为切分边界。
            piece = parent_text[start:end].strip()  # 截取起点与所选边界之间的子块正文。
            if piece:  # 当片段有效时进入此分支。
                children.append(piece)  # 将片段加入子块列表。
            if end >= length:  # 当终点至少为长度时进入此分支。
                break  # 结束当前循环。
            start = max(end - overlap, start + 1)  # 下一块从重叠区起步，同时保证起点至少前进一个字符。
        return children or [parent_text.strip()]  # 若未取得非空片段，则回退到去除首尾空白的父块正文。


def _pack(units: list[str], max_chars: int) -> list[str]:  # 依次装入片段，并将每个结果限制在指定字符数内。
    """依次装入片段，并将每个结果限制在指定字符数内。"""
    packed: list[str] = []  # 初始化已打包片段为空列表。
    buffer: list[str] = []  # 初始化缓冲区为空列表。
    length = 0  # 将长度设为0。
    for unit in units:  # 遍历片段，逐项取得单元。
        if len(unit) > max_chars:  # 当单元的长度大于字符上限时进入此分支。
            if buffer:  # 当缓冲区有效时进入此分支。
                packed.append(" ".join(buffer).strip())  # 将清理首尾空白后的拼接后的缓冲区加入已打包片段。
                buffer, length = [], 0  # 将缓冲区、长度设为组合值。
            packed.extend(_hard_split(unit, max_chars))  # 将按长度切分的结果逐项加入已打包片段。
            continue  # 跳过本项，继续处理下一项。
        if length + len(unit) + 1 > max_chars and buffer:  # 将分隔空格也计入长度，判断是否该先输出缓冲。
            packed.append(" ".join(buffer).strip())  # 将清理首尾空白后的拼接后的缓冲区加入已打包片段。
            buffer, length = [], 0  # 将缓冲区、长度设为组合值。
        buffer.append(unit)  # 将单元加入缓冲区。
        length += len(unit) + 1  # 累计当前单元和分隔空格的长度。
    if buffer:  # 当缓冲区有效时进入此分支。
        packed.append(" ".join(buffer).strip())  # 将清理首尾空白后的拼接后的缓冲区加入已打包片段。
    return [p for p in packed if p]  # 返回从已打包片段中提取的各项。


def _merge_short(pieces: list[str], max_chars: int) -> list[str]:  # 把过短且难以独立检索的片段与邻块合并。
    """把过短且难以独立检索的片段与邻块合并。"""
    if not pieces:  # 当片段列表为空或未启用时进入此分支。
        return []  # 返回空列表。
    minimum = max(80, max_chars // 6)  # 过短片段的阈值取八十字符与上限六分之一中的较大值。
    merged: list[str] = [pieces[0]]  # 将合并结果设为列表。
    for piece in pieces[1:]:  # 遍历片段列表的指定区间，逐项取得片段。
        if len(merged[-1]) < minimum and len(merged[-1]) + len(piece) <= max_chars:  # 前段偏短且合并后不超限时连接前后两段。
            merged[-1] = f"{merged[-1]} {piece}".strip()  # 将合并结果中对应的项设为清理首尾空白后的格式化文本。
        else:  # 处理其余情况。
            merged.append(piece)  # 将片段加入合并结果。
    return merged  # 返回合并结果。


_SENTENCE_ENDINGS = ".!?。！？；;…"  # 将_SENTENCE_ENDINGS设为“.!?。！？；;…”。

def _boundary_quality(pieces: list[str]) -> float:  # 计算在句末标点处结束的片段比例，以此比较切分方案。
    """计算在句末标点处结束的片段比例，以此比较切分方案。"""
    if not pieces:  # 当片段列表为空或未启用时进入此分支。
        return float("-inf")  # 返回“-inf”的浮点数形式。
    clean = sum(1 for p in pieces[:-1] if p.rstrip() and p.rstrip()[-1] in _SENTENCE_ENDINGS)  # 统计最后一段以外有多少片段以句末标点收束。
    return clean / max(1, len(pieces) - 1)  # 用自然句末数除以实际切口数；只有一段时避免除零。


def _hard_split(text: str, max_chars: int) -> list[str]:  # 结构切分仍然超长时按字符数切分，优先从空白处断开。
    """结构切分仍然超长时按字符数切分，优先从空白处断开。"""
    pieces: list[str] = []  # 初始化片段列表为空列表。
    start = 0  # 将起点设为0。
    length = len(text)  # 将长度设为文本的长度。
    while start < length:  # 在起点小于长度期间重复处理。
        end = min(start + max_chars, length)  # 将终点设为各项的较小值。
        if end < length:  # 当终点小于长度时进入此分支。
            window = text.rfind(" ", start + max_chars // 2, end)  # 在允许长度的后半段反向寻找空格，尽量避免切断单词。
            if window > start:  # 当window大于起点时进入此分支。
                end = window  # 将终点设为window。
        piece = text[start:end].strip()  # 将片段设为清理首尾空白后的文本的指定区间。
        if piece:  # 当片段有效时进入此分支。
            pieces.append(piece)  # 将片段加入片段列表。
        start = end  # 将起点设为终点。
    return pieces  # 返回片段列表。


def _find_boundary(text: str, start: int, end: int, target: int) -> int | None:  # 从目标末尾向前查找句子或段落的边界。
    """从目标末尾向前查找句子或段落的边界。"""
    lower_bound = max(start + target // 2, start)  # 限定向后查找范围，避免为寻找边界产生过短子块。
    for pattern in ("\n\n", "。", "！", "？", ". ", "! ", "? ", "\n", " "):  # 先找段末与句末标点，再退回换行和普通空格。
        position = text.rfind(pattern, lower_bound, end)  # 寻找目标区间内该分隔符最后出现的位置。
        if position != -1:  # 当位置不等于-1时进入此分支。
            return position + len(pattern)  # 将分隔符也包含在当前子块中。
    return None  # 返回空值。


def _cosine(left: list[float], right: list[float]) -> float:  # 定义计算余弦相似度的处理逻辑。
    dot = sum(a * b for a, b in zip(left, right, strict=True))  # 逐维相乘求和，得到两个向量的点积。
    left_norm = sum(a * a for a in left) ** 0.5  # 对左向量各维平方求和再开方，得到向量模长。
    right_norm = sum(b * b for b in right) ** 0.5  # 用相同方法求右向量的模长。
    if left_norm == 0 or right_norm == 0:  # 存在零向量时单独处理，避免除零。
        return 0.0  # 返回0.0。
    return dot / (left_norm * right_norm)  # 以点积除以两向量模长之积得到余弦相似度。


def _percentile(values: list[float], percentile: int) -> float:  # 定义计算百分位数的处理逻辑。
    if not values:  # 当值列表为空或未启用时进入此分支。
        return 0.0  # 返回0.0。
    ordered = sorted(values)  # 将排序结果设为排序后的值列表。
    clamped = min(max(percentile, 0), 100)  # 将请求百分位约束在零到一百之间。
    index = int(round((clamped / 100) * (len(ordered) - 1)))  # 将百分位换算为有序样本上的最近下标。
    return ordered[index]  # 返回排序结果中对应的项。


def parent_breadcrumb(parent_text: str) -> str:  # 没有标题路径时，用父块首行生成简短的位置提示。
    """没有标题路径时，用父块首行生成简短的位置提示。"""
    first_line = parent_text.strip().split("\n", 1)[0]  # 取父块首行作为位置提示的标识来源。
    digest = hashlib.sha256(first_line.encode("utf-8")).hexdigest()[:8]  # 截取首行摘要的前八个十六进制字符。
    return f"segment-{digest}"  # 为没有标题路径的片段生成简短的位置标识。