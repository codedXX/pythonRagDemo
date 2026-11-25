"""用倒数排名融合向量和关键词两路召回。每路按 1 / (k + 排名) 贡献分数，从而可以组合量纲不同的两种检索结果。

重排时将标题路径和正文一起提交，帮助模型理解短片段。多取一些候选，使共享父块的子块去重后仍能凑足生成所需的不同上下文。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

from dataclasses import dataclass, field  # 从 dataclasses 引入本模块使用的类型和工具。

from rag_demo.config import RetrievalSettings  # 从 rag_demo.config 导入检索配置。
from rag_demo.embedding.provider import Embedder  # 从 rag_demo.embedding.provider 导入向量编码器。
from rag_demo.embedding.rerank import Reranker  # 从 rag_demo.embedding.rerank 导入重排服务。
from rag_demo.errors import ProviderError  # 从 rag_demo.errors 导入供应商异常。
from rag_demo.retrieval.lexical import LexicalIndex  # 从 rag_demo.retrieval.lexical 导入关键词索引。
from rag_demo.store.milvus_store import MilvusStore  # 从 rag_demo.store.milvus_store 导入向量库客户端。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class Candidate:  # 保留候选子块及生成引用所需的来源信息。
    """保留候选子块及生成引用所需的来源信息。"""

    id: str  # 声明标识字段。
    parent_id: str  # 声明所属父块标识字段。
    text: str  # 声明文本字段。
    breadcrumb: str  # 声明标题路径字段。
    page: int  # 声明页码字段。
    source_name: str  # 声明来源文件名字段。
    doc_id: str = ""  # 设置文档标识字段的默认值。
    dense_rank: int | None = None  # 设置向量召回名次字段的默认值。
    sparse_rank: int | None = None  # 设置关键词召回名次字段的默认值。
    fused_score: float = 0.0  # 设置融合分数字段的默认值。
    fused_rank: int = 0  # 设置融合名次字段的默认值。
    rerank_score: float | None = None  # 设置重排分数字段的默认值。
    rerank_rank: int | None = None  # 设置重排名次字段的默认值。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def channels(self) -> list[str]:  # 记录命中此候选的检索通道，供诊断展示。
        """记录命中此候选的检索通道，供诊断展示。"""
        found = []  # 收集命中此子块的检索通道。
        if self.dense_rank is not None:  # 有向量排名说明该子块被向量检索召回。
            found.append("dense")  # 将“dense”加入命中通道列表。
        if self.sparse_rank is not None:  # 有关键词排名说明该子块也出现在关键词结果中。
            found.append("sparse")  # 将“sparse”加入命中通道列表。
        return found  # 返回命中通道列表。

    def rerank_text(self) -> str:  # 将标题路径和正文组合为重排输入。
        """将标题路径和正文组合为重排输入。"""
        if self.breadcrumb:  # 有标题路径时把章节信息加入重排输入。
            return f"{self.source_name} > {self.breadcrumb} | {self.text}"  # 将文件名、标题路径和片段正文拼在一起。
        return f"{self.source_name} | {self.text}"  # 没有章节信息时仍保留文件名与正文。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class Context:  # 记录提供给生成模型的父块及实际命中的子块。
    """记录提供给生成模型的父块及实际命中的子块。"""

    parent_id: str  # 声明所属父块标识字段。
    text: str  # 声明文本字段。
    breadcrumb: str  # 声明标题路径字段。
    page: int  # 声明页码字段。
    source_name: str  # 声明来源文件名字段。
    matched_child: str = ""  # 设置命中的子块标识字段的默认值。
    matched_text: str = ""  # 设置命中的子块正文字段的默认值。
    rerank_score: float | None = None  # 设置重排分数字段的默认值。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class RetrievalOutcome:  # 汇总一次检索的上下文、各阶段候选和重排状态。
    """汇总一次检索的上下文、各阶段候选和重排状态。"""

    query: str  # 声明查询字段。
    contexts: list[Context] = field(default_factory=list)  # 设置上下文列表字段的默认值。
    dense: list[Candidate] = field(default_factory=list)  # 设置向量召回字段的默认值。
    sparse: list[Candidate] = field(default_factory=list)  # 设置关键词召回字段的默认值。
    fused: list[Candidate] = field(default_factory=list)  # 设置融合结果字段的默认值。
    reranked: list[Candidate] = field(default_factory=list)  # 设置重排结果字段的默认值。
    rerank_applied: bool = True  # 设置是否采用重排字段的默认值。
    rerank_error: str | None = None  # 设置重排失败原因字段的默认值。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def empty(self) -> bool:  # 判断是否缺少可供回答的上下文。
        return not self.contexts  # 没有选出上下文时，本次检索为空。


def rrf_fuse(  # 按倒数排名合并多个有序列表，并按总分降序排列。
    rankings: list[list[str]],  # 接收rankings参数。
    rrf_k: int = 60,  # 接收倒数排名融合常数参数。
    weights: list[float] | None = None,  # 接收weights参数。
) -> list[tuple[str, float]]:  # 结束参数声明，进入函数或语句主体。
    """按倒数排名合并多个有序列表，并按总分降序排列。某路未命中时，该路贡献为零。"""
    scores: dict[str, float] = {}  # 按子块标识累计各通道的分数。
    for channel_index, ranking in enumerate(rankings):  # 逐路读取候选排名，并用通道序号查权重。
        weight = weights[channel_index] if weights else 1.0  # 未配置权重时，各通道均取一。
        # 每个标识在同一通道只按最佳名次计一次，避免重复项抬高分数。
        best_rank: dict[str, int] = {}  # 为当前通道单独记录最佳名次。
        for rank, identifier in enumerate(ranking):  # 从零开始给该通道的结果编号。
            if identifier not in best_rank:  # 首次出现的标识才写入。
                best_rank[identifier] = rank  # 保留第一次出现的位置，即本通道内的最佳名次。
        for identifier, rank in best_rank.items():  # 按该通道内的最佳名次累计分数。
            scores[identifier] = scores.get(identifier, 0.0) + weight / (rrf_k + rank + 1)  # 将零起始名次加一，再用通道权重除以平滑后的名次。

    # 总分相同时按标识排序，使不同参数下的结果可以稳定对比。
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))  # 先按分数从高到低排，同分时按标识升序排。


class HybridRetriever:  # 执行双路召回、排名融合、重排与父块扩展。
    """执行双路召回、排名融合、重排与父块扩展。"""

    def __init__(  # 保存实例所需的配置和依赖。
        self,  # 通过实例读取本方法所需状态。
        settings: RetrievalSettings,  # 接收配置参数。
        store: MilvusStore,  # 接收向量库参数。
        embedder: Embedder,  # 接收向量服务参数。
        lexical: LexicalIndex,  # 接收关键词索引参数。
        reranker: Reranker | None = None,  # 接收重排器参数。
    ) -> None:  # 结束参数声明，进入函数或语句主体。
        self._settings = settings  # 将测试配置设为配置。
        self._store = store  # 保存传入的向量库。
        self._embedder = embedder  # 保存传入的向量服务。
        self._lexical = lexical  # 保存传入的关键词索引。
        self._reranker = reranker  # 保存传入的重排器。

    # 向量与关键词召回。

    def _dense(self, query: str) -> list[Candidate]:  # 定义向量召回的处理逻辑。
        vectors = self._embedder.embed_texts([query])  # 将检索问题编码为一个向量。
        if not vectors:  # 当向量列表为空或未启用时进入此分支。
            return []  # 返回空列表。
        rows = self._store.search_children(vectors[0], limit=self._settings.dense_top_k)  # 用查询向量检索最相似的子块，条数取向量通道上限。
        candidates: list[Candidate] = []  # 初始化candidates为空列表。
        for rank, row in enumerate(rows):  # 遍历记录列表及各项的下标，逐项取得排名、记录。
            candidates.append(  # 将检索候选的结果加入candidates。
                Candidate(  # 为检索候选提供输入。
                    id=row["id"],  # 传入标识参数。
                    parent_id=row["parent_id"],  # 传入所属父块标识参数。
                    text=row.get("text", ""),  # 传入文本参数。
                    breadcrumb=row.get("breadcrumb", ""),  # 传入标题路径参数。
                    page=int(row.get("page", 0) or 0),  # 传入页码参数。
                    source_name=row.get("source_name", ""),  # 传入来源文件名参数。
                    doc_id=row.get("doc_id", ""),  # 传入文档标识参数。
                    dense_rank=rank,  # 传入向量召回名次参数。
                )  # 结束当前多行表达式。
            )  # 结束当前多行表达式。
        return candidates  # 返回candidates。

    def _sparse(self, query: str) -> list[Candidate]:  # 定义关键词召回的处理逻辑。
        hits = self._lexical.search(query, limit=self._settings.sparse_top_k)  # 用相同的问题执行关键词召回。
        if not hits:  # 当命中项为空或未启用时进入此分支。
            return []  # 返回空列表。

        ids = [hit.id for hit in hits]  # 按关键词排名收集命中标识。
        rows = self._fetch_children(ids)  # 从存储补回每个子块的正文和来源信息。
        candidates: list[Candidate] = []  # 初始化candidates为空列表。
        for rank, hit in enumerate(hits):  # 遍历命中项及各项的下标，逐项取得排名、hit。
            row = rows.get(hit.id)  # 将记录设为记录列表中的对应值。
            if row is None:  # 当记录为空值时进入此分支。
                # 关键词索引短暂滞后时，跳过存储中已经消失的条目。
                continue  # 跳过本项，继续处理下一项。
            candidates.append(  # 将检索候选的结果加入candidates。
                Candidate(  # 为检索候选提供输入。
                    id=hit.id,  # 传入标识参数。
                    parent_id=row["parent_id"],  # 传入所属父块标识参数。
                    text=row.get("text", ""),  # 传入文本参数。
                    breadcrumb=row.get("breadcrumb", ""),  # 传入标题路径参数。
                    page=int(row.get("page", 0) or 0),  # 传入页码参数。
                    source_name=row.get("source_name", ""),  # 传入来源文件名参数。
                    doc_id=row.get("doc_id", ""),  # 传入文档标识参数。
                    sparse_rank=rank,  # 传入关键词召回名次参数。
                )  # 结束当前多行表达式。
            )  # 结束当前多行表达式。
        return candidates  # 返回candidates。

    def _fetch_children(self, ids: list[str]) -> dict[str, dict]:  # 定义批量读取子块的处理逻辑。
        if not ids:  # 当标识列表为空或未启用时进入此分支。
            return {}  # 返回空字典。
        quoted = ", ".join(f'"{i}"' for i in ids)  # 将标识排列为查询表达式需要的带引号列表。
        rows = self._store.client.query(  # 批量查询符合条件的子块。
            self._store.child_name,  # 在子块集合中查找。
            filter=f"id in [{quoted}] and active == true",  # 同时限制标识范围和激活状态。
            output_fields=[  # 只取后续融合与引用需要的字段。
                "id",  # 读取子块主键。
                "doc_id",  # 读取文档标识。
                "parent_id",  # 读取所属父块标识。
                "text",  # 读取命中片段正文。
                "breadcrumb",  # 读取标题路径。
                "page",  # 读取页码。
                "source_name",  # 读取来源文件名。
            ],  # 结束列表。
            limit=max(1, len(ids)),  # 查询上限按待查标识数设置。
        )  # 结束当前多行表达式。
        return {row["id"]: row for row in rows}  # 以子块标识为键，方便按关键词排名取回记录。

    # 融合两路候选。

    def _fuse(  # 定义融合候选的处理逻辑。
        self, dense: list[Candidate], sparse: list[Candidate]  # 组合当前结果所需的各项数据。
    ) -> list[Candidate]:  # 结束参数声明，进入函数或语句主体。
        by_id: dict[str, Candidate] = {}  # 初始化按标识归集的候选为空字典。
        for candidate in dense:  # 遍历向量召回，逐项取得候选项。
            by_id[candidate.id] = candidate  # 按标识登记向量通道候选。
        for candidate in sparse:  # 遍历关键词召回，逐项取得候选项。
            if candidate.id in by_id:  # 当标识包含在按标识归集的候选时进入此分支。
                # 保留两路的命中名次，使同一子块可以获得两路排名贡献。
                by_id[candidate.id].sparse_rank = candidate.sparse_rank  # 同一子块保留两路排名，以便在诊断结果中对照。
            else:  # 处理其余情况。
                by_id[candidate.id] = candidate  # 只在关键词通道出现的子块也加入候选集合。

        fused = rrf_fuse(  # 用各通道的名次计算融合分数。
            [  # 开始列出各项。
                [c.id for c in dense],  # 传入向量通道按相关性排列的子块标识。
                [c.id for c in sparse],  # 传入关键词通道按相关性排列的子块标识。
            ],  # 结束列表。
            rrf_k=self._settings.rrf_k,  # 传入倒数排名融合常数参数。
        )  # 结束当前多行表达式。

        result: list[Candidate] = []  # 初始化结果为空列表。
        for rank, (identifier, score) in enumerate(fused):  # 按融合后的顺序取出子块标识及得分。
            candidate = by_id[identifier]
            candidate.fused_score = score  # 在候选上记下两路合计分数。
            candidate.fused_rank = rank  # 保存该候选在融合列表中的位置。
            result.append(candidate)  # 将候选项加入结果。
        return result  # 返回结果。

    # 候选重排。

    def _rerank(self, query: str, fused: list[Candidate]) -> tuple[list[Candidate], bool, str | None]:  # 对融合排名靠前的候选重新打分。
        """对融合排名靠前的候选重新打分；失败时返回原融合顺序和错误信息，让检索继续。"""
        if self._reranker is None:  # 当重排器为空值时进入此分支。
            return fused, False, "no reranker configured"  # 未配置重排服务时保留融合顺序并说明原因。

        subset = fused[: self._settings.rerank_candidates]  # 截取融合列表前部作为重排候选。
        if not subset:  # 当重排候选子集为空或未启用时进入此分支。
            return fused, False, None  # 没有候选时直接返回原列表。

        try:  # 在此捕获后续操作可能产生的异常。
            results = self._reranker.rerank(query, [c.rerank_text() for c in subset])  # 将问题及带来源的候选正文一起交给重排模型。
        except ProviderError as error:  # 捕获供应商异常。
            return fused, False, str(error)  # 供应商请求失败时保留融合结果并附上说明。
        except Exception as error:  # pragma: no cover  # 防御性处理。
            return fused, False, f"{type(error).__name__}: {error}"  # 把意外异常的类型和原因放入降级信息。

        reranked: list[Candidate] = []  # 初始化重排结果为空列表。
        for rank, result in enumerate(results):  # 遍历结果列表及各项的下标，逐项取得排名、结果。
            candidate = subset[result.index]  # 通过模型返回的下标找到原来的候选。
            candidate.rerank_score = result.score  # 保存模型给出的相关性分数。
            candidate.rerank_rank = rank  # 保存模型排序后的名次。
            reranked.append(candidate)  # 将候选项加入重排结果。

        # 未获重排分数的候选按原融合顺序附在末尾。
        scored = {c.id for c in reranked}  # 收集已被模型打过分的子块标识。
        reranked.extend(c for c in fused if c.id not in scored)  # 将剩余子块按原融合顺序补到末尾。
        return reranked, True, None  # 返回新顺序并标记已应用重排。

    # 扩展父块上下文。

    def _expand(self, candidates: list[Candidate]) -> list[Context]:  # 将子块扩展为互不重复的父块，从后续候选补足上下文，避免重复消耗提示词预算。
        """将子块扩展为互不重复的父块，从后续候选补足上下文，避免重复消耗提示词预算。"""
        contexts: list[Context] = []  # 初始化上下文列表为空列表。
        seen: set[str] = set()  # 将已处理项设为空集合。
        pending: list[Candidate] = []  # 初始化待处理项为空列表。

        for candidate in candidates:  # 遍历candidates，逐项取得候选项。
            if candidate.parent_id in seen:  # 每个父块只接受第一个命中的子块。
                continue  # 跳过本项，继续处理下一项。
            seen.add(candidate.parent_id)  # 标记该父块已经入选。
            pending.append(candidate)  # 将候选项加入待处理项。
            if len(pending) >= self._settings.context_count:  # 不同父块数量足够时停止选取。
                break  # 结束当前循环。

        if not pending:  # 当待处理项为空或未启用时进入此分支。
            return []  # 返回空列表。

        parents = self._store.fetch_parents([c.parent_id for c in pending])  # 一次读回入选子块对应的父块。

        for candidate in pending:  # 遍历待处理项，逐项取得候选项。
            parent = parents.get(candidate.parent_id)  # 用命中子块携带的父块标识查找正文。
            if parent is None:  # 当父块为空值时进入此分支。
                # 父块缺失时以命中的子块作为上下文，保留可用内容。
                text = candidate.text  # 父块缺失时保留子块内容。
                breadcrumb = candidate.breadcrumb  # 取出当前使用的标题路径。
                page = candidate.page  # 取出当前使用的页码。
            else:  # 处理其余情况。
                text = parent.get("text", "")  # 将文本设为父块中的文本。
                breadcrumb = parent.get("breadcrumb") or candidate.breadcrumb  # 父块没有标题路径时沿用子块位置。
                page = int(parent.get("page_start", 0) or candidate.page)  # 优先使用父块起始页，必要时退回子块页码。
            contexts.append(  # 将引用上下文的结果加入上下文列表。
                Context(  # 为引用上下文提供输入。
                    parent_id=candidate.parent_id,  # 传入所属父块标识参数。
                    text=self._bound(text, candidate.text),  # 按截取上限保留命中子块附近的正文。
                    breadcrumb=breadcrumb,  # 传入标题路径参数。
                    page=page,  # 传入页码参数。
                    source_name=candidate.source_name,  # 传入来源文件名参数。
                    matched_child=candidate.id,  # 保留触发本次父块扩展的子块标识。
                    matched_text=candidate.text,  # 保留命中原文，供引用摘要展示。
                    rerank_score=candidate.rerank_score,  # 传入重排分数参数。
                )  # 结束当前多行表达式。
            )  # 结束当前多行表达式。

        # 对所有上下文合计应用字符预算。
        return self._apply_budget(contexts)  # 在返回生成上下文前应用总字符预算。

    def _bound(self, parent_text: str, child_text: str) -> str:  # 父块过长时截取命中子块附近的内容，为其他来源留下空间。
        """父块过长时截取命中子块附近的内容，为其他来源留下空间。"""
        limit = self._settings.parent_excerpt_chars  # 将数量上限设为父块摘录字符数。
        if len(parent_text) <= limit:  # 当父块正文的长度不超过数量上限时进入此分支。
            return parent_text  # 返回父块正文。

        position = parent_text.find(child_text) if child_text else -1  # 查找命中子块在父块正文中的起点。
        if position < 0:  # 当位置小于0时进入此分支。
            return parent_text[:limit].rstrip() + "…"  # 找不到命中位置时取开头并用省略号标示截断。

        half = max(1, (limit - len(child_text)) // 2)  # 从摘录上限扣除命中长度，把剩余空间分给两侧。
        start = max(0, position - half)  # 截取起点最早为父块开头。
        end = min(len(parent_text), position + len(child_text) + half)  # 截取终点最晚为父块末尾。
        excerpt = parent_text[start:end].strip()  # 截取命中位置及两侧上下文。
        if start > 0:  # 当起点大于0时进入此分支。
            excerpt = "…" + excerpt  # 将摘录设为“…”加上摘录。
        if end < len(parent_text):  # 当终点小于父块正文的长度时进入此分支。
            excerpt = excerpt + "…"  # 将摘录设为摘录加上“…”。
        return excerpt  # 返回摘录。

    def _apply_budget(self, contexts: list[Context]) -> list[Context]:  # 按相关顺序保留上下文，并裁剪到配置的总字符数。
        """按相关顺序保留上下文，并裁剪到配置的总字符数。"""
        budget = self._settings.context_char_budget  # 将预算设为上下文总字符预算。
        kept: list[Context] = []  # 初始化已保留的上下文为空列表。
        used = 0  # 将已用字符数设为0。
        for context in contexts:  # 遍历上下文列表，逐项取得上下文。
            remaining = budget - used  # 计算本轮还可分配的上下文字符数。
            if remaining <= 200:  # 剩余空间不足以保留一段有用上下文时停止。
                break  # 结束当前循环。
            if len(context.text) > remaining:  # 当前段落超过剩余预算时裁剪。
                context.text = context.text[:remaining].rstrip() + "…"  # 保留能容纳的前缀，并用省略号标明截断。
            kept.append(context)  # 将上下文加入已保留的上下文。
            used += len(context.text)  # 把本段实际长度计入已用预算。
            if len(kept) >= self._settings.context_count:  # 当已保留的上下文的长度至少为上下文数量时进入此分支。
                break  # 结束当前循环。
        return kept  # 返回已保留的上下文。

    # 检索入口。

    def retrieve(self, query: str) -> RetrievalOutcome:  # 针对一个查询运行完整的混合检索流程。
        """针对一个查询运行完整的混合检索流程。"""
        outcome = RetrievalOutcome(query=query)  # 新建本次检索结果，后续逐步填入各阶段输出。

        # 未上传文档时直接返回空结果，省去无匹配可能的向量调用。
        if self._lexical.size == 0:  # 关键词索引为空时先判断是否确为空库。
            if self._store_stats_are_empty():  # 只有持久化存储也为空时才直接返回。
                return outcome  # 返回处理结果。
            # 存储有数据但关键词索引为空时重建索引，避免把索引过期误报为空库。
            self.rebuild_lexical_index()  # 从存储中的有效子块恢复过期的关键词索引。
            if self._lexical.size == 0:  # 当self._lexical.size等于0时进入此分支。
                return outcome  # 返回处理结果。

        outcome.dense = self._dense(query)  # 执行向量召回并保留原始候选。
        outcome.sparse = self._sparse(query)  # 执行关键词召回并保留原始候选。

        if not outcome.dense and not outcome.sparse:  # 当向量召回为假且关键词召回为假时进入此分支。
            outcome.reranked = []  # 初始化重排结果为空列表。
            return outcome  # 返回处理结果。

        outcome.fused = self._fuse(outcome.dense, outcome.sparse)  # 对两路候选执行倒数排名融合。
        outcome.reranked, outcome.rerank_applied, outcome.rerank_error = self._rerank(  # 记录重排后的顺序、是否启用及降级原因。
            query, outcome.fused  # 将本次查询与融合候选传给重排阶段。
        )  # 结束当前多行表达式。
        outcome.contexts = self._expand(outcome.reranked)  # 对子块按父块去重并提取生成所需的上下文。
        return outcome  # 返回处理结果。

    def rebuild_lexical_index(self) -> int:  # 用存储中处于激活状态的子块重建关键词索引。
        """用存储中处于激活状态的子块重建关键词索引。"""
        rows = self._store.all_active_children()  # 将记录列表设为读取有效子块的结果。
        self._lexical.rebuild(rows)  # 重建关键词索引。
        return self._lexical.size  # 返回self._lexical.size。

    def _store_stats_are_empty(self) -> bool:  # 判断存储中是否确实没有有效子块。
        """判断存储中是否确实没有有效子块；查询失败时允许调用方尝试同步索引。"""
        try:  # 在此捕获后续操作可能产生的异常。
            stats = self._store.collection_stats()  # 将统计信息设为统计集合记录的结果。
        except Exception:  # 捕获通用异常。
            return False  # 返回假。
        child_count = stats.get(self._store.child_name, -1)  # 将子块数设为统计信息中的对应值。
        return child_count == 0  # 返回子块数等于0。