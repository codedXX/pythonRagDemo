"""提供问答、文档管理和依赖检查接口。

问答流先发送来源，方便用户在等待首段回答时查看命中的文档。上传接口按文件推送处理进度，健康接口返回各依赖的状态。响应中应避免包含供应商凭据和上游地址。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import json  # 引入结构化数据的编解码工具。
import logging  # 引入日志模块。
import queue as queue_module  # 引入队列。
import threading  # 引入线程工具。
import uuid  # 引入唯一标识生成器。
from typing import Any  # 从 typing 导入任意类型。

from fastapi import APIRouter, File, HTTPException, Request, UploadFile  # 从 fastapi 引入本模块使用的类型和工具。
from fastapi.responses import StreamingResponse  # 从 fastapi.responses 导入流式响应。

from rag_demo.api.services import Services, iter_sse  # 从 rag_demo.api.services 引入本模块使用的类型和工具。
from rag_demo.errors import ProviderError  # 从 rag_demo.errors 导入供应商异常。
from rag_demo.generation.answer import NO_CONTEXT_ANSWER, describe_generation_error  # 从 rag_demo.generation.answer 引入本模块使用的类型和工具。
from rag_demo.ingestion import IngestionError, Stage  # 从 rag_demo.ingestion 引入本模块使用的类型和工具。
from rag_demo.parsers.base import supported_extensions_display  # 从 rag_demo.parsers.base 导入允许格式的提示文本。
from rag_demo.retrieval.hybrid import Context  # 从 rag_demo.retrieval.hybrid 导入引用上下文。

logger = logging.getLogger(__name__)  # 将日志记录器设为获取当前模块的日志记录器的结果。
router = APIRouter()  # 创建文档、问答和状态接口共用的路由器。


def _services(request: Request) -> Services:  # 取得该请求所在应用的共享服务。
    return request.app.state.services  # 从应用状态中读取初始化时保存的服务对象。


def _sse(event: str, payload: dict[str, Any]) -> str:  # 把事件名和结构化载荷编码为一帧推送事件。
    return next(iter_sse(event, json.dumps(payload, ensure_ascii=False)))  # 保留中文序列化载荷，并取出编码好的事件帧。


def _context_payload(context: Context, index: int) -> dict[str, Any]:  # 将一个检索上下文转换为客户端可展示的引用信息。
    return {  # 返回字段映射。
        "index": index,  # 为正文中的引用保留对应编号。
        "source": context.source_name,  # 填入来源字段。
        "page": context.page,  # 填入页码字段。
        "breadcrumb": context.breadcrumb,  # 填入标题路径字段。
        "matched_text": context.matched_text,  # 提供实际命中的子块文本作为来源摘要。
        "score": round(context.rerank_score, 4) if context.rerank_score is not None else None,  # 有重排分数时保留四位小数，没有时返回空值。
    }  # 结束本组字段。


@router.get("/api/health")  # 把下方处理函数绑定到 GET 路由。
def health(request: Request) -> dict[str, Any]:  # 报告各项依赖的状态和知识库大小。
    """报告各项依赖的状态和知识库大小。"""
    services = _services(request)  # 取出此应用的服务以读取自检和知识库状态。
    report = services.selfcheck  # 将报告设为自检。
    payload: dict[str, Any] = {  # 将载荷设为字段映射。
        "healthy": report.healthy if report else None,  # 尚无自检报告时保留未知状态。
        "dimension": services.dimension,  # 填入向量维度字段。
        "embedding_model": services.settings.embedding.model,  # 填入向量模型名字段。
        "documents": 0,  # 先提供默认文档数，稍后从登记表填充。
        "units": services.lexical.size,  # 统计当前关键词索引中的有效条目数。
        "dependencies": [d.as_dict() for d in report.dependencies] if report else [],  # 有报告时逐项导出依赖状态，否则返回空列表。
    }  # 结束本组字段。
    try:  # 在此捕获后续操作可能产生的异常。
        payload["documents"] = len(services.registry.list_usable())  # 只将可用于问答的登记记录计入可用文档数。
        payload["documents_failed"] = len(  # 单独计算未处于可用状态的文档条数。
            [d for d in services.registry.list_documents() if not d.usable]  # 从所有登记记录中过滤出不可用项。
        )  # 结束当前多行表达式。
    except Exception as error:  # pragma: no cover  # 防御性处理。
        payload["registry_error"] = str(error)  # 将载荷中的registry_error设为错误的字符串形式。
    return payload  # 返回载荷。


@router.get("/api/docs")  # 把下方处理函数绑定到 GET 路由。
def list_documents(request: Request) -> dict[str, Any]:  # 定义列出登记文档的处理逻辑。
    services = _services(request)  # 将服务组件设为共享服务的结果。
    return {  # 返回字段映射。
        "documents": [record.as_dict() for record in services.registry.list_documents()]  # 将登记记录逐项转换为供文档列表使用的字典。
    }  # 结束本组字段。


@router.delete("/api/docs/{doc_id}")  # 把下方处理函数绑定到 DELETE 路由。
def delete_document(doc_id: str, request: Request) -> dict[str, Any]:  # 定义删除文档的处理逻辑。
    services = _services(request)  # 将服务组件设为共享服务的结果。
    from rag_demo.ingestion import remove_document  # 从 rag_demo.ingestion 导入清除文档及片段。

    try:  # 在此捕获后续操作可能产生的异常。
        removed = remove_document(doc_id, services.store, services.registry)  # 将已删除项设为清除文档及片段的结果。
    except Exception as error:  # 捕获通用异常。
        raise HTTPException(status_code=500, detail=f"could not remove document: {error}") from error  # 向上抛出接口异常。

    if not removed:  # 当已删除项为空或未启用时进入此分支。
        raise HTTPException(status_code=404, detail="no such document")  # 向上抛出接口异常。

    # 删除已生效；重建关键词索引以移除过期条目，即使重建失败也应保留删除成功的结果。
    try:  # 在此捕获后续操作可能产生的异常。
        services.refresh_lexical_index()  # 刷新关键词索引。
    except Exception as error:  # pragma: no cover  # 防御性处理。
        logger.warning("lexical index refresh failed after delete: %s", error)  # 记录警告。
    return {"removed": doc_id}  # 返回字段映射。


@router.post("/api/docs")  # 把下方处理函数绑定到 POST 路由。
async def upload_documents(request: Request, files: list[UploadFile] = File(...)):  # 接收上传文档，并逐个推送处理进度。
    """接收上传文档，并逐个推送处理进度。"""
    services = _services(request)  # 将服务组件设为共享服务的结果。

    accepted: list[tuple[str, bytes]] = []  # 初始化已接受的文件为空列表。
    rejected: list[dict[str, str]] = []  # 初始化被拒绝的文件为空列表。
    limit = services.settings.storage.max_upload_mb * 1024 * 1024  # 把配置的兆字节上限换算为字节数。

    for upload in files:  # 遍历文件列表，逐项取得upload。
        name = upload.filename or "unnamed"  # 将名称设为文件名或“unnamed”。
        content = await upload.read()  # 异步读出当前上传文件的原始字节。
        if len(content) > limit:  # 超出单文件上限时在入库前拒绝。
            rejected.append(  # 将字段映射加入被拒绝的文件。
                {  # 开始构造字段映射。
                    "name": name,  # 填入名称字段。
                    "error": f"file exceeds the {services.settings.storage.max_upload_mb} MB limit",  # 填入错误字段。
                }  # 结束本组字段。
            )  # 结束当前多行表达式。
            continue  # 跳过本项，继续处理下一项。
        accepted.append((name, content))  # 同时保留已接受文件的名称和内容。

    def stream():  # 定义数据流的处理逻辑。
        for item in rejected:  # 遍历被拒绝的文件，逐项取得条目。
            yield _sse("rejected", item)  # 向客户端推送“rejected”事件。

        for name, content in accepted:  # 遍历已接受的文件，逐项取得名称、内容。
            # 提前确定文档标识，让客户端能把后续进度关联到同一个文件。
            job_id = uuid.uuid4().hex[:12]  # 截取随机标识的前十二位作为本次文件任务号。
            yield _sse("start", {"name": name, "job": job_id})  # 向客户端推送“start”事件。

            # 在工作线程执行入库，同时从队列读取进度；等入库返回才读队列会使所有进度同时到达。
            events: queue_module.Queue = queue_module.Queue()  # 用线程安全队列把入库进度交给响应生成器。
            outcome: dict[str, Any] = {}  # 在工作线程和推送逻辑之间共享最终结果。

            def on_progress(progress, job=job_id, filename=name):  # 把本次任务号和文件名绑定到进度回调。
                payload = progress.as_dict()  # 将载荷设为进度的字典表示。
                payload.update({"name": filename, "job": job})  # 将进度与客户端对应的上传项关联。
                events.put(("progress", payload))  # 把进度通知放入事件队列供立即转发。

            def worker(filename=name, payload=content):  # 为当前文件绑定参数，在后台线程执行入库。
                try:  # 在此捕获后续操作可能产生的异常。
                    outcome["result"] = services.pipeline.ingest(  # 保存正常完成的入库结果。
                        filename, payload, on_progress=on_progress  # 将文件名、原始字节及实时进度回调传给入库流程。
                    )  # 结束当前多行表达式。
                except IngestionError as error:  # 捕获入库异常。
                    outcome["error"] = {  # 将处理结果中的错误设为字段映射。
                        "stage": error.stage.value,  # 填入阶段字段。
                        "error": error.message,  # 填入错误字段。
                    }  # 结束本组字段。
                except Exception as error:  # 捕获通用异常。
                    outcome["error"] = {  # 将处理结果中的错误设为字段映射。
                        "stage": Stage.PERSISTING.value,  # 填入阶段字段。
                        "error": str(error),  # 填入错误字段。
                    }  # 结束本组字段。
                finally:  # 无论前面是否成功，都执行收尾操作。
                    events.put(("finished", None))  # 用完成标记通知读取方退出队列循环。

            thread = threading.Thread(target=worker, daemon=True)  # 创建随进程退出的入库工作线程。
            thread.start()  # 启动线程，使主响应能同步推送进度。

            while True:  # 持续处理，直到循环内的退出条件成立。
                kind, payload = events.get()  # 等待工作线程的下一条通知并拆出类型和载荷。
                if kind == "finished":  # 收到内部完成标记时停止等待进度。
                    break  # 结束当前循环。
                yield _sse(kind, payload)  # 向客户端推送类别事件。

            if "error" in outcome:  # 当“error”包含在处理结果时进入此分支。
                yield _sse(  # 向客户端推送“failed”事件。
                    "failed",  # 使用文件入库失败的事件名。
                    {"name": name, "job": job_id, **outcome["error"]},  # 把文件、任务号与失败阶段和原因一起发送。
                )  # 结束当前多行表达式。
                continue  # 跳过本项，继续处理下一项。

            result = outcome["result"]  # 将结果设为处理结果中的结果。
            try:  # 在此捕获后续操作可能产生的异常。
                services.refresh_lexical_index()  # 刷新关键词索引。
            except Exception as error:  # pragma: no cover  # 防御性处理。
                logger.warning("lexical index refresh failed: %s", error)  # 记录警告。

            yield _sse(  # 向客户端推送“done”事件。
                "done",  # 使用单份文档入库完成的事件名。
                {  # 开始构造字段映射。
                    "name": name,  # 填入名称字段。
                    "job": job_id,  # 填入任务标识字段。
                    "doc_id": result.doc_id,  # 填入文档标识字段。
                    "parents": result.parent_count,  # 填入父块列表字段。
                    "children": result.child_count,  # 填入子块列表字段。
                    "duration": round(result.duration_seconds, 2),  # 填入耗时字段。
                },  # 结束本组字段。
            )  # 结束当前多行表达式。

        yield _sse("complete", {"accepted": len(accepted), "rejected": len(rejected)})  # 批次结束时同时报告接收和拒绝的文件数。

    return StreamingResponse(  # 返回流式响应的结果。
        stream(),  # 把逐条产生的事件帧作为响应正文。
        media_type="text/event-stream",  # 传入响应内容类型参数。
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 禁用缓存和代理缓冲，让事件能及时到达客户端。
    )  # 结束当前多行表达式。


@router.post("/api/chat")  # 把下方处理函数绑定到 POST 路由。
async def chat(request: Request):  # 处理提问，先推送来源，再逐段推送回答。
    """处理提问，先推送来源，再逐段推送回答。"""
    services = _services(request)  # 将服务组件设为共享服务的结果。
    body = await request.json()  # 解析问答请求的结构化正文。

    question = (body.get("question") or "").strip()  # 读取问题并去除首尾空白，缺少字段时按空串处理。
    if not question:  # 当问题为空或未启用时进入此分支。
        raise HTTPException(status_code=400, detail="question is required")  # 向上抛出接口异常。

    history = body.get("history") or []  # 将对话历史设为正文中的对话历史或空列表。
    if not isinstance(history, list):  # 对话历史不属于列表类型时进入此分支。
        raise HTTPException(status_code=400, detail="history must be a list")  # 向上抛出接口异常。
    debug = bool(body.get("debug"))  # 读取用户是否要求显示调试详情。

    conversation = _to_conversation(services, history)  # 按配置的轮数上限将客户端历史恢复为对话对象。

    def stream():  # 定义数据流的处理逻辑。
        yield _sse("status", {"stage": "rewriting"})  # 通知客户端进入问题改写阶段。

        query, rewritten = services.answerer.rewrite(question, conversation)  # 取得实际检索用问题及是否改写的标记。
        if rewritten:  # 当改写标记有效时进入此分支。
            yield _sse("rewritten", {"query": query})  # 只有改写已采用时才推送改写后的查询。

        yield _sse("status", {"stage": "retrieving"})  # 通知客户端已开始检索资料。
        try:  # 在此捕获后续操作可能产生的异常。
            outcome = services.retriever.retrieve(query)  # 使用改写或原始问题执行混合召回。
        except ProviderError as error:  # 捕获供应商异常。
            yield _sse("error", {"message": describe_retrieval_error(error)})  # 向客户端推送“error”事件。
            return  # 结束当前处理。
        except Exception as error:  # 捕获通用异常。
            logger.exception("retrieval failed")  # 记录异常及堆栈。
            yield _sse("error", {"message": f"Retrieval failed: {error}"})  # 向客户端推送“error”事件。
            return  # 结束当前处理。

        # 先发送检索来源，让用户在首段回答到达前就能看到依据。
        yield _sse(  # 向客户端推送“sources”事件。
            "sources",  # 使用检索来源事件名。
            {  # 开始构造字段映射。
                "sources": [  # 填入来源列表字段。
                    _context_payload(context, index)  # 把选中的上下文转换为带编号的来源记录。
                    for index, context in enumerate(outcome.contexts, start=1)  # 从一开始编号，使界面与提示词中的引用一致。
                ],  # 结束列表。
                "rerank_applied": outcome.rerank_applied,  # 填入是否采用重排字段。
                "rerank_error": outcome.rerank_error,  # 填入重排失败原因字段。
            },  # 结束本组字段。
        )  # 结束当前多行表达式。

        if debug:  # 当调试开关有效时进入此分支。
            yield _sse("debug", _diagnostics(outcome))  # 向请求了调试信息的客户端推送各阶段候选。

        if not outcome.contexts:  # 当上下文列表为空或未启用时进入此分支。
            yield _sse("token", {"text": NO_CONTEXT_ANSWER})  # 发送没有可引用内容时的约定回答。
            yield _sse("done", {"answer": NO_CONTEXT_ANSWER, "empty": True})  # 结束本次回答，并显式标记知识库未命中。
            return  # 结束当前处理。

        collected: list[str] = []  # 初始化已输出的回答片段为空列表。
        try:  # 在此捕获后续操作可能产生的异常。
            for piece in services.answerer.stream_answer(question, outcome.contexts, conversation):  # 将问题、检索上下文及对话历史交给流式回答器。
                collected.append(piece)  # 记下已交付片段，以便最后还原完整回答。
                yield _sse("token", {"text": piece})  # 有新文字即推送，减少客户端等待。
        except Exception as error:  # 捕获通用异常。
            # 通知客户端回答中途截断，以便区分正常结束和生成失败。
            message = describe_generation_error(error)  # 将消息设为整理生成错误提示的结果。
            yield _sse("error", {"message": message, "truncated": True})  # 向客户端推送“error”事件。
            yield _sse("done", {"answer": "".join(collected), "truncated": True})  # 将已经交付的部分回答和截断标记一起返回。
            return  # 结束当前处理。

        answer = "".join(collected)  # 按到达顺序拼出最终回答。
        yield _sse("done", {"answer": answer, "truncated": False})  # 向客户端推送“done”事件。

    return StreamingResponse(  # 返回流式响应的结果。
        stream(),  # 使用问答事件生成器作为持续输出的响应正文。
        media_type="text/event-stream",  # 传入响应内容类型参数。
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 关闭缓存与代理缓冲，避免回答片段积压。
    )  # 结束当前多行表达式。


def _to_conversation(services: Services, history: list) -> Any:  # 根据客户端传来的记录恢复对话历史。
    """根据客户端传来的记录恢复对话历史；服务端无需保存会话。"""
    from rag_demo.generation.answer import Conversation  # 从 rag_demo.generation.answer 导入对话历史。

    conversation = Conversation(max_turns=services.settings.llm.history_turns)  # 建立具有配置中轮数上限的历史容器。
    for item in history:  # 遍历对话历史，逐项取得条目。
        if not isinstance(item, dict):  # 条目不属于字典类型时进入此分支。
            continue  # 跳过本项，继续处理下一项。
        question = str(item.get("question", "")).strip()  # 将问题设为清理首尾空白后的条目中的问题的字符串形式。
        answer = str(item.get("answer", "")).strip()  # 将回答设为清理首尾空白后的条目中的回答的字符串形式。
        if question and answer:  # 只保留提问与回答都非空的一轮。
            conversation.append(question, answer)  # 将问题加入对话历史。
    return conversation  # 返回对话历史。


def describe_retrieval_error(error: Exception) -> str:  # 将检索错误转成可操作的提示，引导用户检查发生故障的检索环节。
    """将检索错误转成可操作的提示，引导用户检查发生故障的检索环节。"""
    from rag_demo.errors import ConfigurationIssue, ContractMismatch, Unavailable  # 从 rag_demo.errors 引入本模块使用的类型和工具。

    if isinstance(error, ConfigurationIssue):  # 当错误属于供应商配置异常类型时进入此分支。
        return f"Retrieval is misconfigured: {error.message}"  # 返回格式化文本。
    if isinstance(error, Unavailable):  # 当错误属于服务不可达异常类型时进入此分支。
        return f"A retrieval dependency is unreachable: {error.message}"  # 返回格式化文本。
    if isinstance(error, ContractMismatch):  # 当错误属于响应协议异常类型时进入此分支。
        return f"A retrieval dependency returned an unexpected response: {error.message}"  # 返回格式化文本。
    return f"Retrieval failed: {type(error).__name__}: {error}"  # 返回格式化文本。


def _diagnostics(outcome) -> dict[str, Any]:  # 构造各检索阶段的诊断信息，便于对比参数效果。
    """构造各检索阶段的诊断信息，便于对比参数效果。"""

    def summarise(candidates, limit=10):  # 定义summarise的处理逻辑。
        return [  # 返回从candidates的指定区间中提取的各项。
            {  # 开始构造字段映射。
                "id": candidate.id,  # 填入标识字段。
                "parent_id": candidate.parent_id,  # 填入所属父块标识字段。
                "channels": candidate.channels,  # 填入命中通道字段。
                "dense_rank": candidate.dense_rank,  # 填入向量召回名次字段。
                "sparse_rank": candidate.sparse_rank,  # 填入关键词召回名次字段。
                "fused_rank": candidate.fused_rank,  # 填入融合名次字段。
                "fused_score": round(candidate.fused_score, 6),  # 填入融合分数字段。
                "rerank_rank": candidate.rerank_rank,  # 填入重排名次字段。
                "rerank_score": round(candidate.rerank_score, 6)  # 有效重排分数保留六位小数以便比较。
                if candidate.rerank_score is not None  # 按条件筛选或选择当前值。
                else None,  # 采用备用值。
                "preview": candidate.text[:120],  # 最多提供一百二十字符的候选正文预览。
            }  # 结束本组字段。
            for candidate in candidates[:limit]  # 逐项处理推导式中的候选项。
        ]  # 结束列表。

    return {  # 返回字段映射。
        "query": outcome.query,  # 填入查询字段。
        "dense": summarise(outcome.dense),  # 填入向量召回字段。
        "sparse": summarise(outcome.sparse),  # 填入关键词召回字段。
        "fused": summarise(outcome.fused),  # 填入融合结果字段。
        "reranked": summarise(outcome.reranked),  # 填入重排结果字段。
    }  # 结束本组字段。