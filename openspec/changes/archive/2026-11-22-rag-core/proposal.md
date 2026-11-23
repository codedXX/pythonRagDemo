## Why

需要一个可运行、可观测、可调参的 RAG 学习型项目：把「混合检索（BM25 + 向量）、语义分块、父子文档、RRF、重排」这条工业界主流链路完整地跑通一遍，并配一个能上传 PDF/MD 并直接问答的静态页面。现有环境已有 OrbStack 中的 Milvus standalone，中转站与阿里云百炼均可访问，具备立刻动手的条件。

## What Changes

- 新增基于 `uv` 管理的 Python 项目骨架，采用分层包结构（parsers / chunking / embedding / store / retrieval / generation / api）。
- 新增**文档解析**能力：PDF 走 PyMuPDF、Markdown 走 AST 解析，并对疑似扫描件显式拒收（不做 OCR）。
- 新增**两层分块**能力：结构优先（标题层级 / 版面块）+ 语义兜底（句子嵌入断点检测），产出「父块 ⊃ 子块」双层结构。
- 新增**向量化**能力：调用阿里云百炼 OpenAI 兼容端点，带批量、退避重试与按内容哈希的磁盘缓存。
- 新增**向量库存储**：Milvus 双 collection（`rag_child` / `rag_parent`），Strong 一致性，`active` 标志实现原子入库；SQLite 维护文档注册表。
- 新增**混合检索**：`bm25s` + `jieba` 稀疏召回 与 Milvus 稠密召回并行，RRF 融合，DashScope 原生端点重排，父块去重展开。
- 新增**问答生成**：经中转站调用 `claude-sonnet-5` 流式生成，支持多轮 `query rewrite`（带启发式跳过）。
- 新增 **FastAPI 接口层**：`/api/chat`(SSE)、`/api/docs`(SSE 上传进度)、`/api/health`(依赖自检)。
- 新增**静态问答页面**：纯 HTML/CSS/JS，FastAPI 托管，含对话区、拖拽上传进度、引用卡片与文档列表。
- 新增**启动自检**：探测 embedding 维度与模型有效性、rerank 模型与响应结构、生成模型与流式能力，将「模型名未知」风险降级为配置错误。
- **非目标**：不做 OCR、不做多知识库隔离、不做用户鉴权、不做 RAGAS 评测（拆到后续独立变更）。

## Capabilities

### New Capabilities
- `document-ingestion`: 文档解析（PDF/MD）、扫描件拒收、两层分块、向量化与原子入库、文档生命周期管理。
- `hybrid-retrieval`: 稠密与稀疏双路召回、RRF 融合、重排、父子文档展开与去重。
- `answer-generation`: 检索增强的问答生成、多轮 query rewrite、流式输出与引用溯源。
- `qa-web-ui`: 静态问答页面，含上传、进度反馈、引用展示与文档管理。
- `runtime-selfcheck`: 启动期对 embedding / rerank / 生成三类外部依赖的有效性与参数探测。

### Modified Capabilities

（无：本项目为全新仓库，`openspec/specs/` 当前为空。）

## Impact

- **代码**：全新目录 `src/rag_demo/`、`static/`、`tests/`、`scripts/`、`data/`。
- **依赖**：`fastapi`、`uvicorn`、`pymilvus`、`bm25s`、`jieba`、`pymupdf`、`openai`、`httpx`、`pydantic-settings`、`pytest`。RAGAS 相关依赖不进主依赖树。
- **外部系统**：
  - Milvus standalone v2.6.23，`127.0.0.1:19530`（已验证连通）。
  - 阿里云百炼：embedding 走 `/compatible-mode/v1/embeddings`，rerank 走 `/api/v1/services/rerank/text-rerank/text-rerank`（**两者协议不同**，已验证）。
  - 中转站 `http://8.134.19.201:20089/v1`：`/chat/completions` 可用，`/v1/rerank` 不存在。
- **配置**：新增 `config.example.yaml`（入版本库）与 `config.yaml`（**不入版本库**，含手动填写的密钥）。
- **运行时约束**：collection 必须使用 `consistency_level="Strong"`，否则入库后立即查询不可见（已实测复现）。
- **对既有数据的影响**：Milvus 中已存在 `rag_demo`、`rag_chunks` 两个历史实验 collection，本变更使用独立命名，不读取、不修改、不删除它们。