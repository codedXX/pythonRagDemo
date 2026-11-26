# RAG Demo

混合检索问答系统：BM25 + 向量双路召回、语义父子分块、RRF 融合、交叉重排，
配一个可以上传 PDF/Markdown 并直接提问的静态页面。

## 它能做什么

- 上传 PDF 或 Markdown，自动解析、分块、向量化、入库
- 用**稀疏（BM25）+ 稠密（向量）**两路召回，RRF 融合后用交叉重排精选
- **父子文档**：小片段负责"找得准"，大段落负责"读得全"
- 每个回答附来源（文档名 / 页码 / 标题路径 / 命中片段）
- 支持多轮追问，自动把指代性追问改写成独立问题
- 可查看检索各阶段结果（两路原始排序、融合顺序、重排前后变化）

## 快速开始

### 1. 环境

需要 Python 3.13（由 `uv` 管理）、本机原生 MySQL 8.4 和可访问的 Milvus 实例。

```bash
# 安装 uv（若尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# 安装依赖
uv sync
```

### 2. Milvus

需要一个 Milvus standalone（本项目在 v2.6.23 上验证）。

```bash
# 若用 Docker Compose 部署
docker compose up -d    # 默认监听 127.0.0.1:19530
```

确认可达：

```bash
nc -z 127.0.0.1 19530 && echo "milvus reachable"
```

### 3. 本机 MySQL 登记库

在应用所在机器上启动原生安装的 MySQL。macOS 使用官方安装包时，可在「系统设置 → MySQL」中启动服务。使用 Homebrew 安装的 mysql@8.4 时，运行 `brew services start mysql@8.4`。应用默认通过 TCP 连接 `127.0.0.1:3306`。

用管理员账号登录 MySQL，在首次部署时创建库和应用账号。以下 SQL 会返回应用账号的随机密码，请将它填入本地配置的 `mysql.password` 或环境变量 `RAG__MYSQL__PASSWORD`。

```sql
CREATE DATABASE rag_demo CHARACTER SET utf8mb4;
CREATE USER 'rag_demo'@'127.0.0.1' IDENTIFIED BY RANDOM PASSWORD;
GRANT CREATE, SELECT, INSERT, UPDATE, DELETE ON `rag\_demo`.* TO 'rag_demo'@'127.0.0.1';
```

通过 TCP 验证账号和库名，按提示输入密码。若本机使用其他端口，请一并调整 `mysql.port` 与命令中的端口。

```bash
mysql --protocol=TCP -h 127.0.0.1 -P 3306 -u rag_demo -p rag_demo -e 'SELECT DATABASE(), VERSION()'
```

### 4. 配置密钥

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，填入**两个密钥**（都手动填写）：

| 配置项 | 用途 | 去哪拿 |
|---|---|---|
| `embedding.api_key` | 向量化（阿里云百炼） | 百炼控制台 |
| `rerank.api_key` | 重排（阿里云百炼） | 同上，通常同一个 key |
| `llm.api_key` | 问答生成（中转站） | 你的中转站 |

`config.yaml` 已在 `.gitignore` 中，不会被提交。

### 5. 跑自检（**强烈建议**）

```bash
uv run python scripts/selfcheck.py
```

它会验证三件事，并在失败时直接告诉你改哪里：

```
[OK  ] embedding       model available, dimension 1024
[OK  ] rerank          model available, ranking plausible
[OK  ] milvus          server 2.6.23 reachable
[FAIL] llm             credentials rejected (HTTP 401): ...
```

自检同时会**探测并记录向量维度**。维度是建 collection 时锁死的，所以不靠猜。

> ⚠️ 若 embedding / rerank 的模型名不存在，自检会明确报出模型名与错误码。
> 改 `config.yaml` 里的 `model` 一行即可，不需要改代码。

### 6. 启动

```bash
uv run python scripts/serve.py
```

打开 http://127.0.0.1:8000 即可使用。

## 架构

```
浏览器静态页 (HTML/CSS/ES modules，零构建)
    │
    │ POST /api/chat (SSE)        · sources 先于 token 到达
    │ POST /api/docs (JSON)       · 逐文件返回导入结果
    │ GET  /api/health            · 依赖状态
    ▼
FastAPI
    │
    ├── 解析      PyMuPDF / Markdown AST，扫描件拒收
    ├── 分块      结构优先 + 语义兜底 → 父块 ⊃ 子块
    ├── 向量化    百炼 OpenAI 兼容端点，批处理 + 退避重试 + 内容哈希缓存
    ├── 存储      Milvus 双 collection（Strong 一致性）+ 本机 MySQL 文档登记表
    ├── 检索      bm25s + jieba ∥ Milvus ANN → RRF → 重排 → 父块去重展开
    └── 生成      中转站 /v1/chat/completions，流式
```

### 上传接口

向 `POST /api/docs` 提交 multipart 表单，一个或多个文件均使用 `files` 字段。请求结束后一次返回 JSON，结果顺序与提交顺序一致。有效表单返回 HTTP 200，每份文件的 `status` 表示该文件是否成功。下面是同批导入一份 Markdown 和一份扫描 PDF 的响应示例。

```json
{
  "results": [
    {"name": "guide.md", "status": "ok", "doc_id": "4e5cbdd70482aa99bfb296746154821a", "parents": 2, "children": 4, "duration": 0.4},
    {"name": "scan.pdf", "status": "failed", "error": "parsing: PDF appears to be image-based: 0 extractable characters across 1 page(s), which is 0 per page (minimum 40). Such a file would need OCR, which is not supported."}
  ]
}
```

### 关键设计点

**父子文档**。检索用子块（约 320 字）命中精确，生成用父块（约 1200 字）上下文完整。
每个子块带 `parent_id`，命中后展开父块。

**重排输入带面包屑**。喂给重排的是 `文档名 > 标题路径 | 正文`，而不是裸正文。
否则"如下表所示"这类片段脱离标题会被误判为不相关。

**重排候选数 > 最终供给数**。重排 top-30 里可能有多条来自同一父块，去重后会缩水，
所以要从剩余候选回填，否则配置的上下文数形同虚设。

**BM25 从 Milvus 重建**。不读磁盘原文。若两路召回的语料不一致，融合结果将无法解释。

**Milvus 一致性必须 Strong**。实测：默认一致性下 insert 后立即查询返回 0 条，
要等 flush 或时间窗口。那会表现为"上传完成后立刻提问却查不到"。

## 调参入口

全在 `config.yaml` 里，改完重启生效。

| 参数 | 含义 | 调整方向 |
|---|---|---|
| `retrieval.rrf_k` | RRF 的排名常数（默认 60） | 调小 → 更看重头部名次 |
| `retrieval.dense_top_k` / `sparse_top_k` | 每路召回条数 | 调大 → 召回更全，重排更慢 |
| `retrieval.rerank_candidates` | 送重排的候选数 | 应明显大于 `context_count` |
| `retrieval.context_count` | 最终给模型的父块数 | 调大 → 上下文更全，更费 token |
| `retrieval.context_char_budget` | 上下文总字符预算 | 防止超长父块挤占预算 |
| `chunking.parent_max_chars` | 父块上限 | 越小 → 切得越碎，语义完整性下降 |
| `chunking.semantic_break_percentile` | 语义断点阈值百分位 | 调低 → 更激进地切分 |
| `embedding.batch_size` | 批量大小 | 遇到限流就调小 |

调检索时建议打开页面上的「调试」勾选框，可以看到：

```
dense    ▸ 每条的向量召回名次
sparse   ▸ 每条的 BM25 名次
fused    ▸ RRF 融合后的名次与分数，以及每条被哪几路召回到
reranked ▸ 重排后的名次与分数
```

重排是否有用，就看同一个 `parent_id` 在 `fused` 与 `reranked` 里的名次差。

## 将已有 SQLite 登记库切换到本机 MySQL

1. 暂停上传和删除，等在途任务结束后停止旧服务。保留旧版程序、配置及原 SQLite 文件。使用旧配置的实际路径备份登记库，例如 `sqlite3 data/rag.db ".backup 'data/rag-before-mysql.db'"`。
2. 按快速开始准备 MySQL 空库并配置账号，继续使用原 Milvus 集合。执行迁移命令。它以只读方式打开备份，按原 ID 导入全部状态，在同一事务内逐条核对九个字段，并报告总行数和各状态计数。

   ```bash
   uv run python scripts/migrate_registry.py data/rag-before-mysql.db --config config.yaml
   ```

3. 对比 `sqlite3 -readonly data/rag-before-mysql.db 'SELECT state, COUNT(*) FROM documents GROUP BY state'` 与命令输出。完全相同的目标可重复执行。目标已有差异或导入失败时，命令报告原因并以非零状态退出，修复后可重试。
4. 仅在校验成功后启动新版本，保持写入口关闭。检查 `GET /api/docs` 的 ID、状态与数量，通过问答核对已有 active 文档的引用，再开放上传和删除。
5. 开放写入前若需回退，停止新版，用旧版、旧配置和已备份的 SQLite 库启动。开放写入后若需回退，先停写并核对期间新增、替换或删除的文档及其 Milvus 片段，确保登记库与当前片段一致。

## 开发

```bash
uv sync --locked --extra dev
uv run pytest -m "not integration"
```

运行 MySQL 集成测试时，在本机 MySQL 实例准备独立测试账号。夹具为每个用例创建并清理以 `ragtest_` 开头的随机数据库。下面的权限仅授予这组测试库。

```sql
CREATE USER 'ragtest'@'127.0.0.1' IDENTIFIED BY RANDOM PASSWORD;
GRANT ALL ON `ragtest\_%`.* TO 'ragtest'@'127.0.0.1';
```

设置 `RAG_TEST_MYSQL_USER=ragtest`、`RAG_TEST_MYSQL_PASSWORD`，按需设置 `RAG_TEST_MYSQL_HOST` 和 `RAG_TEST_MYSQL_PORT`。配置这些环境变量后执行 `uv run pytest -q -rs`。入库和端到端测试还需要本机 Milvus。未指定测试账号时，MySQL 用例会跳过。显式指定了账号但连接失败时，用例会报错。

## 已知限制

- **不支持扫描件 PDF**。没有文字层就无法抽取，系统会明确拒收并提示需要 OCR，
  而不是产出一个永远答不出问题的空文档。
- **单知识库**。所有文档混在一个全局索引里检索，没有库隔离。
- **无鉴权**。接口不设认证，只适合本机使用。不要直接暴露到公网。
- **BM25 索引在内存中**。启动时和每次入库后从 Milvus 全量重建。
  目标规模是几十份文档 / 万级 chunk；再大会需要改为增量或落盘索引。
- **检索质量没有量化基线**。原生 IR 指标与 RAGAS 评测在后续变更里，
  当前调参依赖「调试」面板的人工观察。

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 自检报 `model ... may not exist` | 模型名不对，或账号没开通 | 改 `config.yaml` 里的 `model` |
| 自检报 `credentials rejected` | key 填错或没填 | 检查 `api_key`，注意别带空格 |
| 自检报 `endpoint not found` 且涉及 rerank | 把 rerank 配到了 `/compatible-mode/v1/rerank` | 该路径不存在，必须用原生端点（示例配置已正确） |
| 上传后立刻提问查不到 | 有人改掉了一致性级别 | `milvus.consistency_level` 必须是 `Strong` |
| 自检报维度不一致 | 换了 embedding 模型，但库里还是旧向量 | 删除 `rag_child`/`rag_parent`/`rag_meta` 三个 collection 后重新入库 |
| 上传报「疑似扫描件」 | PDF 没有文字层 | 需要先做 OCR，本系统不支持 |

## 目录

```
src/rag_demo/
├── config.py            配置加载（环境变量 > config.yaml > 默认值）
├── selfcheck.py         启动期依赖验证
├── ingestion.py         入库流水线
├── errors.py            错误分类（配置问题 vs 临时故障）
├── parsers/             PDF / Markdown 解析、扫描件检测
├── chunking/            父子双层分块、语义断点检测
├── embedding/           向量化适配器、重排适配器（两套协议）
├── store/               Milvus 双 collection、MySQL 登记表
├── retrieval/           词法索引、RRF 融合、重排、父块展开
├── generation/          流式生成、prompt 组装、追问改写
└── api/                 FastAPI 路由与装配
static/                  index.html / style.css / app.js
scripts/                 selfcheck.py / serve.py
tests/                   单元 + 集成 + 端到端
```