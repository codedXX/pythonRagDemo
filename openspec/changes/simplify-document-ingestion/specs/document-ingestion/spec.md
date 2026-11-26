## REMOVED Requirements

### Requirement: Ingestion progress reporting

**Reason**: 用户只需要知道本次上传的最终结果。上传过程中的阶段事件与百分比已取消。
**Migration**: 原先消费上传 SSE 的调用方改为读取同一接口的 JSON 响应，按文件检查结果。

## ADDED Requirements

### Requirement: Batch ingestion results

服务 MUST 在处理完 `POST /api/docs` 接收的多文件表单后，一次返回 `application/json`。有效表单的响应为 HTTP 200，`results` 数组按提交顺序为每个文件各保留一项。文件处理失败时 MUST 返回文件名与具体原因，并继续处理后续文件。

#### Scenario: Successful import

- **WHEN** 用户上传一份可提取正文的 PDF
- **THEN** 结果包含 `name`、`status: "ok"`、`doc_id`、父块数 `parents`、子块数 `children` 和耗时秒数 `duration`，并可立即提问检索该文档

#### Scenario: Mixed batch

- **WHEN** 表单依次包含扫描 PDF、超过大小限制的文件和可用 Markdown
- **THEN** 前两项的 `status` 为 `"failed"`，各自的 `error` 分别说明需要 OCR 和大小限制，第三项报告成功

#### Scenario: Duplicate filenames

- **WHEN** 同一批次包含两份内容不同的同名文件且都导入成功
- **THEN** 响应按顺序返回两个结果，文档列表与检索只提供后一版本
