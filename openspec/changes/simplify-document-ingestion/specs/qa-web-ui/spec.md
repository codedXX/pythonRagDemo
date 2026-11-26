## REMOVED Requirements

### Requirement: Ingestion progress feedback

**Reason**: 批量上传改为在请求结束时返回结果，页面无需消费进度事件。
**Migration**: 用一次请求的等待提示和每个文件的最终结果替换各文件的进度条。

## ADDED Requirements

### Requirement: Upload result feedback

页面 MUST 支持通过选择或拖入文件发起上传。请求进行时 MUST 显示“正在导入”等待提示，结束后 MUST 显示每个文件的结果并刷新文档列表。

#### Scenario: Waiting for import

- **WHEN** 文件已提交，服务尚未返回结果
- **THEN** 页面显示正在导入，省略处理阶段、进度条与百分比

#### Scenario: Completed batch with partial failure

- **WHEN** 同一次上传中，有文件成功入库，有文件处理失败
- **THEN** 等待提示结束，页面逐项标明可提问的文件和失败文件的具体原因，并刷新文档列表

#### Scenario: Request fails before results arrive

- **WHEN** 上传请求发生网络错误或服务返回 HTTP 错误
- **THEN** 页面结束等待提示并报告请求失败
