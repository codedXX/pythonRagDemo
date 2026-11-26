## Why

现有 Markdown 解析器会漏掉下划线式标题，也会把长代码围栏内的示例标题当成真实章节。这两种情况已经复现，会让后续正文缺少章节来源或引用错误章节。

## What Changes

- 使用 `markdown-it-py` 识别 CommonMark 块结构并启用表格扩展，替换逐行判断语法的正则和围栏状态。
- 将解析结果映射为有序结构片段。文档标题决定引用路径，代码示例、列表、引用和表格按各自的块边界保留原文。
- 用解析与分块样例验证标题归属、代码完整性及嵌套内容只收录一次。

## Capabilities

### New Capabilities

### Modified Capabilities

- `document-ingestion` 增加 Markdown 块识别和章节归属的可验收要求，覆盖新上传及重新导入的文档。

## Impact

涉及 Markdown 解析模块、Python 运行依赖及锁文件、解析与分块测试，以及 README 中的解析说明。PDF 已由 PyMuPDF 提取文本和页码，本次范围集中在 Markdown。已有 Markdown 文档可通过重新上传获得修正后的片段和引用。
