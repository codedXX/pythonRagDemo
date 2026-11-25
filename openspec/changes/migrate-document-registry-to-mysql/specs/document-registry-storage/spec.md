## Purpose

定义文档登记数据在 MySQL 中的持久化行为，以及部署者如何从已有 SQLite 登记库切换，确保用户能继续查看文档状态、管理原有文档并使用已有检索内容。

## ADDED Requirements

### Requirement: MySQL registry persistence

The service and the migration command SHALL connect to MySQL installed directly on the application host, using `127.0.0.1:3306` as the default endpoint. The service SHALL preserve the existing document API representation and lifecycle semantics.

#### Scenario: 新部署与服务重启
- **WHEN** 本机 MySQL 在默认端口运行，部署者仅配置空数据库的库名和凭据，应用登记文档后使用同一配置重启
- **THEN** 文档清单仍包含原 ID、名称、状态、父子块数、字符数、错误原因及 UTC 秒级时间字符串，健康接口的可用文档数只计入 active 记录

#### Scenario: 重试同一标识
- **WHEN** 已有文档以相同 ID 重新开始入库
- **THEN** 文档回到 staging，保留原创建时间，清零计数、清除旧错误并更新修改时间

#### Scenario: 精确匹配文件名
- **WHEN** 库中已有 `Guide.md` 与 `guide.md`，用户上传新版 `Guide.md`
- **THEN** 新版替换同名旧版及其片段，`guide.md` 的记录和内容仍可访问

### Requirement: Offline migration of existing registrations

The migration command SHALL preserve all registration fields and the identifiers linking documents to Milvus. It MUST leave the source database and Milvus data intact and MUST work without model services.

#### Scenario: 迁入已有数据
- **WHEN** 服务停止写入后，部署者将含有 active、staging 和 failed 记录的 SQLite 登记库迁入空的 MySQL 登记表
- **THEN** 源、目标的 ID 集合和每条记录的所有字段均一致，包括历史时间、空错误值和失败原因
- **AND** 服务连接原 Milvus 集合后，已有可检索的文档可按原 ID 检索，无需重新上传

#### Scenario: 重复执行
- **WHEN** 对已完整导入且与源库内容一致的目标再次执行迁移
- **THEN** 命令报告数据一致并成功退出，行数、字段值及时间均保持原值

#### Scenario: 目标已经产生不同的数据
- **WHEN** 目标登记表非空且与源快照的 ID 集合或字段值不同
- **THEN** 命令报告差异并以非零状态退出，保留目标原有记录

#### Scenario: 导入失败
- **WHEN** 向空目标导入时某行写入失败
- **THEN** 命令以非零状态退出，目标表没有本次导入的部分记录，部署者可在修复原因后重试
