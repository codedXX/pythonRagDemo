## Why

当前文档清单依赖服务所在机器上的 SQLite 文件。项目希望使用 MySQL 管理登记数据，同时让已有文档继续通过原来的标识关联 Milvus 中的内容。

## What Changes

- **BREAKING** 文档登记库使用直接安装在应用所在机器上的 MySQL 服务。应用和迁移命令默认通过 `127.0.0.1:3306` 连接。部署者通过项目配置或环境变量提供库名、账号、密码和实际端口，现有部署需从 `storage.sqlite_path` 切换。
- 提供一次性离线迁移命令，复制全部登记字段，涵盖已激活、入库中和失败的记录。迁移能在失败后重试。
- 文档 API 继续提供原字段与时间格式，上传、同名替换、删除及健康接口中的文档计数沿用现有语义。数据迁移只读写登记表。

## Capabilities

### New Capabilities

- `document-registry-storage` — 定义服务如何使用 MySQL 持久化文档登记，以及已有部署如何验证和迁移 SQLite 数据。

### Modified Capabilities

## Impact

涉及 `DocumentRegistry`、配置加载与脱敏、服务组装、Python 驱动和锁文件、登记表及入库测试、部署说明和迁移命令。Milvus 中已有片段通过原 `doc_id` 继续使用，迁移无需调用嵌入模型。
