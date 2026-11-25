"""在 Milvus 中分别保存用于向量召回的子块、按主键读取的父块，以及建库元数据。

子块需要真实向量；父块和元数据使用二维常量占位，以满足每个集合必须有带索引向量字段、且维度至少为二的要求。项目固定使用强一致性，确保入库成功后立即可以检索；在 Milvus 2.6.23 上，默认的有界一致性可能暂时查不到刚写入的数据。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import json  # 引入结构化数据的编解码工具。
from collections.abc import Iterable, Sequence  # 从 collections.abc 引入本模块使用的类型和工具。
from dataclasses import dataclass  # 从 dataclasses 导入数据类装饰器。

from pymilvus import DataType, MilvusClient  # 从 pymilvus 引入本模块使用的类型和工具。

from rag_demo.config import Settings  # 从 rag_demo.config 导入项目配置。

# 限制字段长度，使超长内容在写入时暴露问题。
MAX_ID_LEN = 96  # 将MAX_ID_LEN设为96。
MAX_TEXT_LEN = 65535  # 将MAX_TEXT_LEN设为65535。
MAX_BREADCRUMB_LEN = 4096  # 将MAX_BREADCRUMB_LEN设为4096。
MAX_NAME_LEN = 512  # 将MAX_NAME_LEN设为512。

# 父块只按主键读取，用二维常量满足数据库对向量及索引的最低要求。
PLACEHOLDER_DIM = 2  # 将PLACEHOLDER_DIM设为2。
PLACEHOLDER_VECTOR = [0.0, 0.0]  # 将PLACEHOLDER_VECTOR设为列表。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class ChildUnit:  # 保存一条参与检索的子块记录。
    """保存一条参与检索的子块记录。"""

    id: str  # 声明标识字段。
    doc_id: str  # 声明文档标识字段。
    parent_id: str  # 声明所属父块标识字段。
    child_idx: int  # 声明子块序号字段。
    text: str  # 声明文本字段。
    breadcrumb: str  # 声明标题路径字段。
    page: int  # 声明页码字段。
    source_name: str  # 声明来源文件名字段。
    embedding: list[float]  # 声明向量字段。

    def as_row(self) -> dict:  # 组装待写入数据库的一行数据。
        return {  # 返回字段映射。
            "id": self.id,  # 填入标识字段。
            "doc_id": self.doc_id,  # 填入文档标识字段。
            "parent_id": self.parent_id,  # 填入所属父块标识字段。
            "child_idx": self.child_idx,  # 填入子块序号字段。
            "text": self.text[:MAX_TEXT_LEN],  # 按子块文本字段的上限截取正文。
            "breadcrumb": self.breadcrumb[:MAX_BREADCRUMB_LEN],  # 限制保存的标题路径长度。
            "page": self.page,  # 填入页码字段。
            "source_name": self.source_name[:MAX_NAME_LEN],  # 限制保存的来源名称长度。
            "active": False,  # 先写成未激活状态，整份上传完成后再开放检索。
            "embedding": self.embedding,  # 填入向量字段。
        }  # 结束本组字段。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class ParentUnit:  # 保存一条提供给生成模型的父块记录。
    """保存一条提供给生成模型的父块记录。"""

    parent_id: str  # 声明所属父块标识字段。
    doc_id: str  # 声明文档标识字段。
    text: str  # 声明文本字段。
    breadcrumb: str  # 声明标题路径字段。
    page_start: int  # 声明起始页字段。
    source_name: str  # 声明来源文件名字段。

    def as_row(self) -> dict:  # 组装待写入数据库的一行数据。
        return {  # 返回字段映射。
            "parent_id": self.parent_id,  # 填入所属父块标识字段。
            "doc_id": self.doc_id,  # 填入文档标识字段。
            "text": self.text[:MAX_TEXT_LEN],  # 按父块字段长度上限保存正文。
            "breadcrumb": self.breadcrumb[:MAX_BREADCRUMB_LEN],  # 按字段上限保存父块标题路径。
            "page_start": self.page_start,  # 填入起始页字段。
            "source_name": self.source_name[:MAX_NAME_LEN],  # 填入来源文件名字段。
            "active": False,  # 父块也要等文档写完后才激活。
            "placeholder": list(PLACEHOLDER_VECTOR),  # 为每行复制占位向量，父块无需模型编码。
        }  # 结束本组字段。


class InvalidStoreMetadata(ValueError):
    """已有建库记录无法描述向量模型和维度。"""


class MilvusStore:  # 管理三个集合的结构和底层读写。
    """管理三个集合的结构和底层读写。"""

    def __init__(self, settings: Settings, client: MilvusClient | None = None) -> None:  # 保存实例所需的配置和依赖。
        self._settings = settings  # 将测试配置设为配置。
        self._client = client  # 将测试客户端设为客户端。
        self._dimension: int | None = None  # 将向量维度设为空值。

    # 连接向量数据库。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def client(self) -> MilvusClient:  # 定义客户端的处理逻辑。
        if self._client is None:  # 当测试客户端为空值时进入此分支。
            self._client = MilvusClient(  # 将测试客户端设为数据库客户端的结果。
                uri=self._settings.milvus.uri,  # 传入连接地址参数。
                token=self._settings.milvus.token or None,  # 传入词元参数。
            )  # 结束当前多行表达式。
        return self._client  # 返回测试客户端。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def child_name(self) -> str:  # 定义子块集合名的处理逻辑。
        return self._settings.milvus.child_collection  # 返回子块集合名。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def parent_name(self) -> str:  # 定义父块集合名的处理逻辑。
        return self._settings.milvus.parent_collection  # 返回父块集合名。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def meta_name(self) -> str:  # 定义元数据集合名的处理逻辑。
        return self._settings.milvus.meta_collection  # 返回元数据集合名。

    # 创建集合与索引。

    def ensure_collections(self, dimension: int) -> None:  # 按实测向量维度创建尚不存在的集合。
        """按实测向量维度创建尚不存在的集合。"""
        self._dimension = dimension  # 保存传入的向量维度。
        if not self.client.has_collection(self.child_name):  # 检查子块集合名是否尚未创建。
            self._create_child(dimension)  # 调用_create_child。
        if not self.client.has_collection(self.parent_name):  # 检查父块集合名是否尚未创建。
            self._create_parent()  # 调用_create_parent。
        if not self.client.has_collection(self.meta_name):  # 检查元数据集合名是否尚未创建。
            self._create_meta()  # 调用_create_meta。

    def _create_child(self, dimension: int) -> None:  # 定义_create_child的处理逻辑。
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)  # 使用应用提供的主键，并要求字段符合显式结构。
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=MAX_ID_LEN)  # 声明集合的标识字段。
        schema.add_field("doc_id", DataType.VARCHAR, max_length=MAX_ID_LEN)  # 声明集合的文档标识字段。
        schema.add_field("parent_id", DataType.VARCHAR, max_length=MAX_ID_LEN)  # 声明集合的所属父块标识字段。
        schema.add_field("child_idx", DataType.INT32)  # 声明集合的子块序号字段。
        schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LEN)  # 声明集合的文本字段。
        schema.add_field("breadcrumb", DataType.VARCHAR, max_length=MAX_BREADCRUMB_LEN)  # 声明集合的标题路径字段。
        schema.add_field("page", DataType.INT32)  # 声明集合的页码字段。
        schema.add_field("source_name", DataType.VARCHAR, max_length=MAX_NAME_LEN)  # 声明集合的来源文件名字段。
        schema.add_field("active", DataType.BOOL)  # 声明集合的激活状态字段。
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)  # 用模型实测的维数声明子块向量字段。

        index_params = self.client.prepare_index_params()  # 将索引参数设为准备索引参数的结果。
        # 使用余弦距离，写入前在客户端归一化。
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")  # 对真实子块向量按余弦相似度建立自动索引。
        for scalar in ("doc_id", "parent_id", "active"):  # 遍历组合值，逐项取得scalar。
            index_params.add_index(scalar, index_type="INVERTED")  # 为文档、父块及激活状态等过滤字段建立倒排索引。

        self.client.create_collection(  # 调用create_collection。
            self.child_name,  # 传入self.child_name。
            schema=schema,  # 传入集合结构参数。
            index_params=index_params,  # 传入索引参数参数。
            consistency_level=self._settings.milvus.consistency_level,  # 传入一致性级别参数。
        )  # 结束当前多行表达式。

    def _create_parent(self) -> None:  # 定义_create_parent的处理逻辑。
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)  # 将集合结构设为创建集合结构的结果。
        schema.add_field(  # 声明集合的所属父块标识字段。
            "parent_id", DataType.VARCHAR, is_primary=True, max_length=MAX_ID_LEN  # 采用应用生成的父块标识作为主键并限制长度。
        )  # 结束当前多行表达式。
        schema.add_field("doc_id", DataType.VARCHAR, max_length=MAX_ID_LEN)  # 声明集合的文档标识字段。
        schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LEN)  # 声明集合的文本字段。
        schema.add_field("breadcrumb", DataType.VARCHAR, max_length=MAX_BREADCRUMB_LEN)  # 声明集合的标题路径字段。
        schema.add_field("page_start", DataType.INT32)  # 声明集合的起始页字段。
        schema.add_field("source_name", DataType.VARCHAR, max_length=MAX_NAME_LEN)  # 声明集合的来源文件名字段。
        schema.add_field("active", DataType.BOOL)  # 声明集合的激活状态字段。
        # 仅用于满足集合结构要求的占位向量。
        schema.add_field("placeholder", DataType.FLOAT_VECTOR, dim=PLACEHOLDER_DIM)  # 用二维向量字段满足父块集合的结构要求。

        index_params = self.client.prepare_index_params()  # 将索引参数设为准备索引参数的结果。
        index_params.add_index("placeholder", index_type="AUTOINDEX", metric_type="COSINE")  # 为“placeholder”字段建立指定类型的索引。
        for scalar in ("doc_id", "active"):  # 遍历组合值，逐项取得scalar。
            index_params.add_index(scalar, index_type="INVERTED")  # 为父块的常用过滤字段建立倒排索引。

        self.client.create_collection(  # 调用create_collection。
            self.parent_name,  # 传入self.parent_name。
            schema=schema,  # 传入集合结构参数。
            index_params=index_params,  # 传入索引参数参数。
            consistency_level=self._settings.milvus.consistency_level,  # 传入一致性级别参数。
        )  # 结束当前多行表达式。

    def _create_meta(self) -> None:  # 定义_create_meta的处理逻辑。
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)  # 将集合结构设为创建集合结构的结果。
        schema.add_field("key", DataType.VARCHAR, is_primary=True, max_length=MAX_ID_LEN)  # 声明集合的键字段。
        schema.add_field("value", DataType.VARCHAR, max_length=MAX_TEXT_LEN)  # 声明集合的值字段。
        # 元数据集合也需要向量字段和索引，使用占位值满足要求。
        schema.add_field("placeholder", DataType.FLOAT_VECTOR, dim=PLACEHOLDER_DIM)  # 声明集合的占位向量字段。

        index_params = self.client.prepare_index_params()  # 将索引参数设为准备索引参数的结果。
        index_params.add_index("placeholder", index_type="AUTOINDEX", metric_type="COSINE")  # 为“placeholder”字段建立指定类型的索引。

        self.client.create_collection(  # 调用create_collection。
            self.meta_name,  # 传入self.meta_name。
            schema=schema,  # 传入集合结构参数。
            index_params=index_params,  # 传入索引参数参数。
            consistency_level=self._settings.milvus.consistency_level,  # 传入一致性级别参数。
        )  # 结束当前多行表达式。

    # 读写建库元数据。

    def read_meta(self) -> dict:  # 返回已保存的建库元数据。
        """返回已保存的建库元数据；无记录时返回空字典。"""
        if not self.client.has_collection(self.meta_name):  # 集合尚未创建时没有可读取的建库元数据。
            return {}  # 返回空字典。
        rows = self.client.query(
            self.meta_name, filter='key == "schema"', output_fields=["value"], limit=1
        )
        if not rows:
            return {}
        try:
            schema = json.loads(rows[0]["value"])
        except json.JSONDecodeError as exc:
            raise InvalidStoreMetadata("stored schema contains invalid JSON") from exc
        if not (
            isinstance(schema, dict)
            and isinstance(schema.get("embedding_model"), str)
            and schema["embedding_model"]
            and type(schema.get("dimension")) is int
            and schema["dimension"] > 0
        ):
            raise InvalidStoreMetadata("stored schema requires an embedding model and a positive dimension")
        return schema

    def write_meta(self, payload: dict) -> None:  # 写入建库元数据，覆盖旧值。
        """写入建库元数据，覆盖旧值。"""
        if not self.client.has_collection(self.meta_name):  # 检查元数据集合名是否尚未创建。
            self._create_meta()  # 调用_create_meta。
        row = {  # 将记录设为字段映射。
            "key": "schema",  # 填入键字段。
            "value": json.dumps(payload, ensure_ascii=False),  # 将元数据编码成文本以写入键值集合。
            "placeholder": list(PLACEHOLDER_VECTOR),  # 填入占位向量字段。
        }  # 结束本组字段。
        self.client.upsert(self.meta_name, [row])  # 按固定主键写回，使新元数据覆盖旧值。

    # 写入与激活文档。

    def insert_units(  # 先以未激活状态写入父块和子块。
        self,  # 通过实例读取本方法所需状态。
        children: Sequence[ChildUnit],  # 接收子块列表参数。
        parents: Sequence[ParentUnit],  # 接收父块列表参数。
    ) -> None:  # 结束参数声明，进入函数或语句主体。
        """先以未激活状态写入父块和子块；整份文档写完后才允许检索，以免暴露半成品。"""
        if children:  # 当子块列表有效时进入此分支。
            self.client.insert(self.child_name, [c.as_row() for c in children])  # 将子块转换为行并批量插入子块集合。
        if parents:  # 当父块列表有效时进入此分支。
            self.client.insert(self.parent_name, [p.as_row() for p in parents])  # 父块独立批量写入父块集合。

    def activate_document(self, doc_id: str) -> None:  # 使文档的所有片段可以被检索。
        """使文档的所有片段可以被检索。"""
        # 批量更新需要整行数据，因此先取回记录，修改状态后再写回。
        for name, key in ((self.child_name, "id"), (self.parent_name, "parent_id")):  # 依次处理子块与父块两个集合。
            rows = self.client.query(  # 将记录列表设为查询的结果。
                name, filter=f'doc_id == "{doc_id}"', output_fields=["*"]  # 按文档标识取回全部字段，准备更新可见状态。
            )  # 结束当前多行表达式。
            if not rows:  # 当记录列表为空或未启用时进入此分支。
                continue  # 跳过本项，继续处理下一项。
            for row in rows:  # 遍历记录列表，逐项取得记录。
                # 请求全部字段，以保留子块中的真实向量和父块自身的字段。
                row["active"] = True  # 把属于该文档的记录改为可检索。
            self.client.upsert(name, rows)  # 按原主键写回整行，使激活状态生效。

    def deactivate_document(self, doc_id: str) -> None:  # 隐藏文档片段，保留记录以备后续处理。
        """隐藏文档片段，保留记录以备后续处理。"""
        for name in (self.child_name, self.parent_name):  # 遍历子块集合名、父块集合名，逐项取得名称。
            rows = self.client.query(  # 将记录列表设为查询的结果。
                name, filter=f'doc_id == "{doc_id}"', output_fields=["*"]  # 取回本次要隐藏的文档记录及原始字段。
            )  # 结束当前多行表达式。
            if not rows:  # 当记录列表为空或未启用时进入此分支。
                continue  # 跳过本项，继续处理下一项。
            for row in rows:  # 遍历记录列表，逐项取得记录。
                row["active"] = False  # 将记录标记为未激活。
            self.client.upsert(name, rows)  # 按主键写回状态变更后的完整记录。

    def delete_document(self, doc_id: str) -> None:  # 删除属于同一文档的全部片段。
        """删除属于同一文档的全部片段。"""
        for name in (self.child_name, self.parent_name):  # 遍历子块集合名、父块集合名，逐项取得名称。
            self.client.delete(name, filter=f'doc_id == "{doc_id}"')  # 按文档标识删除当前集合中的片段。

    # 查询与检索。

    def search_children(  # 仅在已激活的子块中执行近似最近邻检索。
        self,  # 通过实例读取本方法所需状态。
        embedding: Sequence[float],  # 接收向量参数。
        limit: int,  # 接收数量上限参数。
        doc_id: str | None = None,  # 接收文档标识参数。
    ) -> list[dict]:  # 结束参数声明，进入函数或语句主体。
        """仅在已激活的子块中执行近似最近邻检索。"""
        expression = "active == true"  # 默认只检索已激活的子块。
        if doc_id:  # 当文档标识有效时进入此分支。
            expression += f' and doc_id == "{doc_id}"'  # 指定文档时将其标识加入过滤条件。
        hits = self.client.search(  # 将命中项设为搜索的结果。
            self.child_name,  # 传入self.child_name。
            data=[list(embedding)],  # 将单个查询向量封装成一批搜索输入。
            anns_field="embedding",  # 选择真实的子块向量作为近似搜索字段。
            limit=limit,  # 传入数量上限参数。
            filter=expression,  # 传入查询过滤条件参数。
            output_fields=["id", "doc_id", "parent_id", "text", "breadcrumb", "page", "source_name"],  # 一起返回正文和来源，供重排及引用使用。
        )  # 结束当前多行表达式。
        results = []  # 初始化结果列表为空列表。
        for hit in hits[0] if hits else []:  # 遍历按条件选取的值，逐项取得hit。
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}  # 将entity设为按条件选取的值。
            results.append(  # 将字段映射加入结果列表。
                {  # 开始构造字段映射。
                    "id": entity.get("id"),  # 填入标识字段。
                    "doc_id": entity.get("doc_id"),  # 填入文档标识字段。
                    "parent_id": entity.get("parent_id"),  # 填入所属父块标识字段。
                    "text": entity.get("text", ""),  # 填入文本字段。
                    "breadcrumb": entity.get("breadcrumb", ""),  # 填入标题路径字段。
                    "page": entity.get("page", 0),  # 填入页码字段。
                    "source_name": entity.get("source_name", ""),  # 填入来源文件名字段。
                    "score": float(hit.get("distance", 0.0)) if isinstance(hit, dict) else 0.0,  # 从向量库命中结果取出余弦度量分值。
                }  # 结束本组字段。
            )  # 结束当前多行表达式。
        return results  # 返回结果列表。

    def fetch_parents(self, parent_ids: Iterable[str]) -> dict[str, dict]:  # 按主键批量取回父块，并按标识建立查找表。
        """按主键批量取回父块，并按标识建立查找表。"""
        ids = [str(p) for p in dict.fromkeys(parent_ids)]  # 按首次出现的顺序去除重复父块标识并转成字符串。
        if not ids:  # 当标识列表为空或未启用时进入此分支。
            return {}  # 返回空字典。
        quoted = ", ".join(f'"{p}"' for p in ids)  # 将带引号的标识列表设为拼接后的从标识列表中提取的各项。
        rows = self.client.query(  # 将记录列表设为查询的结果。
            self.parent_name,  # 传入self.parent_name。
            filter=f"parent_id in [{quoted}]",  # 用主键列表限制要返回的父块。
            output_fields=["parent_id", "doc_id", "text", "breadcrumb", "page_start", "source_name", "active"],  # 传入返回字段参数。
        )  # 结束当前多行表达式。
        return {row["parent_id"]: row for row in rows}  # 按父块主键整理结果，以便从命中的子块回查。

    def all_active_children(self) -> list[dict]:  # 读取所有有效子块以重建关键词索引，使两路召回共享同一数据源。
        """读取所有有效子块以重建关键词索引，使两路召回共享同一数据源。"""
        rows = self.client.query(  # 将记录列表设为查询的结果。
            self.child_name,  # 传入self.child_name。
            filter="active == true",  # 关键词索引仅重建当前可检索的子块。
            output_fields=["id", "doc_id", "parent_id", "text", "breadcrumb", "page", "source_name"],  # 传入返回字段参数。
            limit=16384,  # 本次查询最多读取一万六千三百八十四行。
        )  # 结束当前多行表达式。
        return rows  # 返回记录列表。

    def collection_stats(self) -> dict[str, int]:  # 定义统计集合记录的处理逻辑。
        stats: dict[str, int] = {}  # 初始化统计信息为空字典。
        for name in (self.child_name, self.parent_name):  # 遍历子块集合名、父块集合名，逐项取得名称。
            if not self.client.has_collection(name):  # 检查名称是否尚未创建。
                stats[name] = 0  # 将统计信息中对应的项设为0。
                continue  # 跳过本项，继续处理下一项。
            rows = self.client.query(name, filter="", output_fields=["count(*)"])
            stats[name] = int(rows[0]["count(*)"])
        return stats  # 返回统计信息。

    def drop_all(self) -> None:  # 删除本项目的集合，供测试清理或重建使用。
        """删除本项目的集合，供测试清理或重建使用。"""
        for name in (self.child_name, self.parent_name, self.meta_name):  # 遍历子块集合名、父块集合名、元数据集合名，逐项取得名称。
            if self.client.has_collection(name):  # 当集合是否存在的结果时进入此分支。
                self.client.drop_collection(name)  # 调用drop_collection。


def normalise(vector: Sequence[float]) -> list[float]:  # 将向量缩放为单位长度，使点积也能用于比较余弦相似度。
    """将向量缩放为单位长度，使点积也能用于比较余弦相似度。"""
    magnitude = sum(x * x for x in vector) ** 0.5  # 对分量平方和开方，得到向量的欧氏长度。
    if magnitude == 0:  # 零向量无法除以其长度。
        return list(vector)  # 返回零向量的副本以避免除零。
    return [x / magnitude for x in vector]  # 每一维除以模长，使非零向量具有单位长度。