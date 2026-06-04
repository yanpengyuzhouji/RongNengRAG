"""
Milvus Lite 向量数据库 — Schema 创建 + 批量入库
支持稠密+稀疏混合搜索、元数据标量索引
"""

import yaml
from pymilvus import (
    MilvusClient, DataType, Function, AnnSearchRequest, RRFRanker
)
from typing import List, Dict, Optional


class MilvusStore:
    """Milvus Lite 向量存储封装"""

    COLLECTION_NAME = "power_design_chunks"
    DENSE_DIM = 1024  # BGE-M3 稠密向量维度

    def __init__(self, config_path: str = "D:/rag-system/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.db_path = self.config["paths"]["milvus_db"]
        self.client = MilvusClient(self.db_path)

    def create_collection(self, drop_existing: bool = False):
        """创建 Milvus 集合（Schema 定义）"""
        if drop_existing and self.client.has_collection(self.COLLECTION_NAME):
            self.client.drop_collection(self.COLLECTION_NAME)
            print(f"🗑️ 已删除旧集合: {self.COLLECTION_NAME}")

        if self.client.has_collection(self.COLLECTION_NAME):
            print(f"📦 集合已存在: {self.COLLECTION_NAME}")
            return

        # Schema 定义
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        # 主键
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            max_length=256,
            is_primary=True,
        )

        # 文本内容
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="embedding_text",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )

        # 向量字段
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.DENSE_DIM,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # 元数据标量字段（支持过滤索引）
        schema.add_field(field_name="domain", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="subcategory", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="publish_level", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="voltage_level", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="discipline", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="equipment_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="project_stage", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="year", datatype=DataType.INT16)
        schema.add_field(field_name="region", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="file_type", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="doc_number", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="drawing_code", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="is_drawing", datatype=DataType.BOOL)
        schema.add_field(field_name="page_num", datatype=DataType.INT16)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT16)
        schema.add_field(field_name="chunk_strategy", datatype=DataType.VARCHAR, max_length=32)

        # 创建集合
        index_params = self.client.prepare_index_params()

        # 稠密向量索引 (COSINE 相似度)
        index_params.add_index(
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 1024},
        )

        # 稀疏向量索引
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 标量字段索引（加速过滤）
        index_params.add_index(field_name="domain", index_type="TRIE")
        index_params.add_index(field_name="category", index_type="TRIE")
        index_params.add_index(field_name="voltage_level", index_type="TRIE")
        index_params.add_index(field_name="publish_level", index_type="TRIE")
        index_params.add_index(field_name="year", index_type="STL_SORT")

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )

        print(f"✅ 集合创建完成: {self.COLLECTION_NAME}")

    def insert(self, chunks: List, dense_vectors: List[List[float]],
               sparse_vectors: List[dict], embedding_texts: List[str],
               batch_size: int = 500):
        """
        批量插入 chunks 及其向量

        Args:
            chunks: Chunk 对象列表
            dense_vectors: 稠密向量列表
            sparse_vectors: 稀疏向量列表
            embedding_texts: 嵌入优化的文本列表
            batch_size: 每批插入数量
        """
        total = len(chunks)
        inserted = 0

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_dense = dense_vectors[i:i + batch_size]
            batch_sparse = sparse_vectors[i:i + batch_size]
            batch_emb_texts = embedding_texts[i:i + batch_size]

            rows = []
            for j, chunk in enumerate(batch_chunks):
                row = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "embedding_text": batch_emb_texts[j],
                    "dense_vector": batch_dense[j],
                    "sparse_vector": batch_sparse[j] if batch_sparse[j] else {},
                    "domain": chunk.domain or "",
                    "category": chunk.category or "",
                    "subcategory": chunk.subcategory or "",
                    "publish_level": chunk.publish_level or "",
                    "voltage_level": chunk.voltage_level or "",
                    "discipline": chunk.discipline or "",
                    "equipment_type": chunk.equipment_type or "",
                    "project_stage": "",
                    "year": chunk.year or 0,
                    "region": chunk.region or "全国",
                    "file_type": chunk.file_type or "",
                    "file_path": chunk.file_path or "",
                    "doc_number": chunk.doc_number or "",
                    "drawing_code": chunk.drawing_code or "",
                    "is_drawing": chunk.is_drawing,
                    "page_num": chunk.page_num or 0,
                    "chunk_index": chunk.chunk_index,
                    "chunk_strategy": chunk.chunk_strategy,
                }
                rows.append(row)

            self.client.insert(
                collection_name=self.COLLECTION_NAME,
                data=rows,
            )
            inserted += len(rows)

            if total > batch_size:
                print(f"   📥 入库进度: {inserted}/{total} ({inserted * 100 // total}%)")

        # 刷新索引
        self.client.flush(self.COLLECTION_NAME)
        print(f"   📊 已入库 {inserted} 条记录，索引刷新完成")

    def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: dict,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[dict]:
        """
        混合搜索 — 稠密 + 稀疏 RRF 融合

        Args:
            dense_vector: 稠密查询向量
            sparse_vector: 稀疏查询向量
            filter_expr: 元数据过滤表达式
            output_fields: 返回字段列表
            limit: 返回结果数
            rrf_k: RRF 融合参数
            dense_weight: 稠密搜索权重
            sparse_weight: 稀疏搜索权重

        Returns:
            List[dict] — 排序后的搜索结果
        """
        if output_fields is None:
            output_fields = [
                "chunk_id", "text", "domain", "category", "file_path",
                "doc_number", "voltage_level", "publish_level",
                "discipline", "equipment_type", "year", "region",
                "drawing_code", "page_num", "chunk_index", "is_drawing"
            ]

        # 稠密搜索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=limit * 2,
        )

        # 稀疏搜索请求
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=limit * 2,
        )

        # RRF 混合搜索
        results = self.client.hybrid_search(
            collection_name=self.COLLECTION_NAME,
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=rrf_k),
            filter=filter_expr,
            limit=limit,
            output_fields=output_fields,
        )

        return results[0] if results else []

    def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        if not self.client.has_collection(self.COLLECTION_NAME):
            return {"exists": False, "count": 0}

        stats = self.client.get_collection_stats(self.COLLECTION_NAME)
        return {"exists": True, "count": stats.get("row_count", 0)}

    def delete_by_file_hash(self, file_hash: str):
        """删除指定文件的所有 chunks（用于增量更新）"""
        expr = f'chunk_id like "{file_hash}%"'
        self.client.delete(collection_name=self.COLLECTION_NAME, filter=expr)


def build_filter_expression(
    domain: Optional[str] = None,
    category: Optional[str] = None,
    voltage_level: Optional[str] = None,
    publish_level: Optional[str] = None,
    discipline: Optional[str] = None,
    equipment_type: Optional[str] = None,
    year: Optional[int] = None,
    region: Optional[str] = None,
    exclude_drawings: bool = False,
    doc_number: Optional[str] = None,
) -> Optional[str]:
    """
    构建 Milvus 标量过滤表达式
    多个条件用 AND 组合
    """
    conditions = []

    if domain:
        conditions.append(f'domain == "{domain}"')
    if category:
        conditions.append(f'category == "{category}"')
    if voltage_level:
        conditions.append(f'voltage_level == "{voltage_level}"')
    if publish_level:
        conditions.append(f'publish_level == "{publish_level}"')
    if discipline:
        conditions.append(f'discipline == "{discipline}"')
    if equipment_type:
        conditions.append(f'equipment_type == "{equipment_type}"')
    if year:
        conditions.append(f"year == {year}")
    if region:
        conditions.append(f'region == "{region}"')
    if exclude_drawings:
        conditions.append("is_drawing == false")
    if doc_number:
        conditions.append(f'doc_number like "%{doc_number}%"')

    if not conditions:
        return None

    return " and ".join(conditions)
