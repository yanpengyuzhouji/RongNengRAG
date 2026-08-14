"""
Milvus Lite 向量数据库 — Schema 创建 + 批量入库
支持稠密+稀疏混合搜索、元数据标量索引
"""

import os
import time
import uuid
import json
from pymilvus import (
    MilvusClient, DataType, Function, AnnSearchRequest, RRFRanker, WeightedRanker
)
from typing import List, Dict, Optional
from pathlib import Path

from ingestion.query_paging import iter_query_batches

# ===== 修复 milvus-lite 3.0 Windows os.rename bug =====
# 问题: milvus-lite manifest.save() 使用 os.rename(tmp, target)
#       Windows 上 os.rename 不能覆盖已存在文件，报 WinError 183
# 修复: 全局替换 os.rename 为 os.replace (原子替换，跨平台安全)
_os_rename = os.rename

def _safe_rename(src: str, dst: str):
    """os.replace 在 Windows/Linux 上均可原子替换目标文件"""
    try:
        _os_rename(src, dst)
    except FileExistsError:
        os.replace(src, dst)

os.rename = _safe_rename
# =====


def _try_clean_stale_lock(db_path: str):
    """清理因进程异常退出残留的 Milvus LOCK 文件"""
    lock_file = os.path.join(db_path, "LOCK")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            print(f"[修复] 已清理残留锁文件: {lock_file}")
        except PermissionError:
            pass  # 锁被其他存活进程持有，不强行删除


class MilvusStore:
    """Milvus Lite 向量存储封装"""

    COLLECTION_NAME = "power_design_chunks_active"
    LEGACY_COLLECTION_NAME = "power_design_chunks"
    GENERATION_PREFIX = "power_design_chunks_gen_"
    DENSE_DIM = 1024  # 默认稠密向量维度 (BGE-M3), 实际值从 config embedding.dimensions 读取
    _SNAPSHOT_FIELDS = [
        "chunk_id", "text", "embedding_text", "dense_vector", "sparse_vector",
        "domain", "category", "subcategory", "publish_level", "voltage_level",
        "discipline", "equipment_type", "project_stage", "year", "region",
        "file_type", "file_path", "doc_number", "drawing_code", "is_drawing",
        "page_num", "chunk_index", "chunk_strategy",
    ]
    # Milvus Lite's query API is reliable for a bounded result set, whereas
    # query_iterator may repeat rows when the primary key is VARCHAR.  Keep
    # every single direct request below the server-side query limit and split
    # collection snapshots by the SHA-256 prefix when necessary.
    _DIRECT_QUERY_LIMIT = 16_000
    _HASH_PREFIX_CHARS = "0123456789abcdef"

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        self.db_path = self.config["paths"]["milvus_db"]
        # 稠密向量维度可配置 (与 embedding 模型一致); 新库按此创建, 改维度需删库重建
        self.DENSE_DIM = self.config["embedding"].get("dimensions", self.DENSE_DIM)
        self.client = self._connect()
        self._active_collection = None
        self._active_state_path = Path(self.db_path).with_name(
            "milvus_active_collection.json"
        )
        self._ensure_active_alias()
        persisted = self._load_active_collection()
        if persisted:
            self._active_collection = persisted
            # Reattach the stable alias after a process restart.  This is
            # idempotent and repairs Milvus Lite alias visibility drift.
            try:
                self.client.alter_alias(
                    collection_name=persisted,
                    alias=self.COLLECTION_NAME,
                )
            except Exception:
                pass
        else:
            self._active_collection = self._alias_target() or self.LEGACY_COLLECTION_NAME

    def _connect(self):
        """连接 Milvus Lite，失败时自动清理锁重试一次"""
        # pymilvus 默认把 gRPC keepalive 设为 10 秒。Milvus Lite（尤其是
        # Windows 下的嵌入式服务）会把空闲连接上的高频 ping 视为滥用，
        # 返回 GOAWAY/ENHANCE_YOUR_CALM (too_many_pings)，随后连接管理器
        # 会不断尝试恢复连接。降低保活频率并禁止无 RPC 时的 ping；这些
        # 选项会同时用于初次连接和 reconnect。
        grpc_options = {
            "grpc.keepalive_time_ms": 300_000,
            "grpc.keepalive_timeout_ms": 20_000,
            "grpc.keepalive_permit_without_calls": 0,
            "grpc.http2.max_pings_without_data": 2,
        }

        def create_client():
            return MilvusClient(self.db_path, grpc_options=grpc_options)

        try:
            return create_client()
        except Exception as e:
            # 尝试清理锁文件后重试
            _try_clean_stale_lock(self.db_path)
            time.sleep(0.5)
            return create_client()

    def _alias_target(self) -> Optional[str]:
        try:
            return self.client.describe_alias(
                alias=self.COLLECTION_NAME
            ).get("collection_name")
        except Exception:
            return None

    def _load_active_collection(self) -> Optional[str]:
        try:
            state = json.loads(self._active_state_path.read_text(encoding="utf-8"))
            value = str(state.get("collection") or "").strip()
            return value or None
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def _persist_active_collection(self, collection_name: str) -> None:
        """Persist the last validated physical collection for restart recovery."""
        path = getattr(self, "_active_state_path", None)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
        pending.write_text(
            json.dumps({"collection": collection_name}, ensure_ascii=False),
            encoding="utf-8",
        )
        pending.replace(path)

    def _active_target(self) -> str:
        """Return the physical collection selected for this client.

        Milvus Lite may resolve an alias differently for scalar queries and
        vector searches after an alias switch. Keeping the physical target on
        the store makes all reads/writes in this process use one generation.
        """
        target = getattr(self, "_active_collection", None)
        if target:
            return target
        target = self._alias_target()
        if target:
            self._active_collection = target
            return target
        return self.COLLECTION_NAME

    def _ensure_active_alias(self) -> None:
        """Place the legacy collection behind the stable read alias."""
        if self._alias_target():
            return
        if not self.client.has_collection(self.LEGACY_COLLECTION_NAME):
            self.create_collection(collection_name=self.LEGACY_COLLECTION_NAME)
        self.client.create_alias(
            collection_name=self.LEGACY_COLLECTION_NAME,
            alias=self.COLLECTION_NAME,
        )

    def create_collection(self, drop_existing: bool = False,
                          collection_name: Optional[str] = None):
        """创建 Milvus 集合（Schema 定义），索引创建为尽力而为"""
        target = collection_name or self.LEGACY_COLLECTION_NAME
        if self.client.has_collection(target):
            if drop_existing:
                self.client.drop_collection(target)
                print(f"[drop] 已删除旧集合: {target}")
            else:
                print(f"[info] 集合已存在: {target}")
                return

        # Schema 定义
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding_text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.DENSE_DIM)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
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

        # Step 1: 创建集合（不带索引，避免 milvus-lite Windows os.rename bug）
        self.client.create_collection(
            collection_name=target,
            schema=schema,
        )

        # Step 2: 创建索引（仅向量字段创建索引，标量字段 FLAT 搜索即可）
        # pymilvus 3.0 create_index 对 WAL 集合需要特殊处理
        for field_name, index_type, metric_type, params in [
            ("dense_vector", "FLAT", "COSINE", None),
            ("sparse_vector", "SPARSE_INVERTED_INDEX", "IP", None),
        ]:
            try:
                idx = self.client.prepare_index_params()
                kwargs = {"field_name": field_name, "index_type": index_type, "metric_type": metric_type}
                if params:
                    kwargs["params"] = params
                idx.add_index(**kwargs)
                self.client.create_index(
                    collection_name=target,
                    index_params=idx,
                )
            except Exception:
                pass  # 索引创建失败不影响功能

        print(f"[OK] 集合创建完成: {target}")

    def insert(self, chunks: List, dense_vectors: List[List[float]],
               sparse_vectors: List[dict], embedding_texts: List[str],
               batch_size: int = 500,
               collection_name: Optional[str] = None):
        """
        批量插入 chunks 及其向量

        Args:
            chunks: Chunk 对象列表
            dense_vectors: 稠密向量列表
            sparse_vectors: 稀疏向量列表
            embedding_texts: 嵌入优化的文本列表
            batch_size: 每批插入数量
        """
        if collection_name is None:
            self._ensure_collection()  # 自动创建集合（如果不存在）
        target = collection_name or self._active_target()

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
                collection_name=target,
                data=rows,
            )
            inserted += len(rows)

            if total > batch_size:
                print(f"   📥 入库进度: {inserted}/{total} ({inserted * 100 // total}%)")

        # 刷新索引并加载集合以确保可搜索
        self.client.flush(target)
        try:
            self.client.load_collection(target)
        except Exception:
            pass  # milvus-lite 可能自动加载
        print(f"   [OK] 已入库 {inserted} 条记录")

    def _ensure_collection(self):
        """确保集合存在，不存在则创建"""
        self._ensure_active_alias()

    def collection_exists(self) -> bool:
        """检查集合是否存在且有数据"""
        target = self._active_target()
        if not self.client.has_collection(target):
            return False
        stats = self.client.get_collection_stats(target)
        return stats.get("row_count", 0) > 0

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
            List[dict] — 排序后的搜索结果；集合不存在或无数据时返回空列表
        """
        # 集合不存在时返回空结果
        target = self._active_target()
        if not self.client.has_collection(target):
            return []

        # 确保集合已加载（防止 released 状态）
        try:
            self.client.load_collection(target)
        except Exception:
            pass

        if output_fields is None:
            output_fields = [
                "chunk_id", "text", "domain", "category", "file_path",
                "doc_number", "voltage_level", "publish_level",
                "discipline", "equipment_type", "year", "region",
                "drawing_code", "page_num", "chunk_index", "is_drawing"
            ]

        # 稀疏向量为空时退化为稠密单路搜索 (如纯稠密模式或空查询)
        if not sparse_vector:
            results = self.client.search(
                collection_name=target,
                data=[dense_vector],
                anns_field="dense_vector",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=limit,
                filter=filter_expr,
                output_fields=output_fields,
            )
            return results[0] if results else []

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

        # 加权融合搜索 (dense_weight + sparse_weight = 1.0)
        # 使用 WeightedRanker 替代 RRFRanker，让 config.yaml 中的
        # dense_weight/sparse_weight 配置真正生效
        ranker = WeightedRanker(dense_weight, sparse_weight)

        results = self.client.hybrid_search(
            collection_name=target,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            filter=filter_expr,
            limit=limit,
            output_fields=output_fields,
        )

        return results[0] if results else []

    def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        target = self._active_target()
        if not self.client.has_collection(target):
            return {"exists": False, "count": 0}

        stats = self.client.get_collection_stats(target)
        return {"exists": True, "count": stats.get("row_count", 0)}

    def query_rows_for_bm25(self, filter_expr: Optional[str] = None) -> List[dict]:
        """Read current chunk text/metadata for the in-process BM25 branch.

        This intentionally omits vector fields.  The active physical
        collection is used instead of the alias so the BM25 snapshot follows
        the same generation as dense/BGE-M3 Milvus search.
        """
        target = self._active_target()
        if not self.client.has_collection(target):
            return []
        fields = [
            "chunk_id", "text", "embedding_text", "domain", "category",
            "file_path", "doc_number", "voltage_level", "publish_level",
            "discipline", "equipment_type", "year", "region", "is_drawing",
            "page_num", "chunk_index",
        ]
        return self._query_collection_rows(
            target, fields, filter_expr=filter_expr or ""
        )

    def iter_scalar_query(self, filter_expr: str, output_fields: List[str],
                          batch_size: int = 1000):
        """Yield every matching scalar row in bounded batches."""
        target = self._active_target()
        if not self.client.has_collection(target):
            return
        try:
            self.client.load_collection(target)
        except Exception:
            pass
        yield from iter_query_batches(
            self.client,
            collection_name=target,
            filter_expr=filter_expr or "",
            output_fields=output_fields,
            batch_size=batch_size,
        )

    def delete_by_file_hash(self, file_hash: str,
                            collection_name: Optional[str] = None):
        """删除指定文件的所有 chunks（用于增量更新）"""
        target = collection_name or self._active_target()
        if not self.client.has_collection(target):
            return  # 集合不存在，无需删除
        # ponytail: 确保已加载，防 released 状态下 delete 静默失败
        try:
            self.client.load_collection(target)
        except Exception:
            pass
        expr = f'chunk_id like "{file_hash}%"'
        self.client.delete(collection_name=target, filter=expr)

    def purge_file_generations(self, file_hash: str) -> None:
        """Delete one file from the active index and every retained generation.

        A normal reindex deliberately keeps the previous physical collection
        as a rollback source.  That is not appropriate after the user deletes
        a document: retaining its rows would make a future alias rollback (or
        an accidental alias switch) able to surface deleted content again.
        """
        targets = {self._active_target()}
        try:
            targets.update(self.client.list_collections())
        except Exception:
            # Older Milvus Lite clients may not expose list_collections; the
            # active alias still covers normal retrieval in that environment.
            pass

        for target in targets:
            if not target or (
                target != self.COLLECTION_NAME
                and not target.startswith(self.GENERATION_PREFIX)
            ):
                continue
            if not self.client.has_collection(target):
                continue
            self.delete_by_file_hash(file_hash, collection_name=target)
            try:
                self.client.flush(target)
            except Exception:
                pass

    def _direct_query(self, collection_name: str, filter_expr: str,
                      output_fields: List[str], limit: int) -> List[dict]:
        """Issue one bounded non-iterator query with strong consistency."""
        if limit <= 0 or limit > self._DIRECT_QUERY_LIMIT:
            raise RuntimeError(
                f"Unsafe direct Milvus query limit: {limit}; "
                f"maximum is {self._DIRECT_QUERY_LIMIT}"
            )
        try:
            return self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=limit,
                consistency_level="Strong",
            )
        except TypeError:
            # Older pymilvus/test clients do not support consistency_level.
            return self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=limit,
            )

    def _query_hash_prefix_rows(self, collection_name: str,
                                output_fields: List[str], prefix: str) -> List[dict]:
        """Read one SHA-256 key prefix, recursively splitting full buckets.

        A direct Milvus query has no pagination-order dependency.  SHA-256
        chunk IDs make the 16-way prefix partition balanced; if a bucket ever
        reaches the direct-query ceiling, split it again instead of accepting a
        truncated snapshot.
        """
        rows = self._direct_query(
            collection_name,
            f'chunk_id like "{prefix}%"',
            output_fields,
            self._DIRECT_QUERY_LIMIT,
        )
        if len(rows) < self._DIRECT_QUERY_LIMIT:
            return rows
        if len(prefix) >= 64:
            raise RuntimeError(
                f"Milvus prefix bucket is too large to snapshot safely: {prefix}"
            )
        nested: List[dict] = []
        for char in self._HASH_PREFIX_CHARS:
            nested.extend(self._query_hash_prefix_rows(
                collection_name, output_fields, prefix + char
            ))
        return nested

    def _query_file_rows(self, file_hash: str, output_fields: List[str],
                         expected_count: Optional[int] = None,
                         collection_name: Optional[str] = None) -> List[dict]:
        """Read every row for one file without VARCHAR iterator pagination."""
        target = collection_name or self._active_target()
        if not self.client.has_collection(target):
            return []
        # A collection can be released after a staged generation is created.
        # Explicitly load before either query API; otherwise Milvus returns
        # code 101 ("call load() before search/get/query").
        try:
            self.client.load_collection(target)
        except Exception:
            pass
        expr = f'chunk_id like "{file_hash}%"'
        # expected+1 detects a real excess without silently accepting it.
        limit = (expected_count + 1) if expected_count is not None else self._DIRECT_QUERY_LIMIT
        rows = self._direct_query(target, expr, output_fields, limit)
        if len(rows) >= limit:
            raise RuntimeError(
                f"Milvus file query reached safe limit for {file_hash}: {len(rows)}"
            )
        return rows

    def restore_file_snapshot(self, rows: List[dict]) -> None:
        """Restore previously snapshotted rows by primary key."""
        if not rows:
            return
        target = self._active_target()
        self.client.upsert(collection_name=target, data=rows)
        self.client.flush(target)

    def replace_file_metadata(self, file_hash: str, updates: Dict[str, str],
                              expected_count: Optional[int] = None) -> List[dict]:
        """Update scalar metadata on all chunks and verify the write.

        The old complete rows (including vectors) are returned for compensation.
        A failed write is restored before the exception reaches the caller.
        """
        allowed = {"domain", "category", "doc_number"}
        if not updates or not set(updates).issubset(allowed):
            raise ValueError("Unsupported or empty chunk metadata update")
        snapshot = self._query_file_rows(
            file_hash, self._SNAPSHOT_FIELDS, expected_count=expected_count
        )
        if expected_count is not None and len(snapshot) != expected_count:
            raise RuntimeError(
                f"Milvus chunk count mismatch: expected {expected_count}, "
                f"found {len(snapshot)}"
            )
        changed = [{**row, **updates} for row in snapshot]
        target = self._active_target()
        try:
            if changed:
                self.client.upsert(
                    collection_name=target, data=changed
                )
                self.client.flush(target)
            verification = self._query_file_rows(
                file_hash,
                ["chunk_id", *sorted(updates)],
                expected_count=expected_count,
            )
            if len(verification) != len(snapshot) or any(
                row.get(field) != value
                for row in verification
                for field, value in updates.items()
            ):
                raise RuntimeError("Milvus metadata verification failed")
        except BaseException:
            self.restore_file_snapshot(snapshot)
            raise
        return snapshot

    def _query_collection_rows(self, collection_name: str,
                               output_fields: List[str],
                               filter_expr: str = "") -> List[dict]:
        """Read a complete physical collection without iterator pagination."""
        try:
            self.client.load_collection(collection_name)
        except Exception:
            pass
        stats = self.client.get_collection_stats(collection_name)
        expected = int(stats.get("row_count", 0))
        if filter_expr:
            rows = self._direct_query(
                collection_name, filter_expr, output_fields, self._DIRECT_QUERY_LIMIT
            )
            if len(rows) >= self._DIRECT_QUERY_LIMIT:
                raise RuntimeError("Filtered collection snapshot reached safe query limit")
            return rows
        if "chunk_id" not in output_fields:
            raise ValueError("Complete collection snapshots must include chunk_id")
        rows: List[dict] = []
        for prefix in self._HASH_PREFIX_CHARS:
            rows.extend(self._query_hash_prefix_rows(
                collection_name, output_fields, prefix
            ))
        unique_ids = {row.get("chunk_id") for row in rows}
        if len(rows) != expected or len(unique_ids) != expected or None in unique_ids:
            raise RuntimeError(
                f"Collection snapshot mismatch: expected {expected}, found {len(rows)}, "
                f"unique {len(unique_ids)}"
            )
        return rows

    def begin_file_generation(self, file_hash: str):
        """Clone active data and remove one file only from an invisible clone."""
        source = self._alias_target()
        if not source:
            self._ensure_active_alias()
            source = self._alias_target()
        if not source:
            raise RuntimeError("Active Milvus collection alias is unavailable")

        staging = f"{self.GENERATION_PREFIX}{uuid.uuid4().hex}"
        self.create_collection(collection_name=staging)
        try:
            source_rows = self._query_collection_rows(
                source, self._SNAPSHOT_FIELDS
            )
            for offset in range(0, len(source_rows), 500):
                self.client.insert(
                    collection_name=staging,
                    data=source_rows[offset:offset + 500],
                )
            self.client.flush(staging)
            old_file_rows = self._query_file_rows(
                file_hash,
                ["chunk_id"],
                collection_name=staging,
            )
            self.delete_by_file_hash(file_hash, collection_name=staging)
            self.client.flush(staging)
            return IndexGeneration(
                store=self,
                source_collection=source,
                staging_collection=staging,
                source_count=len(source_rows),
                replaced_count=len(old_file_rows),
                file_hash=file_hash,
            )
        except BaseException:
            if self.client.has_collection(staging):
                self.client.drop_collection(staging)
            raise

    def query_by_file_path(self, file_path: str,
                           sort_by_page: bool = True) -> List[dict]:
        """
        查询指定文件的所有 chunks（完整文档内容）

        Args:
            file_path: 文件路径（支持精确匹配或 LIKE 匹配）
            sort_by_page: 是否按页码和 chunk_index 排序

        Returns:
            chunks 列表，按 chunk_index 或 page_num 排序
        """
        target = self._active_target()
        if not self.client.has_collection(target):
            return []

        try:
            self.client.load_collection(target)
        except Exception:
            pass

        # 用 file_path 字段精确匹配
        # Milvus 标量过滤不支持 LIKE，使用 == 精确匹配
        # Windows 路径中的反斜杠会被 Milvus 表达式解析器当作转义序列（如 \R, \d）
        # 需要把 \ 转义为 \\ 才能在表达式中安全使用
        escaped_path = file_path.replace("\\", "\\\\")
        expr = f'file_path == "{escaped_path}"'

        output_fields = [
            "chunk_id", "text", "domain", "category", "file_path",
            "doc_number", "voltage_level", "publish_level",
            "discipline", "equipment_type", "year", "region",
            "drawing_code", "page_num", "chunk_index", "is_drawing"
        ]

        try:
            results = self.client.query(
                collection_name=target,
                filter=expr,
                output_fields=output_fields,
                limit=10000,  # 一个文件可能有大量 chunks
            )
        except Exception:
            return []

        if sort_by_page and results:
            # 按 page_num 排序，同页按 chunk_index 排序
            results.sort(key=lambda x: (
                x.get("page_num", 0) or 0,
                x.get("chunk_index", 0) or 0,
            ))

        return results

    def query_by_file_hash(self, file_hash: str,
                           sort_by_page: bool = True) -> List[dict]:
        """按文件哈希前缀查询所有 chunks。"""
        rows = self._query_file_rows(
            file_hash,
            [
                "chunk_id", "text", "domain", "category", "file_path",
                "doc_number", "voltage_level", "publish_level",
                "discipline", "equipment_type", "year", "region",
                "drawing_code", "page_num", "chunk_index", "is_drawing",
            ],
        )
        if sort_by_page and rows:
            rows.sort(key=lambda row: (
                row.get("page_num", 0) or 0,
                row.get("chunk_index", 0) or 0,
            ))
        return rows


class IndexGeneration:
    """A physical collection that stays invisible until alias activation."""

    def __init__(self, *, store: MilvusStore, source_collection: str,
                 staging_collection: str, source_count: int,
                 replaced_count: int, file_hash: str):
        self.store = store
        self.source_collection = source_collection
        self.staging_collection = staging_collection
        self.source_count = source_count
        self.replaced_count = replaced_count
        self.file_hash = file_hash
        self.activated = False

    def insert(self, *, chunks, dense_vectors, sparse_vectors,
               embedding_texts) -> None:
        self.store.insert(
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            embedding_texts=embedding_texts,
            batch_size=len(chunks),
            collection_name=self.staging_collection,
        )

    def validate(self, expected_file_count: int, expected_chunks=None) -> None:
        expected_total = (
            self.source_count - self.replaced_count + expected_file_count
        )
        file_rows: List[dict] = []
        stat_count = -1
        # Validate precisely what changed (the file prefix) and the physical
        # staging row count.  Do not enumerate every row through query_iterator:
        # Milvus Lite repeats VARCHAR primary keys on later iterator pages.
        for attempt in range(4):
            file_rows = self.store._query_file_rows(
                self.file_hash,
                ["chunk_id"],
                expected_count=expected_file_count,
                collection_name=self.staging_collection,
            )
            stats = self.store.client.get_collection_stats(self.staging_collection)
            stat_count = int(stats.get("row_count", -1))
            if len(file_rows) == expected_file_count and stat_count == expected_total:
                break
            time.sleep(0.25 * (attempt + 1))
        expected_other = self.source_count - self.replaced_count
        actual_other = max(-1, stat_count - len(file_rows))
        if len(file_rows) != expected_file_count or stat_count != expected_total:
            raise RuntimeError(
                "Staged index validation failed: "
                f"file={len(file_rows)}/{expected_file_count}, "
                f"other={actual_other}/{expected_other}, "
                f"total={stat_count}/{expected_total}"
            )

        if expected_chunks is not None:
            expected_rows = {}
            for chunk in expected_chunks:
                chunk_id = str(getattr(chunk, "chunk_id", "") or "")
                if not chunk_id or chunk_id in expected_rows:
                    raise RuntimeError(
                        "Staged index validation failed: expected chunk IDs are not unique"
                    )
                expected_rows[chunk_id] = {
                    "text": str(getattr(chunk, "text", "") or ""),
                    "page_num": int(getattr(chunk, "page_num", 0) or 0),
                    "chunk_index": int(getattr(chunk, "chunk_index", 0) or 0),
                }

            actual_rows = self.store._query_file_rows(
                self.file_hash,
                ["chunk_id", "text", "page_num", "chunk_index"],
                expected_count=expected_file_count,
                collection_name=self.staging_collection,
            )
            actual_ids = [str(row.get("chunk_id") or "") for row in actual_rows]
            actual_by_id = {chunk_id: row for chunk_id, row in zip(actual_ids, actual_rows)}
            mismatches = []
            if len(actual_rows) != len(actual_by_id) or set(actual_by_id) != set(expected_rows):
                mismatches.append("chunk_id")
            else:
                for chunk_id, expected in expected_rows.items():
                    actual = actual_by_id[chunk_id]
                    actual_values = {
                        "text": str(actual.get("text") or ""),
                        "page_num": int(actual.get("page_num") or 0),
                        "chunk_index": int(actual.get("chunk_index") or 0),
                    }
                    if actual_values != expected:
                        mismatches.append(chunk_id)
            if mismatches:
                sample = ", ".join(mismatches[:5])
                raise RuntimeError(
                    "Staged index text verification failed; refusing to publish "
                    f"stale or mismatched chunks: {sample}"
                )

    def activate(self) -> None:
        self.store.client.load_collection(self.staging_collection)
        self.store.client.alter_alias(
            collection_name=self.staging_collection,
            alias=self.store.COLLECTION_NAME,
        )
        self.store._active_collection = self.staging_collection
        persist = getattr(self.store, "_persist_active_collection", None)
        if callable(persist):
            persist(self.staging_collection)
        self.activated = True

    def rollback(self) -> None:
        if self.activated:
            self.store.client.alter_alias(
                collection_name=self.source_collection,
                alias=self.store.COLLECTION_NAME,
            )
            self.store._active_collection = self.source_collection
            persist = getattr(self.store, "_persist_active_collection", None)
            if callable(persist):
                persist(self.source_collection)
            self.activated = False
        if self.store.client.has_collection(self.staging_collection):
            self.store.client.drop_collection(self.staging_collection)

    def finalize(self) -> None:
        # Keep the immediately previous generation as a ready rollback source.
        self.activated = False


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
