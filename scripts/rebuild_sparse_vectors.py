"""Migrate existing Milvus rows from hashed-TF to BGE-M3 sparse weights.

Dense vectors and all scalar metadata are read back and written unchanged;
only ``sparse_vector`` is replaced.  Stop the API process before running this
script because Milvus Lite permits one writer per database file.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.embedder import Embedder
from ingestion.milvus_store import MilvusStore


def main() -> None:
    store = MilvusStore()
    target = store._active_target()
    fields = list(store._SNAPSHOT_FIELDS)
    rows = store._query_collection_rows(target, fields)
    print(f"[sparse-migrate] collection={target} rows={len(rows)}", flush=True)
    if not rows:
        print("[sparse-migrate] no rows")
        return

    embedder = Embedder()
    embedder._ensure_loaded()
    batch_size = int(embedder.config["embedding"].get("sparse_batch_size", 4))
    started = time.time()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        texts = [row.get("embedding_text") or row.get("text") or "" for row in batch]
        sparse = embedder._bge_m3_sparse_encode(texts)
        if len(sparse) != len(batch) or any(not item for item in sparse):
            raise RuntimeError(f"BGE-M3 sparse output mismatch at offset {offset}")
        for row, vector in zip(batch, sparse):
            row["sparse_vector"] = vector
        store.client.upsert(collection_name=target, data=batch)
        print(f"[sparse-migrate] {min(offset + len(batch), len(rows))}/{len(rows)}", flush=True)

    store.client.flush(target)
    verify = store._query_collection_rows(target, ["chunk_id", "sparse_vector"])
    if len(verify) != len(rows) or any(not row.get("sparse_vector") for row in verify):
        raise RuntimeError("sparse migration verification failed")
    print(f"[sparse-migrate] complete elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

