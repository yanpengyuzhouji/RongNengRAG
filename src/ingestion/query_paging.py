"""Complete Milvus scalar-query iteration without a silent fixed limit."""

from __future__ import annotations


def iter_query_batches(
    client, *, collection_name: str, filter_expr: str,
    output_fields: list[str], batch_size: int = 1000,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not hasattr(client, "query_iterator"):
        raise RuntimeError(
            "Milvus client does not support query_iterator; complete statistics "
            "cannot be guaranteed"
        )
    iterator = client.query_iterator(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=output_fields,
        batch_size=batch_size,
    )
    seen_primary_keys: set[str] = set()
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            # Milvus Lite can repeat VARCHAR primary-key rows on a later
            # iterator page.  Never silently return an inflated aggregate.
            if isinstance(batch, list):
                for row in batch:
                    if not isinstance(row, dict) or "chunk_id" not in row:
                        continue
                    chunk_id = str(row["chunk_id"])
                    if chunk_id in seen_primary_keys:
                        raise RuntimeError(
                            "Milvus query_iterator returned a duplicate chunk_id; "
                            "refusing partial or inflated results"
                        )
                    seen_primary_keys.add(chunk_id)
            yield batch
    finally:
        iterator.close()
