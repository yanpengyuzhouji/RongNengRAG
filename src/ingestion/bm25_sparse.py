"""Deprecated compatibility wrapper.

This stateless helper is retained only for old imports/tests. It is not used
by the active pipeline: real BM25 is implemented in
``retrieval.bm25_index`` with corpus IDF and document-length statistics.
"""

from .hashed_tf_sparse import compute_hashed_tf_sparse, token_id, tokenize


def compute_bm25_sparse(text: str):
    return compute_hashed_tf_sparse(text)
