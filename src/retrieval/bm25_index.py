"""Small in-process BM25 index for lexical hybrid retrieval.

Milvus' sparse vector index is used for BGE-M3 lexical weights.  Those weights
are useful learned lexical signals, but they are not BM25 because they do not
contain corpus IDF or document-length statistics.  This module provides the
missing corpus-aware BM25 branch without adding a second service or a native
dependency.  The index is rebuilt when the active Milvus generation changes.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, List, Mapping, Optional


_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./\-][a-z0-9]+)*")
_CJK_CHAR_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> List[str]:
    """Tokenize technical English/Chinese text deterministically.

    Chinese character bigrams preserve standard identifiers such as ``断路器``
    and ``运行中`` while ASCII runs preserve GB/DL numbers and model names.
    """
    if not text:
        return []
    lowered = str(text).lower()
    tokens = list(_ASCII_TOKEN_RE.findall(lowered))
    chars = _CJK_CHAR_RE.findall(lowered)
    if len(chars) >= 2:
        tokens.extend(a + b for a, b in zip(chars, chars[1:]))
    elif chars:
        tokens.append(chars[0])
    return tokens


class BM25Index:
    """In-memory Okapi BM25 index over current Milvus chunks."""

    def __init__(self, rows: Iterable[Mapping], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.rows = []
        self.doc_tokens: List[Counter] = []
        self.postings = defaultdict(dict)

        for row in rows or []:
            row = dict(row)
            chunk_id = str(row.get("chunk_id") or "")
            if not chunk_id:
                continue
            # Metadata is indexed as well so exact standard numbers and names
            # can be recalled even when the body chunk is short.
            text = " ".join(
                str(row.get(field) or "")
                for field in ("embedding_text", "text", "file_path", "doc_number")
            )
            counts = Counter(tokenize(text))
            idx = len(self.rows)
            self.rows.append(row)
            self.doc_tokens.append(counts)
            for term, frequency in counts.items():
                self.postings[term][idx] = frequency

        self.doc_count = len(self.rows)
        self.doc_lengths = [sum(tokens.values()) for tokens in self.doc_tokens]
        self.avgdl = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        # BM25+ style IDF with a positive floor.  This prevents a term present
        # in half the corpus from producing a negative lexical contribution.
        self.idf = {
            term: math.log(1.0 + (self.doc_count - len(posting) + 0.5) /
                           (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }
        self._ids = {str(row.get("chunk_id")): idx for idx, row in enumerate(self.rows)}

    @property
    def empty(self) -> bool:
        return self.doc_count == 0

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        allowed_ids: Optional[set[str]] = None,
    ) -> List[dict]:
        """Return Milvus-compatible candidates sorted by BM25 score."""
        if self.empty or limit <= 0:
            return []
        query_terms = set(tokenize(query))
        scores = defaultdict(float)
        for term in query_terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf.get(term, 0.0)
            for idx, frequency in posting.items():
                row = self.rows[idx]
                if allowed_ids is not None and str(row.get("chunk_id")) not in allowed_ids:
                    continue
                length = self.doc_lengths[idx]
                norm = 1.0 - self.b + self.b * (length / self.avgdl if self.avgdl else 1.0)
                scores[idx] += idf * (frequency * (self.k1 + 1.0)) / (
                    frequency + self.k1 * norm
                )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            {
                "id": self.rows[idx].get("chunk_id"),
                "distance": float(score),
                "entity": self.rows[idx],
                "_bm25_score": float(score),
            }
            for idx, score in ranked
            if score > 0
        ]

