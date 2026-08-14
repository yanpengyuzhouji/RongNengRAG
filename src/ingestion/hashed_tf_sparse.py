"""Hashed sublinear-TF sparse vectors for exact lexical recall.

This is intentionally *not* BM25: a stateless encoder has no corpus document
frequency or average document length, so it cannot compute BM25 IDF/length
normalisation. Tokens are crc32-hashed, weighted with ``log(1 + tf)``, and L2
normalised for inner-product fusion with dense vectors.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Dict, List


_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./\-][a-z0-9]+)*")
_CJK_CHAR_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    lowered = text.lower()
    tokens = list(_ASCII_TOKEN_RE.findall(lowered))
    characters = _CJK_CHAR_RE.findall(lowered)
    if len(characters) >= 2:
        tokens.extend(a + b for a, b in zip(characters, characters[1:]))
    elif characters:
        tokens.append(characters[0])
    return tokens


def token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) or 1


def compute_hashed_tf_sparse(text: str) -> Dict[int, float]:
    term_frequencies: Dict[str, int] = {}
    for token in tokenize(text):
        term_frequencies[token] = term_frequencies.get(token, 0) + 1
    if not term_frequencies:
        return {}

    # Sum on the (rare) crc32 collision instead of silently overwriting it.
    weights: Dict[int, float] = {}
    for token, frequency in term_frequencies.items():
        key = token_id(token)
        weights[key] = weights.get(key, 0.0) + math.log1p(frequency)
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    if not norm:
        return {}
    return {key: weight / norm for key, weight in weights.items()}
