import math
import unittest

from src.ingestion.bm25_sparse import compute_bm25_sparse
from src.ingestion.hashed_tf_sparse import (
    compute_hashed_tf_sparse,
    token_id,
    tokenize,
)


def dot(left, right):
    return sum(value * right.get(key, 0.0) for key, value in left.items())


class HashedTfSparseTests(unittest.TestCase):
    def test_exact_standard_number_has_explainable_lexical_match(self):
        query = compute_hashed_tf_sparse("GB/T-11022")
        exact = compute_hashed_tf_sparse("依据 GB/T-11022 执行")
        unrelated = compute_hashed_tf_sparse("变压器接地要求")
        self.assertGreater(dot(query, exact), 0)
        self.assertEqual(0, dot(query, unrelated))

    def test_weights_are_l2_normalized_and_term_frequency_is_sublinear(self):
        vector = compute_hashed_tf_sparse("alpha alpha alpha beta")
        self.assertAlmostEqual(1.0, math.sqrt(sum(v * v for v in vector.values())))
        ratio = vector[token_id("alpha")] / vector[token_id("beta")]
        self.assertAlmostEqual(2.0, ratio)
        self.assertLess(ratio, 3.0)

    def test_chinese_bigrams_and_legacy_wrapper_are_stable(self):
        self.assertEqual(["变压", "压器"], tokenize("变压器"))
        self.assertEqual(
            compute_hashed_tf_sparse("10kV 变压器"),
            compute_bm25_sparse("10kV 变压器"),
        )


if __name__ == "__main__":
    unittest.main()
