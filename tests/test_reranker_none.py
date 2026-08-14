import unittest
from types import SimpleNamespace

from src.retrieval.reranker import Reranker


def reranker(**overrides):
    instance = Reranker.__new__(Reranker)
    values = {
        "top_k": 10,
        "metadata_boosts": {},
        "max_metadata_boost": 1.20,
        "min_score_threshold": 0.0,
        "softmax_temperature": 1.0,
        "rrf_k": 60,
        "none_rrf_weight": 0.15,
    }
    values.update(overrides)
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


class NoModelRerankerTests(unittest.TestCase):
    def test_retrieval_relevance_remains_the_primary_signal(self):
        candidates = [
            {"distance": 0.85, "entity": {"category": "其他"}},
            {"distance": 0.20, "entity": {"category": "标准规范"}},
        ]
        ranked = reranker().rerank_without_model(candidates, query="question")
        self.assertIs(candidates[0], ranked[0])

    def test_configured_metadata_boost_changes_a_tie(self):
        candidates = [
            {"distance": 0.5, "entity": {"doc_number": ""}},
            {"distance": 0.5, "entity": {"doc_number": "DOC-1"}},
        ]
        ranked = reranker(
            metadata_boosts={"exact_doc_number_match": 1.20},
            none_rrf_weight=0.0,
        ).rerank_without_model(candidates, query="DOC-1")
        self.assertIs(candidates[1], ranked[0])

    def test_threshold_temperature_and_rrf_k_affect_scores(self):
        candidates_a = [{"distance": 0.4}, {"distance": 0.4}]
        small_k = reranker(rrf_k=1, none_rrf_weight=0.5)
        score_small_k = small_k.rerank_without_model(candidates_a)[1]["_rerank_score"]

        candidates_b = [{"distance": 0.4}, {"distance": 0.4}]
        large_k = reranker(rrf_k=100, none_rrf_weight=0.5)
        score_large_k = large_k.rerank_without_model(candidates_b)[1]["_rerank_score"]
        self.assertNotEqual(score_small_k, score_large_k)

        cold = reranker(softmax_temperature=0.5, none_rrf_weight=0.0)
        cold_score = cold.rerank_without_model([{"distance": 0.8}])[0]["_rerank_score"]
        self.assertGreater(cold_score, 0.8)

        filtered = reranker(
            min_score_threshold=0.9, none_rrf_weight=0.0
        ).rerank_without_model([{"distance": 0.5}])
        self.assertEqual([], filtered)


if __name__ == "__main__":
    unittest.main()
