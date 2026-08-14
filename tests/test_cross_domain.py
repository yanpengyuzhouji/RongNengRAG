import unittest
from types import SimpleNamespace

from src.generation.prompt_templates import get_prompt
from src.retrieval.cross_domain import apply_domain_override, retrieve_cross_domain


class CrossDomainTests(unittest.TestCase):
    def test_explicit_domain_filter_overrides_automatic_domain(self):
        analyzed = SimpleNamespace(
            domain="变电", filter_expr='domain == "变电"'
        )
        result = apply_domain_override(
            analyzed,
            "配电",
            lambda query: f'domain == "{query.domain}"',
        )

        self.assertEqual("配电", result.domain)
        self.assertEqual('domain == "配电"', result.filter_expr)

    def test_cross_domain_contexts_reach_both_prompt_slots(self):
        result_a = SimpleNamespace(
            results=[SimpleNamespace(text="A requirement")],
            total_candidates=2,
            elapsed_ms=1.5,
        )
        result_b = SimpleNamespace(
            results=[SimpleNamespace(text="B requirement")],
            total_candidates=3,
            elapsed_ms=2.5,
        )
        retriever = SimpleNamespace(
            search_cross_domain=lambda query, top_k: {
                "变电": result_a,
                "配电": result_b,
            },
            format_context_for_llm=lambda rows, max_chunks: rows[0].text,
        )

        response = retrieve_cross_domain(retriever, "compare", 5)
        prompt = get_prompt(
            response.query_type,
            response.context,
            response.query,
            response.context_domain1,
            response.context_domain2,
        )

        self.assertEqual(["变电", "配电"], response.domain_names)
        self.assertEqual(5, response.total_candidates)
        self.assertIn("【专业域：变电】\nA requirement", prompt)
        self.assertIn("【专业域：配电】\nB requirement", prompt)

    def test_cross_domain_prompt_never_raises_for_missing_second_context(self):
        prompt = get_prompt(
            "cross_domain_comparison", "first context", "compare"
        )
        self.assertIn("first context", prompt)
        self.assertIn("（域2无资料）", prompt)


if __name__ == "__main__":
    unittest.main()
