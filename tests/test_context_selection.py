import unittest
from types import SimpleNamespace

from src.retrieval.context_selection import select_context_chunks


def row(chunk_id, file_path, text=None):
    return SimpleNamespace(
        chunk_id=chunk_id, file_path=file_path,
        page_num=1, text=text or chunk_id,
    )


class ContextSelectionTests(unittest.TestCase):
    def test_multiple_relevant_chunks_from_one_file_are_kept_up_to_cap(self):
        results = [
            row("a1", "a.pdf"), row("a2", "a.pdf"),
            row("a3", "a.pdf"), row("a4", "a.pdf"),
            row("b1", "b.pdf"), row("b2", "b.pdf"),
        ]
        selected = select_context_chunks(results, max_chunks=5, max_chunks_per_file=3)
        self.assertEqual(["a1", "a2", "a3", "b1", "b2"], [r.chunk_id for r in selected])

    def test_duplicate_chunk_ids_are_removed_and_later_results_fill_budget(self):
        results = [
            row("same", "a.pdf", "first"), row("same", "a.pdf", "duplicate"),
            row("b1", "b.pdf"), row("c1", "c.pdf"),
        ]
        selected = select_context_chunks(results, max_chunks=3, max_chunks_per_file=2)
        self.assertEqual(["same", "b1", "c1"], [r.chunk_id for r in selected])

    def test_empty_or_zero_limits_return_no_context(self):
        self.assertEqual([], select_context_chunks([row("a", "a")], 0, 3))
        self.assertEqual([], select_context_chunks([row("a", "a")], 3, 0))


if __name__ == "__main__":
    unittest.main()
