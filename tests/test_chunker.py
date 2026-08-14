import unittest

from src.ingestion.chunker import Chunker


class ForceSplitTests(unittest.TestCase):
    def setUp(self):
        # _force_split 不依赖配置，避免在算法单测中加载模型配置。
        self.chunker = Chunker.__new__(Chunker)

    def test_long_text_terminates_and_preserves_overlap(self):
        text = "".join(str(index % 10) for index in range(2000))

        chunks = self.chunker._force_split(text, max_chars=300, overlap=60)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 300 for chunk in chunks))
        self.assertEqual(text[:300], chunks[0])
        self.assertEqual(text[-len(chunks[-1]):], chunks[-1])
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous[-60:], current[:60])

    def test_exact_boundary_finishes_after_last_chunk(self):
        text = "x" * 200

        chunks = self.chunker._force_split(text, max_chars=100, overlap=20)

        self.assertEqual([100, 100, 40], [len(chunk) for chunk in chunks])

    def test_oversized_overlap_is_clamped_to_keep_progress(self):
        chunks = self.chunker._force_split("abcdefgh", max_chars=4, overlap=99)

        self.assertEqual(5, len(chunks))
        self.assertEqual("abcdefgh", chunks[0] + "".join(chunk[-1] for chunk in chunks[1:]))

    def test_negative_overlap_is_treated_as_zero(self):
        self.assertEqual(
            ["abcd", "efgh"],
            self.chunker._force_split("abcdefgh", max_chars=4, overlap=-1),
        )

    def test_non_positive_chunk_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_chars"):
            self.chunker._force_split("text", max_chars=0, overlap=0)


if __name__ == "__main__":
    unittest.main()
