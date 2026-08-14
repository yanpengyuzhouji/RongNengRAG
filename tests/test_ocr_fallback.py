import unittest

from src.ingestion.ocr_fallback import choose_page_text


class OcrFallbackTests(unittest.TestCase):
    def test_empty_ocr_preserves_short_pdf_text(self):
        self.assertEqual(
            ("short source text", "pdf_text_fallback"),
            choose_page_text(" short source text ", "", lambda _text: False),
        )

    def test_garbage_or_detector_exception_preserves_pdf_text(self):
        self.assertEqual(
            ("source", "pdf_text_fallback"),
            choose_page_text("source", "hallucinated", lambda _text: True),
        )

        def broken(_text):
            raise RuntimeError("detector failed")

        self.assertEqual(
            ("source", "pdf_text_fallback"),
            choose_page_text("source", "uncertain", broken),
        )

    def test_valid_ocr_replaces_weaker_source_text(self):
        self.assertEqual(
            ("recognized full text", "ocr"),
            choose_page_text(
                "short", " recognized full text ", lambda _text: False
            ),
        )

    def test_no_source_and_no_valid_ocr_is_empty(self):
        self.assertEqual(
            ("", "empty"),
            choose_page_text("", "garbage", lambda _text: True),
        )


if __name__ == "__main__":
    unittest.main()
