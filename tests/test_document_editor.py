import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.document_editor import (
    LayoutEditError,
    apply_layout_edits,
    count_visual_assets,
    layout_page_texts,
    layout_text_fingerprints,
    layout_revision,
)


class DocumentEditorTests(unittest.TestCase):
    def setUp(self):
        self.layout = {
            "0": [
                {"block_type": "title", "block_content": "第一章", "bbox": [0, 0, 500, 50]},
                {"block_type": "text", "block_content": "原始正文", "bbox": [20, 80, 480, 130]},
                {
                    "block_type": "chart",
                    "block_content": "",
                    "bbox": [20, 160, 480, 420],
                    "visual_data_uri": "data:image/png;base64,AAAA",
                },
                {"block_type": "text", "block_content": "图内识别文字", "bbox": [80, 220, 300, 250]},
                {"block_type": "figure_title", "block_content": "图 1 曲线", "bbox": [120, 430, 380, 460]},
            ]
        }

    def test_text_update_changes_revision_and_rebuilt_page_text(self):
        before = layout_revision(self.layout)
        edited, audit = apply_layout_edits(self.layout, [{
            "page_num": 1,
            "block_index": 1,
            "op": "update",
            "content": "修改后的正文",
            "content_format": "text",
        }])
        self.assertNotEqual(before, layout_revision(edited))
        self.assertEqual("原始正文", self.layout["0"][1]["block_content"])
        self.assertEqual("修改后的正文", audit[0]["after"])
        self.assertIn("修改后的正文", layout_page_texts(edited)[0][1])
        self.assertNotIn("原始正文", layout_page_texts(edited)[0][1])

    def test_visual_delete_removes_covered_ocr_but_keeps_caption(self):
        edited, _ = apply_layout_edits(self.layout, [{
            "page_num": 1,
            "block_index": 2,
            "op": "delete",
        }])
        text = layout_page_texts(edited)[0][1]
        self.assertEqual(0, count_visual_assets(edited))
        self.assertNotIn("图内识别文字", text)
        self.assertIn("图 1 曲线", text)
        self.assertIn("原始正文", text)

    def test_all_renderer_visual_kinds_cascade_delete_covered_ocr(self):
        for kind in ("photo", "seal", "header_image", "footer_image"):
            layout = {"0": [
                {
                    "block_type": kind,
                    "block_content": "",
                    "bbox": [20, 160, 480, 420],
                    "visual_data_uri": "data:image/png;base64,AAAA",
                },
                {
                    "block_type": "text",
                    "block_content": f"{kind} 内 OCR 旧内容",
                    "bbox": [80, 220, 300, 250],
                },
            ]}
            edited, _ = apply_layout_edits(layout, [{
                "page_num": 1,
                "block_index": 0,
                "op": "delete",
            }])
            self.assertNotIn(f"{kind} 内 OCR 旧内容", layout_page_texts(edited)[0][1])

    def test_duplicate_block_edit_is_rejected(self):
        with self.assertRaises(LayoutEditError):
            apply_layout_edits(self.layout, [
                {"page_num": 1, "block_index": 1, "op": "update", "content": "A"},
                {"page_num": 1, "block_index": 1, "op": "update", "content": "B"},
            ])

    def test_table_html_is_passive_and_becomes_indexable_text(self):
        layout = {"0": [{
            "block_type": "table",
            "block_content": "<table><tr><td>旧值</td></tr></table>",
            "bbox": [0, 0, 300, 100],
        }]}
        edited, _ = apply_layout_edits(layout, [{
            "page_num": 1,
            "block_index": 0,
            "op": "update",
            "content_format": "html",
            "content": '<table onclick="bad()"><tr><td>新值</td></tr></table><script>bad()</script>',
        }])
        stored = edited["0"][0]["block_content"]
        self.assertNotIn("onclick", stored)
        self.assertNotIn("script", stored)
        self.assertIn("新值", layout_page_texts(edited)[0][1])

    def test_table_image_edit_keeps_safe_image_but_drops_editor_controls(self):
        layout = {"0": [{
            "block_type": "table",
            "block_content": "<table><tr><td>图片</td></tr></table>",
            "bbox": [0, 0, 300, 100],
        }]}
        edited, _ = apply_layout_edits(layout, [{
            "page_num": 1,
            "block_index": 0,
            "op": "update",
            "content_format": "html",
            "content": (
                '<table><tr><td><span class="layout-inline-image">'
                '<img src="data:image/png;base64,AAAA"><button class="layout-delete-inline-image">删除图片</button>'
                '</span></td></tr></table>'
            ),
        }])
        stored = edited["0"][0]["block_content"]
        self.assertIn('src="data:image/png;base64,AAAA"', stored)
        self.assertNotIn("layout-inline-image", stored)
        self.assertNotIn("layout-delete-inline-image", stored)
        self.assertEqual(1, count_visual_assets(edited))
        self.assertNotIn("data:image", layout_page_texts(edited)[0][1])

    def test_delete_table_images_is_visual_only_and_handles_multiple_indexes(self):
        layout = {"0": [{
            "block_type": "table",
            "block_content": (
                '<table><tr><td>固定文字</td><td>'
                '<img src="data:image/png;base64,AAAA"><img src="data:image/png;base64,BBBB">'
                '</td></tr></table>'
            ),
            "bbox": [0, 0, 300, 100],
        }]}
        before = layout_text_fingerprints(layout)
        edited, audit = apply_layout_edits(layout, [
            {"page_num": 1, "block_index": 0, "op": "delete_table_image", "image_index": 0},
            {"page_num": 1, "block_index": 0, "op": "delete_table_image", "image_index": 1},
        ])
        self.assertEqual(before, layout_text_fingerprints(edited))
        self.assertEqual(0, count_visual_assets(edited))
        self.assertEqual(2, len(audit))


if __name__ == "__main__":
    unittest.main()
