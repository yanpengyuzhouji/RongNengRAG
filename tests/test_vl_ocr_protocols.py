import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("fitz", Mock())
sys.modules.setdefault("httpx", Mock())
from ingestion.vl_ocr import (
    VLOcrClient,
    _clean_fragmentary_html,
    extract_layout_outline,
    render_layout_html,
    render_layout_pages_html,
)


class VLOcrProtocolTests(unittest.TestCase):
    def test_fragmentary_table_markup_is_removed_but_complete_table_kept(self):
        text = "说明\n<td>415</td></tr><tr><td>335</td></tr></table>\n<table><tr><td>完整</td></tr></table>"
        cleaned = _clean_fragmentary_html(text)
        self.assertNotIn("415", cleaned)
        self.assertIn("<table><tr><td>完整</td></tr></table>", cleaned)

    def test_glued_numeric_row_is_removed(self):
        self.assertNotIn("415>50, ≤100335300175", _clean_fragmentary_html("注：\n415>50, ≤100335300175"))

    def test_formula_number_and_connector_share_layout_identity(self):
        html = render_layout_html([
            {
                "bbox": [100, 100, 300, 140],
                "block_type": "formula",
                "block_content": "$$x^2+y^2=z^2$$",
            },
            {
                "bbox": [500, 112, 540, 132],
                "block_type": "formula_number",
                "block_content": "(1)",
            },
        ])
        self.assertIn('data-layout-id="layout-0"', html)
        self.assertIn('data-layout-id="layout-1"', html)
        self.assertIn('data-formula-id="layout-0"', html)
        self.assertIn('data-number-id="layout-1"', html)
        self.assertIn('function alignFormulaRows()', html)
        self.assertIn('function fitPage()', html)
        self.assertIn('rongneng-layout-size', html)

    def test_main_title_uses_page_width_and_center_alignment(self):
        html = render_layout_html([
            {
                "bbox": [280, 40, 460, 80],
                "block_type": "title",
                "block_content": "文档大标题",
            },
        ])
        self.assertIn("ocr-main-title", html)
        self.assertIn("ocr-centered-heading", html)
        self.assertIn("left:0!important;right:0!important;width:auto!important", html)
        self.assertIn('data-x1="0" data-x2="740.0"', html)

    def test_section_title_is_centered_across_the_page(self):
        html = render_layout_html([
            {
                "bbox": [80, 180, 250, 215],
                "block_type": "paragraph_title",
                "block_content": "6 蓄电池组配置",
            },
        ])
        self.assertIn("ocr-centered-heading", html)
        self.assertIn('data-x1="0" data-x2="330.0"', html)

    def test_flow_text_uses_its_own_symmetric_page_margin(self):
        html = render_layout_html([
            # A page decoration touching x=0 must not make body text extend
            # all the way to the right edge.
            {
                "bbox": [0, 10, 80, 30],
                "block_type": "header",
                "block_content": "页眉",
            },
            {
                "bbox": [120, 80, 560, 130],
                "block_type": "text",
                "block_content": "一段需要在正文右边界前换行的长文本。",
            },
        ])
        self.assertIn('left:120.0px;top:80.0px;right:120.0px;', html)
        self.assertIn('overflow-wrap:anywhere', html)
        self.assertIn('.ocr-block-content{width:100%;max-width:100%', html)

    def test_contents_rows_use_aligned_leaders_and_page_numbers(self):
        html = render_layout_html([
            {
                "bbox": [40, 100, 700, 260],
                "block_type": "content",
                "block_content": (
                    "1 总则 ..... (1)\n"
                    "1.1 一般规定 ..... (2)\n"
                    "2 General provisions ..... (3)"
                ),
            },
        ])
        self.assertIn('class="layout-toc"', html)
        self.assertEqual(3, html.count('class="layout-toc-row"'))
        self.assertEqual(3, html.count('class="layout-toc-leader"'))
        self.assertIn('class="layout-toc-page">1</span>', html)
        self.assertIn("1.1 一般规定", html)
        self.assertNotIn(".....", html)

    def test_contents_is_detected_when_layout_label_is_generic_text(self):
        html = render_layout_html([
            {
                "bbox": [40, 100, 700, 180],
                "block_type": "text",
                "block_content": "1 Introduction ..... 1\n2 Scope ..... 2",
            },
        ])
        self.assertIn('<div class="layout-toc">', html)
        self.assertIn('class="layout-toc-page">2</span>', html)

    def test_visual_chart_block_is_rendered_at_original_bbox(self):
        html = render_layout_html([
            {
                "bbox": [120, 220, 420, 520],
                "block_type": "chart",
                "block_content": "",
                "visual_data_uri": "data:image/png;base64,AAAA",
            },
            {
                "bbox": [150, 530, 390, 555],
                "block_type": "figure_title",
                "block_content": "图 1 励磁系统响应曲线",
            },
        ])
        self.assertIn("ocr-visual-block", html)
        self.assertIn("layout-visual-asset", html)
        self.assertIn("data:image/png;base64,AAAA", html)
        self.assertIn("left:120.0px;top:220.0px;width:300.0px", html)
        self.assertIn("图 1 励磁系统响应曲线", html)
        self.assertIn("ocr-block:not(.ocr-visual-block)", html)

    def test_editable_layout_uses_same_renderer_with_block_controls(self):
        blocks = [
            {
                "block_type": "text",
                "block_content": "可编辑正文",
                "bbox": [10, 10, 300, 60],
            },
            {
                "block_type": "chart",
                "block_content": "",
                "bbox": [10, 80, 300, 260],
                "visual_data_uri": "data:image/png;base64,AAAA",
            },
        ]
        preview = render_layout_html(blocks, page_num=2)
        editable = render_layout_html(blocks, page_num=2, editable=True)
        self.assertNotIn('contenteditable="true"', preview)
        self.assertNotIn("layout-delete-visual", preview)
        self.assertIn('contenteditable="true"', editable)
        self.assertIn('data-source-index="0"', editable)
        self.assertIn("layout-delete-visual", editable)
        self.assertIn("rongneng-layout-edit", editable)

    def test_pdf_layout_cache_is_rendered_as_one_document_per_page(self):
        pages = render_layout_pages_html({
            "0": [{
                "bbox": [10, 20, 100, 50],
                "block_type": "text",
                "block_content": "第一页",
            }],
            "1": [{
                "bbox": [10, 20, 100, 50],
                "block_type": "text",
                "block_content": "第二页",
            }],
        })
        self.assertEqual([1, 2], [page["page_num"] for page in pages])
        self.assertIn("第一页", pages[0]["layout_html"])
        self.assertIn("第二页", pages[1]["layout_html"])

    def test_layout_outline_keeps_heading_levels_and_rejects_numbered_body(self):
        blocks = [{
            "block_id": 1,
            "block_label": "paragraph_title",
            "block_content": "6 蓄电池组配置",
            "block_bbox": [20, 20, 300, 50],
        }, {
            "block_id": 2,
            "block_label": "text",
            "block_content": "6.1 蓄电池容量计算",
            "block_bbox": [150, 60, 350, 90],
        }, {
            "block_id": 3,
            "block_label": "text",
            "block_content": "1 电缆长期允许载流量不应小于计算电流量；",
            "block_bbox": [20, 100, 500, 130],
        }]
        outline = extract_layout_outline(blocks)
        self.assertEqual([1, 2], [item["level"] for item in outline])
        self.assertEqual("outline-p1-b1", outline[0]["anchor"])
        self.assertNotIn("电缆长期允许载流量", " ".join(item["title"] for item in outline))
        html = render_layout_html(blocks)
        self.assertIn('id="outline-p1-b1"', html)
        self.assertIn('id="outline-p1-b2"', html)

    @patch("httpx.post")
    def test_pipeline_multipart_response_is_flattened_in_layout_order(self, post):
        response = Mock()
        response.json.return_value = {
            "result": [{
                "res": {
                    "parsing_res_list": [
                        {"block_order": 2, "block_content": "第二段"},
                        {"block_order": 1, "block_content": "标题"},
                    ]
                }
            }]
        }
        post.return_value = response
        client = VLOcrClient("http://192.168.0.201:8001", protocol="pipeline")
        self.assertEqual("标题\n\n第二段", client.recognize_image(b"png"))
        post.assert_called_once()
        self.assertTrue(post.call_args.kwargs["files"]["file"][2] == "image/png")

    @patch("httpx.post")
    def test_pipeline_prefers_server_markdown_when_available(self, post):
        response = Mock()
        response.json.return_value = {"result": [{"res": {"markdown": "# 标题\n\n|A|B|"}}]}
        post.return_value = response
        client = VLOcrClient("http://192.168.0.201:8001", protocol="pipeline")
        self.assertEqual("# 标题\n\n|A|B|", client.recognize_image(b"png"))

    @patch("httpx.post")
    def test_pipeline_accepts_null_layout_order_fields(self, post):
        response = Mock()
        response.json.return_value = {
            "result": [{
                "res": {
                    "parsing_res_list": [
                        {"block_order": None, "block_id": None, "block_content": "正文"},
                        {"block_order": 1, "block_content": "标题"},
                    ]
                }
            }]
        }
        post.return_value = response
        client = VLOcrClient("http://192.168.0.201:8001", protocol="pipeline")
        self.assertEqual("标题\n\n正文", client.recognize_image(b"png"))

    @patch("httpx.post")
    def test_pipeline_prefers_coordinates_and_deduplicates_blocks(self, post):
        response = Mock()
        response.json.return_value = {
            "result": [{
                "res": {
                    "parsing_res_list": [
                        {"block_order": 1, "bbox": [0, 700, 100, 740], "block_content": "3.0.4 页尾"},
                        {"block_order": 2, "bbox": [0, 100, 100, 140], "block_content": "表格"},
                        {"block_order": 3, "bbox": [0, 100, 100, 140], "block_content": "表格"},
                    ]
                }
            }]
        }
        post.return_value = response
        client = VLOcrClient("http://192.168.0.201:8001", protocol="pipeline")
        self.assertEqual("表格\n\n3.0.4 页尾", client.recognize_image(b"png"))


if __name__ == "__main__":
    unittest.main()
