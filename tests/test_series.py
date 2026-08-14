import unittest

from src.retrieval.series import extract_series_key


class SeriesKeyTests(unittest.TestCase):
    def test_chinese_and_numeric_ordinals_share_the_same_chapter_key(self):
        self.assertEqual("第*章", extract_series_key("第1章 总则.pdf"))
        self.assertEqual("第*章", extract_series_key("第十二章 附则.pdf"))
        self.assertEqual("第*章节", extract_series_key("第二章节 说明.docx"))

    def test_different_section_units_do_not_collide(self):
        self.assertEqual("第*部分", extract_series_key("第2部分.pdf"))
        self.assertEqual("第*篇", extract_series_key("第三篇.pdf"))
        self.assertNotEqual(
            extract_series_key("第2部分.pdf"),
            extract_series_key("第2章.pdf"),
        )

    def test_meeting_material_and_unrelated_names(self):
        self.assertEqual("会议材料之", extract_series_key("01会议材料之一.pdf"))
        self.assertIsNone(extract_series_key("设计规范.pdf"))


if __name__ == "__main__":
    unittest.main()
