import unittest

from src.api.download_utils import download_filename, download_media_type


class DownloadFormatTests(unittest.TestCase):
    def test_media_type_follows_original_extension(self):
        self.assertEqual("application/pdf", download_media_type("规范.pdf"))
        self.assertEqual(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            download_media_type("台账.xlsx"),
        )
        self.assertEqual("image/png", download_media_type("截图.png"))
        self.assertEqual("text/markdown", download_media_type("说明.md"))
        self.assertEqual("application/x-ceb", download_media_type("通知.ceb"))

    def test_unknown_extension_uses_safe_binary_fallback(self):
        self.assertEqual("application/octet-stream", download_media_type("file.unknown"))

    def test_download_filename_is_a_basename(self):
        self.assertEqual("报告.pdf", download_filename(r"C:\tmp\报告.pdf", "fallback.bin"))
        self.assertEqual("fallback.xlsx", download_filename("", "/tmp/fallback.xlsx"))
        self.assertEqual("fallback.xlsx", download_filename("", r"C:\tmp\fallback.xlsx"))


if __name__ == "__main__":
    unittest.main()
