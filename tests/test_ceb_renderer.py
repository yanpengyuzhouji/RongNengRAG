import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep this test independent from optional Milvus/embedding installations.
pdf_stub = types.ModuleType("ingestion.pdf_parser")
pdf_stub.PDFParser = type("PDFParser", (), {})
embed_stub = types.ModuleType("ingestion.embedder")
embed_stub.Embedder = type("Embedder", (), {})
embed_stub.create_text_for_embedding = lambda chunk: chunk.text
store_stub = types.ModuleType("ingestion.milvus_store")
store_stub.MilvusStore = type("MilvusStore", (), {})
sys.modules.setdefault("ingestion.pdf_parser", pdf_stub)
sys.modules.setdefault("ingestion.embedder", embed_stub)
sys.modules.setdefault("ingestion.milvus_store", store_stub)

from ingestion.ceb_renderer import CEBRenderer, CEBRenderResult
from ingestion.file_processor import FileProcessor, FileStatus, ProcessResult


class CEBRendererTests(unittest.TestCase):
    def test_valid_manifest_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            page_dir = root / "hash"
            page_dir.mkdir()
            (page_dir / "0001.png").write_bytes(b"png")
            (page_dir / "manifest.json").write_text(json.dumps({
                "file_hash": "hash",
                "source_name": "sample.ceb",
                "page_count": 1,
                "width": 800,
                "height": 1000,
            }), encoding="utf-8")
            renderer = CEBRenderer({
                "paths": {"ceb_rendered": str(root)},
                "ceb": {"render_width": 800, "render_height": 1000},
            })
            result = renderer._load_valid_cache(
                page_dir, "hash", Path("sample.ceb")
            )
            self.assertIsNotNone(result)
            self.assertEqual(1, result.page_count)

    def test_power_shell_single_quote_is_escaped(self):
        self.assertEqual("'a''b'", CEBRenderer._ps_quote("a'b"))


class CEBIngestionTests(unittest.TestCase):
    def _processor(self, cache_dir, page_paths, edited=False):
        processor = FileProcessor.__new__(FileProcessor)
        processor.config = {"paths": {"parsed_cache": str(cache_dir)}}
        processor.ceb_renderer = SimpleNamespace(
            render=lambda *_args, **_kwargs: CEBRenderResult(
                file_hash="h" * 64,
                source_path="sample.ceb",
                output_dir=cache_dir,
                page_paths=page_paths,
                width=800,
                height=1000,
            ),
            cleanup=lambda _file_hash: None,
        )

        class FakeOCR:
            def ocr_page_with_layout(self, path):
                page = Path(path).stem
                return (
                    f"正文 {page}",
                    [{
                        "block_type": "text",
                        "block_content": f"正文 {page}",
                        "bbox": [10, 10, 300, 50],
                    }],
                )

        class FakeChunker:
            def chunk_page_text(self, text, page_num, total_pages, file_meta):
                return [SimpleNamespace(
                    text=text,
                    char_count=len(text),
                    page_num=page_num,
                    total_pages=total_pages,
                    file_hash=file_meta["file_hash"],
                )]

        processor.pdf_parser = FakeOCR()
        processor.chunker = FakeChunker()
        processor._upsert_registry = lambda *args, **kwargs: None

        def finish(chunks, result, *args, **kwargs):
            result.status = FileStatus.COMPLETED
            result.chunks_created = len(chunks)
            result.chars_extracted = sum(c.char_count for c in chunks)
            return result

        processor._embed_and_insert = finish
        return processor

    def test_ceb_pages_share_layout_and_page_chunk_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pages = []
            for index in range(2):
                path = root / f"{index + 1:04d}.png"
                path.write_bytes(b"png")
                pages.append(path)
            processor = self._processor(root, pages)
            result = ProcessResult(
                file_path="sample.ceb",
                file_hash="h" * 64,
                file_name="sample.ceb",
                status=FileStatus.PROCESSING,
            )
            meta = {
                "file_hash": "h" * 64,
                "file_name": "sample.ceb",
                "domain": "配电",
                "category": "标准规范",
                "doc_number": "",
            }
            result = processor._process_ceb_png(
                "sample.ceb", meta, "h" * 64, "sample.ceb", 3, ".ceb", result
            )
            self.assertEqual(FileStatus.COMPLETED, result.status)
            self.assertEqual(2, result.chunks_created)
            layout = json.loads((root / ("h" * 64 + ".layout.json")).read_text(encoding="utf-8"))
            self.assertEqual(["0", "1"], sorted(layout))
            self.assertEqual("正文 0001", layout["0"][0]["block_content"])

    def test_published_edit_skips_ceb_render_and_ocr(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file_hash = "h" * 64
            edited = {
                "0": [{
                    "block_type": "text",
                    "block_content": "用户修订后的正文",
                    "bbox": [10, 10, 300, 50],
                }]
            }
            (root / f"{file_hash}.layout.edited.json").write_text(
                json.dumps(edited, ensure_ascii=False), encoding="utf-8"
            )
            processor = self._processor(root, [])
            processor.ceb_renderer.render = lambda *_args, **_kwargs: (
                (_ for _ in ()).throw(AssertionError("不应重新渲染 CEB"))
            )
            processor.pdf_parser.ocr_page_with_layout = lambda _path: (
                (_ for _ in ()).throw(AssertionError("不应重新 OCR"))
            )
            result = ProcessResult(
                file_path="sample.ceb",
                file_hash=file_hash,
                file_name="sample.ceb",
                status=FileStatus.PROCESSING,
            )
            meta = {
                "file_hash": file_hash,
                "file_name": "sample.ceb",
                "domain": "配电",
                "category": "标准规范",
                "doc_number": "",
            }
            result = processor._process_ceb_png(
                "sample.ceb", meta, file_hash, "sample.ceb", 3, ".ceb", result
            )
            self.assertEqual(FileStatus.COMPLETED, result.status)
            self.assertEqual(1, result.chunks_created)


if __name__ == "__main__":
    unittest.main()
