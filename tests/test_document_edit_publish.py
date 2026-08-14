import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.chunker import Chunk
from ingestion.document_editor import LayoutRevisionConflict, layout_revision

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

from ingestion.file_processor import FileProcessor

# Keep this module's lightweight dependency stubs local so unittest discovery
# can import the real Milvus modules (or another test's own stubs) afterwards.
for module_name in (
    "ingestion.file_processor",
    "ingestion.pdf_parser",
    "ingestion.embedder",
    "ingestion.milvus_store",
):
    sys.modules.pop(module_name, None)


FILE_HASH = "a" * 64


class FakeGeneration:
    def __init__(self):
        self.inserted = []
        self.validated = None
        self.activated = False
        self.finalized = False
        self.rolled_back = False

    def insert(self, **kwargs):
        self.inserted.extend(kwargs["chunks"])

    def validate(self, count, expected_chunks=None):
        self.validated = count

    def activate(self):
        self.activated = True

    def finalize(self):
        self.finalized = True

    def rollback(self):
        self.rolled_back = True


class FakeStore:
    def __init__(self):
        self.generation = FakeGeneration()
        self.purged_hashes = []

    def begin_file_generation(self, file_hash):
        self.file_hash = file_hash
        return self.generation

    def purge_file_generations(self, file_hash):
        self.purged_hashes.append(file_hash)


class FakeChunker:
    def chunk_page_text(self, text, page_num, total_pages, file_meta):
        return [Chunk(
            chunk_id=f"{file_meta['file_hash']}_{page_num}",
            file_hash=file_meta["file_hash"],
            text=text,
            char_count=len(text),
            page_num=page_num,
            total_pages=total_pages,
            file_name=file_meta["file_name"],
        )]


class DocumentEditPublishTests(unittest.TestCase):
    def make_processor(self, cache_dir: Path):
        processor = FileProcessor.__new__(FileProcessor)
        processor.config = {"paths": {"parsed_cache": str(cache_dir)}}
        processor.config_path = "unused"
        processor.uploads_dir = cache_dir
        processor._metadata_lock = threading.RLock()
        processor._index_lock = threading.RLock()
        processor.store = FakeStore()
        processor.chunker = FakeChunker()
        processor.embedder = SimpleNamespace(encode=lambda texts, show_progress=False: SimpleNamespace(
            dense_vectors=[[0.1] for _ in texts],
            sparse_vectors=[{} for _ in texts],
        ))
        processor._resolve_hash = lambda identifier: FILE_HASH
        processor._get_registry = lambda file_hash: {
            "file_hash": FILE_HASH,
            "file_name": "sample.pdf",
            "original_path": str(cache_dir / "sample.pdf"),
            "stored_path": "",
            "file_size": 10,
            "file_type": ".pdf",
            "status": "completed",
            "domain": "标准规范",
            "category": "国标",
            "doc_number": "TEST-1",
        }
        processor._build_file_meta = lambda path, file_hash, domain, category: {
            "file_hash": file_hash,
            "file_name": "sample.pdf",
            "extension": ".pdf",
            "relative_path": "sample.pdf",
            "domain": domain or "",
            "category": category or "",
        }
        processor._upsert_registry = lambda *args, **kwargs: None
        return processor

    def test_success_publishes_cache_sidecar_and_vector_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            layout = {"0": [{"block_type": "text", "block_content": "旧正文", "bbox": [0, 0, 100, 20]}]}
            path = cache_dir / f"{FILE_HASH}.layout.json"
            path.write_text(json.dumps(layout, ensure_ascii=False), encoding="utf-8")
            processor = self.make_processor(cache_dir)

            result = processor.save_layout_edits(FILE_HASH, layout_revision(layout), [{
                "page_num": 1,
                "block_index": 0,
                "op": "update",
                "content": "新正文",
                "content_format": "text",
            }])

            self.assertTrue(result["success"])
            self.assertEqual("新正文", json.loads(path.read_text(encoding="utf-8"))["0"][0]["block_content"])
            self.assertTrue((cache_dir / f"{FILE_HASH}.layout.edited.json").exists())
            self.assertTrue(processor.store.generation.activated)
            self.assertTrue(processor.store.generation.finalized)
            self.assertEqual(1, processor.store.generation.validated)

    def test_revision_conflict_does_not_start_vector_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            layout = {"0": [{"block_type": "text", "block_content": "正文", "bbox": [0, 0, 100, 20]}]}
            path = cache_dir / f"{FILE_HASH}.layout.json"
            path.write_text(json.dumps(layout, ensure_ascii=False), encoding="utf-8")
            processor = self.make_processor(cache_dir)
            with self.assertRaises(LayoutRevisionConflict):
                processor.save_layout_edits(FILE_HASH, "b" * 64, [{
                    "page_num": 1, "block_index": 0, "op": "update", "content": "覆盖"
                }])
            self.assertFalse(processor.store.generation.activated)
            self.assertEqual(layout, json.loads(path.read_text(encoding="utf-8")))

    def test_registry_failure_rolls_back_alias_and_both_cache_files(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            layout = {"0": [{"block_type": "text", "block_content": "旧正文", "bbox": [0, 0, 100, 20]}]}
            path = cache_dir / f"{FILE_HASH}.layout.json"
            path.write_text(json.dumps(layout, ensure_ascii=False), encoding="utf-8")
            processor = self.make_processor(cache_dir)
            processor._upsert_registry = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("registry failed"))

            with self.assertRaisesRegex(RuntimeError, "registry failed"):
                processor.save_layout_edits(FILE_HASH, layout_revision(layout), [{
                    "page_num": 1, "block_index": 0, "op": "update", "content": "新正文"
                }])

            self.assertTrue(processor.store.generation.rolled_back)
            self.assertEqual(layout, json.loads(path.read_text(encoding="utf-8")))
            self.assertFalse((cache_dir / f"{FILE_HASH}.layout.edited.json").exists())

    def test_delete_purges_vectors_and_both_hash_keyed_layout_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            for suffix in (".layout.json", ".layout.edited.json"):
                (cache_dir / f"{FILE_HASH}{suffix}").write_text("{}", encoding="utf-8")
            pending = cache_dir / f".{FILE_HASH}.layout.json.123.pending"
            pending.write_text("{}", encoding="utf-8")
            processor = self.make_processor(cache_dir)

            self.assertTrue(processor.delete(FILE_HASH, remove_file=False))

            self.assertEqual([FILE_HASH], processor.store.purged_hashes)
            self.assertFalse((cache_dir / f"{FILE_HASH}.layout.json").exists())
            self.assertFalse((cache_dir / f"{FILE_HASH}.layout.edited.json").exists())
            self.assertFalse(pending.exists())


if __name__ == "__main__":
    unittest.main()
