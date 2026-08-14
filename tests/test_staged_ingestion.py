import sys
import threading
import types
import unittest
from types import SimpleNamespace


pdf_stub = types.ModuleType("ingestion.pdf_parser")
pdf_stub.PDFParser = type("PDFParser", (), {})
embed_stub = types.ModuleType("ingestion.embedder")
embed_stub.Embedder = type("Embedder", (), {})
embed_stub.create_text_for_embedding = lambda chunk: chunk.text
store_stub = types.ModuleType("ingestion.milvus_store")
store_stub.MilvusStore = type("MilvusStore", (), {})

sys.path.insert(0, "src")
sys.modules.setdefault("ingestion.pdf_parser", pdf_stub)
sys.modules.setdefault("ingestion.embedder", embed_stub)
sys.modules.setdefault("ingestion.milvus_store", store_stub)

from ingestion.file_processor import FileProcessor, FileStatus, ProcessResult


class FakeEmbedder:
    def encode(self, texts, show_progress=False):
        del show_progress
        return SimpleNamespace(
            dense_vectors=[[float(len(text))] for text in texts],
            sparse_vectors=[{} for _ in texts],
        )


class FakeGeneration:
    def __init__(self, fail_insert=False):
        self.fail_insert = fail_insert
        self.inserted = 0
        self.validated = None
        self.activated = False
        self.rolled_back = False
        self.finalized = False

    def insert(self, **values):
        if self.fail_insert:
            raise OSError("staging insert failed")
        self.inserted += len(values["chunks"])

    def validate(self, expected_file_count, expected_chunks=None):
        self.validated = expected_file_count

    def activate(self):
        self.activated = True

    def rollback(self):
        self.rolled_back = True
        self.activated = False

    def finalize(self):
        self.finalized = True


class FakeStore:
    def __init__(self, generation):
        self.generation = generation

    def begin_file_generation(self, file_hash):
        self.file_hash = file_hash
        return self.generation


class StagedIngestionTests(unittest.TestCase):
    def _processor(self, generation, registry_error=None):
        processor = FileProcessor.__new__(FileProcessor)
        processor.embedder = FakeEmbedder()
        processor.store = FakeStore(generation)
        processor._index_lock = threading.RLock()
        processor.registry_calls = []

        def upsert(*args, **kwargs):
            processor.registry_calls.append((args, kwargs))
            if registry_error:
                raise registry_error

        processor._upsert_registry = upsert
        return processor

    @staticmethod
    def _result():
        return ProcessResult(
            file_path="file.txt",
            file_hash="hash",
            file_name="file.txt",
            status=FileStatus.PROCESSING,
            chars_extracted=3,
            domain="domain",
        )

    def test_partial_insert_failure_rolls_back_before_activation(self):
        generation = FakeGeneration(fail_insert=True)
        processor = self._processor(generation)
        with self.assertRaisesRegex(OSError, "staging insert failed"):
            processor._embed_and_insert(
                [SimpleNamespace(text="abc")], self._result(),
                "hash", "file.txt", "file.txt", 3, ".txt", 0,
            )
        self.assertTrue(generation.rolled_back)
        self.assertFalse(generation.activated)
        self.assertEqual([], processor.registry_calls)

    def test_registry_failure_reverts_activated_alias(self):
        generation = FakeGeneration()
        processor = self._processor(generation, OSError("registry failed"))
        with self.assertRaisesRegex(OSError, "registry failed"):
            processor._embed_and_insert(
                [SimpleNamespace(text="abc")], self._result(),
                "hash", "file.txt", "file.txt", 3, ".txt", 0,
            )
        self.assertTrue(generation.rolled_back)
        self.assertFalse(generation.activated)
        self.assertFalse(generation.finalized)

    def test_registry_commits_only_after_validation_and_activation(self):
        generation = FakeGeneration()
        processor = self._processor(generation)
        result = processor._embed_and_insert(
            [SimpleNamespace(text="a"), SimpleNamespace(text="bc")],
            self._result(), "hash", "file.txt", "file.txt", 3, ".txt", 0,
        )
        self.assertEqual(FileStatus.COMPLETED, result.status)
        self.assertEqual(2, generation.validated)
        self.assertTrue(generation.finalized)
        self.assertEqual(1, len(processor.registry_calls))


if __name__ == "__main__":
    unittest.main()
