import unittest

from src.ingestion.metadata_sync import (
    inherit_registry_metadata,
    normalize_metadata_updates,
    synchronize_metadata,
)


class FakeStore:
    def __init__(self):
        self.rows = [
            {"chunk_id": "hash_0", "domain": "old", "category": "a", "doc_number": "1"},
            {"chunk_id": "hash_1", "domain": "old", "category": "a", "doc_number": "1"},
        ]
        self.restored = False

    def replace_file_metadata(self, file_hash, updates, expected_count):
        if len(self.rows) != expected_count:
            raise RuntimeError("count mismatch")
        snapshot = [dict(row) for row in self.rows]
        self.rows = [{**row, **updates} for row in self.rows]
        return snapshot

    def restore_file_snapshot(self, snapshot):
        self.rows = [dict(row) for row in snapshot]
        self.restored = True


class MetadataSyncTests(unittest.TestCase):
    def test_both_stores_receive_the_same_metadata(self):
        store = FakeStore()
        registry = {"domain": "old"}

        count = synchronize_metadata(
            file_hash="hash",
            updates={"domain": "reviewed", "doc_number": "DOC-2"},
            expected_chunks=2,
            store=store,
            commit_registry=lambda values: registry.update(values),
        )

        self.assertEqual(2, count)
        self.assertEqual("reviewed", registry["domain"])
        self.assertTrue(all(row["domain"] == "reviewed" for row in store.rows))
        self.assertTrue(all(row["doc_number"] == "DOC-2" for row in store.rows))

    def test_registry_failure_restores_vector_snapshot(self):
        store = FakeStore()

        def fail(_values):
            raise OSError("registry unavailable")

        with self.assertRaisesRegex(OSError, "registry unavailable"):
            synchronize_metadata(
                file_hash="hash",
                updates={"domain": "new"},
                expected_chunks=2,
                store=store,
                commit_registry=fail,
            )

        self.assertTrue(store.restored)
        self.assertTrue(all(row["domain"] == "old" for row in store.rows))

    def test_chunk_count_mismatch_prevents_registry_commit(self):
        store = FakeStore()
        committed = []
        with self.assertRaisesRegex(RuntimeError, "count mismatch"):
            synchronize_metadata(
                file_hash="hash",
                updates={"domain": "new"},
                expected_chunks=3,
                store=store,
                commit_registry=lambda values: committed.append(values),
            )
        self.assertEqual([], committed)

    def test_rebuild_preserves_reviewed_values_but_explicit_input_wins(self):
        merged = inherit_registry_metadata(
            {"domain": "detected", "category": "detected", "doc_number": "auto"},
            {"domain": "reviewed", "category": "manual", "doc_number": "DOC-X"},
            explicit_domain="request-domain",
        )
        self.assertEqual("request-domain", merged["domain"])
        self.assertEqual("manual", merged["category"])
        self.assertEqual("DOC-X", merged["doc_number"])

    def test_values_are_schema_bounded(self):
        self.assertEqual({"domain": "配电"}, normalize_metadata_updates({"domain": " 配电 "}))
        with self.assertRaisesRegex(ValueError, "domain exceeds"):
            normalize_metadata_updates({"domain": "x" * 33})


if __name__ == "__main__":
    unittest.main()
