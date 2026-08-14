import sys
import types
import unittest
from pathlib import Path


try:
    import pymilvus  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("pymilvus")
    for name in (
        "MilvusClient", "DataType", "Function", "AnnSearchRequest",
        "RRFRanker", "WeightedRanker",
    ):
        setattr(stub, name, type(name, (), {}))
    sys.modules["pymilvus"] = stub

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.milvus_store import IndexGeneration, MilvusStore


class FakeMilvusClient:
    def __init__(self):
        self.rows = [
            {
                "chunk_id": "abc_0", "text": "zero", "dense_vector": [0.1],
                "sparse_vector": {1: 0.2}, "domain": "old", "category": "cat",
                "doc_number": "D1",
            },
            {
                "chunk_id": "abc_1", "text": "one", "dense_vector": [0.2],
                "sparse_vector": {2: 0.3}, "domain": "old", "category": "cat",
                "doc_number": "D1",
            },
        ]
        self.flushes = 0
        self.corrupt_next_upsert = False

    def has_collection(self, _name):
        return True

    def query(self, *, collection_name, filter, output_fields, limit):
        del collection_name, filter, limit
        return [
            {field: row.get(field) for field in output_fields if field in row}
            for row in self.rows
        ]

    def upsert(self, *, collection_name, data):
        del collection_name
        data = [dict(row) for row in data]
        if self.corrupt_next_upsert and data:
            data[0]["domain"] = "corrupted"
            self.corrupt_next_upsert = False
        by_id = {row["chunk_id"]: dict(row) for row in self.rows}
        for row in data:
            by_id[row["chunk_id"]] = dict(row)
        self.rows = [by_id[key] for key in sorted(by_id)]

    def flush(self, _collection_name):
        self.flushes += 1


class MilvusMetadataTests(unittest.TestCase):
    def setUp(self):
        self.store = MilvusStore.__new__(MilvusStore)
        self.store.client = FakeMilvusClient()

    def test_replace_verifies_scalars_and_preserves_vectors_for_rollback(self):
        snapshot = self.store.replace_file_metadata(
            "abc", {"domain": "new", "doc_number": "D2"}, expected_count=2
        )

        self.assertEqual(2, len(snapshot))
        self.assertEqual([0.1], snapshot[0]["dense_vector"])
        self.assertTrue(all(row["domain"] == "new" for row in self.store.client.rows))
        self.store.restore_file_snapshot(snapshot)
        self.assertTrue(all(row["domain"] == "old" for row in self.store.client.rows))

    def test_count_mismatch_does_not_write(self):
        before = [dict(row) for row in self.store.client.rows]
        with self.assertRaisesRegex(RuntimeError, "chunk count mismatch"):
            self.store.replace_file_metadata(
                "abc", {"category": "new"}, expected_count=3
            )
        self.assertEqual(before, self.store.client.rows)
        self.assertEqual(0, self.store.client.flushes)

    def test_failed_verification_automatically_restores_snapshot(self):
        self.store.client.corrupt_next_upsert = True
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            self.store.replace_file_metadata(
                "abc", {"domain": "new"}, expected_count=2
            )
        self.assertTrue(all(row["domain"] == "old" for row in self.store.client.rows))
        self.assertEqual(2, self.store.client.flushes)


class FakeGenerationClient:
    def __init__(self):
        self.collections = {
            "source": [
                {"chunk_id": "abc_0", "text": "old", "dense_vector": [0.1], "sparse_vector": {}},
                {"chunk_id": "def_0", "text": "other", "dense_vector": [0.2], "sparse_vector": {}},
            ]
        }
        self.aliases = {MilvusStore.COLLECTION_NAME: "source"}

    def _target(self, name):
        return self.aliases.get(name, name)

    def describe_alias(self, *, alias):
        if alias not in self.aliases:
            raise KeyError(alias)
        return {"alias": alias, "collection_name": self.aliases[alias]}

    def has_collection(self, name):
        return self._target(name) in self.collections

    def get_collection_stats(self, name):
        return {"row_count": len(self.collections[self._target(name)])}

    def query(self, *, collection_name, filter, output_fields, limit):
        rows = self.collections[self._target(collection_name)]
        if filter.startswith('chunk_id like "'):
            prefix = filter.split('"', 2)[1].removesuffix("%")
            rows = [row for row in rows if row["chunk_id"].startswith(prefix)]
        return [
            {field: row.get(field) for field in output_fields if field in row}
            for row in rows[:limit]
        ]

    def query_iterator(self, **_request):
        raise AssertionError("staged validation must not use query_iterator")

    def insert(self, *, collection_name, data):
        self.collections[self._target(collection_name)].extend(
            dict(row) for row in data
        )

    def delete(self, *, collection_name, filter):
        prefix = filter.split('"', 2)[1].removesuffix("%")
        target = self._target(collection_name)
        self.collections[target] = [
            row for row in self.collections[target]
            if not row["chunk_id"].startswith(prefix)
        ]

    def flush(self, _name):
        pass

    def load_collection(self, _name):
        pass

    def alter_alias(self, *, collection_name, alias):
        self.aliases[alias] = collection_name

    def drop_collection(self, name):
        del self.collections[name]


class IndexGenerationTests(unittest.TestCase):
    def setUp(self):
        self.store = MilvusStore.__new__(MilvusStore)
        self.store.client = FakeGenerationClient()
        self.store.create_collection = lambda drop_existing=False, collection_name=None: (
            self.store.client.collections.setdefault(collection_name, [])
        )

    def test_partial_staging_never_changes_active_search_results(self):
        generation = self.store.begin_file_generation("abc")
        self.store.client.collections[generation.staging_collection].append(
            {"chunk_id": "abc_0", "text": "partial"}
        )

        active = self.store._query_file_rows("abc", ["chunk_id", "text"])
        self.assertEqual([{"chunk_id": "abc_0", "text": "old"}], active)
        with self.assertRaisesRegex(RuntimeError, "Staged index validation failed"):
            generation.validate(expected_file_count=2)
        generation.rollback()
        self.assertEqual("source", self.store.client.aliases[self.store.COLLECTION_NAME])

    def test_registry_failure_after_activation_switches_back_to_old_generation(self):
        generation = self.store.begin_file_generation("abc")
        self.store.client.collections[generation.staging_collection].append(
            {"chunk_id": "abc_0", "text": "new"}
        )
        generation.validate(expected_file_count=1)
        generation.activate()
        self.assertEqual(
            generation.staging_collection,
            self.store.client.aliases[self.store.COLLECTION_NAME],
        )

        generation.rollback()
        self.assertEqual("source", self.store.client.aliases[self.store.COLLECTION_NAME])
        self.assertNotIn(generation.staging_collection, self.store.client.collections)

    def test_staged_validation_uses_direct_file_query_and_total_stat(self):
        generation = self.store.begin_file_generation("abc")
        self.store.client.collections[generation.staging_collection].append(
            {"chunk_id": "abc_0", "text": "new"}
        )
        generation.validate(expected_file_count=1)


if __name__ == "__main__":
    unittest.main()
