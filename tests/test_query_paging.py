import unittest

from src.ingestion.query_paging import iter_query_batches


class FakeIterator:
    def __init__(self, batches):
        self.batches = list(batches)
        self.closed = False

    def next(self):
        return self.batches.pop(0) if self.batches else []

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, batches):
        self.iterator = FakeIterator(batches)
        self.request = None

    def query_iterator(self, **request):
        self.request = request
        return self.iterator


class QueryPagingTests(unittest.TestCase):
    def test_all_batches_beyond_legacy_20000_limit_are_returned(self):
        client = FakeClient([
            list(range(10_000)), list(range(10_000, 20_000)),
            list(range(20_000, 25_001)),
        ])
        rows = [
            row
            for batch in iter_query_batches(
                client,
                collection_name="active",
                filter_expr='domain == "配电"',
                output_fields=["chunk_id"],
                batch_size=10_000,
            )
            for row in batch
        ]
        self.assertEqual(25_001, len(rows))
        self.assertEqual(25_000, rows[-1])
        self.assertTrue(client.iterator.closed)

    def test_iterator_is_closed_when_consumer_fails(self):
        client = FakeClient([[1], [2]])
        generator = iter_query_batches(
            client, collection_name="active", filter_expr="",
            output_fields=["chunk_id"],
        )
        self.assertEqual([1], next(generator))
        generator.close()
        self.assertTrue(client.iterator.closed)

    def test_old_client_fails_instead_of_returning_partial_statistics(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be guaranteed"):
            list(iter_query_batches(
                object(), collection_name="active", filter_expr="",
                output_fields=["chunk_id"],
            ))

    def test_duplicate_chunk_id_fails_closed_and_closes_iterator(self):
        client = FakeClient([
            [{"chunk_id": "a"}, {"chunk_id": "b"}],
            [{"chunk_id": "b"}],
        ])
        with self.assertRaisesRegex(RuntimeError, "duplicate chunk_id"):
            list(iter_query_batches(
                client, collection_name="active", filter_expr="",
                output_fields=["chunk_id"],
            ))
        self.assertTrue(client.iterator.closed)


if __name__ == "__main__":
    unittest.main()
