import unittest

from src.api.stream_limits import StreamLimitExceeded, limited_stream


async def chunks(*values):
    for value in values:
        yield value


class StreamLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_is_forwarded_chunk_by_chunk(self):
        result = [part async for part in limited_stream(chunks(b"ab", b"cd"), 4)]
        self.assertEqual([b"ab", b"cd"], result)

    async def test_oversized_stream_stops_without_buffering_remainder(self):
        consumed = []

        async def source():
            for value in (b"abc", b"def", b"never"):
                consumed.append(value)
                yield value

        with self.assertRaisesRegex(StreamLimitExceeded, "proxy limit"):
            [part async for part in limited_stream(source(), 5)]
        self.assertEqual([b"abc", b"def"], consumed)


if __name__ == "__main__":
    unittest.main()
