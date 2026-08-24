import unittest

from src.generation.providers.openai_compat_provider import (
    LLMServiceError,
    OpenAICompatProvider,
)


class _NotFoundError(Exception):
    status_code = 404


class _Completions:
    def create(self, **_kwargs):
        raise OSError("connection refused")


class _Client:
    class chat:
        completions = _Completions()


class LLMServiceErrorTests(unittest.TestCase):
    def setUp(self):
        # Bypass SDK construction; these tests only exercise local error handling.
        self.provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
        self.provider.base_url = "http://192.168.0.201:18000/v1"
        self.provider._model_name = "Qwen3-14B"
        self.provider.timeout_seconds = 30

    def test_connection_error_has_actionable_message(self):
        error = self.provider._service_error(OSError("connection refused"))
        self.assertEqual("llm_unreachable", error.code)
        self.assertIn("18000", str(error))

    def test_not_found_mentions_endpoint_and_model(self):
        error = self.provider._service_error(_NotFoundError())
        self.assertEqual("llm_not_found", error.code)
        self.assertIn("Qwen3-14B", str(error))

    def test_generate_wraps_connection_error(self):
        self.provider.client = _Client()
        self.provider.temperature = 0.1
        self.provider.max_tokens_default = 32
        self.provider.enable_thinking = False

        with self.assertRaises(LLMServiceError) as context:
            self.provider.generate([{"role": "user", "content": "hello"}])
        self.assertEqual("llm_unreachable", context.exception.code)


if __name__ == "__main__":
    unittest.main()
