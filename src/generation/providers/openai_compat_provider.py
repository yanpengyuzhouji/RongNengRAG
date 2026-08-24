"""OpenAI 兼容 LLM Provider — 调用任何 OpenAI 兼容的 /v1/chat/completions 服务
适用于: vLLM / Xinference / LM Studio / PaddleX serving / 本地 llama.cpp server 等
"""
import os
import sys
from typing import Any, Generator, List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from .base import BaseProvider


class LLMServiceError(RuntimeError):
    """A safe, actionable error raised when the configured LLM cannot serve a request."""

    def __init__(self, message: str, code: str = "llm_unavailable",
                 retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class OpenAICompatProvider(BaseProvider):
    """
    OpenAI 兼容大模型后端
    通过 base_url / model / api_key 三项配置指向任意 OpenAI 兼容服务
    api_key 可留空 (本地服务通常不校验)
    """

    def __init__(self, config: dict):
        from openai import OpenAI

        self._model_name = config.get("model", "Qwen3-4B")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens_default = config.get("max_tokens", 4096)
        self.timeout_seconds = config.get("timeout", 300)
        self.health_timeout_seconds = config.get("health_timeout", 5)
        # Qwen3-compatible servers (including vLLM) accept this OpenAI
        # extension. Keep it configurable because unrelated providers may
        # reject extra request fields.
        self.enable_thinking = bool(config.get("enable_thinking", False))

        self.base_url = config.get(
            "base_url",
            "http://localhost:8000/v1",
        ).rstrip("/")

        # API Key: 从配置的 env var 名称或直接取环境变量
        api_key_env = config.get("api_key_env", "LLM_API_KEY")
        self.api_key = os.environ.get(api_key_env, "") or config.get("api_key", "")

        # 本地服务通常不校验 key, 空 key 用占位符避免 SDK 报错
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "sk-empty",
            timeout=self.timeout_seconds,
        )

        print(f"[llm] OpenAI兼容服务: {self.base_url}, model={self._model_name}"
              f"{' (无API Key)' if not self.api_key else ''}, "
              f"thinking={'on' if self.enable_thinking else 'off'}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def _service_error(self, exc: Exception) -> LLMServiceError:
        """Convert SDK/network errors to messages safe to show in the UI."""
        name = type(exc).__name__
        response = getattr(exc, "response", None)
        status_code = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
        if name in {"APITimeoutError", "TimeoutException", "ReadTimeout", "ConnectTimeout"}:
            return LLMServiceError(
                f"LLM 服务响应超时（{self.timeout_seconds} 秒）。请检查模型负载或调小输出长度。",
                code="llm_timeout",
            )
        if name in {"APIConnectionError", "ConnectError", "ConnectionError"} or isinstance(exc, OSError):
            return LLMServiceError(
                f"LLM 服务不可达：{self.base_url}。请检查地址、端口和 vLLM 服务状态。",
                code="llm_unreachable",
            )
        if status_code in (401, 403):
            return LLMServiceError("LLM 鉴权失败，请检查 API Key 配置。", code="llm_auth", retryable=False)
        if status_code == 404:
            return LLMServiceError(
                f"LLM 接口或模型不存在，请检查 base_url={self.base_url} 和 model={self._model_name}。",
                code="llm_not_found",
                retryable=False,
            )
        if status_code == 429:
            return LLMServiceError("LLM 服务繁忙，请稍后重试或降低并发。", code="llm_busy")
        if isinstance(status_code, int) and status_code >= 500:
            return LLMServiceError(
                f"LLM 服务返回 {status_code}，请检查 vLLM 日志和显存状态。",
                code="llm_server_error",
            )
        return LLMServiceError(
            f"LLM 调用失败（{name}）。请检查 LLM 服务配置和运行日志。",
            code="llm_error",
        )

    def health_check(self) -> Dict[str, Any]:
        """Probe the OpenAI-compatible model endpoint with a short timeout."""
        import httpx

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = httpx.get(
                f"{self.base_url}/models", headers=headers,
                timeout=self.health_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            models = [item.get("id", "") for item in payload.get("data", [])]
            return {
                "available": True,
                "model": self._model_name,
                "configured_model_found": self._model_name in models if models else None,
                "models": models,
            }
        except Exception as exc:
            error = self._service_error(exc)
            return {
                "available": False,
                "model": self._model_name,
                "error_code": error.code,
                "message": str(error),
            }

    def generate(self, messages: List[Dict[str, str]], temperature: float = None,
                 max_tokens: int = None) -> str:
        """非流式生成"""
        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens_default

        request = dict(
            model=self._model_name, messages=messages, temperature=temp,
            max_tokens=mt, stream=False,
        )
        if not self.enable_thinking:
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            response = self.client.chat.completions.create(**request)
            content = response.choices[0].message.content if response.choices else None
            if not content:
                raise LLMServiceError("LLM 服务未返回有效回答。", code="llm_invalid_response")
            return content
        except LLMServiceError:
            raise
        except Exception as exc:
            raise self._service_error(exc) from exc

    def generate_stream(self, messages: List[Dict[str, str]], temperature: float = None,
                        max_tokens: int = None) -> Generator[str, None, None]:
        """流式生成"""
        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens_default

        request = dict(
            model=self._model_name, messages=messages, temperature=temp,
            max_tokens=mt, stream=True,
        )
        if not self.enable_thinking:
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            response = self.client.chat.completions.create(**request)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except LLMServiceError:
            raise
        except Exception as exc:
            raise self._service_error(exc) from exc
