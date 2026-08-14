"""OpenAI 兼容 LLM Provider — 调用任何 OpenAI 兼容的 /v1/chat/completions 服务
适用于: vLLM / Xinference / LM Studio / PaddleX serving / 本地 llama.cpp server 等
"""
import os
import sys
from typing import Generator, List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from .base import BaseProvider


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
            timeout=config.get("timeout", 300),
        )

        print(f"[llm] OpenAI兼容服务: {self.base_url}, model={self._model_name}"
              f"{' (无API Key)' if not self.api_key else ''}, "
              f"thinking={'on' if self.enable_thinking else 'off'}")

    @property
    def model_name(self) -> str:
        return self._model_name

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
        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content

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
        response = self.client.chat.completions.create(**request)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
