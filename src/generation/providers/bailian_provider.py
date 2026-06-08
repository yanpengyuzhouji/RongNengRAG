"""阿里云百炼 (DashScope) LLM Provider — 通过 OpenAI 兼容 API 调用"""
import os
import sys
from typing import Generator, List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from .base import BaseProvider


class BailianProvider(BaseProvider):
    """
    阿里云百炼大模型后端 (OpenAI 兼容接口)
    需设置环境变量 DASHSCOPE_API_KEY
    API 地址: https://dashscope.aliyuncs.com/compatible-mode/v1
    """

    # 常用模型列表
    KNOWN_MODELS = {
        "qwen-turbo": "通义千问-Turbo (经济型)",
        "qwen-plus": "通义千问-Plus (均衡型)",
        "qwen-max": "通义千问-Max (最强)",
        "qwen3-235b-a22b": "通义千问3-235B",
        "deepseek-r1": "DeepSeek-R1 (百炼)",
        "deepseek-v3": "DeepSeek-V3 (百炼)",
    }

    def __init__(self, config: dict):
        from openai import OpenAI

        self._model_name = config.get("model", "qwen-plus")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens_default = config.get("max_tokens", 4096)

        # API Key: 从配置的 env var 名称或直接取环境变量
        api_key_env = config.get("api_key_env", "DASHSCOPE_API_KEY")
        self.api_key = os.environ.get(api_key_env, "")

        if not self.api_key:
            print(f"[warn] 百炼 API Key 未设置!")
            print(f"       请在 .env 文件中设置 {api_key_env}=sk-your-key")
            print(f"       或设置环境变量: set {api_key_env}=sk-your-key")

        self.base_url = config.get(
            "base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "missing-api-key",
        )

        print(f"[bailian] 模型: {self._model_name} "
              f"({'已配置' if self.api_key else '⚠ 未配置 API Key'})")

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, messages: List[Dict[str, str]], temperature: float = None,
                 max_tokens: int = None) -> str:
        """非流式生成"""
        if not self.api_key:
            raise RuntimeError(
                "百炼 API Key 未设置。请在 .env 文件中添加 DASHSCOPE_API_KEY=sk-your-key"
            )

        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens_default

        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temp,
            max_tokens=mt,
            stream=False,
        )
        return response.choices[0].message.content

    def generate_stream(self, messages: List[Dict[str, str]], temperature: float = None,
                        max_tokens: int = None) -> Generator[str, None, None]:
        """流式生成"""
        if not self.api_key:
            raise RuntimeError(
                "百炼 API Key 未设置。请在 .env 文件中添加 DASHSCOPE_API_KEY=sk-your-key"
            )

        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens_default

        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temp,
            max_tokens=mt,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
