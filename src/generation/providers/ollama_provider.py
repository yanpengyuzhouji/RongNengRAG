"""Ollama LLM Provider — 原生 Ollama /api/chat 端点
支持 Qwen3.5 think:false 完全关闭思维链
"""

import sys, os, json, re
import requests
from typing import Generator, List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """Ollama 后端 (原生 /api/chat 端点)"""

    def __init__(self, config: dict):
        self._model_name = config.get("model", "qwen3:8b")
        self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
        self.temperature = config.get("temperature", 0.1)
        self.num_ctx = config.get("num_ctx", 8192)
        self.num_predict = config.get("num_predict", 2048)
        # Qwen3.5: think=false 完全关闭思维链, None=使用模型默认
        self.think = config.get("think", None)
        self._is_thinking_model = "qwen3." in self._model_name.lower() or "qwen3:" in self._model_name.lower()

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                think_info = f"think={'OFF' if self.think is False else 'ON'}"
                print(f"[ollama] {self.base_url}, model={self._model_name} "
                      f"ctx={self.num_ctx} {think_info}")
            else:
                print(f"[warn] Ollama 连接异常: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[warn] Ollama 连接失败 ({self.base_url}): {e}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_options(self, max_tokens: int) -> dict:
        return {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": max_tokens if max_tokens else self.num_predict,
        }

    @staticmethod
    def _clean_thinking(text: str) -> str:
        """移除 <think>...</think> 标签 (Qwen3旧模型兼容)"""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    def generate(self, messages: List[Dict[str, str]], temperature: float = None,
                 max_tokens: int = 4096) -> str:
        temp = temperature if temperature is not None else self.temperature
        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "options": self._build_options(max_tokens),
        }
        if temp != self.temperature:
            payload["options"]["temperature"] = temp
        if self.think is not None:
            payload["think"] = self.think

        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama 错误: HTTP {resp.status_code}")

        data = resp.json()
        msg = data.get("message", {})
        content = msg.get("content", "") or msg.get("thinking", "")
        return self._clean_thinking(content)

    def generate_stream(self, messages: List[Dict[str, str]], temperature: float = None,
                        max_tokens: int = 4096) -> Generator[str, None, None]:
        temp = temperature if temperature is not None else self.temperature
        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": True,
            "options": self._build_options(max_tokens),
        }
        if temp != self.temperature:
            payload["options"]["temperature"] = temp
        if self.think is not None:
            payload["think"] = self.think

        resp = requests.post(f"{self.base_url}/api/chat", json=payload,
                             stream=True, timeout=(30, 600))
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama 流式错误: HTTP {resp.status_code}")

        # think=false 时直接输出, 否则先缓冲到 </think> 再输出
        if self.think is False:
            for line in resp.iter_lines():
                if not line: continue
                try:
                    chunk = json.loads(line if isinstance(line, str) else line.decode())
                    c = chunk.get("message", {}).get("content", "")
                    if c: yield c
                    if chunk.get("done", False): break
                except json.JSONDecodeError: continue
        else:
            buffer = ""; past_think = False
            for line in resp.iter_lines():
                if not line: continue
                try:
                    chunk = json.loads(line if isinstance(line, str) else line.decode())
                    c = chunk.get("message", {}).get("content", "")
                    if not c: continue
                    if not past_think:
                        buffer += c
                        if "</think>" in buffer:
                            past_think = True
                            after = buffer.split("</think>", 1)[1]
                            buffer = ""
                            if after.strip(): yield after
                    else:
                        yield c
                    if chunk.get("done", False): break
                except json.JSONDecodeError: continue
