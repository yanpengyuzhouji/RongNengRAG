"""
LLM 推理引擎 — 支持 Ollama 和 llama-cpp-python 两种后端
默认使用 Ollama + Qwen2.5-14B-Instruct
"""

import yaml
import re
from typing import Optional, Generator


class LLMEngine:
    """LLM 推理引擎封装"""

    def __init__(self, config_path: str = "D:/rag-system/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        llm_config = self.config["llm"]
        self.provider = llm_config["provider"]

        if self.provider == "ollama":
            self._init_ollama(llm_config["ollama"])
        elif self.provider == "llama_cpp":
            self._init_llama_cpp(llm_config)
        else:
            raise ValueError(f"不支持的 LLM 后端: {self.provider}")

        self._loaded = True

    def _init_ollama(self, ollama_config: dict):
        """初始化 Ollama 后端"""
        from openai import OpenAI

        self.ollama_config = ollama_config
        self.model_name = ollama_config["model"]
        self.client = OpenAI(
            base_url=ollama_config["base_url"] + "/v1",
            api_key="ollama",  # Ollama 不需要真实 key
        )

    def _init_llama_cpp(self, llm_config: dict):
        """初始化 llama-cpp-python 后端"""
        from llama_cpp import Llama

        self.model = Llama(
            model_path=f"D:/rag-system/models/{llm_config.get('model_file', 'qwen2.5-7b-instruct-q4_k_m.gguf')}",
            n_ctx=llm_config.get("num_ctx", 32768),
            n_threads=llm_config.get("n_threads", 16),
            n_gpu_layers=llm_config.get("n_gpu_layers", -1),
            verbose=False,
        )
        self.model_name = "qwen2.5-7b (llama.cpp)"

    def generate(self, prompt: str, system: str = None, temperature: float = 0.1,
                 max_tokens: int = 4096, stream: bool = False) -> str:
        """
        生成回答
        """
        if self.provider == "ollama":
            return self._generate_ollama(prompt, system, temperature, max_tokens, stream)
        else:
            return self._generate_llama_cpp(prompt, system, temperature, max_tokens, stream)

    def _generate_ollama(self, prompt: str, system: str = None,
                         temperature: float = 0.1, max_tokens: int = 4096,
                         stream: bool = False) -> str:
        """通过 Ollama API 生成"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if stream:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            full_text = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
            return full_text
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

    def _generate_llama_cpp(self, prompt: str, system: str = None,
                            temperature: float = 0.1, max_tokens: int = 4096,
                            stream: bool = False) -> str:
        """通过 llama-cpp-python 生成"""
        full_prompt = ""
        if system:
            full_prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        output = self.model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],
            echo=False,
            stream=stream,
        )

        if stream:
            full_text = ""
            for chunk in output:
                text = chunk["choices"][0].get("text", "")
                full_text += text
            return full_text
        else:
            return output["choices"][0]["text"]

    def generate_rag_answer(self, query: str, context: str, query_type: str,
                            context_domain1: str = None,
                            context_domain2: str = None) -> str:
        """
        基于检索结果生成 RAG 回答
        使用领域提示词模板
        """
        from generation.prompt_templates import get_prompt

        prompt = get_prompt(
            query_type=query_type,
            context=context,
            query=query,
            context_domain1=context_domain1,
            context_domain2=context_domain2,
        )

        return self.generate(prompt, temperature=0.1)

    def extract_citations(self, answer: str) -> list:
        """从回答中提取引用信息"""
        # 匹配引用格式: 【XXX】
        citations = re.findall(r'【([^】]+)】', answer)
        return list(set(citations))
