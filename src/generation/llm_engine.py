"""
LLM 推理引擎 — 支持 Ollama / 阿里云百炼 / llama-cpp-python 三种后端
通过 Provider 模式实现后端解耦
"""

import os
import re
from typing import Optional, Generator, List, Dict


class LLMEngine:
    """LLM 推理引擎 (Provider 模式 facade)"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        llm_config = self.config["llm"]
        self.provider_name = llm_config.get("provider", "ollama")

        if self.provider_name == "ollama":
            from generation.providers.ollama_provider import OllamaProvider
            self._provider = OllamaProvider(llm_config.get("ollama", {}))
        elif self.provider_name == "bailian":
            from generation.providers.bailian_provider import BailianProvider
            self._provider = BailianProvider(llm_config.get("bailian", {}))
        elif self.provider_name == "llama_cpp":
            self._init_llama_cpp(llm_config)
            self._provider = None  # llama-cpp 不走 provider 接口
        else:
            raise ValueError(f"不支持的 LLM 后端: {self.provider_name}")

        # 上下文字段
        self.max_context = llm_config.get("num_ctx", 32768)
        if self.provider_name == "ollama":
            self.max_context = llm_config.get("ollama", {}).get("num_ctx", 32768)
        elif self.provider_name == "bailian":
            self.max_context = llm_config.get("bailian", {}).get("max_tokens", 4096) * 4

        self._loaded = True

    def _init_llama_cpp(self, llm_config: dict):
        """初始化 llama-cpp-python 后端"""
        from llama_cpp import Llama

        model_dir = self.config["paths"]["models_dir"]
        self._llama_model = Llama(
            model_path=os.path.join(
                model_dir, llm_config.get("model_file", "qwen2.5-7b-instruct-q4_k_m.gguf")
            ),
            n_ctx=llm_config.get("num_ctx", 32768),
            n_threads=llm_config.get("n_threads", 16),
            n_gpu_layers=llm_config.get("n_gpu_layers", -1),
            verbose=False,
        )
        self._model_name = "qwen2.5-7b (llama.cpp)"

    @property
    def model_name(self) -> str:
        if self.provider_name == "llama_cpp":
            return self._model_name
        return self._provider.model_name

    def generate(self, prompt: str, system: str = None, temperature: float = 0.1,
                 max_tokens: int = 4096, stream: bool = False) -> str:
        """生成回答
        max_tokens=4096: Qwen3 推理模型需要足够 token 预算给 thinking + content
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.provider_name == "llama_cpp":
            return self._generate_llama_cpp(prompt, system, temperature, max_tokens, stream)

        if stream:
            full = ""
            for token in self._provider.generate_stream(messages, temperature, max_tokens):
                full += token
            return full
        else:
            return self._provider.generate(messages, temperature, max_tokens)

    def generate_chat(self, messages: List[Dict[str, str]], temperature: float = 0.1,
                      max_tokens: int = 3072) -> str:
        """多轮对话生成 (传入完整消息列表)"""
        if self.provider_name == "llama_cpp":
            # 将 messages 转换为 llama-cpp prompt 格式
            prompt = self._messages_to_prompt(messages)
            return self._generate_llama_cpp(prompt, None, temperature, max_tokens, False)

        return self._provider.generate(messages, temperature, max_tokens)

    def generate_chat_stream(self, messages: List[Dict[str, str]], temperature: float = 0.1,
                             max_tokens: int = 3072) -> Generator[str, None, None]:
        """多轮对话流式生成"""
        if self.provider_name == "llama_cpp":
            prompt = self._messages_to_prompt(messages)
            for chunk in self._generate_llama_cpp_stream(prompt, None, temperature, max_tokens):
                yield chunk
            return

        for token in self._provider.generate_stream(messages, temperature, max_tokens):
            yield token

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将 OpenAI 格式消息列表转为 llama-cpp 的 prompt 格式"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _generate_llama_cpp(self, prompt: str, system: str = None,
                            temperature: float = 0.1, max_tokens: int = 4096,
                            stream: bool = False) -> str:
        """通过 llama-cpp-python 生成"""
        full_prompt = ""
        if system:
            full_prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        output = self._llama_model(
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

    def _generate_llama_cpp_stream(self, prompt: str, system: str = None,
                                   temperature: float = 0.1, max_tokens: int = 4096):
        """llama-cpp 流式生成"""
        full_prompt = ""
        if system:
            full_prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        output = self._llama_model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],
            echo=False,
            stream=True,
        )
        for chunk in output:
            text = chunk["choices"][0].get("text", "")
            if text:
                yield text

    def generate_rag_answer(self, query: str, context: str, query_type: str,
                            context_domain1: str = None,
                            context_domain2: str = None,
                            max_tokens: int = None) -> str:
        """
        基于检索结果生成 RAG 回答
        使用领域提示词模板 + 系统提示抑制过度思考
        """
        from generation.prompt_templates import get_prompt, get_system_prompt

        prompt = get_prompt(
            query_type=query_type,
            context=context,
            query=query,
            context_domain1=context_domain1,
            context_domain2=context_domain2,
        )

        # Qwen3 推理模型: thinking 消耗大部分 token, 需要足够的 token 预算
        mt = max_tokens if max_tokens is not None else 4096
        return self.generate(prompt, system=get_system_prompt(query_type),
                           temperature=0.1, max_tokens=mt)

    def generate_rag_answer_stream(self, query: str, context: str, query_type: str,
                                   context_domain1: str = None,
                                   context_domain2: str = None,
                                   max_tokens: int = None
                                   ) -> Generator[str, None, None]:
        """基于检索结果流式生成 RAG 回答"""
        from generation.prompt_templates import get_prompt, get_system_prompt

        prompt = get_prompt(
            query_type=query_type,
            context=context,
            query=query,
            context_domain1=context_domain1,
            context_domain2=context_domain2,
        )

        mt = max_tokens if max_tokens is not None else 4096

        # 构建消息: 系统提示 + 用户prompt
        system = get_system_prompt(query_type)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.provider_name == "llama_cpp":
            for token in self._generate_llama_cpp_stream(prompt, system, 0.1, mt):
                yield token
        else:
            for token in self._provider.generate_stream(messages, 0.1, mt):
                yield token

    def extract_citations(self, answer: str) -> list:
        """从回答中提取引用信息"""
        citations = re.findall(r'【([^】]+)】', answer)
        return list(set(citations))
