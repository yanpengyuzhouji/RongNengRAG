"""
LLM 推理引擎 — 通过 Provider 模式调用 OpenAI 兼容接口
支持任何暴露 /v1/chat/completions 的服务 (vLLM / Xinference / PaddleX serving / 云厂商兼容API)
"""

import re
from typing import Optional, Generator, List, Dict


class LLMEngine:
    """LLM 推理引擎 (Provider 模式 facade)"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        llm_config = self.config["llm"]
        self.provider_name = llm_config.get("provider", "openai")

        if self.provider_name == "openai":
            from generation.providers.openai_compat_provider import OpenAICompatProvider
            self._provider = OpenAICompatProvider(llm_config.get("openai", {}))
        else:
            raise ValueError(f"不支持的 LLM 后端: {self.provider_name} "
                             f"(仅支持 'openai' OpenAI兼容接口)")

        # 上下文字段
        self.max_context = llm_config.get("openai", {}).get("max_tokens", 4096) * 4
        self._loaded = True

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    def health_check(self) -> dict:
        """Check whether the configured LLM endpoint is reachable."""
        return self._provider.health_check()

    def generate(self, prompt: str, system: str = None, temperature: float = 0.1,
                 max_tokens: int = 4096, stream: bool = False) -> str:
        """生成回答"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

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
        return self._provider.generate(messages, temperature, max_tokens)

    def generate_chat_stream(self, messages: List[Dict[str, str]], temperature: float = 0.1,
                             max_tokens: int = 3072) -> Generator[str, None, None]:
        """多轮对话流式生成"""
        for token in self._provider.generate_stream(messages, temperature, max_tokens):
            yield token

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

        for token in self._provider.generate_stream(messages, 0.1, mt):
            yield token

    def extract_citations(self, answer: str) -> list:
        """从回答中提取引用信息"""
        citations = re.findall(r'【([^】]+)】', answer)
        return list(set(citations))
