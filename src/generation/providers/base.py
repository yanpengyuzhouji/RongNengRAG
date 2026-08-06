"""LLM Provider abstract base class"""
from abc import ABC, abstractmethod
from typing import Generator, List, Dict


class BaseProvider(ABC):
    """所有 LLM 后端的统一接口"""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.1,
                 max_tokens: int = 4096) -> str:
        """非流式生成"""
        ...

    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float = 0.1,
                        max_tokens: int = 4096) -> Generator[str, None, None]:
        """流式生成，逐 token yield"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回当前模型名称"""
        ...
