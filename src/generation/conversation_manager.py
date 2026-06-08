"""
多轮对话管理器 — 会话存储、上下文压缩、Token 估算
- 每个会话独立存储消息历史
- 超过最大上下文窗口时自动压缩
- 保留最近 N 轮详细内容
- 旧消息压缩为摘要
"""

import uuid
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 北京时间时区
BEIJING_TZ = timezone(timedelta(hours=8))


def beijing_now() -> str:
    """返回北京时间 ISO 格式字符串"""
    return datetime.now(BEIJING_TZ).isoformat(timespec="seconds")


def beijing_now_display() -> str:
    """返回北京时间的显示格式"""
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Message:
    """单条消息"""
    role: str           # "user" | "assistant"
    content: str
    timestamp: str = ""      # 北京时间 ISO
    citations: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp or beijing_now(),
            "citations": self.citations,
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", ""),
            citations=d.get("citations", []),
            sources=d.get("sources", []),
        )


@dataclass
class Conversation:
    """会话"""
    conv_id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    messages: List[Message] = field(default_factory=list)
    summary: str = ""         # 旧消息的压缩摘要
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "conv_id": self.conv_id,
            "title": self.title or self._auto_title(),
            "created_at": self.created_at or beijing_now(),
            "updated_at": self.updated_at or beijing_now(),
            "message_count": len(self.messages),
            "total_tokens": self.total_tokens,
            "has_summary": bool(self.summary),
            "preview": self._preview(),
        }

    def _auto_title(self) -> str:
        """自动从第一条用户消息生成标题"""
        for m in self.messages:
            if m.role == "user":
                return m.content[:30] + ("..." if len(m.content) > 30 else "")
        return "新对话"

    def _preview(self) -> str:
        """会话预览文本"""
        if self.messages:
            last = self.messages[-1]
            return last.content[:50] + ("..." if len(last.content) > 50 else "")
        return ""


class ConversationManager:
    """
    多轮对话管理器
    - 内存存储 (重启丢失，后续可扩展为 SQLite)
    - 超过上下文窗口时自动压缩
    - 保留最近 N 轮详细内容
    """

    def __init__(self, config_path: str = None):
        from config import load_config
        config = load_config(config_path)

        conv_config = config.get("conversation", {})
        self.max_context_tokens = conv_config.get("max_context_tokens", 32768)
        self.keep_detail_rounds = conv_config.get("keep_detail_rounds", 3)
        self.compress_threshold = conv_config.get("compress_threshold", 0.85)

        self._conversations: Dict[str, Conversation] = {}

        print(f"[conv] 对话管理器就绪 (max_ctx={self.max_context_tokens}, "
              f"keep_rounds={self.keep_detail_rounds})")

    def create_conversation(self, title: str = "") -> str:
        """创建新会话，返回 conv_id"""
        conv_id = uuid.uuid4().hex[:12]
        now = beijing_now()
        conv = Conversation(
            conv_id=conv_id,
            title=title,
            created_at=now,
            updated_at=now,
        )
        self._conversations[conv_id] = conv
        return conv_id

    def add_message(self, conv_id: str, role: str, content: str,
                    citations: list = None, sources: list = None):
        """添加消息到会话"""
        conv = self._get_or_create(conv_id)
        msg = Message(
            role=role,
            content=content,
            timestamp=beijing_now(),
            citations=citations or [],
            sources=sources or [],
            token_count=self._estimate_tokens(content),
        )
        conv.messages.append(msg)
        conv.total_tokens += msg.token_count
        conv.updated_at = beijing_now()

        # 自动设置标题
        if not conv.title and role == "user" and len(conv.messages) == 1:
            conv.title = content[:30] + ("..." if len(content) > 30 else "")

        # 检查是否需要压缩
        if conv.total_tokens > self.max_context_tokens * self.compress_threshold:
            self._compress(conv)

    def get_context_messages(self, conv_id: str) -> List[Dict[str, str]]:
        """
        获取用于 LLM 的消息上下文
        包含: 系统摘要(如有) + 最近 N 轮详细消息
        """
        conv = self._get_or_create(conv_id)
        messages = []

        # 如果有旧消息摘要，作为 system 消息插入
        if conv.summary:
            messages.append({
                "role": "system",
                "content": f"[历史对话摘要] 以下是之前讨论的要点:\n{conv.summary}"
            })

        # 最近 N 轮详细消息 (N轮 = N*2 条消息)
        detail_count = self.keep_detail_rounds * 2
        recent = conv.messages[-detail_count:] if len(conv.messages) > detail_count else conv.messages

        for m in recent:
            messages.append({
                "role": m.role,
                "content": m.content,
            })

        return messages

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        """获取会话完整数据"""
        conv = self._conversations.get(conv_id)
        if not conv:
            return None
        d = conv.to_dict()
        d["messages"] = [m.to_dict() for m in conv.messages]
        d["summary"] = conv.summary
        return d

    def list_conversations(self) -> List[dict]:
        """列出所有会话 (摘要)"""
        result = []
        for conv in self._conversations.values():
            result.append(conv.to_dict())
        # 按更新时间倒序
        result.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return result

    def delete_conversation(self, conv_id: str) -> bool:
        """删除会话"""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            return True
        return False

    def _get_or_create(self, conv_id: str) -> Conversation:
        """获取或创建会话"""
        if conv_id not in self._conversations:
            conv = Conversation(
                conv_id=conv_id,
                created_at=beijing_now(),
                updated_at=beijing_now(),
            )
            self._conversations[conv_id] = conv
        return self._conversations[conv_id]

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数 (中文~1.5字符/token, 英文~4字符/token)"""
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def _compress(self, conv: Conversation):
        """
        压缩历史: 保留最近 N 轮详细消息, 旧消息合并为摘要
        使用简单的提取式摘要 (后续可升级为 LLM 摘要)
        """
        keep_count = self.keep_detail_rounds * 2  # N轮 = N*2 条消息
        if len(conv.messages) <= keep_count:
            return

        old_messages = conv.messages[:-keep_count]

        # 提取式摘要: 收集旧消息中的关键信息
        summary_parts = []
        for m in old_messages:
            if m.role == "user":
                summary_parts.append(f"用户询问: {m.content[:100]}")
            else:
                # assistant: 提取可能的要点 (按句号分)
                sentences = m.content.replace("；", "。").replace("；", "。").split("。")
                key_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
                if key_sentences:
                    summary_parts.append(f"回答要点: {'; '.join(key_sentences[:2])}")

        conv.summary = "\n".join(summary_parts[:20])  # 限制摘要长度

        # 更新消息列表
        old_token_count = sum(m.token_count for m in old_messages)
        conv.messages = conv.messages[-keep_count:]
        conv.total_tokens -= old_token_count
        conv.total_tokens += self._estimate_tokens(conv.summary)

        print(f"   [conv] 压缩会话 {conv.conv_id}: "
              f"tokens {conv.total_tokens + old_token_count} → {conv.total_tokens}")
