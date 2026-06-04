"""
嵌入引擎 — 使用 BGE-M3 生成稠密 + 稀疏向量
支持批量处理、进度显示、ONNX 加速
"""

import time
import yaml
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    chunk_ids: List[str]
    dense_vectors: list         # List[List[float]]  稠密向量
    sparse_vectors: list        # List[dict] 稀疏向量 (词ID: 权重)


class Embedder:
    """BGE-M3 嵌入模型封装"""

    def __init__(self, config_path: str = "D:/rag-system/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        emb_config = self.config["embedding"]
        self.model_name = emb_config["model_name"]
        self.device = emb_config["device"]
        self.batch_size = emb_config["batch_size"]
        self.normalize = emb_config["normalize"]
        self.max_length = emb_config["max_length"]
        self.use_onnx = emb_config.get("use_onnx", False)

        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._loaded:
            return

        print(f"📥 加载嵌入模型: {self.model_name} ...")
        print(f"   设备: {self.device}, 批量大小: {self.batch_size}")

        if self.use_onnx:
            self._load_onnx()
        else:
            self._load_sentence_transformers()

        self._loaded = True

    def _load_sentence_transformers(self):
        """使用 sentence-transformers 加载"""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
        )

        # 预热
        print("   🔥 预热模型...")
        _ = self.model.encode(["预热文本"], show_progress_bar=False)

    def _load_onnx(self):
        """使用 ONNX Runtime 加载（CPU 优化）"""
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            self.model = ORTModelForFeatureExtraction.from_pretrained(
                self.model_name,
                export=True,
                provider="CPUExecutionProvider",
                file_name="model.onnx",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except ImportError:
            print("⚠ ONNX Runtime 未安装，回退到 sentence-transformers")
            self._load_sentence_transformers()

    def encode(self, texts: List[str], show_progress: bool = True) -> EmbeddingResult:
        """
        批量生成稠密 + 稀疏嵌入向量
        返回 EmbeddingResult
        """
        self._ensure_loaded()

        all_dense = []
        all_sparse = []

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # 稠密向量
            dense = self.model.encode(
                batch,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                batch_size=len(batch),
            )
            all_dense.extend(dense.tolist() if hasattr(dense, 'tolist') else dense)

            # 稀疏向量 (BGE-M3 特有)
            try:
                sparse = self.model.encode(
                    batch,
                    return_sparse=True,
                    show_progress_bar=False,
                    batch_size=len(batch),
                )
                all_sparse.extend(sparse)
            except Exception:
                # 回退：生成空的稀疏向量
                all_sparse.extend([{} for _ in batch])

            if show_progress and total_batches > 1:
                progress = min(i + self.batch_size, len(texts))
                print(f"   📊 嵌入进度: {progress}/{len(texts)} "
                      f"({progress * 100 // len(texts)}%)")

        return EmbeddingResult(
            chunk_ids=[],  # 由调用方填充
            dense_vectors=all_dense,
            sparse_vectors=all_sparse,
        )

    def encode_query(self, query: str) -> tuple:
        """
        对单条查询编码
        返回 (dense_vector, sparse_vector)
        """
        self._ensure_loaded()

        dense = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        dense_vec = dense[0].tolist() if hasattr(dense[0], 'tolist') else dense[0]

        try:
            sparse = self.model.encode(
                [query],
                return_sparse=True,
                show_progress_bar=False,
            )
            sparse_vec = sparse[0]
        except Exception:
            sparse_vec = {}

        return dense_vec, sparse_vec


def create_text_for_embedding(chunk) -> str:
    """
    为嵌入生成优化的文本表示
    拼接元数据 + 正文，提升检索质量
    """
    from chunker import Chunk

    parts = []

    # 元数据富化（帮助向量模型理解上下文）
    meta_str = chunk.get_metadata_str()
    if meta_str:
        parts.append(f"【{meta_str}】")

    # 正文
    parts.append(chunk.text)

    return " ".join(parts)
