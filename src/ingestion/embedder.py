"""
嵌入引擎 — 使用本地 Ollama 模型生成稠密向量
支持: Ollama API (默认) / sentence-transformers (回退)
"""

import time
import yaml
import requests
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    chunk_ids: List[str]
    dense_vectors: list         # List[List[float]]  稠密向量
    sparse_vectors: list        # List[dict] 稀疏向量 (Ollama 不支持，返回空字典)


class Embedder:
    """嵌入模型封装 (Ollama 优先)"""

    def __init__(self, config_path: str = "D:/rag-system/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        emb_config = self.config["embedding"]
        self.provider = emb_config.get("provider", "ollama")

        if self.provider == "ollama":
            ollama_cfg = emb_config.get("ollama", {})
            self.ollama_model = ollama_cfg.get("model", "bge-m3:latest")
            self.ollama_url = ollama_cfg.get("base_url", "http://localhost:11434")
        else:
            # sentence_transformers 回退
            self.model_name = emb_config["model_name"]
            self.device = emb_config["device"]
            self.batch_size = emb_config["batch_size"]
            self.normalize = emb_config["normalize"]
            self.max_length = emb_config["max_length"]
            self.use_onnx = emb_config.get("use_onnx", False)

        self.batch_size = emb_config.get("batch_size", 32)
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._loaded:
            return

        if self.provider == "ollama":
            self._init_ollama()
        else:
            self._load_sentence_transformers()

        self._loaded = True

    def _init_ollama(self):
        """验证 Ollama 服务可用"""
        print(f"[Ollama] 连接嵌入服务: {self.ollama_url}")
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama 返回状态码 {resp.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"无法连接 Ollama ({self.ollama_url})。请先启动 Ollama 服务。"
            )

        # 验证嵌入模型可用
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.ollama_model, "prompt": "测试"},
                timeout=30,
            )
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama 嵌入接口返回 {resp.status_code}: {resp.text}")
            data = resp.json()
            dim = len(data.get("embedding", []))
            print(f"   [OK] Ollama 嵌入模型就绪: {self.ollama_model} (维度: {dim})")
        except Exception as e:
            raise ConnectionError(
                f"Ollama 嵌入模型 {self.ollama_model} 不可用: {e}\n"
                f"请确保已安装: ollama pull {self.ollama_model}"
            )

    def _load_sentence_transformers(self):
        """使用 sentence-transformers 加载 (HuggingFace 回退)"""
        from sentence_transformers import SentenceTransformer

        print(f"[加载] 嵌入模型: {self.model_name} ...")
        print(f"   设备: {self.device}, 批量大小: {self.batch_size}")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
        )

        # 预热
        print("   [预热] 模型预热中...")
        _ = self.model.encode(["预热文本"], show_progress_bar=False)

    def _ollama_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """通过 Ollama API 批量生成嵌入向量"""
        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.ollama_model, "input": texts},
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama 嵌入失败: {resp.status_code} {resp.text}")
        return resp.json()["embeddings"]

    def encode(self, texts: List[str], show_progress: bool = True) -> EmbeddingResult:
        """
        批量生成稠密 + 稀疏嵌入向量
        返回 EmbeddingResult
        """
        self._ensure_loaded()

        all_dense = []
        all_sparse = []

        if self.provider == "ollama":
            # Ollama 批量嵌入
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                dense_batch = self._ollama_embed_batch(batch)
                all_dense.extend(dense_batch)
                # Ollama 不支持稀疏向量
                all_sparse.extend([{} for _ in batch])

                if show_progress and total_batches > 1:
                    progress = min(i + self.batch_size, len(texts))
                    print(f"   [嵌入] 进度: {progress}/{len(texts)} "
                          f"({progress * 100 // len(texts)}%)")
        else:
            # sentence_transformers 路径
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]

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
                    all_sparse.extend([{} for _ in batch])

                if show_progress and total_batches > 1:
                    progress = min(i + self.batch_size, len(texts))
                    print(f"   [嵌入] 进度: {progress}/{len(texts)} "
                          f"({progress * 100 // len(texts)}%)")

        return EmbeddingResult(
            chunk_ids=[],
            dense_vectors=all_dense,
            sparse_vectors=all_sparse,
        )

    def encode_query(self, query: str) -> tuple:
        """
        对单条查询编码
        返回 (dense_vector, sparse_vector)
        """
        self._ensure_loaded()

        if self.provider == "ollama":
            resp = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.ollama_model, "prompt": query},
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama 嵌入失败: {resp.status_code} {resp.text}")
            dense_vec = resp.json()["embedding"]
            sparse_vec = {}  # Ollama 不支持稀疏向量
        else:
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
    parts = []

    meta_str = chunk.get_metadata_str()
    if meta_str:
        parts.append(f"【{meta_str}】")

    parts.append(chunk.text)

    return " ".join(parts)
