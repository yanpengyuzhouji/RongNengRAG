"""
嵌入引擎 — 使用本地 BGE-M3 生成稠密 + 稀疏向量
默认后端: sentence_transformers (GPU, dense + sparse)
回退后端: Ollama (dense only)
"""

import os
import time
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    chunk_ids: List[str]
    dense_vectors: list          # List[List[float]]  稠密向量 (1024维)
    sparse_vectors: list         # List[dict] 或 List[ndarray] 稀疏向量


class Embedder:
    """BGE-M3 嵌入模型封装 (本地sentence-transformers优先, 稀疏+稠密)"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        emb_config = self.config["embedding"]
        self.provider = emb_config.get("provider", "sentence_transformers")

        # HF 镜像设置
        hf_home = emb_config.get("hf_home", "")
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
        hf_endpoint = emb_config.get("hf_endpoint", "")
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        self.model_name = emb_config.get("model_name", "BAAI/bge-m3")
        self.device = emb_config.get("device", "cuda")
        self.batch_size = emb_config.get("batch_size", 32)
        self.normalize = emb_config.get("normalize", True)
        self.max_length = emb_config.get("max_length", 8192)

        # Ollama 回退
        if self.provider == "ollama":
            ollama_cfg = emb_config.get("ollama", {})
            self.ollama_model = ollama_cfg.get("model", "bge-m3:latest")
            self.ollama_url = ollama_cfg.get("base_url", "http://localhost:11434")

        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        if self.provider == "ollama":
            self._init_ollama()
        else:
            self._load_sentence_transformers()

        self._loaded = True
        print(f"[embed] 模型就绪: {self.model_name} ({self.provider})")

    def unload(self):
        """释放 BGE-M3 显存 — OCR进程池/LLM调度前调用

        参照 Reranker.unload() 模式:
          - del model + gc.collect() + torch.cuda.empty_cache()
          - Windows WDDM下 empty_cache() 不会将显存归还OS，
            但释放后的内存在同一CUDA context内可被后续分配复用。
            对OCR子进程而言，WDDM会将释放的显存标记为可回收缓存，
            子进程的独立CUDA context可以申请使用这部分显存。
        """
        if not self._loaded or self.model is None:
            return
        try:
            import torch
            import gc
            del self.model
            self.model = None
            self._loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("   [embed] BGE-M3 已卸载，显存已释放")
        except Exception as e:
            print(f"   [embed] 卸载异常: {e}")

    def reload(self):
        """重新加载 BGE-M3 — OCR完成后调用，等同于 _ensure_loaded()"""
        self._ensure_loaded()

    def _init_ollama(self):
        import requests
        print(f"[embed] 使用 Ollama: {self.ollama_url}")
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama status={resp.status_code}")
        except requests.ConnectionError:
            raise ConnectionError(f"无法连接 Ollama ({self.ollama_url})")
        print(f"   [OK] Ollama 嵌入: {self.ollama_model}")

    def _load_sentence_transformers(self):
        from sentence_transformers import SentenceTransformer
        print(f"[embed] 加载本地模型: {self.model_name}")
        print(f"   device={self.device} batch={self.batch_size}")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
        )

        # 预热
        print("   [预热] ...")
        _ = self.model.encode(["预热"], show_progress_bar=False)
        print("   [OK] 模型加载完成")

    def encode(self, texts: List[str], show_progress: bool = True) -> EmbeddingResult:
        """批量生成稠密 + 稀疏嵌入向量"""
        self._ensure_loaded()

        all_dense = []
        all_sparse = []

        if self.provider == "ollama":
            return self._encode_ollama(texts, show_progress)

        # sentence_transformers 路径
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]

            # 稠密向量
            dense = self.model.encode(
                batch,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                batch_size=len(batch),
            )
            all_dense.extend(dense.tolist() if hasattr(dense, 'tolist') else dense)

            # 稀疏向量 (BGE-M3 原生支持)
            try:
                sparse_result = self.model.encode(
                    batch,
                    return_sparse=True,
                    show_progress_bar=False,
                    batch_size=len(batch),
                )
                # 统一转为 {int: float} dict 格式 (Milvus 兼容)
                for s in sparse_result:
                    if isinstance(s, dict):
                        all_sparse.append(s)
                    elif hasattr(s, 'todense'):
                        arr = s.todense().flatten()
                        all_sparse.append({j: float(arr[j]) for j in range(len(arr)) if arr[j] != 0})
                    elif isinstance(s, np.ndarray):
                        all_sparse.append({j: float(s[j]) for j in range(len(s)) if s[j] != 0})
                    else:
                        all_sparse.append({})
            except Exception as e:
                print(f"   [warn] 稀疏向量生成失败: {e}")
                all_sparse.extend([{} for _ in batch])

            if show_progress and total > self.batch_size:
                progress = min(i + self.batch_size, total)
                print(f"   [嵌入] {progress}/{total} ({progress*100//total}%)")

        return EmbeddingResult(
            chunk_ids=[],
            dense_vectors=all_dense,
            sparse_vectors=all_sparse,
        )

    def _encode_ollama(self, texts: List[str], show_progress: bool) -> EmbeddingResult:
        import requests
        all_dense = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = requests.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.ollama_model, "input": batch},
                timeout=120,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama embed failed: {resp.status_code}")
            all_dense.extend(resp.json()["embeddings"])
        return EmbeddingResult(chunk_ids=[], dense_vectors=all_dense, sparse_vectors=[{} for _ in texts])

    def encode_query(self, query: str) -> tuple:
        """对单条查询编码，返回 (dense_vector, sparse_vector)"""
        self._ensure_loaded()

        if self.provider == "ollama":
            import requests
            resp = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.ollama_model, "prompt": query},
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama embed failed: {resp.status_code}")
            return resp.json()["embedding"], {}

        # sentence_transformers
        dense = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        dense_vec = dense[0].tolist() if hasattr(dense[0], 'tolist') else dense[0]

        sparse_vec = {}
        try:
            sparse_result = self.model.encode(
                [query],
                return_sparse=True,
                show_progress_bar=False,
            )
            s = sparse_result[0]
            if isinstance(s, dict):
                sparse_vec = s
            elif hasattr(s, 'todense'):
                arr = s.todense().flatten()
                sparse_vec = {j: float(arr[j]) for j in range(len(arr)) if arr[j] != 0}
            elif isinstance(s, np.ndarray):
                sparse_vec = {j: float(s[j]) for j in range(len(s)) if s[j] != 0}
        except Exception:
            pass

        return dense_vec, sparse_vec


def create_text_for_embedding(chunk) -> str:
    """为嵌入生成优化的文本表示，拼接元数据 + 正文"""
    parts = []
    meta_str = chunk.get_metadata_str()
    if meta_str:
        parts.append(f"[{meta_str}]")
    parts.append(chunk.text)
    return " ".join(parts)
