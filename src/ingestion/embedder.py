"""
嵌入引擎 — 稠密向量 (+ BGE-M3 稀疏向量) 生成

provider:
  - openai: 稠密向量调用 OpenAI 兼容 embedding API (当前为 Qwen3-Embedding),
      稀疏向量可由本地 BGE-M3 生成。两者互不影响。
  - sentence_transformers: 兼容旧配置的本地 BGE-M3 dense + sparse 路径
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
    """嵌入模型封装 — 支持 OpenAI 兼容 API 与本地 sentence-transformers 两种 provider"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        emb_config = self.config["embedding"]
        self.provider = emb_config.get("provider", "openai")
        self.sparse_provider = emb_config.get("sparse_provider", "hashed_tf")

        # HF 镜像设置 (本地模型/重排器下载用; API 模式不影响)
        hf_home = emb_config.get("hf_home", "")
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
        hf_endpoint = emb_config.get("hf_endpoint", "")
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        self.model_name = emb_config.get("model_name", "BAAI/bge-m3")
        self.device = emb_config.get("device", "cpu")
        self.batch_size = emb_config.get("batch_size", 32)
        self.normalize = emb_config.get("normalize", True)
        self.max_length = emb_config.get("max_length", 8192)
        self.dimensions = emb_config.get("dimensions", 1024)

        # OpenAI 兼容 API 配置
        oai = emb_config.get("openai", {})
        self.api_base_url = oai.get("base_url", "https://api-inference.modelscope.cn/v1")
        self.api_model = oai.get("model", self.model_name)
        self.api_timeout = oai.get("timeout", 60)
        self.api_max_retries = oai.get("max_retries", 3)
        self.api_key = self._resolve_api_key(oai)

        self.client = None
        self.model = None
        self.sparse_model = None
        self._loaded = False

    @property
    def is_api(self) -> bool:
        """是否走 OpenAI 兼容 API (不加载本地模型)"""
        return self.provider == "openai"

    def _resolve_api_key(self, oai: dict) -> str:
        """API key 优先级: 环境变量(api_key_env) > 配置 api_key"""
        api_key_env = oai.get("api_key_env", "")
        if api_key_env:
            env_key = os.environ.get(api_key_env, "")
            if env_key:
                return env_key
        return oai.get("api_key", "")

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.is_api:
            self._load_openai_client()
            if self.sparse_provider == "bge_m3":
                self._load_bge_m3_sparse()
        else:
            self._load_sentence_transformers()
        self._loaded = True

    def _load_openai_client(self):
        """惰性创建 OpenAI 兼容客户端 (无本地模型加载)

        api_key 为空时用占位符, 避免 openai SDK 抛 "Missing credentials",
        让魔搭服务端返回清晰的鉴权错误。
        """
        from openai import OpenAI
        key = self.api_key or "empty-key-not-set"
        self.client = OpenAI(
            base_url=self.api_base_url,
            api_key=key,
            timeout=self.api_timeout,
            max_retries=self.api_max_retries,
        )
        if not self.api_key:
            print(f"[embed][WARN] 未配置 embedding API key "
                  f"(env {self.config['embedding']['openai'].get('api_key_env', 'MODELSCOPE_API_KEY')} 或 config api_key)")
        print(f"[embed] OpenAI 兼容客户端就绪: base_url={self.api_base_url}, model={self.api_model}")

    def _load_bge_m3_sparse(self):
        """Load the BGE-M3 lexical head locally without replacing dense API embeddings.

        BGE-M3 returns lexical weights keyed by tokenizer ids.  Milvus
        SPARSE_FLOAT_VECTOR uses the same integer-key representation, so the
        weights can be stored directly after converting string ids to ints.
        CPU is deliberately explicit: this model is independent from the
        Qwen dense embedding endpoint.
        """
        from FlagEmbedding import BGEM3FlagModel

        sparse_cfg = self.config["embedding"]
        model_name = sparse_cfg.get("sparse_model_name", "BAAI/bge-m3")
        sparse_device = sparse_cfg.get("sparse_device", "cpu")
        if sparse_device != "cpu":
            raise ValueError(
                f"BGE-M3 sparse provider requires CPU in the current deployment, got {sparse_device!r}"
            )
        batch_size = int(sparse_cfg.get("sparse_batch_size", 4))
        max_length = int(sparse_cfg.get("sparse_max_length", 512))
        cache_dir = sparse_cfg.get("hf_home") or None
        print(
            f"[embed] 加载本地 BGE-M3 稀疏模型: {model_name} "
            f"(device=cpu, batch={batch_size}, max_length={max_length})"
        )
        self.sparse_model = BGEM3FlagModel(
            model_name,
            use_fp16=False,
            devices="cpu",
            cache_dir=cache_dir,
            batch_size=batch_size,
            query_max_length=max_length,
            passage_max_length=max_length,
            return_dense=False,
            return_sparse=True,
        )
        print("   [OK] BGE-M3 稀疏模型就绪 (CPU)")

    @staticmethod
    def _milvus_sparse(lexical_weights) -> dict:
        """Convert FlagEmbedding lexical weights to Milvus sparse format."""
        if not lexical_weights:
            return {}
        converted = {}
        for key, value in lexical_weights.items():
            try:
                weight = float(value)
                if weight > 0:
                    converted[int(key)] = weight
            except (TypeError, ValueError):
                continue
        return converted

    def _bge_m3_sparse_encode(self, texts: List[str], *, query: bool = False) -> List[dict]:
        if not self.sparse_model:
            return [{} for _ in texts]
        sparse_cfg = self.config["embedding"]
        batch_size = int(sparse_cfg.get("sparse_batch_size", 4))
        max_length = int(sparse_cfg.get("sparse_max_length", 512))
        if query and hasattr(self.sparse_model, "encode_queries"):
            result = self.sparse_model.encode_queries(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=False,
                return_sparse=True,
            )
        else:
            result = self.sparse_model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=False,
                return_sparse=True,
            )
        weights = result.get("lexical_weights") or []
        if isinstance(weights, dict):
            weights = [weights]
        return [self._milvus_sparse(item) for item in weights]

    def unload(self):
        """释放模型 — API 模式无本地模型, 仅复位状态; 本地模式释放显存"""
        if self.is_api:
            self.client = None
            self.sparse_model = None
            self._loaded = False
            return

        if not self._loaded or self.model is None:
            return
        try:
            import torch
            import gc
            del self.model
            self.model = None
            self.sparse_model = None
            self._loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("   [embed] BGE-M3 已卸载，显存已释放")
        except Exception as e:
            print(f"   [embed] 卸载异常: {e}")

    def reload(self):
        """重新加载 — 等同 _ensure_loaded()"""
        self._ensure_loaded()

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

    def _sparse_for(self, text: str) -> dict:
        """Generate one API-mode sparse vector."""
        if self.sparse_provider == "bge_m3":
            return self._bge_m3_sparse_encode([text], query=True)[0]
        if self.sparse_provider in {"hashed_tf", "local_bm25"}:
            from ingestion.hashed_tf_sparse import compute_hashed_tf_sparse
            return compute_hashed_tf_sparse(text)
        return {}

    def _api_embeddings(self, batch: List[str]) -> List[List[float]]:
        """调用 /v1/embeddings 生成稠密向量, 按输入顺序返回 (可选 L2 归一化)

        encoding_format="float": 魔搭 Qwen3-Embedding 要求显式指定编码格式。
        """
        resp = self.client.embeddings.create(
            model=self.api_model,
            input=batch,
            encoding_format="float",
        )
        ordered = sorted(resp.data, key=lambda d: d.index)
        vecs = [d.embedding for d in ordered]
        if self.normalize:
            vecs = self._l2_normalize(vecs)
        return vecs

    @staticmethod
    def _l2_normalize(vecs: List[List[float]]) -> List[List[float]]:
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()

    def encode(self, texts: List[str], show_progress: bool = True) -> EmbeddingResult:
        """批量生成稠密 + 稀疏嵌入向量"""
        self._ensure_loaded()

        all_dense = []
        all_sparse = []
        total = len(texts)

        # ===== OpenAI 兼容 API 路径 =====
        if self.is_api:
            for i in range(0, total, self.batch_size):
                batch = texts[i:i + self.batch_size]
                all_dense.extend(self._api_embeddings(batch))
                if self.sparse_provider == "bge_m3":
                    all_sparse.extend(self._bge_m3_sparse_encode(batch))
                else:
                    all_sparse.extend(self._sparse_for(t) for t in batch)

                if show_progress and total > self.batch_size:
                    progress = min(i + self.batch_size, total)
                    print(f"   [嵌入] {progress}/{total} ({progress * 100 // total}%)")

            return EmbeddingResult(
                chunk_ids=[],
                dense_vectors=all_dense,
                sparse_vectors=all_sparse,
            )

        # ===== sentence_transformers 本地路径 =====
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

    def encode_query(self, query: str) -> tuple:
        """对单条查询编码，返回 (dense_vector, sparse_vector)"""
        self._ensure_loaded()

        # ===== OpenAI 兼容 API 路径 =====
        if self.is_api:
            dense_vec = self._api_embeddings([query])[0]
            if self.sparse_provider == "bge_m3":
                sparse_vec = self._bge_m3_sparse_encode([query], query=True)[0]
            else:
                sparse_vec = self._sparse_for(query)
            return dense_vec, sparse_vec

        # ===== sentence_transformers 本地路径 =====
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
