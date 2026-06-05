"""
重排序器 — 嵌入相似度 + 元数据加权
Ollama 模式: 用嵌入模型计算 query-doc 余弦相似度 + 元数据加权
FlagEmbedding 回退: BGE-Reranker 交叉编码器精排
"""

import yaml
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class Reranker:
    """重排序器: Ollama 嵌入相似度 或 交叉编码器"""

    def __init__(self, config_path: str = "D:/rag-system/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        rerank_config = self.config["reranker"]
        self.provider = rerank_config.get("provider", "ollama")
        self.top_k = rerank_config.get("top_k", 15)
        self.metadata_boosts = rerank_config.get("metadata_boosts", {})

        if self.provider == "ollama":
            ollama_cfg = rerank_config.get("ollama", {})
            self.ollama_model = ollama_cfg.get("model", "bge-m3:latest")
            self.ollama_url = ollama_cfg.get("base_url", "http://localhost:11434")
        else:
            self.model_name = rerank_config["model_name"]
            self.device = rerank_config.get("device", "cpu")
            self.batch_size = rerank_config.get("batch_size", 16)

        self.model = None
        self._loaded = False
        self._query_embedding_cache = {}  # 缓存查询嵌入

    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._loaded:
            return

        if self.provider == "ollama":
            self._init_ollama()
        else:
            self._load_flag_reranker()

        self._loaded = True

    def _init_ollama(self):
        """验证 Ollama 服务可用"""
        print(f"[Ollama] 重排序使用: {self.ollama_model}")
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama 返回状态码 {resp.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"无法连接 Ollama ({self.ollama_url})。请先启动 Ollama 服务。"
            )
        print(f"   [OK] Ollama 重排序模型就绪: {self.ollama_model}")

    def _load_flag_reranker(self):
        """使用 FlagEmbedding 加载交叉编码器 (HuggingFace 回退)"""
        print(f"[加载] 重排序模型: {self.model_name} ...")

        try:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(
                self.model_name,
                use_fp16=True,
                device=self.device,
            )
        except ImportError:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )

    def _ollama_embed(self, texts: List[str]) -> List[List[float]]:
        """通过 Ollama API 批量生成嵌入向量"""
        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.ollama_model, "input": texts},
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama 嵌入失败: {resp.status_code} {resp.text}")
        return resp.json()["embeddings"]

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询向量（带缓存）"""
        if query not in self._query_embedding_cache:
            resp = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.ollama_model, "prompt": query},
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama 嵌入失败: {resp.status_code} {resp.text}")
            self._query_embedding_cache[query] = resp.json()["embedding"]
        return self._query_embedding_cache[query]

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        analyzed_query=None,
        top_k: int = None,
    ) -> List[dict]:
        """
        重排序候选结果

        Ollama 模式:
          1. 获取查询向量
          2. 批量获取候选文档向量
          3. 计算余弦相似度
          4. 元数据加权
          5. 排序返回 Top-K

        Args:
            query: 用户原始查询
            candidates: Milvus 返回的候选结果列表
            analyzed_query: 解析后的查询对象（用于元数据加权）
            top_k: 返回结果数（默认使用配置值）

        Returns:
            排序后的结果列表
        """
        if top_k is None:
            top_k = self.top_k

        if not candidates:
            return []

        self._ensure_loaded()

        if self.provider == "ollama":
            return self._rerank_ollama(query, candidates, analyzed_query, top_k)
        else:
            return self._rerank_flag(query, candidates, analyzed_query, top_k)

    def _rerank_ollama(
        self,
        query: str,
        candidates: List[dict],
        analyzed_query=None,
        top_k: int = None,
    ) -> List[dict]:
        """使用 Ollama 嵌入 + 余弦相似度重排序"""
        # Step 1: 获取查询向量
        query_vec = self._get_query_embedding(query)

        # Step 2: 提取候选文本
        texts = [
            c.get("text", c.get("entity", {}).get("text", ""))
            for c in candidates
        ]

        # Step 3: 批量获取候选向量并计算相似度
        scores = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_vecs = self._ollama_embed(batch_texts)
            for vec in batch_vecs:
                scores.append(_cosine_similarity(query_vec, vec))

        # Step 4: 元数据加权
        for i, candidate in enumerate(candidates):
            boost = self._compute_metadata_boost(query, candidate, analyzed_query)
            if i < len(scores):
                scores[i] *= boost

        # Step 5: 排序
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [item[0] for item in ranked[:top_k]]

    def _rerank_flag(
        self,
        query: str,
        candidates: List[dict],
        analyzed_query=None,
        top_k: int = None,
    ) -> List[dict]:
        """使用交叉编码器重排序 (FlagEmbedding 回退)"""
        # Step 1: 交叉编码器打分
        pairs = [[query, c.get("text", c.get("entity", {}).get("text", ""))]
                 for c in candidates]

        try:
            scores = self.model.compute_score(
                pairs,
                batch_size=self.batch_size,
                normalize=True,
            )
        except Exception:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()

        if not isinstance(scores, list):
            scores = [scores]

        # Step 2: 元数据加权
        for i, candidate in enumerate(candidates):
            boost = self._compute_metadata_boost(query, candidate, analyzed_query)
            if i < len(scores):
                scores[i] *= boost

        # Step 3: 排序
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [item[0] for item in ranked[:top_k]]

    def _compute_metadata_boost(self, query: str, candidate: dict,
                                analyzed_query) -> float:
        """计算元数据加权系数"""
        boost = 1.0

        entity = candidate.get("entity", candidate)

        doc_number = entity.get("doc_number", "")
        if doc_number and doc_number in query:
            boost *= self.metadata_boosts.get("exact_doc_number_match", 1.5)

        category = entity.get("category", "")
        if category == "标准规范":
            boost *= self.metadata_boosts.get("category_standard", 1.2)

        publish_level = entity.get("publish_level", "")
        if publish_level in ("国标", "行标"):
            boost *= self.metadata_boosts.get("publish_level_national", 1.1)

        is_drawing = entity.get("is_drawing", False)
        if is_drawing and not self._is_drawing_query(query):
            boost *= 0.85

        if analyzed_query and analyzed_query.domain:
            candidate_domain = entity.get("domain", "")
            if candidate_domain == analyzed_query.domain:
                boost *= 1.1

        if analyzed_query and analyzed_query.voltage_level:
            candidate_voltage = entity.get("voltage_level", "")
            if candidate_voltage == analyzed_query.voltage_level:
                boost *= 1.15

        return boost

    def _is_drawing_query(self, query: str) -> bool:
        """判断是否为查图查询"""
        drawing_keywords = ["图纸", "方案图", "布置图", "接线图", "主接线",
                           "平面图", "剖面图", "设计图", "CAD", "dwg"]
        return any(kw in query for kw in drawing_keywords)

    def rerank_without_model(
        self,
        candidates: List[dict],
        analyzed_query=None,
        top_k: int = None,
    ) -> List[dict]:
        """
        无模型重排序（仅用元数据加权）
        用于轻量级场景或模型不可用时
        """
        if top_k is None:
            top_k = self.top_k

        scored = []
        for candidate in candidates:
            boost = self._compute_metadata_boost("", candidate, analyzed_query)
            scored.append((candidate, boost))

        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked[:top_k]]
