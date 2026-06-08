"""
重排序器 — 交叉编码器精排 + 元数据加权
默认: FlagEmbedding BGE-Reranker-v2-m3 (本地GPU)
回退: Ollama 嵌入相似度
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class Reranker:
    """交叉编码器重排序 + 元数据加权"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        rerank_config = self.config["reranker"]
        self.provider = rerank_config.get("provider", "flagembedding")
        self.top_k = rerank_config.get("top_k", 15)
        self.metadata_boosts = rerank_config.get("metadata_boosts", {})

        # 置信度校准参数
        retrieval_config = self.config.get("retrieval", {})
        confidence_config = retrieval_config.get("confidence", {})
        self.min_score_threshold = confidence_config.get("min_score_threshold", 0.3)
        self.softmax_temperature = confidence_config.get("softmax_temperature", 1.0)

        # HF 镜像 (从 embedding 配置复用)
        emb_config = self.config.get("embedding", {})
        hf_home = emb_config.get("hf_home", "")
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
        hf_endpoint = emb_config.get("hf_endpoint", "")
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        self.model_name = rerank_config.get("model_name", "BAAI/bge-reranker-v2-m3")
        self.device = rerank_config.get("device", "cuda")
        self.batch_size = rerank_config.get("batch_size", 16)

        # Ollama 回退
        if self.provider == "ollama":
            ollama_cfg = rerank_config.get("ollama", {})
            self.ollama_model = ollama_cfg.get("model", "bge-m3:latest")
            self.ollama_url = ollama_cfg.get("base_url", "http://localhost:11434")

        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        if self.provider == "flagembedding":
            self._load_flag_reranker()
        else:
            self._init_ollama()

        self._loaded = True

    def _load_flag_reranker(self):
        print(f"[rerank] 加载本地交叉编码器: {self.model_name}")
        try:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(
                self.model_name,
                use_fp16=True,
                device=self.device,
            )
        except ImportError:
            from sentence_transformers import CrossEncoder
            print(f"   FlagEmbedding 未安装，回退到 CrossEncoder")
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
        print(f"   [OK] 重排序模型就绪")

    def _init_ollama(self):
        import requests
        print(f"[rerank] 使用 Ollama: {self.ollama_url}")
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama status={resp.status_code}")
        except requests.ConnectionError:
            raise ConnectionError(f"无法连接 Ollama ({self.ollama_url})")

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        analyzed_query=None,
        top_k: int = None,
    ) -> List[dict]:
        """
        重排序: 交叉编码器打分 + 元数据加权 → Top-K
        每个结果附加 confidence 字段 (0~1)
        """
        if top_k is None:
            top_k = self.top_k

        if not candidates:
            return []

        self._ensure_loaded()

        if self.provider == "flagembedding":
            ranked = self._rerank_cross_encoder(query, candidates, analyzed_query, top_k)
        else:
            ranked = self._rerank_ollama(query, candidates, analyzed_query, top_k)

        # 附加置信度到每个结果
        for item in ranked:
            score = item.get("_rerank_score", item.get("distance", 0.0))
            item["confidence"] = round(score, 4)

        return ranked

    def _rerank_cross_encoder(
        self, query: str, candidates: List[dict], analyzed_query, top_k: int
    ) -> List[dict]:
        """交叉编码器精排"""
        texts = [
            c.get("text", c.get("entity", {}).get("text", ""))
            for c in candidates
        ]

        # 构建 query-doc pairs
        pairs = [[query, t] for t in texts]

        # 交叉编码器打分
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

        # 元数据加权
        for i, candidate in enumerate(candidates):
            boost = self._compute_metadata_boost(query, candidate, analyzed_query)
            if i < len(scores):
                scores[i] *= boost

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = []
        for item in ranked[:top_k]:
            item[0]["_rerank_score"] = float(item[1])
            result.append(item[0])
        return result

    def _rerank_ollama(
        self, query: str, candidates: List[dict], analyzed_query, top_k: int
    ) -> List[dict]:
        """Ollama 嵌入相似度重排 (回退方案)"""
        import requests

        # 查询向量
        resp = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.ollama_model, "prompt": query},
            timeout=30,
        )
        query_vec = resp.json()["embedding"]

        texts = [
            c.get("text", c.get("entity", {}).get("text", ""))
            for c in candidates
        ]

        # 批量获取候选向量
        scores = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            resp = requests.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.ollama_model, "input": batch},
                timeout=120,
            )
            for vec in resp.json()["embeddings"]:
                scores.append(_cosine_similarity(query_vec, vec))

        # 元数据加权
        for i, candidate in enumerate(candidates):
            boost = self._compute_metadata_boost(query, candidate, analyzed_query)
            if i < len(scores):
                scores[i] *= boost

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = []
        for item in ranked[:top_k]:
            item[0]["_rerank_score"] = float(item[1])
            result.append(item[0])
        return result

    def _compute_metadata_boost(self, query: str, candidate: dict, analyzed_query) -> float:
        """计算元数据加权系数（降低过度加权风险）"""
        boost = 1.0
        entity = candidate.get("entity", candidate)

        # 文档编号精确匹配
        doc_number = entity.get("doc_number", "")
        if doc_number and doc_number in query:
            boost *= self.metadata_boosts.get("exact_doc_number_match", 1.05)

        # 文件名匹配 (新增)
        file_name = entity.get("file_path", "")
        if file_name:
            fname = os.path.basename(file_name).lower()
            # 检查查询中是否包含文件名片段
            query_lower = query.lower()
            # 用文件名中的中文/英文/数字片段匹配
            import re
            fname_tokens = re.split(r'[_\-\.\s]+', fname)
            matched = 0
            for token in fname_tokens:
                if len(token) >= 2 and token.lower() in query_lower:
                    matched += 1
            if matched >= 2:
                boost *= self.metadata_boosts.get("file_name_match", 1.30)

        # 标准规范类目
        category = entity.get("category", "")
        if category == "标准规范":
            boost *= self.metadata_boosts.get("category_standard", 1.05)

        # 国标/行标
        publish_level = entity.get("publish_level", "")
        if publish_level in ("国标", "行标"):
            boost *= self.metadata_boosts.get("publish_level_national", 1.05)

        # 图纸降权
        is_drawing = entity.get("is_drawing", False)
        if is_drawing and not self._is_drawing_query(query):
            boost *= 0.85

        # 域匹配
        if analyzed_query and analyzed_query.domain:
            if entity.get("domain", "") == analyzed_query.domain:
                boost *= 1.05

        # 电压等级匹配
        if analyzed_query and analyzed_query.voltage_level:
            if entity.get("voltage_level", "") == analyzed_query.voltage_level:
                boost *= 1.10

        return boost

    def _is_drawing_query(self, query: str) -> bool:
        drawing_kw = ["图纸", "方案图", "布置图", "接线图", "主接线",
                      "平面图", "剖面图", "设计图", "CAD", "dwg"]
        return any(kw in query for kw in drawing_kw)

    def rerank_without_model(
        self, candidates: List[dict], analyzed_query=None, top_k: int = None
    ) -> List[dict]:
        """纯元数据排序兜底"""
        if top_k is None:
            top_k = self.top_k
        scored = [(c, self._compute_metadata_boost("", c, analyzed_query)) for c in candidates]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        result = []
        for item in ranked[:top_k]:
            item[0]["_rerank_score"] = float(item[1])
            result.append(item[0])
        return result
