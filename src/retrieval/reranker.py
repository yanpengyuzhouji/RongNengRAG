"""
重排序器 — 交叉编码器精排 + 元数据加权
BGE-Reranker-v2-m3 作为主力，支持元数据提权
"""

import yaml
from typing import List, Dict, Optional, Tuple


class Reranker:
    """交叉编码器重排序 + 元数据加权"""

    def __init__(self, config_path: str = "D:/rag-system/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        rerank_config = self.config["reranker"]
        self.model_name = rerank_config["model_name"]
        self.device = rerank_config["device"]
        self.batch_size = rerank_config["batch_size"]
        self.top_k = rerank_config["top_k"]
        self.metadata_boosts = rerank_config.get("metadata_boosts", {})

        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._loaded:
            return

        print(f"📥 加载重排序模型: {self.model_name} ...")

        try:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(
                self.model_name,
                use_fp16=True,
                device=self.device,
            )
        except ImportError:
            # 回退到 sentence-transformers CrossEncoder
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )

        self._loaded = True

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        analyzed_query=None,
        top_k: int = None,
    ) -> List[dict]:
        """
        重排序候选结果

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

        # Step 1: 交叉编码器打分
        pairs = [[query, c.get("text", c.get("entity", {}).get("text", ""))]
                 for c in candidates]

        texts_for_scoring = [c.get("text", c.get("entity", {}).get("text", ""))
                            for c in candidates]

        try:
            # FlagEmbedding API
            scores = self.model.compute_score(
                pairs,
                batch_size=self.batch_size,
                normalize=True,
            )
        except Exception:
            # Fallback: sentence-transformers API
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()

        # 确保 scores 是列表
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
            reverse=True
        )

        return [item[0] for item in ranked[:top_k]]

    def _compute_metadata_boost(self, query: str, candidate: dict,
                                analyzed_query) -> float:
        """计算元数据加权系数"""
        boost = 1.0

        # 获取候选元数据（兼容不同数据格式）
        entity = candidate.get("entity", candidate)

        # 文档编号精确匹配加权
        doc_number = entity.get("doc_number", "")
        if doc_number and doc_number in query:
            boost *= self.metadata_boosts.get("exact_doc_number_match", 1.5)

        # 标准规范加权
        category = entity.get("category", "")
        if category == "标准规范":
            boost *= self.metadata_boosts.get("category_standard", 1.2)

        # 国标/行标加权（权威性）
        publish_level = entity.get("publish_level", "")
        if publish_level in ("国标", "行标"):
            boost *= self.metadata_boosts.get("publish_level_national", 1.1)

        # 图纸降权（如果不是专门查图纸）
        is_drawing = entity.get("is_drawing", False)
        if is_drawing and not self._is_drawing_query(query):
            boost *= 0.85

        # 域匹配加权（如果查询指定了域）
        if analyzed_query and analyzed_query.domain:
            candidate_domain = entity.get("domain", "")
            if candidate_domain == analyzed_query.domain:
                boost *= 1.1  # 域精确匹配小加分

        # 电压等级匹配
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
            boost = self._compute_metadata_boost(
                "", candidate, analyzed_query
            )
            scored.append((candidate, boost))

        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked[:top_k]]
