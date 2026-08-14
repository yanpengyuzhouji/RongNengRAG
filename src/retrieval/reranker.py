"""
重排序器 — 交叉编码器精排 + 元数据加权
默认: FlagEmbedding BGE-Reranker-v2-m3 (本地 CPU/GPU)
"""

import os
import math
from typing import List, Dict, Optional, Tuple


class Reranker:
    """交叉编码器重排序 + 元数据加权"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        rerank_config = self.config["reranker"]
        self.provider = rerank_config.get("provider", "flagembedding")
        self.top_k = rerank_config.get("top_k", 15)
        self.metadata_boosts = rerank_config.get("metadata_boosts", {})
        self.max_metadata_boost = float(
            rerank_config.get("max_metadata_boost", 1.20)
        )

        # 置信度校准参数
        retrieval_config = self.config.get("retrieval", {})
        confidence_config = retrieval_config.get("confidence", {})
        self.min_score_threshold = confidence_config.get("min_score_threshold", 0.3)
        self.softmax_temperature = confidence_config.get("softmax_temperature", 1.0)
        self.rrf_k = max(1, int(retrieval_config.get("rrf_k", 60)))
        self.none_rrf_weight = min(1.0, max(
            0.0, float(rerank_config.get("none_rrf_weight", 0.15))
        ))

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
        self.cache_dir = emb_config.get("hf_home") or None

        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        self._load_flag_reranker()
        self._loaded = True

    def unload(self):
        """释放重排序模型显存 — LLM/OCR 按需调度"""
        if not self._loaded:
            return
        try:
            del self.model
            self.model = None
            self._loaded = False
            import torch
            torch.cuda.empty_cache()
            print("   [rerank] 模型已卸载，显存已释放")
        except Exception as e:
            print(f"   [rerank] 卸载异常: {e}")

    def _load_flag_reranker(self):
        print(f"[rerank] 加载本地交叉编码器: {self.model_name}")
        # fp16 仅 CUDA 可用; CPU 上强制 fp16 会出问题
        use_fp16 = self.device.startswith("cuda")
        try:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(
                self.model_name,
                use_fp16=use_fp16,
                devices=self.device,
                cache_dir=self.cache_dir,
                batch_size=self.batch_size,
            )
        except ImportError:
            from sentence_transformers import CrossEncoder
            print(f"   FlagEmbedding 未安装，回退到 CrossEncoder")
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
                cache_folder=self.cache_dir,
            )
        # 兼容 transformers 5.x: FlagEmbedding 1.4 依赖 tokenizer.prepare_for_model,
        # 新版 transformers 已移除该方法. 注入等价的 token-id 拼接实现.
        self._patch_tokenizer_compat()
        print(f"   [OK] 重排序模型就绪")

    def _patch_tokenizer_compat(self):
        """给 tokenizer 注入 prepare_for_model 兼容实现 (transformers 5.x 已移除)。

        FlagEmbedding 1.4.0 在 compute_score 中调用 tokenizer.prepare_for_model(q_ids, d_ids),
        新版 transformers 删除了该方法导致 AttributeError。
        此处按 XLMRoberta 的 <s> q </s></s> d </s> 拼接格式复刻等价逻辑。
        """
        try:
            tok = self.model.tokenizer
        except AttributeError:
            return
        if not tok or hasattr(tok, "prepare_for_model"):
            return
        import types

        def _prepare_for_model(self, text, text_pair=None, truncation=None,
                               max_length=None, padding=False, **kwargs):
            max_len = max_length or 512
            q_ids = list(text)
            d_ids = list(text_pair) if text_pair is not None else []
            cls_id = self.cls_token_id
            sep_id = self.sep_token_id
            # XLMRoberta 双序列: <s> q </s></s> d </s>
            input_ids = [cls_id] + q_ids + [sep_id, sep_id] + d_ids + [sep_id]
            if truncation == "only_second":
                while len(input_ids) > max_len and d_ids:
                    d_ids.pop()
                    input_ids = [cls_id] + q_ids + [sep_id, sep_id] + d_ids + [sep_id]
            elif len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }

        tok.prepare_for_model = types.MethodType(_prepare_for_model, tok)
        print(f"   [rerank] 已注入 tokenizer.prepare_for_model 兼容补丁 (transformers 5.x)")

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

        # provider=none: 不加载模型, 直接元数据加权排序 (秒级, 无 GPU 时默认)
        if self.provider == "none":
            ranked = self.rerank_without_model(
                candidates, analyzed_query, top_k, query=query
            )
            for item in ranked:
                score = item.get("_rerank_score", item.get("distance", 0.0))
                item["confidence"] = round(score, 4)
            return ranked

        self._ensure_loaded()

        ranked = self._rerank_cross_encoder(query, candidates, analyzed_query, top_k)

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
        for candidate, raw_score in ranked:
            confidence = self._calibrate_score(raw_score)
            if confidence < self.min_score_threshold:
                continue
            candidate["_rerank_score"] = confidence
            result.append(candidate)
            if len(result) >= top_k:
                break
        return result

    def _compute_metadata_boost(self, query: str, candidate: dict, analyzed_query) -> float:
        """计算元数据加权系数（加法归一化，上限1.15，防止元数据统治语义评分）"""
        boost_delta = 0.0
        entity = candidate.get("entity", candidate)

        def configured_delta(name: str, default: float) -> float:
            return float(self.metadata_boosts.get(name, default)) - 1.0

        # 文档编号精确匹配
        doc_number = entity.get("doc_number", "")
        if doc_number and doc_number in query:
            boost_delta += configured_delta("exact_doc_number_match", 1.05)

        # 文件名匹配
        file_name = entity.get("file_path", "")
        if file_name:
            fname = os.path.basename(file_name).lower()
            query_lower = query.lower()
            import re
            fname_tokens = re.split(r'[_\-\.\s]+', fname)
            matched = 0
            for token in fname_tokens:
                if len(token) >= 2 and token.lower() in query_lower:
                    matched += 1
            if matched >= 2:
                boost_delta += configured_delta("file_name_match", 1.10)

        # 标准规范类目
        category = entity.get("category", "")
        if category == "标准规范":
            boost_delta += configured_delta("category_standard", 1.03)

        # 国标/行标
        publish_level = entity.get("publish_level", "")
        if publish_level in ("国标", "行标"):
            boost_delta += configured_delta("publish_level_national", 1.03)

        # 图纸降权
        is_drawing = entity.get("is_drawing", False)
        if is_drawing and not self._is_drawing_query(query):
            boost_delta += configured_delta("drawing_penalty", 0.90)

        # 域匹配
        if analyzed_query and analyzed_query.domain:
            if entity.get("domain", "") == analyzed_query.domain:
                boost_delta += configured_delta("domain_match", 1.05)

        # 电压等级匹配
        if analyzed_query and analyzed_query.voltage_level:
            if entity.get("voltage_level", "") == analyzed_query.voltage_level:
                boost_delta += configured_delta("voltage_level_match", 1.05)

        return max(0.1, min(self.max_metadata_boost, 1.0 + boost_delta))

    def _calibrate_score(self, score: float) -> float:
        """Temperature-scale a probability-like score without changing rank."""
        probability = min(1.0 - 1e-6, max(1e-6, float(score)))
        temperature = max(1e-6, float(self.softmax_temperature))
        logit = math.log(probability / (1.0 - probability)) / temperature
        return 1.0 / (1.0 + math.exp(-logit))

    def _is_drawing_query(self, query: str) -> bool:
        drawing_kw = ["图纸", "方案图", "布置图", "接线图", "主接线",
                      "平面图", "剖面图", "设计图", "CAD", "dwg"]
        return any(kw in query for kw in drawing_kw)

    def rerank_without_model(
        self, candidates: List[dict], analyzed_query=None, top_k: int = None,
        query: str = "",
    ) -> List[dict]:
        """召回分数 + RRF 名次先验 + 配置化元数据的无模型排序。"""
        if top_k is None:
            top_k = self.top_k
        scored = []
        for rank, candidate in enumerate(candidates, 1):
            base_score = float(candidate.get("distance", 0.0) or 0.0)
            base_score = min(1.0, max(0.0, base_score))
            rrf_score = (self.rrf_k + 1.0) / (self.rrf_k + rank)
            fused = (
                (1.0 - self.none_rrf_weight) * base_score
                + self.none_rrf_weight * rrf_score
            )
            boosted = min(1.0, max(0.0, fused * self._compute_metadata_boost(
                query, candidate, analyzed_query
            )))
            scored.append((candidate, self._calibrate_score(boosted)))
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        result = []
        for candidate, score in ranked:
            if score < self.min_score_threshold:
                continue
            candidate["_rerank_score"] = float(score)
            result.append(candidate)
            if len(result) >= top_k:
                break
        return result
