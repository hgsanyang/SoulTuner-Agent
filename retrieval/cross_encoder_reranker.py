# ============================================================
# Cross-Encoder 精排模块 (bge-reranker-v2-m3)
#
# 在 RRF 粗排融合之后、Graph Affinity 个性化之前，
# 利用 Cross-Encoder 对 (query, document) 进行深度语义匹配精排。
#
# 特性：
#   - 惰性加载单例：首次调用时懒加载模型权重，之后复用
#   - Neo4j 标签增强：从图谱查询 vibe/moods/themes/scenarios 拼接文档文本
#   - 批量推理：支持 batch 推理加速
#   - 加权融合：RRF 分数先 min-max 归一化到 [0,1]，再与 reranker_score 加权
#     final_score = (1 - α) × rrf_normalized + α × reranker_score
#   - 降级处理：模型加载/推理异常时返回原始列表不影响主流程
# ============================================================

import logging
from typing import List, Dict, Any, Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Cross-Encoder 精排器（惰性加载单例）"""

    _instance: Optional["CrossEncoderReranker"] = None
    _model = None
    _model_loaded: bool = False
    _load_failed: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 单例只初始化一次
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

    def _ensure_model(self) -> bool:
        """懒加载 Cross-Encoder 模型，返回是否成功。"""
        if self._model_loaded:
            return True
        if self._load_failed:
            return False

        try:
            from config.settings import settings
            from sentence_transformers import CrossEncoder

            model_name = settings.reranker_model_name
            device = settings.reranker_device
            logger.info(f"[Reranker] 正在加载 Cross-Encoder 模型: {model_name} (device={device})")

            self._model = CrossEncoder(model_name, device=device)
            self._model_loaded = True
            logger.info(f"[Reranker] ✅ 模型加载成功: {model_name}")
            return True
        except Exception as e:
            logger.warning(f"[Reranker] ⚠️ 模型加载失败（降级为不精排）: {e}")
            self._load_failed = True
            return False

    @staticmethod
    def _build_document_text(song: Dict[str, Any], neo4j_tags: Optional[Dict] = None) -> str:
        """
        从候选歌曲的元数据 + Neo4j 标签拼接结构化 Document 文本。

        示例输出:
          "晴天" by 周杰伦 | Album: 叶惠美 | Genre: Pop | Vibe: nostalgic, warm
          | Moods: 怀旧, 温暖 | Scenarios: 下雨天
        """
        title = song.get("title", "未知")
        artist = song.get("artist", "未知")
        album = song.get("album", "")
        genre = song.get("genre", "")

        parts = [f'"{title}" by {artist}']
        if album and album != "未知":
            parts.append(f"Album: {album}")
        if genre:
            parts.append(f"Genre: {genre}")

        # 从 Neo4j 查询到的标签
        if neo4j_tags:
            if neo4j_tags.get("vibe"):
                parts.append(f"Vibe: {neo4j_tags['vibe']}")
            if neo4j_tags.get("moods"):
                parts.append(f"Moods: {', '.join(neo4j_tags['moods'])}")
            if neo4j_tags.get("themes"):
                parts.append(f"Themes: {', '.join(neo4j_tags['themes'])}")
            if neo4j_tags.get("scenarios"):
                parts.append(f"Scenarios: {', '.join(neo4j_tags['scenarios'])}")

        return " | ".join(parts)

    @staticmethod
    def _fetch_neo4j_tags(titles: List[str]) -> Dict[str, Dict]:
        """
        批量从 Neo4j 查询歌曲的 vibe/moods/themes/scenarios 标签。
        返回 {title: {vibe, moods, themes, scenarios}} 映射。
        """
        tag_map: Dict[str, Dict] = {}
        try:
            from retrieval.neo4j_client import get_neo4j_client
            neo4j = get_neo4j_client()
            if not neo4j or not neo4j.driver:
                return tag_map

            query = """
            UNWIND $titles AS t
            OPTIONAL MATCH (s:Song)
              WHERE s.title = t
            OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
            OPTIONAL MATCH (s)-[:HAS_THEME]->(th:Theme)
            OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
            RETURN t AS title,
                   s.vibe AS vibe,
                   collect(DISTINCT m.name) AS moods,
                   collect(DISTINCT th.name) AS themes,
                   collect(DISTINCT sc.name) AS scenarios
            """
            results = neo4j.execute_query(query, {"titles": titles})
            for r in results:
                t = r.get("title", "")
                if t:
                    tag_map[t] = {
                        "vibe": r.get("vibe", "") or "",
                        "moods": [x for x in (r.get("moods") or []) if x],
                        "themes": [x for x in (r.get("themes") or []) if x],
                        "scenarios": [x for x in (r.get("scenarios") or []) if x],
                    }
        except Exception as e:
            logger.warning(f"[Reranker] Neo4j 标签查询失败（不影响精排）: {e}")

        return tag_map

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        对候选列表进行 Cross-Encoder 精排（纯替代模式）。

        Cross-Encoder 的分数直接替代粗排分数（工业标准做法）。
        RRF 的职责到"选出候选池"就已完成，精排层负责最终排序。

        Args:
            query: 用户原始查询
            candidates: RRF 融合后的候选列表，每项需含 song dict 和 similarity_score
            top_k: 精排后保留条数（None 则从 settings 读取）

        Returns:
            精排后的列表（带 _reranker_score 字段），如异常则返回原始列表
        """
        if not candidates:
            return candidates

        # 读取配置
        try:
            from config.settings import settings
            if top_k is None:
                top_k = settings.reranker_top_k
            batch_size = settings.reranker_batch_size
        except Exception:
            if top_k is None:
                top_k = 10
            batch_size = 16

        # 尝试加载模型
        if not self._ensure_model():
            logger.warning("[Reranker] 模型不可用，返回原始列表")
            return candidates

        try:
            # 1. 批量获取 Neo4j 标签
            titles = [c.get("song", {}).get("title", "") for c in candidates if c.get("song")]
            neo4j_tags = self._fetch_neo4j_tags(titles) if titles else {}

            # 2. 构建 (query, document) 对
            pairs = []
            for c in candidates:
                song = c.get("song", {})
                title = song.get("title", "")
                tags = neo4j_tags.get(title)
                doc_text = self._build_document_text(song, tags)
                pairs.append([query, doc_text])

            # 3. 批量推理
            logger.info(f"[Reranker] 开始精排: {len(pairs)} 个 query-doc 对 (batch_size={batch_size})")
            scores = self._model.predict(pairs, batch_size=batch_size)

            # 4. 归一化到 [0, 1] — sigmoid（Cross-Encoder 原始输出是 logit）
            import torch
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            normalized_scores = torch.sigmoid(scores_tensor).tolist()

            # 5. 纯替代：Cross-Encoder 分数直接作为 similarity_score
            #    工业标准做法：Cross-Encoder 的全注意力深度匹配分数
            #    直接替代粗排分数，而非加权混合。RRF 的职责到
            #    "选出候选池"就已完成。
            for i, c in enumerate(candidates):
                reranker_score = normalized_scores[i]
                c["_reranker_score"] = round(reranker_score, 4)
                c["_pre_rerank_score"] = c.get("similarity_score", 0)  # 保留原始 RRF 分数用于调试
                c["similarity_score"] = reranker_score  # 纯 Cross-Encoder 分数 [0, 1]

            # 6. 按新分数降序排列并截取 top_k
            candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
            result = candidates[:top_k]

            logger.info(
                f"[Reranker] ✅ 精排完成: {len(candidates)} → {len(result)} 首 | Top3: "
                f"{[(r['song']['title'], round(r['similarity_score'], 4)) for r in result[:3]]}"
            )
            return result

        except Exception as e:
            logger.warning(f"[Reranker] ⚠️ 精排推理异常（降级返回原始列表）: {e}")
            return candidates
