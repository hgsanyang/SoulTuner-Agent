"""
Profile Synthesizer — 动态用户画像聚合器
=======================================
从 GraphZep 长期记忆 + Neo4j 行为统计中聚合结构化用户画像。

核心能力：
  1. 情绪基线：用户常态情绪倾向
  2. 情境偏好：不同时段/场景/情绪下的音乐偏好模式
  3. 品味进化：品味随时间的变化轨迹
  4. 负面偏好：明确不喜欢的风格
  5. 交互风格：用户表达方式分析

触发时机：
  - 每 PORTRAIT_REFRESH_INTERVAL 轮对话后，后台异步刷新
  - 用户手动点击"刷新画像"按钮
  - 服务启动时从 Neo4j 加载上次画像

使用 explain_llm 配置的模型来运行 LLM 聚合调用。
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field
from config.logging_config import get_logger

logger = get_logger(__name__)

# ---- 触发配置 ----
PORTRAIT_REFRESH_INTERVAL = 10  # 每 N 轮对话后自动触发画像刷新


# ============================================================
# Pydantic 输出 Schema — 定义 LLM 输出的结构化用户画像
# ============================================================

class SituationalPattern(BaseModel):
    """情境-偏好模式（时间/场景/活动/情绪 都算情境）"""
    situation: str = Field(description="情境描述，如'深夜独处''跑步时''心情低落时'")
    preferred_styles: List[str] = Field(default_factory=list, description="该情境下偏好的音乐风格")
    evidence: str = Field(default="", description="判断依据，引用具体的记忆事实")

class TasteShift(BaseModel):
    """品味进化轨迹中的一个阶段"""
    period: str = Field(description="时间段描述，如'4月初'")
    dominant_taste: str = Field(description="该阶段的主导品味")
    evidence: str = Field(default="", description="判断依据")

class UserPortrait(BaseModel):
    """动态用户画像 — Profile Synthesizer 的完整输出"""
    # 情绪基线
    emotional_baseline: str = Field(
        default="数据不足",
        description="用户的常态情绪倾向，如'偏内省忧郁，深夜活跃度高'"
    )
    # 情境偏好模式（合并了时间和场景维度）
    situational_patterns: List[SituationalPattern] = Field(
        default_factory=list,
        description="不同情境下的音乐偏好模式"
    )
    # 品味进化轨迹
    taste_evolution: List[TasteShift] = Field(
        default_factory=list,
        description="品味随时间的变化轨迹，按时间先后排列"
    )
    # 当前主导偏好（最近 2 周的核心品味）
    current_dominant_genres: List[str] = Field(default_factory=list)
    current_dominant_moods: List[str] = Field(default_factory=list)
    # 负面偏好
    dislike_signals: List[str] = Field(
        default_factory=list,
        description="用户明确不喜欢的风格/元素"
    )
    # 交互风格
    interaction_style: str = Field(
        default="数据不足",
        description="用户的交互偏好，如'探索型''精确指定型'"
    )
    # 用户主动设置的偏好（高权重，来自设置面板）
    user_declared_preferences: str = Field(
        default="",
        description="用户在设置面板中主动声明的偏好（带时间戳），权重最高"
    )
    # 聚合置信度
    confidence: str = Field(
        default="low",
        description="画像置信度: low(<5条记忆)/medium(5-20条)/high(>20条)"
    )
    # 一句话摘要（用于注入 prompt）
    one_line_summary: str = Field(
        default="暂无画像数据",
        description="一句话概括此用户的音乐品味特征"
    )


# ============================================================
# Profile Synthesizer 核心类
# ============================================================

class ProfileSynthesizer:
    """动态用户画像聚合器"""

    def __init__(self, user_id: str = "local_admin"):
        self.user_id = user_id
        self._cached_portrait: Optional[UserPortrait] = None
        self._conversation_count: int = 0

    # ---- 对话计数与自动触发 ----

    def increment_conversation(self) -> bool:
        """
        对话计数器 +1，返回是否应该触发画像刷新。
        由 recall_graphzep_memory 在每轮对话结束时调用。
        """
        self._conversation_count += 1
        should_refresh = self._conversation_count >= PORTRAIT_REFRESH_INTERVAL
        if should_refresh:
            self._conversation_count = 0
        return should_refresh

    # ---- 数据收集 ----

    async def _collect_graphzep_facts(self) -> str:
        """从 GraphZep 收集所有可用的带时间戳记忆"""
        try:
            from services.graphzep_client import get_graphzep_client
            client = get_graphzep_client()

            # 用宽泛的查询检索尽可能多的用户记忆
            facts = await client.search_facts(
                query="用户的音乐偏好、情绪、喜好、行为、品味",
                max_facts=30,
                search_type="hybrid",
            )
            if facts == "暂无用户长期记忆" or "服务不可用" in facts:
                return ""
            return facts
        except Exception as e:
            logger.warning(f"[ProfileSynth] GraphZep 数据收集失败: {e}")
            return ""

    async def _collect_neo4j_stats(self) -> Dict[str, Any]:
        """从 Neo4j 收集结构化行为统计"""
        try:
            from retrieval.neo4j_client import get_neo4j_client
            client = get_neo4j_client()
            if not client or not client.driver:
                return {}

            # 综合查询：Top 歌手 + Top 流派 + 不喜欢 + 经常跳过
            stats_query = """
            MATCH (u:User {id: $user_id})

            // 最喜欢的歌手
            OPTIONAL MATCH (u)-[r1:LIKES]->(s1:Song)-[:PERFORMED_BY]->(a:Artist)
            WITH u, a.name AS liked_artist, count(r1) AS like_count
            ORDER BY like_count DESC
            WITH u, collect(DISTINCT {artist: liked_artist, likes: like_count})[..5] AS top_artists

            // 最喜欢的流派（通过 BELONGS_TO_GENRE 关系获取）
            OPTIONAL MATCH (u)-[:LIKES]->(s2:Song)-[:BELONGS_TO_GENRE]->(g:Genre)
            WHERE g.name IS NOT NULL AND g.name <> ''
            WITH u, top_artists, g.name AS genre, count(*) AS genre_count
            ORDER BY genre_count DESC
            WITH u, top_artists, collect(DISTINCT {genre: genre, count: genre_count})[..5] AS top_genres

            // 不喜欢的歌曲
            OPTIONAL MATCH (u)-[:DISLIKES]->(ds:Song)
            WITH u, top_artists, top_genres, collect(DISTINCT ds.title)[..5] AS disliked_songs

            // 经常跳过
            OPTIONAL MATCH (u)-[sk:SKIPPED]->(ss:Song) WHERE sk.count >= 2
            WITH u, top_artists, top_genres, disliked_songs,
                 collect(DISTINCT ss.title)[..5] AS often_skipped

            // 用户主动设置的偏好
            RETURN top_artists, top_genres, disliked_songs, often_skipped,
                   u.preferred_genres AS declared_genres,
                   u.preferred_moods AS declared_moods,
                   u.preferred_scenarios AS declared_scenarios,
                   u.preferred_languages AS declared_languages,
                   u.profile_free_text AS declared_free_text,
                   u.profile_updated_at AS profile_updated_at
            """
            results = client.execute_query(stats_query, {"user_id": self.user_id})
            if results and results[0]:
                row = results[0]
                return {
                    "top_artists": row.get("top_artists", []),
                    "top_genres": row.get("top_genres", []),
                    "disliked_songs": row.get("disliked_songs", []),
                    "often_skipped": row.get("often_skipped", []),
                    "declared_genres": row.get("declared_genres", ""),
                    "declared_moods": row.get("declared_moods", ""),
                    "declared_scenarios": row.get("declared_scenarios", ""),
                    "declared_languages": row.get("declared_languages", ""),
                    "declared_free_text": row.get("declared_free_text", ""),
                    "profile_updated_at": str(row.get("profile_updated_at", "")),
                }
            return {}
        except Exception as e:
            logger.warning(f"[ProfileSynth] Neo4j 数据收集失败: {e}")
            return {}

    # ---- LLM 聚合 ----

    async def synthesize(self) -> UserPortrait:
        """
        核心方法：收集数据 → LLM 聚合 → 返回结构化画像。
        使用 explain_llm 配置。
        """
        import time as _time
        _t0 = _time.time()
        logger.info("[ProfileSynth] 开始画像聚合...")

        # ① 并行收集两个数据源
        graphzep_facts, neo4j_stats = await asyncio.gather(
            self._collect_graphzep_facts(),
            self._collect_neo4j_stats(),
        )

        # ② 构建用户主动声明的偏好文本（高权重）
        declared_text = self._format_declared_preferences(neo4j_stats)

        # ③ 如果两个数据源都为空，返回低置信度画像
        if not graphzep_facts and not neo4j_stats:
            logger.info("[ProfileSynth] 数据源均为空，返回默认画像")
            portrait = UserPortrait(
                confidence="low",
                one_line_summary="新用户，尚无足够数据构建画像",
                user_declared_preferences=declared_text,
            )
            self._cached_portrait = portrait
            return portrait

        # ④ 调用 LLM 聚合
        try:
            portrait = await self._llm_synthesize(graphzep_facts, neo4j_stats, declared_text)
        except Exception as e:
            logger.error(f"[ProfileSynth] LLM 聚合失败: {e}")
            # 降级：构造基础画像
            portrait = self._build_fallback_portrait(neo4j_stats, declared_text)

        self._cached_portrait = portrait
        _elapsed = _time.time() - _t0
        logger.info(f"[ProfileSynth] ✅ 画像聚合完成 ({_elapsed:.1f}s) | confidence={portrait.confidence} | summary={portrait.one_line_summary[:60]}")
        return portrait

    def _format_declared_preferences(self, neo4j_stats: Dict[str, Any]) -> str:
        """格式化用户主动声明的偏好（来自设置面板），标注时间戳和高权重"""
        parts = []
        updated_at = neo4j_stats.get("profile_updated_at", "")

        for key, label in [
            ("declared_genres", "偏好流派"),
            ("declared_moods", "情绪偏向"),
            ("declared_scenarios", "常听场景"),
            ("declared_languages", "语言偏好"),
        ]:
            raw = neo4j_stats.get(key, "")
            if raw:
                try:
                    values = json.loads(raw)
                    if values:
                        parts.append(f"{label}: {', '.join(values)}")
                except (ValueError, TypeError):
                    pass

        free_text = neo4j_stats.get("declared_free_text", "")
        if free_text and free_text.strip():
            parts.append(f"自述: {free_text.strip()}")

        if parts:
            time_note = f" (设置于: {updated_at})" if updated_at else ""
            return f"[用户主动设置，权重最高{time_note}] " + "；".join(parts)
        return ""

    async def _llm_synthesize(
        self,
        graphzep_facts: str,
        neo4j_stats: Dict[str, Any],
        declared_text: str,
    ) -> UserPortrait:
        """调用 LLM 生成结构化画像"""
        from llms.multi_llm import get_explain_chat_model
        from llms.prompts import PROFILE_SYNTHESIZER_PROMPT
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = get_explain_chat_model()

        # 格式化 Neo4j 统计
        top_artists_str = ", ".join(
            f"{a.get('artist', '?')}({a.get('likes', 0)}首)"
            for a in (neo4j_stats.get("top_artists") or [])
            if a.get("artist")
        ) or "无数据"

        top_genres_str = ", ".join(
            f"{g.get('genre', '?')}({g.get('count', 0)}首)"
            for g in (neo4j_stats.get("top_genres") or [])
            if g.get("genre")
        ) or "无数据"

        disliked_str = ", ".join(neo4j_stats.get("disliked_songs") or []) or "无"
        skipped_str = ", ".join(neo4j_stats.get("often_skipped") or []) or "无"

        # 构造 Prompt
        prompt = ChatPromptTemplate.from_template(PROFILE_SYNTHESIZER_PROMPT)
        chain = prompt | llm | StrOutputParser()

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        raw_output = await chain.ainvoke({
            "graphzep_facts": graphzep_facts or "无可用记忆数据",
            "top_artists": top_artists_str,
            "top_genres": top_genres_str,
            "disliked_songs": disliked_str,
            "often_skipped": skipped_str,
            "declared_preferences": declared_text or "用户未主动设置偏好",
            "current_time": current_time,
        })

        # 清理可能的 <think>...</think> 残留
        raw_output = raw_output.strip()
        if "<think>" in raw_output:
            think_end = raw_output.find("</think>")
            if think_end > 0:
                raw_output = raw_output[think_end + 8:].strip()

        # 清理 markdown 代码块包裹
        if raw_output.startswith("```"):
            lines = raw_output.split("\n")
            # 去掉首行 ```json 和末行 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_output = "\n".join(lines).strip()

        logger.info(f"[ProfileSynth] LLM 原始输出 (前200字): {raw_output[:200]}")

        # 解析 JSON（带字段名容错）
        try:
            data = json.loads(raw_output)
            data = self._normalize_llm_output(data)
            portrait = UserPortrait(**data)
            # 确保用户声明的偏好以高权重写入
            if declared_text:
                portrait.user_declared_preferences = declared_text
            return portrait
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[ProfileSynth] JSON 解析失败，尝试提取关键信息: {e}")
            # 尝试从 raw_output 中提取 JSON 子串
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    data = self._normalize_llm_output(data)
                    portrait = UserPortrait(**data)
                    if declared_text:
                        portrait.user_declared_preferences = declared_text
                    return portrait
                except Exception:
                    pass
            # 最终降级
            return self._build_fallback_portrait(
                {"disliked_songs": [], "top_genres": [], "top_artists": []},
                declared_text,
            )

    def _normalize_llm_output(self, data: Dict) -> Dict:
        """
        标准化 LLM 输出的字段名。
        LLM 可能用不同的字段名（如 time_range 代替 period），在这里统一映射。
        """
        # taste_evolution 字段名映射
        if "taste_evolution" in data and isinstance(data["taste_evolution"], list):
            normalized = []
            for item in data["taste_evolution"]:
                if isinstance(item, dict):
                    norm_item = {}
                    # period 的别名
                    norm_item["period"] = (
                        item.get("period")
                        or item.get("time_range")
                        or item.get("time_period")
                        or item.get("时间段", "未知时段")
                    )
                    # dominant_taste 的别名
                    norm_item["dominant_taste"] = (
                        item.get("dominant_taste")
                        or item.get("observation")
                        or item.get("taste")
                        or item.get("style")
                        or item.get("主导品味", "未知")
                    )
                    # evidence
                    norm_item["evidence"] = (
                        item.get("evidence")
                        or item.get("依据", "")
                    )
                    normalized.append(norm_item)
            data["taste_evolution"] = normalized

        # situational_patterns 字段名映射
        if "situational_patterns" in data and isinstance(data["situational_patterns"], list):
            normalized = []
            for item in data["situational_patterns"]:
                if isinstance(item, dict):
                    norm_item = {}
                    norm_item["situation"] = (
                        item.get("situation")
                        or item.get("context")
                        or item.get("scenario")
                        or item.get("time_period")
                        or item.get("情境", "未知情境")
                    )
                    norm_item["preferred_styles"] = (
                        item.get("preferred_styles")
                        or item.get("styles")
                        or item.get("music_styles")
                        or item.get("偏好风格", [])
                    )
                    norm_item["evidence"] = (
                        item.get("evidence")
                        or item.get("依据", "")
                    )
                    normalized.append(norm_item)
            data["situational_patterns"] = normalized

        return data

    def _build_fallback_portrait(self, neo4j_stats: Dict, declared_text: str) -> UserPortrait:
        """LLM 失败时的降级画像（从 Neo4j 统计中构造基础画像）"""
        genres = [g.get("genre", "") for g in (neo4j_stats.get("top_genres") or []) if g.get("genre")]
        artists = [a.get("artist", "") for a in (neo4j_stats.get("top_artists") or []) if a.get("artist")]

        summary_parts = []
        if genres:
            summary_parts.append(f"偏好{'/'.join(genres[:3])}")
        if artists:
            summary_parts.append(f"常听{'/'.join(artists[:3])}")
        summary = "，".join(summary_parts) if summary_parts else "数据不足，无法生成画像"

        return UserPortrait(
            current_dominant_genres=genres[:3],
            one_line_summary=summary,
            user_declared_preferences=declared_text,
            confidence="low",
        )

    # ---- 存储与读取 ----

    async def save_portrait(self, portrait: UserPortrait) -> bool:
        """将画像保存到 Neo4j User 节点"""
        try:
            from retrieval.neo4j_client import get_neo4j_client
            client = get_neo4j_client()
            if not client or not client.driver:
                return False

            portrait_json = portrait.model_dump_json(ensure_ascii=False)
            client.execute_query("""
            MATCH (u:User {id: $user_id})
            SET u.portrait_json = $portrait_json,
                u.portrait_summary = $summary,
                u.portrait_updated_at = datetime()
            """, {
                "user_id": self.user_id,
                "portrait_json": portrait_json,
                "summary": portrait.one_line_summary,
            })

            # 同步更新用户偏好标签（让 Tri-Anchor Jaccard 消费）
            if portrait.current_dominant_genres:
                client.execute_query("""
                MATCH (u:User {id: $user_id})
                SET u.portrait_genres = $genres
                """, {
                    "user_id": self.user_id,
                    "genres": json.dumps(portrait.current_dominant_genres, ensure_ascii=False),
                })

            if portrait.current_dominant_moods:
                client.execute_query("""
                MATCH (u:User {id: $user_id})
                SET u.portrait_moods = $moods
                """, {
                    "user_id": self.user_id,
                    "moods": json.dumps(portrait.current_dominant_moods, ensure_ascii=False),
                })

            logger.info(f"[ProfileSynth] 画像已保存到 Neo4j (user={self.user_id})")
            return True
        except Exception as e:
            logger.error(f"[ProfileSynth] 画像保存失败: {e}")
            return False

    async def load_portrait(self) -> Optional[UserPortrait]:
        """从 Neo4j 加载上次保存的画像"""
        try:
            from retrieval.neo4j_client import get_neo4j_client
            client = get_neo4j_client()
            if not client or not client.driver:
                return None

            result = client.execute_query("""
            MATCH (u:User {id: $user_id})
            RETURN u.portrait_json AS portrait_json,
                   u.portrait_summary AS summary,
                   u.portrait_updated_at AS updated_at
            """, {"user_id": self.user_id})

            if result and result[0] and result[0].get("portrait_json"):
                data = json.loads(result[0]["portrait_json"])
                portrait = UserPortrait(**data)
                self._cached_portrait = portrait
                logger.info(
                    f"[ProfileSynth] 已从 Neo4j 加载画像 | "
                    f"confidence={portrait.confidence} | "
                    f"updated_at={result[0].get('updated_at', '?')}"
                )
                return portrait
            return None
        except Exception as e:
            logger.warning(f"[ProfileSynth] 画像加载失败: {e}")
            return None

    def get_cached_portrait(self) -> Optional[UserPortrait]:
        """获取内存中缓存的画像（不触发任何 I/O）"""
        return self._cached_portrait

    def get_portrait_for_prompt(self) -> str:
        """
        格式化画像为 prompt 注入文本。
        如果有画像，返回丰富的描述；否则返回空字符串。
        """
        portrait = self._cached_portrait
        if not portrait or portrait.confidence == "low" and portrait.one_line_summary == "暂无画像数据":
            return ""

        parts = []

        # ① 用户主动声明的偏好（最高权重，优先展示）
        if portrait.user_declared_preferences:
            parts.append(portrait.user_declared_preferences)

        # ② 一句话摘要
        parts.append(f"画像摘要: {portrait.one_line_summary}")

        # ③ 情境偏好（如果有）
        if portrait.situational_patterns:
            situations = "; ".join(
                f"{sp.situation}→{'/'.join(sp.preferred_styles[:3])}"
                for sp in portrait.situational_patterns[:3]
            )
            parts.append(f"情境偏好: {situations}")

        # ④ 负面偏好（重要，影响排除逻辑）
        if portrait.dislike_signals:
            parts.append(f"不喜欢: {', '.join(portrait.dislike_signals[:5])}")

        return "；".join(parts)


# ============================================================
# 全局单例
# ============================================================

_synthesizer: Optional[ProfileSynthesizer] = None


def get_profile_synthesizer(user_id: str = "local_admin") -> ProfileSynthesizer:
    """获取 ProfileSynthesizer 全局单例"""
    global _synthesizer
    if _synthesizer is None or _synthesizer.user_id != user_id:
        _synthesizer = ProfileSynthesizer(user_id=user_id)
    return _synthesizer


async def trigger_portrait_refresh(user_id: str = "local_admin") -> Optional[UserPortrait]:
    """
    触发画像刷新的便捷函数（异步）。
    收集数据 → LLM 聚合 → 保存 → 返回画像。
    """
    synth = get_profile_synthesizer(user_id)
    portrait = await synth.synthesize()
    await synth.save_portrait(portrait)
    return portrait
