"""
音乐旅程编排器 ── LLM 驱动版===============================
两阶段管线：
  阶段 1: LLM Structured Output → MusicJourneyPlan（故事理解 + 情绪拆解）  阶段 2: 逐段并发检索 → 实时 SSE 推送歌曲"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional, Set

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


# ────── 默认声学提示（当 LLM 未生成 acoustic_hint 时的 fallback）──────
_MOOD_ACOUSTIC_FALLBACK: Dict[str, str] = {
    "平静": "Gentle ambient music with soft piano, light strings, warm reverb. Slow tempo around 60-75 BPM. Calm and peaceful atmosphere.",
    "放松": "Relaxing lounge music with acoustic guitar, soft pads, gentle brushed drums. Around 70-85 BPM. Warm and cozy.",
    "专注": "Minimal ambient with soft synth textures, sparse piano notes, no vocals. Clean and focused, around 85-95 BPM.",
    "活力": "Upbeat pop-rock with driving drums, bright guitar, energetic vocals. Around 120-140 BPM. Positive and motivating.",
    "开心": "Bright cheerful pop with catchy melody, handclaps, uplifting chorus. Around 110-130 BPM. Sunny and joyful.",
    "悲伤": "Melancholic ballad with slow piano, gentle cello, soft vocals. Around 60-70 BPM. Tender and emotional.",
    "浪漫": "Romantic R&B with smooth keys, warm bass, intimate vocals. Around 75-90 BPM. Warm candlelight atmosphere.",
    "疗愈": "Healing ambient with nature sounds, soft harp, gentle flute. Around 65-80 BPM. Soothing and restorative.",
    "怀旧": "Nostalgic folk-pop with acoustic guitar, harmonica, warm analog tone. Around 85-100 BPM. Warm sunset memories.",
    "热血": "Intense rock anthem with heavy drums, power chords, epic vocals. Around 140-160 BPM. Adrenaline rush.",
    "激昂": "High energy electronic dance music with pulsing synths, four-on-the-floor kick. Around 128-140 BPM. Euphoric and explosive.",
}


def _get_llm():
    """获取当前可用的 LLM 实例"""
    try:
        from agent.music_graph import get_llm
        return get_llm()
    except Exception:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=str(settings.OPENAI_BASE_URL),
            temperature=0.7,
        )


async def plan_journey(
    story: Optional[str] = None,
    mood_transitions: Optional[List[Dict[str, Any]]] = None,
    duration: int = 60,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    阶段 1: 使用 LLM Structured Output 将用户输入解析为旅程规划。
    支持两种模式:
    - 故事驱动模式: 用户输入自然语言故事
    - 情绪曲线模式: 用户提供情绪节点列表
    """
    from schemas.journey_plan import MusicJourneyPlan
    from llms.prompts import MUSIC_JOURNEY_PLANNER_PROMPT

    ctx = context or {}
    location = ctx.get("location", "未指定")
    weather = ctx.get("weather", "未指定")

    # 准备 story 输入
    story_input = story or "（用户未提供故事，请根据情绪曲线生成旅程）"

    # 准备情绪曲线输入
    if mood_transitions:
        mood_curve_input = json.dumps(mood_transitions, ensure_ascii=False)
    else:
        mood_curve_input = "（用户使用故事驱动模式，无情绪曲线数据）"

    prompt_text = MUSIC_JOURNEY_PLANNER_PROMPT.format(
        story_input=story_input,
        mood_curve_input=mood_curve_input,
        location=location,
        weather=weather,
        duration=duration,
    )

    llm = _get_llm()

    try:
        # 尝试 Structured Output（一次调用完成解析）
        structured_llm = llm.with_structured_output(MusicJourneyPlan)
        plan: MusicJourneyPlan = await asyncio.get_event_loop().run_in_executor(
            None, lambda: structured_llm.invoke(prompt_text)
        )
        result = plan.model_dump()
        logger.info(f"[Journey] LLM 规划完成: {result['title']} ({result['total_segments']} 段")
        return result

    except Exception as e:
        logger.warning(f"[Journey] Structured Output 失败: {e}，尝试原始 JSON 解析...")

        # Fallback: 直接调用 LLM 获取 JSON
        try:
            raw_response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: llm.invoke(prompt_text)
            )
            content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            # 尝试提取 JSON
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            plan = MusicJourneyPlan(**parsed)
            result = plan.model_dump()
            logger.info(f"[Journey] JSON Fallback 规划完成: {result['title']}")
            return result
        except Exception as e2:
            logger.error(f"[Journey] JSON Fallback 也失败: {e2}")
            # 最终 fallback: 基于规则生成简单旅程            return _rule_based_fallback(story, mood_transitions, duration)


def _rule_based_fallback(
    story: Optional[str],
    mood_transitions: Optional[List[Dict[str, Any]]],
    duration: int
) -> Dict[str, Any]:
    """当 LLM 完全不可用时的规则兜底"""
    logger.info("[Journey] 使用规则兜底生成旅程")

    if mood_transitions and len(mood_transitions) >= 2:
        segments = []
        total = len(mood_transitions)
        for i, mt in enumerate(mood_transitions):
            segments.append({
                "segment_id": i,
                "mood": mt.get("mood", "平静"),
                "description": f"情绪片段 {i + 1}",
                "duration_ratio": round(1.0 / total, 2),
                "acoustic_hint": _MOOD_ACOUSTIC_FALLBACK.get(mt.get("mood", "平静"), ""),
                "graph_genre_filter": None,
                "graph_mood_filter": mt.get("mood", "平静"),
                "graph_scenario_filter": None,
                "songs_count": 3,
            })
        # 修正最后一段使 ratio 总和 = 1.0
        current_sum = sum(s["duration_ratio"] for s in segments[:-1])
        segments[-1]["duration_ratio"] = round(1.0 - current_sum, 2)
        return {
            "title": "你的音乐旅程",
            "total_segments": len(segments),
            "segments": segments,
            "reasoning": "基于情绪曲线的规则生成",
        }

    # 默认 4 段旅程    default_moods = ["平静", "专注", "活力", "放松"]
    segments = []
    for i, mood in enumerate(default_moods):
        segments.append({
            "segment_id": i,
            "mood": mood,
            "description": f"旅程第 {i + 1} 阶段 — {mood}",
            "duration_ratio": 0.25,
            "acoustic_hint": _MOOD_ACOUSTIC_FALLBACK.get(mood, ""),
            "graph_genre_filter": None,
            "graph_mood_filter": mood,
            "graph_scenario_filter": None,
            "songs_count": 2,
        })
    return {
        "title": story[:8] + "..." if story and len(story) > 8 else (story or "默认旅程"),
        "total_segments": 4,
        "segments": segments,
        "reasoning": "LLM 不可用，使用默认规则生成",
    }


async def retrieve_for_segment(
    segment: Dict[str, Any],
    seen_songs: Set[str],
) -> List[Dict[str, Any]]:
    """
    阶段 2: 为单个旅程片段检索歌曲
    Args:
        segment: 单个 JourneySegmentPlan 或 dict
        seen_songs: 已推荐的歌曲标题集合（避免重复）

    Returns:
        歌曲列表 [{title, artist, audio_url, cover_url, ...}, ...]
    """
    from retrieval.hybrid_retrieval import MusicHybridRetrieval

    mood = segment.get("mood", "平静")
    description = segment.get("description", "")
    acoustic_hint = segment.get("acoustic_hint", "") or _MOOD_ACOUSTIC_FALLBACK.get(mood, "")
    songs_count = segment.get("songs_count", 2)

    # 构建检索查询
    query = f"推荐适合「{mood}」氛围的歌曲：{description}"

    # 构建 precomputed_plan（复用混合检索的现有接口）
    plan = {
        "use_graph": True,
        "graph_entities": [],
        "graph_genre_filter": segment.get("graph_genre_filter"),
        "graph_scenario_filter": segment.get("graph_scenario_filter"),
        "graph_mood_filter": segment.get("graph_mood_filter", mood),
        "graph_language_filter": None,
        "graph_region_filter": None,
        "use_vector": bool(acoustic_hint),
        "vector_acoustic_query": acoustic_hint,
        "use_web_search": False,
        "web_search_keywords": "",
    }

    logger.info(f"[Journey Segment {segment.get('segment_id', '?')}] "
                f"mood={mood}, genre={plan['graph_genre_filter']}, "
                f"songs_count={songs_count}")

    try:
        retriever = MusicHybridRetrieval()
        # retrieve() 是同步的，放到线程池执行
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: retriever.retrieve(
                query=query,
                limit=songs_count + 3,  # 多拿几首以便去重
                precomputed_plan=plan,
            )
        )

        songs = []
        raw_data = result.data if hasattr(result, "data") else result
        if isinstance(raw_data, list):
            for item in raw_data:
                song = item.get("song", item) if isinstance(item, dict) else item
                if not isinstance(song, dict):
                    continue
                title = song.get("title", "")
                # 去重：跳过已在旅程中出现的歌曲                dedup_key = f"{title}-{song.get('artist', '')}".lower()
                if dedup_key in seen_songs:
                    continue
                seen_songs.add(dedup_key)
                songs.append(song)
                if len(songs) >= songs_count:
                    break

        logger.info(f"[Journey Segment {segment.get('segment_id', '?')}] "
                    f"检索到 {len(songs)} 首歌曲")
        return songs

    except Exception as e:
        logger.error(f"[Journey Segment {segment.get('segment_id', '?')}] 检索失败: {e}")
        return []


async def stream_journey_events(
    story: Optional[str] = None,
    mood_transitions: Optional[List[Dict[str, Any]]] = None,
    duration: int = 60,
    context: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    完整的旅程生成管线，以 SSE 事件流的形式输出。    由 api/server.py 的 stream_journey() 调用。
    事件类型:
        journey_start    → 旅程开始        thinking         → 思考状态更新        journey_info     → 旅程元信息（段数/总时长/总歌曲数)
        segment_start    → 片段开始（含 segment 元信息）
        song             → 片段内的单首歌曲
        segment_complete → 片段完成
        transition_point → 片段过渡标记
        journey_complete → 旅程完成
        error            → 错误
    """
    try:
        # ── 阶段 1: 旅程规划 ──
        yield {"type": "journey_start", "message": "正在分析你的故事与情绪.."}
        yield {"type": "thinking", "message": "正在让 AI 解读故事中的情绪变化..."}

        plan = await plan_journey(
            story=story,
            mood_transitions=mood_transitions,
            duration=duration,
            context=context,
        )

        segments = plan.get("segments", [])
        total_songs_estimate = sum(s.get("songs_count", 3) for s in segments)
        # 每首歌约 4 分钟，总时长由歌曲数决定        estimated_total_duration = total_songs_estimate * 4

        yield {"type": "thinking", "message": f"规划完成：{len(segments)} 个情绪阶段，预计 {total_songs_estimate} 首歌曲"}

        # 发送旅程元信息
        yield {
            "type": "journey_info",
            "total_segments": len(segments),
            "total_duration": estimated_total_duration,
            "total_songs": total_songs_estimate,
        }

        # ── 阶段 2: 逐段检索 ──
        seen_songs: Set[str] = set()
        all_segments_result = []
        actual_total_songs = 0

        cumulative_time = 0.0
        for i, segment in enumerate(segments):
            songs_count = segment.get("songs_count", 3)
            seg_duration = round(songs_count * 4, 1)  # 每首歌约 4 分钟
            start_time = round(cumulative_time, 1)

            # 发送 segment_start（前端期望的完整结构）
            segment_info = {
                "segment_id": segment.get("segment_id", i),
                "mood": segment.get("mood", ""),
                "description": segment.get("description", ""),
                "duration": seg_duration,
                "start_time": start_time,
                "total_songs": songs_count,
                "songs": [],
            }
            yield {"type": "segment_start", "segment": segment_info}
            cumulative_time += seg_duration

            # 检索歌曲
            songs = await retrieve_for_segment(segment, seen_songs)

            # 逐首推送歌曲
            for song in songs:
                yield {
                    "type": "song",
                    "segment_id": segment.get("segment_id", i),
                    "song": song,
                }
                actual_total_songs += 1
                await asyncio.sleep(0.15)  # 控制推送速度，让前端有动画效果
            # 发送 segment_complete
            yield {
                "type": "segment_complete",
                "segment_id": segment.get("segment_id", i),
            }

            # 发送过渡标记（非最后一段）
            if i < len(segments) - 1:
                yield {
                    "type": "transition_point",
                    "from_segment": segment.get("segment_id", i),
                    "to_segment": segments[i + 1].get("segment_id", i + 1),
                }

            # 保存完整段信息用于最终 complete 事件
            segment_info["songs"] = songs
            all_segments_result.append(segment_info)

        # ── 旅程完成 ──
        actual_total_duration = actual_total_songs * 4
        yield {
            "type": "journey_complete",
            "result": {
                "title": plan.get("title", ""),
                "segments": all_segments_result,
                "total_duration": actual_total_duration,
                "total_songs": actual_total_songs,
                "reasoning": plan.get("reasoning", ""),
            },
        }

    except Exception as e:
        logger.error(f"[Journey] 旅程生成失败: {e}", exc_info=True)
        yield {"type": "error", "error": f"旅程生成失败：{str(e)}"}
