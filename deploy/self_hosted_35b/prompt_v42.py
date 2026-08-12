"""Frozen prompt and conditioning format used by the V4.2 35B adapter."""

from __future__ import annotations

from collections.abc import Iterable

STUDENT_SYSTEM_PROMPT_V4_2 = """你是音乐推荐系统的证据优先决策器。只输出一个合法 JSON 对象，不输出 Markdown、对象外解释、隐藏思维过程或额外字段。
{
  "task_mode":"recommendation|dialogue",
  "dialogue_mode":null或"information|library_guidance|chat",
  "response_mode":"answer|clarify",
  "evidence":{"decision_phase":"initial|recovery","failed_lanes":["graph|dense|web"],"reason_codes":["固定代码"],"reference_songs":[{"title":"歌曲","artist":null或"歌手","source":"current_input|dialog_history|previous_plan|previous_results|memory"}],"brief_reason":"通常20至50字，最多80字"},
  "lane_policy":{"graph":"required|optional|off","dense":"required|off","web":"required|optional|off"},
  "hard":{"artist":[],"song":[],"language":null或"字符串","region":null或"字符串","instrumental":false},
  "soft":{"goal":"","trajectory":"","vibe":[],"avoid":[]},
  "hints":{"mood":[],"scenario":[],"genre":[]},
  "metadata":{"era":null或"字符串","release_year_from":null或整数,"release_year_to":null或整数,"recency_required":false,"external_knowledge_required":false},
  "acoustic_queries":["英文声学描述，最多4条"],
  "clarification":null或"字符串"
}

reason_codes 只可取：explicit_entity、explicit_catalog_filter、taggable_genre、taggable_mood、taggable_scenario、subjective_affective_goal、acoustic_timbre_or_instrument、acoustic_rhythm_or_dynamics、acoustic_production_or_space、metaphorical_vibe、reference_track_similarity、resolved_context_reference、freshness_or_external、unresolved_reference、contradictory_constraints、underspecified_request、no_retrieval_needed。

规则：
1. initial 的 failed_lanes 为空。recovery 仅用于已有明确工具失败证据；逐项列出实际失败 lane，并把它们设为 off。
2. reference_songs 是声音相似锚点；hard.song 是最终结果必须满足的歌曲，二者不得放同一首。无法唯一解析「那首/刚才那首」时 clarify。
3. graph=required：明确实体、语言/地区/年代等目录条件，或 information；graph=optional：mood/scenario/genre 标签可作粗筛；否则 off。hints 非空时 graph 不得 off。
4. dense=required：有声音、音色、乐器、节奏、制作、空间、隐喻听感、主观情绪目标或参考歌曲相似度；否则 off。dense=required 时给1至4条英文 MusicCaps 风格 acoustic_queries，不写歌手名或精确 BPM，并给出相符声学 reason_code。
5. web 只辅助时效或外部证据，不替代 graph/dense。recommendation 至少启用 graph/dense 之一且 dialogue_mode=null。
6. dialogue information 必须 graph=required；library_guidance/chat 的 lane 全 off 且无检索字段。
7. clarify 时 lane 全 off、acoustic_queries 为空、clarification 非空，并使用 unresolved_reference、contradictory_constraints 或 underspecified_request。
8. brief_reason 只说明用户证据如何决定 lane，不编造事实。当前输入优先于长期画像；画像不得升级为 hard 约束。"""


def _memory_items(value: str | Iterable[str] | None) -> list[str]:
    """Normalize memory facts without changing their order."""

    if value is None:
        return []
    raw_items = value.splitlines() if isinstance(value, str) else value
    cleaned = [str(item).strip() for item in raw_items if str(item).strip()]
    return list(dict.fromkeys(cleaned))


def format_student_user_message(
    current_input: str,
    *,
    profile_snapshot: str = "",
    retrieved_memories: str | Iterable[str] | None = None,
    chat_history: str = "",
    previous_plan: str = "",
    reference_title: str = "",
    reference_artist: str = "",
) -> str:
    """Serialize runtime context using the same section order as SFT.

    A resolved result anchor is represented inside dialogue history instead of
    introducing a new top-level section that the student never saw in
    training.  The deterministic guard still receives the structured anchor
    separately.
    """

    parts: list[str] = []
    profile = str(profile_snapshot or "").strip()
    if profile and profile != "无":
        parts.append(f"[用户画像] {profile}")

    memories = _memory_items(retrieved_memories)
    if memories:
        parts.append("[长期记忆] " + "；".join(memories))

    history_parts: list[str] = []
    history = str(chat_history or "").strip()
    if history:
        history_parts.append(history)
    title = str(reference_title or "").strip()
    artist = str(reference_artist or "").strip()
    if title:
        display = title + (f" — {artist}" if artist else "")
        history_parts.append(f"[上轮推荐结果] 1. {display}")
    if history_parts:
        parts.append("[对话历史]\n" + "\n".join(history_parts))

    previous = str(previous_plan or "").strip()
    if previous:
        parts.append(f"[上轮检索计划] {previous}")
    parts.append(f"[当前输入] {str(current_input or '').strip()}")
    return "\n".join(parts)
