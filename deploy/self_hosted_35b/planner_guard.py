"""Deterministic safety layer for the SoulTuner evidence-first planner.

This module intentionally has no third-party dependencies so the public demo,
the Gallery notebook, and unit tests all share the same fail-closed behavior.
The fine-tuned 35B model is treated as a *candidate* planner. A candidate is
accepted only when its task type agrees with deterministic facts and its lane
choices stay inside explicit safety bounds. The deterministic parser defines
minimum obligations; it does not pretend that a small keyword list can fully
classify every musical description. Otherwise this module returns a bounded,
auditable fallback plan.

``brief_reason`` is a short public explanation.  It is not hidden chain of
thought and execution never parses it.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any


PLANNER_VERSION = "soultuner_guarded_v1"
LANE_MODES = {"required", "optional", "off"}
TOP_LEVEL_KEYS = {
    "task_mode",
    "dialogue_mode",
    "response_mode",
    "evidence",
    "lane_policy",
    "hard",
    "soft",
    "hints",
    "metadata",
    "acoustic_queries",
    "clarification",
}
OBJECT_KEYS = {
    "evidence": {
        "decision_phase",
        "failed_lanes",
        "reason_codes",
        "reference_songs",
        "brief_reason",
    },
    "lane_policy": {"graph", "dense", "web"},
    "hard": {"artist", "song", "language", "region", "instrumental"},
    "soft": {"goal", "trajectory", "vibe", "avoid"},
    "hints": {"mood", "scenario", "genre"},
    "metadata": {
        "era",
        "release_year_from",
        "release_year_to",
        "recency_required",
        "external_knowledge_required",
    },
}
REASON_CODES = {
    "explicit_entity",
    "explicit_catalog_filter",
    "taggable_genre",
    "taggable_mood",
    "taggable_scenario",
    "subjective_affective_goal",
    "acoustic_timbre_or_instrument",
    "acoustic_rhythm_or_dynamics",
    "acoustic_production_or_space",
    "metaphorical_vibe",
    "reference_track_similarity",
    "resolved_context_reference",
    "freshness_or_external",
    "unresolved_reference",
    "contradictory_constraints",
    "underspecified_request",
    "no_retrieval_needed",
}

_EMPTY_THINKING_PREFIX = re.compile(r"\A<think>\s*</think>\s*", re.DOTALL)
_JSON_FENCE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL | re.IGNORECASE)


def parse_candidate_content(content: str) -> dict[str, Any]:
    """Parse one public JSON candidate while rejecting non-empty thinking."""
    text = str(content or "").strip()
    if text.startswith("<think>"):
        text, count = _EMPTY_THINKING_PREFIX.subn("", text, count=1)
        if count != 1:
            raise ValueError("model returned non-empty or unterminated thinking")
    fenced = _JSON_FENCE.fullmatch(text)
    if fenced:
        text = fenced.group(1).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("model candidate must be a JSON object")
    return payload

MOODS = {
    "开心": "开心",
    "快乐": "快乐",
    "治愈": "治愈",
    "温暖": "温暖",
    "难过": "难过",
    "伤心": "伤心",
    "心情很差": "低落",
    "低落": "低落",
    "焦虑": "焦虑",
    "平静": "平静",
    "浪漫": "浪漫",
    "孤独": "孤独",
    "热血": "热血",
    "怀旧": "怀旧",
}
SCENARIOS = {
    "跑步": "跑步",
    "健身": "健身",
    "学习": "学习",
    "工作": "工作",
    "睡前": "睡前",
    "通勤": "通勤",
    "开车": "驾驶",
    "聚会": "聚会",
    "旅行": "旅行",
}
GENRES = {
    "摇滚": "摇滚",
    "爵士": "爵士",
    "古典": "古典",
    "民谣": "民谣",
    "电子": "电子",
    "说唱": "说唱",
    "嘻哈": "嘻哈",
    "r&b": "R&B",
    "metal": "金属",
    "金属": "金属",
    "流行": "流行",
}
ACOUSTIC_TERMS = {
    "bass",
    "base",
    "低音",
    "贝斯",
    "鼓声",
    "鼓点",
    "重鼓",
    "人声",
    "女声",
    "男声",
    "吉他",
    "钢琴",
    "弦乐",
    "合成器",
    "失真",
    "混响",
    "空间感",
    "动态",
    "节奏",
    "音色",
    "听感",
    "氛围",
    "速度",
    "bpm",
}
REFERENCE_MARKERS = {
    "刚刚那首",
    "刚才那首",
    "上一首",
    "这首歌",
    "类似这首",
    "和它像",
    "相似的歌",
}
FRESHNESS_TERMS = {"最新", "最近", "刚发行", "本周", "今年", "热榜", "实时", "新闻"}
GUIDANCE_TERMS = {"怎么使用", "如何使用", "怎么导入", "如何导入", "收藏在哪", "歌单在哪", "怎么设置"}
INFO_TERMS = {"是谁", "介绍一下", "资料", "哪年发行", "什么时候发行", "什么专辑", "创作背景"}


def _contains_any(text: str, values: set[str]) -> bool:
    lowered = text.casefold()
    return any(value.casefold() in lowered for value in values)


def _matched_values(text: str, mapping: Mapping[str, str]) -> list[str]:
    lowered = text.casefold()
    return list(dict.fromkeys(value for key, value in mapping.items() if key.casefold() in lowered))


def _quoted_entities(text: str) -> list[str]:
    patterns = [r"《([^》]{1,80})》", r"[\"“]([^\"”]{1,80})[\"”]"]
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.strip() for match in re.findall(pattern, text) if match.strip())
    return list(dict.fromkeys(found))[:4]


def _empty_payload() -> dict[str, Any]:
    return {
        "hard": {"artist": [], "song": [], "language": None, "region": None, "instrumental": False},
        "soft": {"goal": "", "trajectory": "", "vibe": [], "avoid": []},
        "hints": {"mood": [], "scenario": [], "genre": []},
        "metadata": {
            "era": None,
            "release_year_from": None,
            "release_year_to": None,
            "recency_required": False,
            "external_knowledge_required": False,
        },
        "acoustic_queries": [],
        "clarification": None,
    }


def _base_plan() -> dict[str, Any]:
    payload = _empty_payload()
    return {
        "version": PLANNER_VERSION,
        "source": "deterministic_guard",
        "task_mode": "recommendation",
        "dialogue_mode": None,
        "response_mode": "answer",
        "evidence": {
            "decision_phase": "initial",
            "failed_lanes": [],
            "reason_codes": [],
            "reference_songs": [],
            "brief_reason": "",
        },
        "lane_policy": {"graph": "off", "dense": "off", "web": "off"},
        **payload,
    }


def _finalize(plan: dict[str, Any]) -> dict[str, Any]:
    plan["evidence"]["reason_codes"] = list(dict.fromkeys(plan["evidence"]["reason_codes"]))[:8]
    plan["evidence"]["brief_reason"] = str(plan["evidence"]["brief_reason"]).strip()[:80]
    plan["execution"] = compile_execution(plan["lane_policy"])
    return plan


def build_safe_plan(user_text: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a conservative evidence/lane plan from public request features."""

    text = str(user_text or "").strip()
    context = dict(context or {})
    plan = _base_plan()
    if not text:
        plan.update(task_mode="dialogue", dialogue_mode="chat", response_mode="clarify")
        plan["clarification"] = "请告诉我你想听什么样的音乐，或给我一首参考歌曲。"
        plan["evidence"]["reason_codes"] = ["underspecified_request"]
        plan["evidence"]["brief_reason"] = "请求为空，需要先补充音乐需求"
        return _finalize(plan)

    if _contains_any(text, GUIDANCE_TERMS):
        plan.update(task_mode="dialogue", dialogue_mode="library_guidance")
        plan["evidence"]["reason_codes"] = ["no_retrieval_needed"]
        plan["evidence"]["brief_reason"] = "这是产品使用问题，不应触发音乐召回"
        return _finalize(plan)

    if _contains_any(text, {"你好", "谢谢", "你是谁", "早上好", "晚安"}) and len(text) <= 16:
        plan.update(task_mode="dialogue", dialogue_mode="chat")
        plan["evidence"]["reason_codes"] = ["no_retrieval_needed"]
        plan["evidence"]["brief_reason"] = "普通对话无需访问检索系统"
        return _finalize(plan)

    quoted = _quoted_entities(text)
    if _contains_any(text, INFO_TERMS):
        plan.update(task_mode="dialogue", dialogue_mode="information")
        plan["lane_policy"]["graph"] = "required"
        plan["hard"]["song"] = quoted
        plan["evidence"]["reason_codes"] = ["explicit_entity"] if quoted else ["underspecified_request"]
        plan["evidence"]["brief_reason"] = "事实型音乐问题优先查询可追溯的图谱信息"
        return _finalize(plan)

    reference_requested = _contains_any(text, REFERENCE_MARKERS)
    reference_title = str(context.get("reference_title") or "").strip()
    reference_artist = str(context.get("reference_artist") or "").strip() or None
    if reference_requested and not reference_title:
        plan.update(task_mode="recommendation", response_mode="clarify")
        plan["clarification"] = "我还不能唯一确定你指的是哪首歌，请提供歌名或重新播放参考歌曲。"
        plan["evidence"]["reason_codes"] = ["unresolved_reference"]
        plan["evidence"]["brief_reason"] = "上下文中的参考歌曲无法唯一解析"
        return _finalize(plan)

    moods = _matched_values(text, MOODS)
    scenarios = _matched_values(text, SCENARIOS)
    genres = _matched_values(text, GENRES)
    has_acoustic = _contains_any(text, ACOUSTIC_TERMS)
    has_affective = bool(moods) or _contains_any(text, {"感觉", "氛围", "让我", "陪我", "适合我"})
    has_freshness = _contains_any(text, FRESHNESS_TERMS)

    plan["hints"]["mood"] = moods
    plan["hints"]["scenario"] = scenarios
    plan["hints"]["genre"] = genres
    plan["metadata"]["recency_required"] = has_freshness
    plan["metadata"]["external_knowledge_required"] = has_freshness
    plan["soft"]["goal"] = text[:240]
    plan["soft"]["vibe"] = list(dict.fromkeys([*moods, *scenarios]))[:6]

    graph_evidence = bool(moods or scenarios or genres or quoted)
    dense_evidence = bool(has_acoustic or has_affective or reference_title)

    if quoted and not reference_requested:
        plan["hard"]["song"] = quoted
        plan["evidence"]["reason_codes"].append("explicit_entity")
    if genres:
        plan["evidence"]["reason_codes"].append("taggable_genre")
    if moods:
        plan["evidence"]["reason_codes"].append("taggable_mood")
    if scenarios:
        plan["evidence"]["reason_codes"].append("taggable_scenario")
    if has_affective:
        plan["evidence"]["reason_codes"].append("subjective_affective_goal")
    if has_acoustic:
        plan["evidence"]["reason_codes"].append("acoustic_timbre_or_instrument")
    if has_freshness:
        plan["evidence"]["reason_codes"].append("freshness_or_external")
    if reference_title:
        plan["evidence"]["reference_songs"] = [
            {"title": reference_title[:240], "artist": reference_artist, "source": "previous_results"}
        ]
        plan["evidence"]["reason_codes"].extend(
            ["reference_track_similarity", "resolved_context_reference"]
        )

    if dense_evidence:
        plan["lane_policy"]["dense"] = "required"
        if not reference_title:
            plan["acoustic_queries"] = [text[:240]]
    if graph_evidence:
        # Affect/acoustics describe sound, so graph tags are useful secondary
        # evidence but dense remains authoritative.  Pure catalog constraints
        # use graph as the required lane.
        plan["lane_policy"]["graph"] = "optional" if dense_evidence else "required"
    if has_freshness:
        plan["lane_policy"]["web"] = "required"
        if plan["lane_policy"]["graph"] == "off":
            plan["lane_policy"]["graph"] = "optional"

    if all(mode == "off" for mode in plan["lane_policy"].values()):
        plan["lane_policy"]["dense"] = "required"
        plan["acoustic_queries"] = [text[:240]]
        plan["evidence"]["reason_codes"].append("metaphorical_vibe")

    graph_mode = plan["lane_policy"]["graph"]
    dense_mode = plan["lane_policy"]["dense"]
    if graph_mode == "optional" and dense_mode == "required":
        reason = "标签可辅助粗筛，主观听感与相似度由声学检索主导"
    elif graph_mode == "required" and dense_mode == "off":
        reason = "实体、类型或场景可由图谱目录约束直接检索"
    elif graph_mode == "off" and dense_mode == "required":
        reason = "请求描述的是听感或声学特征，应以向量召回为主"
    else:
        reason = "请求同时包含目录约束与声学目标，按角色组合检索"
    if has_freshness:
        reason = "请求包含时效信息，需联网核验并结合本地音乐召回"
    plan["evidence"]["brief_reason"] = reason
    return _finalize(plan)


def compile_execution(lane_policy: Mapping[str, str]) -> dict[str, Any]:
    """Compile semantic lane roles into deterministic tools and weights."""

    graph = str(lane_policy.get("graph", "off"))
    dense = str(lane_policy.get("dense", "off"))
    web = str(lane_policy.get("web", "off"))
    profiles = {
        ("required", "off"): ("graph_only", 1.0, 0.0),
        ("required", "optional"): ("graph_primary", 0.75, 0.25),
        ("required", "required"): ("balanced_hybrid", 0.5, 0.5),
        ("optional", "required"): ("dense_primary", 0.25, 0.75),
        ("off", "required"): ("dense_only", 0.0, 1.0),
        ("off", "off"): ("no_retrieval", 0.0, 0.0),
    }
    profile, graph_weight, dense_weight = profiles.get((graph, dense), ("guarded_hybrid", 0.5, 0.5))
    tools = [lane for lane, mode in (("graph", graph), ("dense", dense), ("web", web)) if mode != "off"]
    return {
        "profile": profile,
        "tools": tools,
        "weights": {"graph": graph_weight, "dense": dense_weight},
        "web_required": web == "required",
    }


def _candidate_findings(candidate: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    candidate_keys = set(candidate)
    if candidate_keys != TOP_LEVEL_KEYS:
        missing = sorted(TOP_LEVEL_KEYS - candidate_keys)
        extra = sorted(candidate_keys - TOP_LEVEL_KEYS)
        if missing:
            findings.append(f"candidate 缺少字段：{', '.join(missing)}")
        if extra:
            findings.append(f"candidate 含额外字段：{', '.join(extra)}")
    if candidate.get("task_mode") not in {"recommendation", "dialogue"}:
        findings.append("candidate.task_mode 非法")
    if candidate.get("dialogue_mode") not in {None, "information", "library_guidance", "chat"}:
        findings.append("candidate.dialogue_mode 非法")
    if candidate.get("response_mode", "answer") not in {"answer", "clarify"}:
        findings.append("candidate.response_mode 非法")

    for field, expected_keys in OBJECT_KEYS.items():
        value = candidate.get(field)
        if not isinstance(value, Mapping):
            findings.append(f"candidate.{field} 缺失或不是对象")
        elif set(value) != expected_keys:
            findings.append(f"candidate.{field} 字段集合不匹配")

    policy = candidate.get("lane_policy")
    if isinstance(policy, Mapping):
        for lane in ("graph", "dense", "web"):
            if policy.get(lane, "off") not in LANE_MODES:
                findings.append(f"candidate.lane_policy.{lane} 非法")
        if policy.get("dense") == "optional":
            findings.append("dense 不允许 optional")

    evidence = candidate.get("evidence")
    if isinstance(evidence, Mapping):
        if evidence.get("decision_phase") not in {"initial", "recovery"}:
            findings.append("candidate.evidence.decision_phase 非法")
        failed_lanes = evidence.get("failed_lanes")
        if not isinstance(failed_lanes, list) or any(lane not in {"graph", "dense", "web"} for lane in failed_lanes):
            findings.append("candidate.evidence.failed_lanes 非法")
        reason_codes = evidence.get("reason_codes")
        if not isinstance(reason_codes, list) or any(code not in REASON_CODES for code in reason_codes):
            findings.append("candidate.evidence.reason_codes 非法")
        reference_songs = evidence.get("reference_songs")
        if not isinstance(reference_songs, list):
            findings.append("candidate.evidence.reference_songs 非法")
        brief_reason = evidence.get("brief_reason")
        if not isinstance(brief_reason, str) or not brief_reason.strip():
            findings.append("candidate.evidence.brief_reason 为空")
        elif len(brief_reason) > 80:
            findings.append("candidate.evidence.brief_reason 超过 80 字")

    acoustic_queries = candidate.get("acoustic_queries")
    if (
        not isinstance(acoustic_queries, list)
        or len(acoustic_queries) > 4
        or any(not isinstance(query, str) for query in acoustic_queries)
    ):
        findings.append("candidate.acoustic_queries 非法")
    elif any(not query.strip() for query in acoustic_queries):
        findings.append("candidate.acoustic_queries 包含空字符串")

    if isinstance(policy, Mapping):
        dense_mode = policy.get("dense", "off")
        graph_mode = policy.get("graph", "off")
        if dense_mode == "required" and not (
            isinstance(acoustic_queries, list) and 1 <= len(acoustic_queries) <= 4
        ):
            findings.append("dense=required 时必须提供 1 至 4 条 acoustic_queries")

        hints = candidate.get("hints")
        if isinstance(hints, Mapping) and any(hints.get(key) for key in ("mood", "scenario", "genre")):
            if graph_mode == "off":
                findings.append("hints 非空时 graph 不得为 off")

        response_mode = candidate.get("response_mode", "answer")
        if response_mode == "clarify":
            if any(policy.get(lane, "off") != "off" for lane in ("graph", "dense", "web")):
                findings.append("clarify 时所有检索通道必须为 off")
            if acoustic_queries:
                findings.append("clarify 时 acoustic_queries 必须为空")
            clarification = candidate.get("clarification")
            if not isinstance(clarification, str) or not clarification.strip():
                findings.append("clarify 时 clarification 必须非空")

        if candidate.get("task_mode") == "recommendation" and response_mode == "answer":
            if graph_mode == "off" and dense_mode == "off":
                findings.append("recommendation 至少启用 graph 或 dense")

        if candidate.get("task_mode") == "dialogue" and candidate.get("dialogue_mode") in {
            "chat",
            "library_guidance",
        }:
            retrieval_values = [
                candidate.get("hard"),
                candidate.get("soft"),
                candidate.get("hints"),
                candidate.get("metadata"),
                acoustic_queries,
            ]

            def has_payload(value: Any) -> bool:
                if isinstance(value, Mapping):
                    return any(has_payload(item) for item in value.values())
                if isinstance(value, list):
                    return any(has_payload(item) for item in value)
                return value not in {None, "", False}

            if any(has_payload(value) for value in retrieval_values):
                findings.append("普通对话或产品指导不得携带检索字段")
    return findings


def _mapping_has_payload(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_mapping_has_payload(item) for item in value.values())
    if isinstance(value, list):
        return any(_mapping_has_payload(item) for item in value)
    return value not in {None, "", False}


def _candidate_graph_evidence(candidate: Mapping[str, Any], *, strong: bool) -> bool:
    hard = candidate.get("hard", {})
    hints = candidate.get("hints", {})
    metadata = candidate.get("metadata", {})
    evidence = candidate.get("evidence", {})
    reason_codes = set(evidence.get("reason_codes", [])) if isinstance(evidence, Mapping) else set()

    hard_evidence = isinstance(hard, Mapping) and any(
        _mapping_has_payload(hard.get(field)) for field in ("artist", "song", "language", "region")
    )
    catalog_metadata = isinstance(metadata, Mapping) and any(
        _mapping_has_payload(metadata.get(field))
        for field in ("era", "release_year_from", "release_year_to")
    )
    genres = isinstance(hints, Mapping) and _mapping_has_payload(hints.get("genre"))
    if strong:
        return bool(hard_evidence or catalog_metadata or genres)

    tag_hints = isinstance(hints, Mapping) and any(
        _mapping_has_payload(hints.get(field)) for field in ("mood", "scenario", "genre")
    )
    graph_reasons = {
        "explicit_entity",
        "explicit_catalog_filter",
        "taggable_genre",
        "taggable_mood",
        "taggable_scenario",
    }
    return bool(hard_evidence or catalog_metadata or tag_hints or reason_codes & graph_reasons)


def _candidate_dense_evidence(candidate: Mapping[str, Any]) -> bool:
    acoustic_queries = candidate.get("acoustic_queries", [])
    evidence = candidate.get("evidence", {})
    soft = candidate.get("soft", {})
    reason_codes = set(evidence.get("reason_codes", [])) if isinstance(evidence, Mapping) else set()
    reference_songs = evidence.get("reference_songs", []) if isinstance(evidence, Mapping) else []
    dense_reasons = {
        "subjective_affective_goal",
        "acoustic_timbre_or_instrument",
        "acoustic_rhythm_or_dynamics",
        "acoustic_production_or_space",
        "metaphorical_vibe",
        "reference_track_similarity",
        "resolved_context_reference",
    }
    semantic_soft = isinstance(soft, Mapping) and any(
        _mapping_has_payload(soft.get(field)) for field in ("goal", "trajectory", "vibe", "avoid")
    )
    return bool(
        isinstance(acoustic_queries, list)
        and acoustic_queries
        and (reason_codes & dense_reasons or reference_songs or semantic_soft)
    )


def _lane_policy_findings(
    safe: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[str]:
    """Check lane obligations without making the keyword parser an oracle."""

    safe_policy = safe["lane_policy"]
    candidate_policy = candidate.get("lane_policy", {})
    findings: list[str] = []

    # External access is a side-effect boundary. It is enabled only when the
    # deterministic parser independently finds a freshness/external request.
    if candidate_policy.get("web") != safe_policy["web"]:
        findings.append(
            f"模型 web 角色为 {candidate_policy.get('web')}，安全边界要求 {safe_policy['web']}"
        )

    safe_dense = safe_policy["dense"]
    candidate_dense = candidate_policy.get("dense")
    dense_evidence = _candidate_dense_evidence(candidate)
    strong_graph_evidence = _candidate_graph_evidence(candidate, strong=True)
    safe_reason_codes = set(safe.get("evidence", {}).get("reason_codes", []))
    generic_dense_fallback = safe_reason_codes == {"metaphorical_vibe"}
    if safe_dense == "required" and candidate_dense != "required":
        # The keyword parser deliberately has no global artist/song dictionary.
        # When its only reason for Dense is the generic unknown-request fallback,
        # a model candidate may replace that fallback with a concrete catalog
        # entity or date/genre constraint. Graph lookup is bounded and carries no
        # external side effect, so this is a safe semantic refinement.
        catalog_refinement = bool(
            generic_dense_fallback
            and candidate_policy.get("graph") == "required"
            and strong_graph_evidence
        )
        if not catalog_refinement:
            findings.append("模型遗漏确定性规则要求的 dense 通道")
    elif safe_dense == "off" and candidate_dense == "required" and not dense_evidence:
        findings.append("模型启用 dense 但未提供可核验的听感、声学或参考曲证据")

    safe_graph = safe_policy["graph"]
    candidate_graph = candidate_policy.get("graph")
    graph_evidence = _candidate_graph_evidence(candidate, strong=False)
    if safe_graph == "required":
        if candidate_graph == "off":
            findings.append("模型遗漏确定性规则要求的 graph 通道")
        elif candidate_graph == "optional" and not (
            candidate_dense == "required" and dense_evidence
        ):
            findings.append("模型降低 graph 角色但未提供可核验的 dense 主通道证据")
    elif safe_graph == "optional":
        if candidate_graph == "off":
            findings.append("模型遗漏确定性规则要求的 graph 辅助通道")
        elif candidate_graph == "required" and not strong_graph_evidence:
            findings.append("模型将 graph 升为 required 但缺少实体、类型或年代约束")
    elif candidate_graph == "optional" and not graph_evidence:
        findings.append("模型启用 graph 辅助通道但未提供标签或目录证据")
    elif candidate_graph == "required" and not strong_graph_evidence:
        findings.append("模型启用 graph 必选通道但缺少实体、类型或年代约束")

    return findings


def guard_candidate(
    user_text: str,
    candidate: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Accept a model candidate only when deterministic invariants agree."""

    safe = build_safe_plan(user_text, context)
    if not isinstance(candidate, Mapping):
        return safe, ["未提供模型候选，已使用确定性安全计划"]

    findings = _candidate_findings(candidate)
    if not findings:
        if candidate.get("task_mode") != safe["task_mode"]:
            findings.append("模型任务类型与确定性分类冲突")
        if candidate.get("dialogue_mode") != safe.get("dialogue_mode"):
            findings.append("模型对话类型与确定性分类冲突")
        if candidate.get("response_mode", "answer") != safe["response_mode"]:
            findings.append("模型回答/澄清决策与安全边界冲突")

        candidate_policy = candidate.get("lane_policy", {})
        if safe["task_mode"] == "recommendation" and safe["response_mode"] == "answer":
            findings.extend(_lane_policy_findings(safe, candidate))
        else:
            for lane, safe_mode in safe["lane_policy"].items():
                candidate_mode = candidate_policy.get(lane)
                if candidate_mode != safe_mode:
                    findings.append(
                        f"模型 {lane} 角色为 {candidate_mode}，安全边界要求 {safe_mode}"
                    )
        if safe["task_mode"] == "dialogue" and safe.get("dialogue_mode") in {"chat", "library_guidance"}:
            if any(candidate_policy.get(lane, "off") != "off" for lane in ("graph", "dense", "web")):
                findings.append("普通对话或产品指导不允许携带检索通道")

    if findings:
        return safe, [*findings, "候选被拒绝，已回退到确定性安全计划"]

    accepted = copy.deepcopy(dict(candidate))
    accepted["version"] = str(accepted.get("version") or PLANNER_VERSION)
    accepted["source"] = "model_candidate_guarded"
    accepted["execution"] = compile_execution(accepted["lane_policy"])
    return accepted, ["模型候选通过结构与策略守卫"]


def format_route_markdown(plan: Mapping[str, Any]) -> str:
    policy = plan.get("lane_policy", {})
    execution = plan.get("execution", {})
    rows = ["| 通道 | 角色 |", "|---|---|"]
    rows.extend(f"| {lane} | {policy.get(lane, 'off')} |" for lane in ("graph", "dense", "web"))
    weights = execution.get("weights", {})
    rows.extend(
        [
            "",
            f"**权重方案：** `{execution.get('profile', 'unknown')}`",
            f"**Graph / Dense：** `{weights.get('graph', 0):.2f} / {weights.get('dense', 0):.2f}`",
            f"**公开短理由：** {plan.get('evidence', {}).get('brief_reason', '')}",
        ]
    )
    return "\n".join(rows)


def dumps_plan(plan: Mapping[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2)
