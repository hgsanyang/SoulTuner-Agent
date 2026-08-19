"""Planner profiles and fail-closed runtime for the public SoulTuner Space."""

from __future__ import annotations

import copy
import json
import os
import re
import urllib.request
from typing import Any


PROFILE_SAFE = "demo-heuristic"
PROFILE_QWEN = "qwen-api"
PROFILE_SOULTUNER = "soultuner-v4.2-35b"

PROFILE_LABELS = {
    PROFILE_SOULTUNER: "SoulTuner V4.2 35B（AMD ROCm）",
    PROFILE_QWEN: "兼容 OpenAI 协议的云端 Planner",
    PROFILE_SAFE: "公开演示（CPU，无需密钥）",
}

GENRES = ["流行", "电子", "摇滚", "爵士", "民谣", "氛围", "放克", "古典跨界"]
MOODS = ["温暖", "治愈", "平静", "明亮", "怀旧", "浪漫", "专注", "活力", "低落"]
SCENARIOS = ["夜晚", "通勤", "学习", "运动", "雨天", "周末", "小聚", "睡前"]
LANGUAGES = ["中文", "英文", "日文", "纯音乐"]
ACOUSTIC_TERMS = [
    "低音",
    "bass",
    "鼓",
    "人声",
    "音色",
    "空间感",
    "混响",
    "听感",
    "温暖",
    "治愈",
    "安静",
    "能量",
]

SYSTEM_PROMPT = """你是 SoulTuner 音乐检索规划器。只输出一个 JSON 对象，不输出 Markdown 或隐藏思维过程。
字段：task_mode, dialogue_mode, response_mode, evidence, lane_policy, hard, soft, hints,
metadata, acoustic_queries, clarification。evidence 必须是包含 decision_phase、failed_lanes、
reason_codes、reference_songs、brief_reason 的对象；hard、soft、hints、metadata 也必须是对象，
不能输出字符串或数组。lane_policy 中 graph/web 取 required|optional|off，dense 取 required|off。
明确目录条件使用 graph；主观情绪、音色、节奏、空间感或参考歌曲相似度使用 dense；二者都存在时
表达主次。brief_reason 最多80字。"""


def profile_choices() -> list[tuple[str, str]]:
    return [(label, value) for value, label in PROFILE_LABELS.items()]


def default_profile() -> str:
    configured = os.getenv("SOULTUNER_MODEL_PROFILE", "").strip()
    if configured in PROFILE_LABELS:
        return configured
    if os.getenv("SOULTUNER_PLANNER_BASE_URL", "").strip():
        return PROFILE_SOULTUNER
    if os.getenv("DASHSCOPE_API_KEY", "").strip():
        return PROFILE_QWEN
    return PROFILE_SAFE


def _tokens(query: str, values: list[str]) -> list[str]:
    lowered = query.casefold()
    return [value for value in values if value.casefold() in lowered]


def safe_plan(query: str) -> dict[str, Any]:
    genres = _tokens(query, GENRES)
    moods = _tokens(query, MOODS)
    scenarios = _tokens(query, SCENARIOS)
    languages = _tokens(query, LANGUAGES)
    acoustic = _tokens(query, ACOUSTIC_TERMS)
    year_match = re.search(r"(?:19|20)\d{2}|(?:80|90|00|10|20)\s*年代", query)
    graph_evidence = bool(genres or languages or year_match)
    tag_evidence = bool(moods or scenarios)
    dense_evidence = bool(acoustic or tag_evidence)

    if graph_evidence and dense_evidence:
        graph, dense = "required", "required"
        brief = "目录条件与主观听感同时存在，图谱过滤后由向量补充相似度"
    elif graph_evidence:
        graph, dense = "required", "off"
        brief = "请求包含语言、年代或流派等目录条件，使用图谱精确过滤"
    elif dense_evidence:
        graph = "optional" if tag_evidence else "off"
        dense = "required"
        brief = "情绪标签可辅助粗筛，主观听感由向量召回主导"
    else:
        graph, dense = "optional", "required"
        brief = "请求约束较少，使用语义召回并让图谱提供轻量标签辅助"

    reason_codes: list[str] = []
    if genres:
        reason_codes.append("taggable_genre")
    if moods:
        reason_codes.append("taggable_mood")
    if scenarios:
        reason_codes.append("taggable_scenario")
    if dense_evidence:
        reason_codes.append("subjective_affective_goal")
    if any(term in acoustic for term in ["低音", "bass", "鼓", "音色", "人声"]):
        reason_codes.append("acoustic_timbre_or_instrument")
    reason_codes = list(dict.fromkeys(reason_codes)) or ["underspecified_request"]

    decade = None
    if year_match:
        decade = year_match.group(0).replace(" ", "")
    return {
        "task_mode": "recommendation",
        "dialogue_mode": None,
        "response_mode": "answer",
        "evidence": {
            "decision_phase": "initial",
            "failed_lanes": [],
            "reason_codes": reason_codes,
            "reference_songs": [],
            "brief_reason": brief,
        },
        "lane_policy": {"graph": graph, "dense": dense, "web": "off"},
        "hard": {
            "artist": [],
            "song": [],
            "language": languages[0] if languages else None,
            "region": None,
            "instrumental": "纯音乐" in languages,
        },
        "soft": {"goal": query, "trajectory": "", "vibe": moods, "avoid": []},
        "hints": {"mood": moods, "scenario": scenarios, "genre": genres},
        "metadata": {
            "era": decade,
            "release_year_from": None,
            "release_year_to": None,
            "recency_required": False,
            "external_knowledge_required": False,
        },
        "acoustic_queries": [f"music with {query}, coherent timbre and dynamics"]
        if dense == "required"
        else [],
        "clarification": None,
    }


def _endpoint(profile: str) -> tuple[str, str, str] | None:
    if profile == PROFILE_SOULTUNER:
        base_url = os.getenv("SOULTUNER_PLANNER_BASE_URL", "").strip().rstrip("/")
        if not base_url:
            return None
        endpoint = (
            base_url
            if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions"
        )
        return (
            endpoint,
            os.getenv("SOULTUNER_PLANNER_MODEL", PROFILE_SOULTUNER),
            os.getenv("SOULTUNER_PLANNER_API_KEY", "").strip(),
        )
    if profile == PROFILE_QWEN:
        token = os.getenv("SOULTUNER_PLANNER_API_KEY", "").strip()
        if not token:
            return None
        base = os.getenv("SOULTUNER_PLANNER_BASE_URL", "").strip().rstrip("/")
        if not base:
            return None
        endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        return (endpoint, os.getenv("SOULTUNER_PLANNER_MODEL", profile), token)
    return None


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response does not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("planner response must be an object")
    return value


def _remote_plan(profile: str, query: str) -> dict[str, Any]:
    resolved = _endpoint(profile)
    if resolved is None:
        raise RuntimeError("selected model endpoint is not configured")
    endpoint, model, token = resolved
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "enable_thinking": False,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    timeout = float(os.getenv("SOULTUNER_PLANNER_TIMEOUT", "180"))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_json(body["choices"][0]["message"]["content"])


def normalize_legacy_candidate(
    candidate: dict[str, Any],
    fallback: dict[str, Any],
    query: str,
) -> tuple[dict[str, Any], bool]:
    """Map the released 35B adapter's legacy JSON shape into the guarded Space contract.

    The released checkpoint emits the training-time ``music_search`` schema where
    evidence is text and soft/hints are arrays.  Keep its lane decision and acoustic
    queries, but project them into the current object-shaped public runtime contract.
    Unknown malformed responses are intentionally left untouched and fail closed.
    """

    is_legacy = (
        candidate.get("task_mode") == "music_search"
        and isinstance(candidate.get("evidence"), str)
        and isinstance(candidate.get("soft"), list)
        and isinstance(candidate.get("hints"), list)
    )
    if not is_legacy:
        return candidate, False

    normalized = copy.deepcopy(fallback)
    normalized["task_mode"] = "recommendation"
    normalized["dialogue_mode"] = None
    normalized["response_mode"] = "answer"
    normalized["lane_policy"] = copy.deepcopy(candidate.get("lane_policy"))

    public_reason = str(candidate["evidence"]).strip()
    normalized["evidence"]["brief_reason"] = public_reason[:80]

    legacy_hard = candidate.get("hard")
    if isinstance(legacy_hard, dict):
        for key in ("artist", "song", "language", "region", "instrumental"):
            if key in legacy_hard:
                normalized["hard"][key] = copy.deepcopy(legacy_hard[key])

    legacy_metadata = candidate.get("metadata")
    if isinstance(legacy_metadata, dict):
        for key in (
            "era",
            "release_year_from",
            "release_year_to",
            "recency_required",
            "external_knowledge_required",
        ):
            if key in legacy_metadata:
                normalized["metadata"][key] = copy.deepcopy(legacy_metadata[key])
        language = legacy_metadata.get("language")
        if isinstance(language, str) and language.strip().casefold() not in {"", "any"}:
            normalized["hard"]["language"] = language.strip()
        genres = legacy_metadata.get("genre")
        if isinstance(genres, list):
            normalized["hints"]["genre"] = [
                str(item).strip() for item in genres if str(item).strip()
            ][:6]

    descriptors = [str(item).strip() for item in candidate["soft"] if str(item).strip()]
    descriptors.extend(
        str(item).strip() for item in candidate["hints"] if str(item).strip()
    )
    if isinstance(legacy_hard, dict):
        descriptors.extend(
            str(legacy_hard[key]).strip()
            for key in ("mood", "tempo", "energy")
            if str(legacy_hard.get(key) or "").strip()
        )
    normalized["soft"]["goal"] = query
    normalized["soft"]["vibe"] = list(dict.fromkeys(descriptors))[:12]
    normalized["acoustic_queries"] = copy.deepcopy(candidate.get("acoustic_queries"))
    normalized["clarification"] = candidate.get("clarification")
    return normalized, True


def validate_plan(candidate: dict[str, Any], fallback: dict[str, Any]) -> tuple[dict[str, Any], str]:
    policy = candidate.get("lane_policy")
    evidence = candidate.get("evidence")
    if not isinstance(policy, dict) or not isinstance(evidence, dict):
        return fallback, "候选缺少 Evidence 或 LanePolicy，已使用安全计划"
    allowed_graph = {"required", "optional", "off"}
    allowed_dense = {"required", "off"}
    if policy.get("graph") not in allowed_graph or policy.get("web") not in allowed_graph:
        return fallback, "候选通道角色非法，已使用安全计划"
    if policy.get("dense") not in allowed_dense:
        return fallback, "Dense 角色非法，已使用安全计划"
    reason = str(evidence.get("brief_reason", "")).strip()
    if not reason or len(reason) > 80:
        return fallback, "公开短理由缺失或过长，已使用安全计划"
    if fallback["lane_policy"]["dense"] == "required" and policy.get("dense") != "required":
        return fallback, "主观听感需要 Dense，候选已被守卫拒绝"
    required_objects = ("hard", "soft", "hints", "metadata")
    if any(not isinstance(candidate.get(key), dict) for key in required_objects):
        return fallback, "候选缺少检索约束，已使用安全计划"
    acoustic_queries = candidate.get("acoustic_queries")
    if (
        not isinstance(acoustic_queries, list)
        or len(acoustic_queries) > 4
        or any(not isinstance(item, str) or not item.strip() for item in acoustic_queries)
    ):
        return fallback, "候选声学查询非法，已使用安全计划"
    if policy.get("dense") == "required" and not acoustic_queries:
        return fallback, "Dense 候选缺少声学查询，已使用安全计划"
    return candidate, "模型候选通过结构与策略守卫"


def compile_route(plan: dict[str, Any]) -> dict[str, Any]:
    graph = plan["lane_policy"]["graph"]
    dense = plan["lane_policy"]["dense"]
    mapping = {
        ("required", "off"): ("graph_only", 1.0, 0.0),
        ("required", "required"): ("balanced_hybrid", 0.5, 0.5),
        ("optional", "required"): ("dense_primary", 0.25, 0.75),
        ("off", "required"): ("dense_only", 0.0, 1.0),
    }
    profile, graph_weight, dense_weight = mapping.get(
        (graph, dense), ("safe_hybrid", 0.25, 0.75)
    )
    return {
        "profile": profile,
        "graph_weight": graph_weight,
        "dense_weight": dense_weight,
        "web_enabled": plan["lane_policy"].get("web") != "off",
    }


def plan_request(profile: str, query: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    fallback = safe_plan(query)
    if profile == PROFILE_SAFE:
        plan, status = fallback, "公开安全演示：确定性计划，不调用外部模型"
    else:
        try:
            candidate, legacy_adapted = normalize_legacy_candidate(
                _remote_plan(profile, query), fallback, query
            )
            plan, status = validate_plan(candidate, fallback)
            if legacy_adapted and status == "模型候选通过结构与策略守卫":
                status = "35B 模型候选经兼容适配后通过结构与策略守卫"
        except Exception as exc:  # The UI must remain available when a model is cold/offline.
            plan, status = fallback, f"模型端点暂不可用，已安全回退（{type(exc).__name__}）"
    return plan, compile_route(plan), status
