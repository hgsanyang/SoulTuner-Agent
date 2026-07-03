"""
结果导向离线评测 —— "尺子"（Outcome-Oriented Offline Eval）
===========================================================

为什么需要它（与 evaluate_intent.py 的本质区别）：
  - evaluate_intent.py 衡量的是「路由器有没有选中作者标注的意图标签」。
    这是"测特性"：它不会因为推荐变好而变好，也不会因为推荐变差而变差——
    一个半循环的尺子。98% 的意图准确率说明不了用户最后有没有拿到对的歌。
  - 本脚本衡量「整条管线最终吐出的歌，是否满足这条 query 的意图」——
    约束满足、可播放性、安全性（不推已拉黑）、多样性、必含/必排。这是"测结果"。

设计纪律（来自踩过的坑）：
  1. 只断言返回字段里**可靠存在**的东西（title/artist/genre/preview_url/language/moods/scenarios）。
  2. 数据缺失的维度 → 标记 SKIP（而非 FAIL），并单列覆盖率，不自欺。
  3. 语言/情绪/场景标签已在 Phase 1 透传进 song dict，可做自动覆盖；更细的声学质感
     （如"安静低动态"）仍保留 manual_review，不把粗标签伪装成听感结论。
  4. 评测集分 dev / holdout：dev 用于日常迭代；holdout 冻结，平时不要边看失败边修。
  5. 评测默认 Planner temperature=0，并在报告中记录 git sha、模型与关键 config。

⚠️ 运行环境：本脚本会拉起完整 Agent（依赖 langchain/neo4j/M2D-CLAP + 已恢复的图数据）。
   请在完整栈下运行（恢复 docker_dump/neo4j.dump、配好 .env），而非纯净的开发机。
   （编写时未在评审环境执行——该环境缺重依赖；首次运行如报字段不符请按实际返回结构微调。）

运行：
    python -m tests.eval.evaluate_outcomes --split smoke
    python -m tests.eval.evaluate_outcomes --split dev
    python -m tests.eval.evaluate_outcomes --split holdout
    python -m tests.eval.evaluate_outcomes --cases tests/eval/outcome_test_cases.json

支持的 check（写在 outcome_test_cases.json 的 "checks" 里）：
    min_results: int                  至少返回 N 首
    artist_any_of: [str]              + artist_match_min_ratio: float   命中指定歌手的比例下限
    genre_any_of: [str]               + genre_match_min_ratio: float    genre 偏向比例下限(仅统计有genre的歌)
    language_any_of: [str]            + language_match_min_ratio: float  语言命中比例下限
    mood_any_of: [str]                + mood_match_min_ratio: float      moods/genre 中情绪命中比例下限
    scenario_any_of: [str]            + scenario_match_min_ratio: float  scenarios/genre 中场景命中比例下限
    must_include_titles: [str]        每个子串都应出现在某首歌标题中
    must_exclude: [{title?,artist?}]  这些歌都不应出现（用于"已拉黑不应推"回归）
    min_playable_ratio: float         有 preview_url 的比例下限（产品级"可播放"信号）
    max_per_artist: int               任一歌手出现次数上限（多样性）
    not_degraded: true                意图分析未触发降级兜底（_intent_degraded / degraded_to）
    objective_soft_judge: {           软意图客观字段初判；只看 song attrs，不看 explanation
        positive_any: [str],
        negative_any: [str],
        min_positive_ratio: float,
        max_negative_ratio: float,
        min_coverage_ratio: float
    }
    expected_clarification: true      期望返回澄清反问（澄清≠失败）
    dialog_state_contains: {path: str}  检查返回 dialog_state 的字段包含指定值
    dialog_delta_contains: {path: str}  检查本轮状态 delta（followup/继承/替换等）
    manual_review: [str]              人工核对项（不计入自动通过/失败）
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(override=True)

from tests.eval.soft_judge import judge_objective_soft_intent  # noqa: E402

EVAL_DIR = Path(__file__).parent
CASES_DIR = EVAL_DIR / "cases"
CASE_SPLITS = {
    "smoke": EVAL_DIR / "outcome_test_cases.json",
    "dev": CASES_DIR / "outcome_dev.json",
    "holdout": CASES_DIR / "outcome_holdout.json",
    "context_dev": CASES_DIR / "context_dev.json",
    "context_holdout": CASES_DIR / "context_holdout.json",
}

HARD_CATEGORIES = {
    "negative_constraint",
    "soft_mood",
    "scenario",
    "multi_turn_context",
    "catalog_gap",
    "web_fallback",
}


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def _percentile(values: List[float], quantile: float) -> float:
    """Linear-interpolated percentile without a NumPy dependency."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarize_timings(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[str, List[float]] = defaultdict(list)
    for report in reports:
        for stage, value in (report.get("timings_ms") or {}).items():
            if isinstance(value, (int, float)) and value >= 0:
                by_stage[stage].append(float(value))
    return {
        "unit": "ms",
        "case_count": len(reports),
        "stages": {
            stage: {
                "count": len(values),
                "mean": round(sum(values) / len(values), 3),
                "p50": round(_percentile(values, 0.50), 3),
                "p95": round(_percentile(values, 0.95), 3),
            }
            for stage, values in sorted(by_stage.items())
        },
    }


def _top_slow_cases(reports: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """Return the slowest cases by measured end-to-end latency."""
    rows = []
    for report in reports:
        timings = report.get("timings_ms") or {}
        elapsed = timings.get("end_to_end_ms")
        if not isinstance(elapsed, (int, float)):
            continue
        rows.append({
            "id": report.get("id"),
            "query": report.get("query"),
            "case_status": report.get("case_status"),
            "end_to_end_ms": round(float(elapsed), 3),
            "intent_ms": timings.get("intent_ms"),
            "retrieval_total_ms": timings.get("retrieval_total_ms"),
            "graphzep_ms": timings.get("graphzep_ms"),
        })
    rows.sort(key=lambda row: row["end_to_end_ms"], reverse=True)
    return rows[:max(0, int(limit))]


def _unwrap_songs(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 agent.get_recommendations 返回值里取出标准化 song dict 列表。

    recommendations 可能是 list，也可能是 ToolOutput（需取 .data）。
    每个元素形如 {"song": {...}, "reason": ..., ...}，也可能直接就是 song dict。
    """
    recs = result.get("recommendations") or []
    recs = getattr(recs, "data", recs)  # ToolOutput -> list
    songs: List[Dict[str, Any]] = []
    if isinstance(recs, list):
        for r in recs:
            if isinstance(r, dict):
                song = r.get("song", r)
                if isinstance(song, dict) and song.get("title"):
                    songs.append(song)
    return songs


def _field_values(song: Dict[str, Any], *names: str) -> List[str]:
    values: List[str] = []
    for name in names:
        val = song.get(name)
        if isinstance(val, list):
            values.extend(_norm(x) for x in val if _norm(x))
        elif _norm(val):
            values.append(_norm(val))
    return values


def _is_degraded(result: Dict[str, Any]) -> bool:
    for e in (result.get("errors") or []):
        if isinstance(e, dict) and e.get("degraded_to"):
            return True
    return False


def _case_difficulty(case: Dict[str, Any]) -> str:
    explicit = _norm(case.get("difficulty"))
    if explicit in {"easy", "hard"}:
        return explicit
    category = _norm(case.get("category", ""))
    checks = case.get("checks") or {}
    if category in HARD_CATEGORIES:
        return "hard"
    if checks.get("objective_soft_judge") or checks.get("expected_clarification"):
        return "hard"
    if case.get("chat_history") or case.get("dialog_state"):
        return "hard"
    return "easy"


def _load_cases(cases_file: str | None = None, split: str = "smoke") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if cases_file:
        path = Path(cases_file)
        return json.loads(path.read_text(encoding="utf-8")), {
            "split": "custom",
            "case_files": [str(path)],
        }
    if split == "all":
        cases: List[Dict[str, Any]] = []
        files = []
        for name in ("dev", "holdout"):
            path = CASE_SPLITS[name]
            files.append(str(path))
            loaded = json.loads(path.read_text(encoding="utf-8"))
            for case in loaded:
                case.setdefault("split", name)
                case["difficulty"] = _case_difficulty(case)
            cases.extend(loaded)
        return cases, {"split": "all", "case_files": files}
    if split == "context_all":
        cases = []
        files = []
        for name in ("context_dev", "context_holdout"):
            path = CASE_SPLITS[name]
            files.append(str(path))
            loaded = json.loads(path.read_text(encoding="utf-8"))
            for case in loaded:
                case.setdefault("split", name)
                case["difficulty"] = _case_difficulty(case)
            cases.extend(loaded)
        return cases, {"split": "context_all", "case_files": files}
    if split in {"dev_easy", "dev_hard", "holdout_easy", "holdout_hard"}:
        base_split, difficulty = split.rsplit("_", 1)
        path = CASE_SPLITS[base_split]
        loaded = json.loads(path.read_text(encoding="utf-8"))
        cases = []
        for case in loaded:
            case.setdefault("split", base_split)
            case["difficulty"] = _case_difficulty(case)
            if case["difficulty"] == difficulty:
                cases.append(case)
        return cases, {
            "split": split,
            "base_split": base_split,
            "difficulty": difficulty,
            "case_files": [str(path)],
        }

    path = CASE_SPLITS[split]
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        case.setdefault("split", split)
        case["difficulty"] = _case_difficulty(case)
    return cases, {"split": split, "case_files": [str(path)]}


def _git_meta() -> Dict[str, Any]:
    root = Path(__file__).resolve().parent.parent.parent

    def _cmd(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "sha": _cmd(["git", "rev-parse", "HEAD"]),
        "branch": _cmd(["git", "branch", "--show-current"]),
        "dirty": bool(_cmd(["git", "status", "--porcelain"])),
    }


def _effective_model_config(settings, provider: str, planner_temperature: float) -> Dict[str, Any]:
    main_provider = settings.llm_default_provider or provider
    main_model = settings.llm_default_model
    intent_provider = settings.intent_llm_provider or main_provider
    intent_model = settings.intent_llm_model or main_model
    explain_provider = settings.explain_llm_provider or main_provider
    explain_model = settings.explain_llm_model or main_model
    compress_provider = settings.compress_llm_provider or main_provider
    compress_model = settings.compress_llm_model or main_model
    return {
        "provider_arg": provider,
        "main": {"provider": main_provider, "model": main_model},
        "intent": {
            "provider": intent_provider,
            "model": intent_model,
            "temperature": planner_temperature,
            "max_tokens": settings.intent_max_tokens,
        },
        "explain": {
            "provider": explain_provider,
            "model": explain_model,
            "fast_mode": bool(settings.explanation_fast_mode),
        },
        "compress": {"provider": compress_provider, "model": compress_model},
        "llm_timeout": settings.llm_timeout,
    }


# ----------------------------------------------------------------------------
# 各 check 处理器：返回 (status, detail)，status ∈ {"pass","fail","skip"}
# ----------------------------------------------------------------------------
def _c_min_results(songs, val, result) -> Tuple[str, str]:
    n = len(songs)
    return ("pass" if n >= int(val) else "fail", f"返回 {n} 首，要求 ≥ {val}")


def _c_artist_ratio(songs, val, result, checks) -> Tuple[str, str]:
    names = [_norm(x) for x in (checks.get("artist_any_of") or [])]
    if not names:
        return ("skip", "未提供 artist_any_of，跳过")
    if not songs:
        return ("fail", "无结果")
    hit = sum(1 for s in songs if any(name and name in _norm(s.get("artist")) for name in names))
    ratio = hit / len(songs)
    return ("pass" if ratio >= float(val) else "fail",
            f"歌手命中 {hit}/{len(songs)} = {ratio:.0%}，要求 ≥ {float(val):.0%}")


def _c_genre_ratio(songs, val, result, checks) -> Tuple[str, str]:
    genres_wanted = [_norm(x) for x in (checks.get("genre_any_of") or [])]
    if not genres_wanted:
        return ("skip", "未提供 genre_any_of，跳过")
    with_genre = [s for s in songs if _norm(s.get("genre"))]
    if not with_genre:
        return ("skip", "返回结果均无 genre 字段，无法评估")
    hit = sum(1 for s in with_genre if any(g and g in _norm(s.get("genre")) for g in genres_wanted))
    ratio = hit / len(with_genre)
    return ("pass" if ratio >= float(val) else "fail",
            f"genre 偏向 {hit}/{len(with_genre)} = {ratio:.0%}(仅统计有genre的歌)，要求 ≥ {float(val):.0%}")


def _c_language_ratio(songs, val, result, checks) -> Tuple[str, str]:
    wanted = [_norm(x) for x in (checks.get("language_any_of") or [])]
    if not wanted:
        return ("skip", "未提供 language_any_of，跳过")
    with_lang = [s for s in songs if _norm(s.get("language"))]
    if not with_lang:
        return ("skip", "返回结果均无 language 字段，无法评估")
    hit = sum(1 for s in with_lang if any(w and w in _norm(s.get("language")) for w in wanted))
    ratio = hit / len(with_lang)
    return ("pass" if ratio >= float(val) else "fail",
            f"language 命中 {hit}/{len(with_lang)} = {ratio:.0%}，要求 ≥ {float(val):.0%}")


def _c_mood_ratio(songs, val, result, checks) -> Tuple[str, str]:
    wanted = [_norm(x) for x in (checks.get("mood_any_of") or [])]
    if not wanted:
        return ("skip", "未提供 mood_any_of，跳过")
    with_mood = [s for s in songs if _field_values(s, "moods", "genre")]
    if not with_mood:
        return ("skip", "返回结果均无 moods/genre 字段，无法评估")
    hit = sum(1 for s in with_mood if any(
        w and any(w in v for v in _field_values(s, "moods", "genre")) for w in wanted
    ))
    ratio = hit / len(with_mood)
    return ("pass" if ratio >= float(val) else "fail",
            f"mood 命中 {hit}/{len(with_mood)} = {ratio:.0%}，要求 ≥ {float(val):.0%}")


def _c_scenario_ratio(songs, val, result, checks) -> Tuple[str, str]:
    wanted = [_norm(x) for x in (checks.get("scenario_any_of") or [])]
    if not wanted:
        return ("skip", "未提供 scenario_any_of，跳过")
    with_scenario = [s for s in songs if _field_values(s, "scenarios", "genre")]
    if not with_scenario:
        return ("skip", "返回结果均无 scenarios/genre 字段，无法评估")
    hit = sum(1 for s in with_scenario if any(
        w and any(w in v for v in _field_values(s, "scenarios", "genre")) for w in wanted
    ))
    ratio = hit / len(with_scenario)
    return ("pass" if ratio >= float(val) else "fail",
            f"scenario 命中 {hit}/{len(with_scenario)} = {ratio:.0%}，要求 ≥ {float(val):.0%}")


def _c_must_include(songs, val, result) -> Tuple[str, str]:
    titles = [_norm(s.get("title")) for s in songs]
    missing = [sub for sub in val if not any(_norm(sub) in t for t in titles)]
    return ("pass" if not missing else "fail",
            "全部命中" if not missing else f"缺失: {missing}")


def _c_must_exclude(songs, val, result) -> Tuple[str, str]:
    # 过滤占位模板（用户尚未填充真实拉黑数据）
    real = [e for e in val if isinstance(e, dict) and "示例" not in (e.get("title", "") + e.get("artist", ""))]
    if not real:
        return ("skip", "must_exclude 仍是占位模板，请填入真实 DISLIKES 数据")
    violated = []
    for e in real:
        t, a = _norm(e.get("title")), _norm(e.get("artist"))
        for s in songs:
            st, sa = _norm(s.get("title")), _norm(s.get("artist"))
            if (not t or t in st) and (not a or a in sa) and (t or a):
                violated.append(e)
                break
    return ("pass" if not violated else "fail",
            "无违规" if not violated else f"出现了应排除的歌: {violated}")


def _c_min_playable(songs, val, result) -> Tuple[str, str]:
    if not songs:
        return ("fail", "无结果")
    playable = sum(1 for s in songs if s.get("preview_url"))
    ratio = playable / len(songs)
    return ("pass" if ratio >= float(val) else "fail",
            f"可播放 {playable}/{len(songs)} = {ratio:.0%}，要求 ≥ {float(val):.0%}")


def _c_max_per_artist(songs, val, result) -> Tuple[str, str]:
    counts = Counter(_norm(s.get("artist")) for s in songs if _norm(s.get("artist")) and _norm(s.get("artist")) != "未知艺术家")
    if not counts:
        return ("skip", "无可识别歌手")
    worst_artist, worst = counts.most_common(1)[0]
    return ("pass" if worst <= int(val) else "fail",
            f"单歌手最多 {worst} 首(‘{worst_artist}’)，上限 {val}")


def _c_not_degraded(songs, val, result) -> Tuple[str, str]:
    deg = _is_degraded(result)
    if not val:
        return ("skip", "未要求")
    return ("fail" if deg else "pass", "意图分析触发了降级兜底" if deg else "未降级")


def _c_objective_soft_judge(songs, val, result) -> Tuple[str, str]:
    decision = judge_objective_soft_intent(songs, val)
    return decision.status, decision.detail


def _c_expected_clarification(songs, val, result) -> Tuple[str, str]:
    if not val:
        return ("skip", "未要求澄清")
    if result.get("intent_type") != "clarification":
        return ("fail", f"期望澄清，实际 intent={result.get('intent_type')}")
    has_prompt = bool(result.get("final_response") or result.get("response"))
    has_options = bool(result.get("clarification_options"))
    if has_prompt:
        detail = "已返回澄清问题"
        if has_options:
            detail += "，并包含 clarification_options"
        return ("pass", detail)
    return ("fail", "澄清路径没有返回问题文本")


def _get_path(data: Dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _value_contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, list):
        return any(_value_contains(item, expected) for item in actual)
    if isinstance(expected, list):
        return any(_value_contains(actual, item) for item in expected)
    return _norm(expected) in _norm(actual)


def _c_dialog_state_contains(songs, val, result) -> Tuple[str, str]:
    state = result.get("dialog_state") or {}
    if not isinstance(val, dict):
        return ("skip", "dialog_state_contains 需要对象配置")
    missing: list[str] = []
    for path, expected in val.items():
        actual = _get_path(state, path)
        if not _value_contains(actual, expected):
            missing.append(f"{path}~={expected!r} (actual={actual!r})")
    return ("pass" if not missing else "fail", "dialog_state 命中" if not missing else "; ".join(missing))


def _c_dialog_delta_contains(songs, val, result) -> Tuple[str, str]:
    delta = result.get("dialog_delta") or ((result.get("dialog_state") or {}).get("last_delta") or {})
    if not isinstance(val, dict):
        return ("skip", "dialog_delta_contains 需要对象配置")
    missing: list[str] = []
    for path, expected in val.items():
        actual = _get_path(delta, path)
        if isinstance(expected, bool):
            ok = actual is expected
        else:
            ok = _value_contains(actual, expected)
        if not ok:
            missing.append(f"{path}~={expected!r} (actual={actual!r})")
    return ("pass" if not missing else "fail", "dialog_delta 命中" if not missing else "; ".join(missing))


# 硬 check 名 → 处理器；带 *_ratio 的需要读取整个 checks（因为它们配对取另一个键）
_HANDLERS = {
    "min_results": _c_min_results,
    "must_include_titles": _c_must_include,
    "must_exclude": _c_must_exclude,
    "min_playable_ratio": _c_min_playable,
    "max_per_artist": _c_max_per_artist,
    "not_degraded": _c_not_degraded,
    "objective_soft_judge": _c_objective_soft_judge,
    "expected_clarification": _c_expected_clarification,
    "dialog_state_contains": _c_dialog_state_contains,
    "dialog_delta_contains": _c_dialog_delta_contains,
}
_HANDLERS_WITH_CHECKS = {
    "artist_match_min_ratio": _c_artist_ratio,
    "genre_match_min_ratio": _c_genre_ratio,
    "language_match_min_ratio": _c_language_ratio,
    "mood_match_min_ratio": _c_mood_ratio,
    "scenario_match_min_ratio": _c_scenario_ratio,
}
# 仅作为配对参数、不单独触发的键
_PAIR_ONLY = {"artist_any_of", "genre_any_of", "language_any_of", "mood_any_of", "scenario_any_of"}

_CHECK_DIMENSIONS = {
    "min_results": "ranking",
    "artist_match_min_ratio": "intent",
    "genre_match_min_ratio": "ranking",
    "language_match_min_ratio": "intent",
    "mood_match_min_ratio": "ranking",
    "scenario_match_min_ratio": "ranking",
    "must_include_titles": "intent",
    "must_exclude": "intent",
    "min_playable_ratio": "ranking",
    "max_per_artist": "ranking",
    "not_degraded": "intent",
    "objective_soft_judge": "ranking",
    "expected_clarification": "intent",
    "dialog_state_contains": "intent",
    "dialog_delta_contains": "intent",
}


def _dimension_status(outcomes: List[Dict[str, str]], dimension: str) -> str:
    relevant = [
        outcome for outcome in outcomes
        if _CHECK_DIMENSIONS.get(outcome.get("name", "")) == dimension
        and outcome.get("status") in {"pass", "fail"}
    ]
    if not relevant:
        return "indeterminate"
    return "fail" if any(outcome["status"] == "fail" for outcome in relevant) else "pass"


def evaluate_case(case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    songs = _unwrap_songs(result)
    checks = case.get("checks", {})
    outcomes = []  # [{name, status, detail}]

    for name, val in checks.items():
        if name == "manual_review" or name in _PAIR_ONLY:
            continue
        if name in _HANDLERS_WITH_CHECKS:
            status, detail = _HANDLERS_WITH_CHECKS[name](songs, val, result, checks)
        elif name in _HANDLERS:
            status, detail = _HANDLERS[name](songs, val, result)
        else:
            status, detail = ("skip", f"未知 check（已忽略）: {name}")
        outcomes.append({"name": name, "status": status, "detail": detail})

    hard = [o for o in outcomes if o["status"] in ("pass", "fail")]
    n_fail = sum(1 for o in hard if o["status"] == "fail")
    if not hard:
        case_status = "indeterminate"  # 全是 skip（数据不足），无法自动判定
    else:
        case_status = "pass" if n_fail == 0 else "fail"

    return {
        "id": case.get("id"),
        "category": case.get("category", "uncategorized"),
        "goal_category": case.get("goal_category") or case.get("category", "uncategorized"),
        "specificity": case.get("specificity", "unspecified"),
        "difficulty": case.get("difficulty") or _case_difficulty(case),
        "split": case.get("split"),
        "query": case.get("query"),
        "has_chat_history": bool(case.get("chat_history")),
        "intent_type": result.get("intent_type"),
        "num_songs": len(songs),
        "case_status": case_status,
        "intent_status": _dimension_status(outcomes, "intent"),
        "ranking_status": _dimension_status(outcomes, "ranking"),
        "outcomes": outcomes,
        "manual_review": checks.get("manual_review", []),
        "dialog_state": result.get("dialog_state", {}),
        "dialog_delta": result.get("dialog_delta", {}),
        "clarification_options": result.get("clarification_options", []),
        "intent_confidence": result.get("intent_confidence", 1.0),
        "refinement_options": result.get("refinement_options", []),
        "sample": [f'{s.get("title")} - {s.get("artist")}' for s in songs[:5]],
        "sample_songs": [{
            "title": s.get("title"),
            "artist": s.get("artist"),
            "genre": s.get("genre"),
            "genres": s.get("genres"),
            "moods": s.get("moods"),
            "scenarios": s.get("scenarios"),
            "language": s.get("language"),
            "region": s.get("region"),
            "preview_url": s.get("preview_url"),
        } for s in songs[:5]],
    }


async def run(
    provider: str,
    cases_file: str | None,
    split: str,
    limit: int,
    verbose: bool,
    planner_temperature: float,
    timing: bool = False,
    fast: bool = False,
    case_timeout: float = 0.0,
) -> Dict[str, Any]:
    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)
    # 延迟导入：让 --help 在无重依赖时也能用
    from config.settings import settings
    settings.intent_llm_provider = provider
    settings.llm_default_provider = provider
    settings.intent_temperature = planner_temperature
    settings.explanation_fast_mode = fast
    settings.eval_disable_side_effects = True
    from agent.music_agent import MusicRecommendationAgent
    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)

    cases, case_meta = _load_cases(cases_file=cases_file, split=split)
    model_meta = _effective_model_config(settings, provider, planner_temperature)
    git_meta = _git_meta()
    print(f"\n{'='*64}\n结果导向离线评测（Outcome Eval）\n{'='*64}")
    print(f"Provider: {provider} | Split: {case_meta['split']} | Cases: {len(cases)} | top-k: {limit}")
    print(
        "Effective Planner: "
        f"{model_meta['intent']['provider']} / {model_meta['intent']['model']} "
        f"(temperature={planner_temperature}, max_tokens={model_meta['intent']['max_tokens']})"
    )
    print(f"Explanation fast-mode: {settings.explanation_fast_mode}")
    print(f"Eval side effects disabled: {settings.eval_disable_side_effects}")
    print(f"Git: {git_meta['branch']} @ {git_meta['sha'][:12]} | dirty={git_meta['dirty']}\n")

    agent = MusicRecommendationAgent()
    reports = []
    t_start = time.time()

    for i, case in enumerate(cases, 1):
        query = case["query"]
        case_started = time.perf_counter()
        try:
            request = agent.get_recommendations(
                query,
                chat_history=case.get("chat_history"),
                dialog_state=case.get("dialog_state"),
            )
            if case_timeout and case_timeout > 0:
                result = await asyncio.wait_for(request, timeout=case_timeout)
            else:
                result = await request
        except asyncio.TimeoutError:
            result = {
                "recommendations": [],
                "intent_type": "TIMEOUT",
                "errors": [{
                    "node": "harness",
                    "error": f"case exceeded timeout {case_timeout:.1f}s",
                    "timeout_seconds": case_timeout,
                }],
            }
        except Exception as e:  # 单 case 失败不应中断整轮
            result = {"recommendations": [], "intent_type": "ERROR",
                      "errors": [{"node": "harness", "error": str(e)}]}
        rep = evaluate_case(case, result)
        if timing:
            case_timings = {
                name: round(float(value), 3)
                for name, value in (result.get("timings") or {}).items()
                if isinstance(value, (int, float)) and value >= 0
            }
            case_timings["end_to_end_ms"] = round(
                (time.perf_counter() - case_started) * 1000,
                3,
            )
            rep["timings_ms"] = case_timings
        reports.append(rep)

        if verbose:
            icon = {"pass": "✅", "fail": "❌", "indeterminate": "➖"}[rep["case_status"]]
            print(f"[{i}/{len(cases)}] {icon} {rep['id']}  «{query}»  → intent={rep['intent_type']}, {rep['num_songs']} 首")
            for o in rep["outcomes"]:
                mark = {"pass": "  ✓", "fail": "  ✗", "skip": "  ·"}[o["status"]]
                print(f"{mark} {o['name']}: {o['detail']}")
            for m in rep["manual_review"]:
                print(f"  👤 人工核对: {m}")
            if rep["sample"]:
                print(f"     样例: {rep['sample']}")
            print()

    # ---- 汇总 ----
    n = len(reports)
    n_pass = sum(1 for r in reports if r["case_status"] == "pass")
    n_fail = sum(1 for r in reports if r["case_status"] == "fail")
    n_indet = sum(1 for r in reports if r["case_status"] == "indeterminate")
    decided = n_pass + n_fail

    # 按 check 类型聚合
    by_check = defaultdict(lambda: {"pass": 0, "fail": 0, "skip": 0})
    by_category = defaultdict(lambda: {"pass": 0, "fail": 0, "indeterminate": 0, "total": 0})
    by_goal_category = defaultdict(lambda: {"pass": 0, "fail": 0, "indeterminate": 0, "total": 0})
    by_specificity = defaultdict(lambda: {"pass": 0, "fail": 0, "indeterminate": 0, "total": 0})
    by_difficulty = defaultdict(lambda: {"pass": 0, "fail": 0, "indeterminate": 0, "total": 0})
    by_dimension = {
        "intent": {"pass": 0, "fail": 0, "indeterminate": 0},
        "ranking": {"pass": 0, "fail": 0, "indeterminate": 0},
    }
    for r in reports:
        cat = r.get("category", "uncategorized")
        by_category[cat][r["case_status"]] += 1
        by_category[cat]["total"] += 1
        goal_category = r.get("goal_category", cat)
        by_goal_category[goal_category][r["case_status"]] += 1
        by_goal_category[goal_category]["total"] += 1
        specificity = r.get("specificity", "unspecified")
        by_specificity[specificity][r["case_status"]] += 1
        by_specificity[specificity]["total"] += 1
        difficulty = r.get("difficulty", "unknown")
        by_difficulty[difficulty][r["case_status"]] += 1
        by_difficulty[difficulty]["total"] += 1
        by_dimension["intent"][r.get("intent_status", "indeterminate")] += 1
        by_dimension["ranking"][r.get("ranking_status", "indeterminate")] += 1
        for o in r["outcomes"]:
            by_check[o["name"]][o["status"]] += 1

    print(f"{'='*64}\n📊 汇总\n{'='*64}")
    print(f"用例: {n} | ✅ 通过 {n_pass} | ❌ 失败 {n_fail} | ➖ 数据不足 {n_indet}")
    if decided:
        print(f"可判定用例通过率: {n_pass}/{decided} = {n_pass/decided:.1%}")
    print(f"总耗时: {time.time()-t_start:.1f}s\n")

    timing_summary = _summarize_timings(reports) if timing else None
    if timing_summary:
        slow_cases = _top_slow_cases(reports)
        timing_summary["slow_cases"] = slow_cases
        print("延迟分阶段 (ms):")
        print(f"  {'stage':24s} {'count':>6s} {'mean':>10s} {'p50':>10s} {'p95':>10s}")
        print(f"  {'-'*64}")
        preferred_order = [
            "end_to_end_ms", "agent_total_ms", "graphzep_ms", "intent_ms",
            "retrieval_total_ms", "recall_graph_ms", "recall_dense_ms",
            "recall_lexical_ms",
            "fusion_filter_ms", "ranking_ms", "web_fallback_ms", "explanation_ms",
        ]
        stages = timing_summary["stages"]
        ordered_stages = [stage for stage in preferred_order if stage in stages]
        ordered_stages.extend(stage for stage in stages if stage not in ordered_stages)
        for stage in ordered_stages:
            stats = stages[stage]
            print(
                f"  {stage:24s} {stats['count']:>6d} {stats['mean']:>10.1f} "
                f"{stats['p50']:>10.1f} {stats['p95']:>10.1f}"
            )
        print()
        if slow_cases:
            print("最慢用例:")
            for row in slow_cases[:10]:
                print(
                    f"  {row['end_to_end_ms']:>9.1f}ms  "
                    f"{row.get('case_status', ''):>13s}  {row.get('id')}"
                )
            print()
    print("按 check 类型:")
    print(f"  {'check':24s} {'pass':>6s} {'fail':>6s} {'skip':>6s}")
    print(f"  {'-'*44}")
    for name, c in sorted(by_check.items()):
        print(f"  {name:24s} {c['pass']:>6d} {c['fail']:>6d} {c['skip']:>6d}")

    print("\n按场景类别:")
    print(f"  {'category':28s} {'pass':>6s} {'fail':>6s} {'indet':>6s} {'total':>6s}")
    print(f"  {'-'*56}")
    for name, c in sorted(by_category.items()):
        print(f"  {name:28s} {c['pass']:>6d} {c['fail']:>6d} {c['indeterminate']:>6d} {c['total']:>6d}")

    print("\n按上下文目标类别:")
    print(f"  {'goal_category':28s} {'pass':>6s} {'fail':>6s} {'indet':>6s} {'total':>6s}")
    print(f"  {'-'*56}")
    for name, c in sorted(by_goal_category.items()):
        print(f"  {name:28s} {c['pass']:>6d} {c['fail']:>6d} {c['indeterminate']:>6d} {c['total']:>6d}")

    print("\n按具体度:")
    print(f"  {'specificity':28s} {'pass':>6s} {'fail':>6s} {'indet':>6s} {'total':>6s}")
    print(f"  {'-'*56}")
    for name, c in sorted(by_specificity.items()):
        print(f"  {name:28s} {c['pass']:>6d} {c['fail']:>6d} {c['indeterminate']:>6d} {c['total']:>6d}")

    print("\n按难度:")
    print(f"  {'difficulty':28s} {'pass':>6s} {'fail':>6s} {'indet':>6s} {'total':>6s}")
    print(f"  {'-'*56}")
    for name, c in sorted(by_difficulty.items()):
        print(f"  {name:28s} {c['pass']:>6d} {c['fail']:>6d} {c['indeterminate']:>6d} {c['total']:>6d}")

    print("\n按链路分层:")
    print(f"  {'dimension':28s} {'pass':>6s} {'fail':>6s} {'indet':>6s}")
    print(f"  {'-'*48}")
    for name, c in sorted(by_dimension.items()):
        print(f"  {name:28s} {c['pass']:>6d} {c['fail']:>6d} {c['indeterminate']:>6d}")

    failed = [r for r in reports if r["case_status"] == "fail"]
    if failed:
        print(f"\n❌ 失败用例 ({len(failed)}):")
        for r in failed:
            bad = [f"{o['name']}({o['detail']})" for o in r["outcomes"] if o["status"] == "fail"]
            print(f"  [{r['id']}] «{r['query']}» → {bad}")

    manual = [(r["id"], r["query"], r["manual_review"]) for r in reports if r["manual_review"]]
    if manual:
        print(f"\n👤 需人工核对（自动尺子覆盖不到的软意图）: {len(manual)} 例")
        for cid, q, items in manual:
            print(f"  [{cid}] «{q}»: {'; '.join(items)}")

    # ---- 落盘 ----
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"outcome_eval_{provider}_{ts}.json"
    report = {
        "provider": provider, "timestamp": ts, "total": n,
        "case_meta": case_meta,
        "git": git_meta,
        "model_config": model_meta,
        "eval_disable_side_effects": bool(settings.eval_disable_side_effects),
        "case_timeout_seconds": round(case_timeout, 3) if case_timeout and case_timeout > 0 else None,
        "passed": n_pass, "failed": n_fail, "indeterminate": n_indet,
        "decided_pass_rate": round(n_pass / decided, 4) if decided else None,
        "by_check": dict(by_check),
        "by_category": dict(by_category),
        "by_goal_category": dict(by_goal_category),
        "by_specificity": dict(by_specificity),
        "by_difficulty": dict(by_difficulty),
        "by_dimension": by_dimension,
        "cases": reports,
    }
    if timing_summary:
        report["timing"] = timing_summary
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 详细结果: {out_file}")
    return report


def main():
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="SoulTuner 结果导向离线评测（尺子）")
    p.add_argument("--provider", default="dashscope")
    p.add_argument("--split", choices=[
        "smoke", "dev", "dev_easy", "dev_hard",
        "holdout", "holdout_easy", "holdout_hard", "all",
        "context_dev", "context_holdout", "context_all",
    ], default="smoke",
                   help="评测切分。holdout 是冻结集，日常迭代不要频繁看详情")
    p.add_argument("--cases", default=None, help="自定义 cases JSON；传入时覆盖 --split")
    p.add_argument("--limit", type=int, default=15, help="请求的 top-k（仅记录，实际条数由管线 FinalCut 决定）")
    p.add_argument("--planner-temperature", type=float, default=0.0,
                   help="评测时 Planner 温度，默认 0 以提升可复现性")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--timing", action="store_true",
                   help="记录并汇总 Agent/检索各阶段的 p50/p95 延迟")
    p.add_argument("--fast", action="store_true",
                   help="跳过解释 LLM，保留歌曲结果并生成确定性简短说明")
    p.add_argument("--case-timeout", type=float, default=0.0,
                   help="单条用例超时秒数；>0 时单 case 超时会标记为 TIMEOUT 并继续整轮")
    p.add_argument("--min-decided-pass-rate", type=float, default=None,
                   help="质量闸门：可判定用例通过率低于该值时退出非 0，例如 0.95")
    p.add_argument("--require-no-failures", action="store_true",
                   help="质量闸门：任一 case 失败即退出非 0（indeterminate 不算失败）")
    args = p.parse_args()
    report = asyncio.run(run(
        args.provider,
        args.cases,
        args.split,
        args.limit,
        verbose=not args.quiet,
        planner_temperature=args.planner_temperature,
        timing=args.timing,
        fast=args.fast,
        case_timeout=args.case_timeout,
    ))
    failed = int(report.get("failed") or 0)
    pass_rate = report.get("decided_pass_rate")
    gate_failed = False
    if args.require_no_failures and failed > 0:
        print(f"\n❌ 质量闸门失败: failed={failed}, require-no-failures=true")
        gate_failed = True
    if args.min_decided_pass_rate is not None:
        actual = 0.0 if pass_rate is None else float(pass_rate)
        if actual < float(args.min_decided_pass_rate):
            print(
                f"\n❌ 质量闸门失败: decided_pass_rate={actual:.1%} "
                f"< {float(args.min_decided_pass_rate):.1%}"
            )
            gate_failed = True
    if gate_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
