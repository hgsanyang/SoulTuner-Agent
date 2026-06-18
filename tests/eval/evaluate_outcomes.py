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

⚠️ 运行环境：本脚本会拉起完整 Agent（依赖 langchain/neo4j/M2D-CLAP + 已恢复的图数据）。
   请在完整栈下运行（恢复 docker_dump/neo4j.dump、配好 .env），而非纯净的开发机。
   （编写时未在评审环境执行——该环境缺重依赖；首次运行如报字段不符请按实际返回结构微调。）

运行：
    python -m tests.eval.evaluate_outcomes
    python -m tests.eval.evaluate_outcomes --provider siliconflow --limit 15
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
    manual_review: [str]              人工核对项（不计入自动通过/失败）
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


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


# 硬 check 名 → 处理器；带 *_ratio 的需要读取整个 checks（因为它们配对取另一个键）
_HANDLERS = {
    "min_results": _c_min_results,
    "must_include_titles": _c_must_include,
    "must_exclude": _c_must_exclude,
    "min_playable_ratio": _c_min_playable,
    "max_per_artist": _c_max_per_artist,
    "not_degraded": _c_not_degraded,
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
        "query": case.get("query"),
        "intent_type": result.get("intent_type"),
        "num_songs": len(songs),
        "case_status": case_status,
        "outcomes": outcomes,
        "manual_review": checks.get("manual_review", []),
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


async def run(provider: str, cases_file: str, limit: int, verbose: bool) -> Dict[str, Any]:
    # 延迟导入：让 --help 在无重依赖时也能用
    from config.settings import settings
    settings.intent_llm_provider = provider
    settings.llm_default_provider = provider
    from agent.music_agent import MusicRecommendationAgent

    cases = json.loads(Path(cases_file).read_text(encoding="utf-8"))
    print(f"\n{'='*64}\n结果导向离线评测（Outcome Eval）\n{'='*64}")
    print(f"Provider: {provider} | Cases: {len(cases)} | top-k: {limit}\n")

    agent = MusicRecommendationAgent()
    reports = []
    t_start = time.time()

    for i, case in enumerate(cases, 1):
        query = case["query"]
        try:
            result = await agent.get_recommendations(query)
        except Exception as e:  # 单 case 失败不应中断整轮
            result = {"recommendations": [], "intent_type": "ERROR",
                      "errors": [{"node": "harness", "error": str(e)}]}
        rep = evaluate_case(case, result)
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
    for r in reports:
        for o in r["outcomes"]:
            by_check[o["name"]][o["status"]] += 1

    print(f"{'='*64}\n📊 汇总\n{'='*64}")
    print(f"用例: {n} | ✅ 通过 {n_pass} | ❌ 失败 {n_fail} | ➖ 数据不足 {n_indet}")
    if decided:
        print(f"可判定用例通过率: {n_pass}/{decided} = {n_pass/decided:.1%}")
    print(f"总耗时: {time.time()-t_start:.1f}s\n")
    print("按 check 类型:")
    print(f"  {'check':24s} {'pass':>6s} {'fail':>6s} {'skip':>6s}")
    print(f"  {'-'*44}")
    for name, c in sorted(by_check.items()):
        print(f"  {name:24s} {c['pass']:>6d} {c['fail']:>6d} {c['skip']:>6d}")

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
        "passed": n_pass, "failed": n_fail, "indeterminate": n_indet,
        "decided_pass_rate": round(n_pass / decided, 4) if decided else None,
        "by_check": dict(by_check), "cases": reports,
    }
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
    p.add_argument("--cases", default=str(Path(__file__).parent / "outcome_test_cases.json"))
    p.add_argument("--limit", type=int, default=15, help="请求的 top-k（仅记录，实际条数由管线 FinalCut 决定）")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args.provider, args.cases, args.limit, verbose=not args.quiet))


if __name__ == "__main__":
    main()
