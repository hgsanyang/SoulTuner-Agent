"""Lightweight smoke checks for the P9-P14 quality flywheel.

The checks avoid LLM calls and external services. They verify that the local
quality loop still has its key deterministic pieces:

P9  context eval pressure cases and bounded post-recall adjustments
P10 catalog gap detector / controlled web fallback decisions
P11 ingest queue and tag hygiene
P12 exposure + slate feedback logs and readiness summary
P13 MemoryGateway hot-path preference mapping
P14 library / pending UI entrypoints
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _ok(name: str, detail: str = "") -> dict[str, Any]:
    return {"name": name, "status": "pass", "detail": detail}


def _warn(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "warn", "detail": detail}


def _fail(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "fail", "detail": detail}


def _load_cases(name: str) -> list[dict[str, Any]]:
    path = REPO_ROOT / "tests" / "eval" / "cases" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _check_context_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dev = _load_cases("context_dev.json")
    holdout = _load_cases("context_holdout.json")
    rows.append(_ok("context_dev_size", f"{len(dev)} cases") if len(dev) >= 52 else _fail("context_dev_size", str(len(dev))))
    rows.append(
        _ok("context_holdout_size", f"{len(holdout)} cases")
        if len(holdout) >= 16
        else _fail("context_holdout_size", str(len(holdout)))
    )
    pressure_ids = {
        "ctx_dev_audio_hh_02_soft_rain_low_dynamic",
        "ctx_dev_refine_lh_03_keep_rain_less_drum",
        "ctx_holdout_audio_hh_02_rainy_soft_no_driving",
    }
    found = {case.get("id") for case in [*dev, *holdout]}
    missing = sorted(pressure_ids - found)
    rows.append(_ok("context_pressure_cases", "rain/soft/multiturn present") if not missing else _fail("context_pressure_cases", ",".join(missing)))
    return rows


def _check_catalog_gap_and_adjustments() -> list[dict[str, Any]]:
    from agent.catalog_gap import analyze_catalog_gap, interleave_online_results
    from retrieval.post_recall_adjustments import apply_post_recall_adjustments

    local = [{"song": {"title": f"Song {i}", "artist": "A", "language": "Chinese", "preview_url": "u"}} for i in range(12)]
    plan = {
        "hard_constraints": {"language": "Chinese"},
        "soft_intent": {"vibe": "classic"},
        "hints": {},
        "metadata_constraints": {"release_year_from": 1980, "release_year_to": 1989, "era": "80s"},
    }
    gap = analyze_catalog_gap(local, plan, "推荐80年代的中文老歌", web_enabled=True)
    rows = [
        _ok("catalog_gap_release_fallback", ",".join(gap.reasons))
        if gap.action == "fallback" and "metadata_release_year_missing" in gap.reasons
        else _fail("catalog_gap_release_fallback", json.dumps(gap.model_dump(), ensure_ascii=False))
    ]
    mixed = interleave_online_results(
        local,
        [{"song": {"title": "Online", "artist": "B", "source": "online_search"}}],
        target_len=len(local),
        first_slot=3,
        stride=4,
    )
    rows.append(_ok("catalog_gap_interleave", f"{len(mixed)} rows") if any((item.get("song") or {}).get("source") == "online_search" for item in mixed) else _fail("catalog_gap_interleave", "no online row"))

    adjusted = apply_post_recall_adjustments(
        [
            {
                "song": {
                    "title": "fresh",
                    "artist": "A",
                    "genres": ["Lo-Fi"],
                    "moods": ["Peaceful"],
                    "scenarios": ["Rainy Day"],
                },
                "similarity_score": 0.5,
                "_graph_affinity": 1.0,
            },
            {
                "song": {
                    "title": "stale",
                    "artist": "A",
                    "genres": ["Dance"],
                    "moods": ["Energetic"],
                    "scenarios": ["Driving"],
                },
                "similarity_score": 0.5,
                "_graph_affinity": 0.0,
            },
        ],
        metadata_by_title={"fresh": {"updated_at": 1_800_000_000_000, "ts_beta": 1.0}},
        hints={"genres": ["Lo-Fi"], "mood": "Peaceful", "scenario": "Rainy Day"},
        soft_intent={"avoid": ["Dance", "Energetic", "Driving"]},
        now_ms=1_800_000_000_000,
    )
    max_delta = max(abs(float(item.get("_post_recall_delta") or 0)) for item in adjusted)
    rows.append(_ok("post_recall_delta_bounded", f"max_delta={max_delta:.3f}") if max_delta <= 0.08 else _fail("post_recall_delta_bounded", str(max_delta)))
    by_title = {item["song"]["title"]: item for item in adjusted}
    rows.append(
        _ok("post_recall_semantic_context", "calm fit up, conflict down")
        if by_title["fresh"]["_post_recall_delta"] > by_title["stale"]["_post_recall_delta"]
        and by_title["stale"]["_post_semantic_conflict_score"] > 0
        else _fail("post_recall_semantic_context", json.dumps(adjusted, ensure_ascii=False))
    )
    return rows


def _check_ingest_feedback_memory_and_tags() -> list[dict[str, Any]]:
    from services.feedback_logger import SLATE_FEEDBACK_FILE, load_jsonl, log_exposure, log_slate_feedback
    from services import ingest_queue
    from services.memory_gateway import derive_preferences_from_slate_feedback
    from services.ranking_policy import summarize_policy_readiness
    from services.tag_policy import clean_tag_payload

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        old_feedback = os.environ.get("MUSIC_FEEDBACK_DIR")
        old_queue = os.environ.get("MUSIC_INGEST_QUEUE_DIR")
        old_queue_paths = (
            ingest_queue.QUEUE_ROOT,
            ingest_queue.PENDING_DIR,
            ingest_queue.PROCESSING_DIR,
            ingest_queue.DONE_DIR,
            ingest_queue.FAILED_DIR,
        )
        try:
            os.environ["MUSIC_FEEDBACK_DIR"] = str(Path(tmp) / "feedback")
            os.environ["MUSIC_INGEST_QUEUE_DIR"] = str(Path(tmp) / "queue")
            queue_root = Path(tmp) / "queue"
            ingest_queue.QUEUE_ROOT = queue_root
            ingest_queue.PENDING_DIR = queue_root / "pending"
            ingest_queue.PROCESSING_DIR = queue_root / "processing"
            ingest_queue.DONE_DIR = queue_root / "done"
            ingest_queue.FAILED_DIR = queue_root / "failed"
            exposure_id = log_exposure(
                query="雨天安静一点",
                request_id="p9p14-smoke",
                recommendations=[{"song": {"title": "A", "artist": "B"}, "similarity_score": 0.9}],
            )
            feedback_id = log_slate_feedback(
                exposure_id=exposure_id,
                rating="too_noisy",
                reasons=["太吵了"],
                note="少一点 EDM",
            )
            slate = load_jsonl(Path(os.environ["MUSIC_FEEDBACK_DIR"]) / SLATE_FEEDBACK_FILE)
            readiness = summarize_policy_readiness(num_exposures=1, num_events=0, num_slate_feedback=len(slate), min_events=3)
            rows.append(_ok("slate_feedback_log", feedback_id) if slate else _fail("slate_feedback_log", "empty"))
            rows.append(_ok("ranking_readiness_safe", readiness["stage"]) if readiness["stage"] == "collect_feedback" else _warn("ranking_readiness_safe", readiness["stage"]))

            prefs = derive_preferences_from_slate_feedback(rating="too_noisy", reasons=["太吵"], note="少点土嗨 EDM")
            rows.append(_ok("memory_slate_mapping", json.dumps(prefs, ensure_ascii=False)) if "avoid_moods" in prefs else _fail("memory_slate_mapping", str(prefs)))

            job_id = ingest_queue.enqueue_songs([
                {
                    "title": "Smoke Song",
                    "artist": "Smoke Artist",
                    "audio_url": "/static/online_audio/Smoke Song - Smoke Artist.mp3",
                    "file_basename": "Smoke Song - Smoke Artist",
                }
            ])
            jobs = ingest_queue.list_jobs()
            rows.append(_ok("ingest_queue_pending", job_id) if jobs and jobs[0]["status"] == "pending" else _fail("ingest_queue_pending", str(jobs)))

            tags = clean_tag_payload({"genres": ["Indie", "indie", "Unknown", "Folk", "Rock", "Pop", "Dream Pop", "EDM"]})
            rows.append(_ok("tag_policy_cap", ",".join(tags["genres"])) if tags["genres"] == ["Indie", "Folk", "Rock", "Pop", "Dream Pop"] else _fail("tag_policy_cap", str(tags)))
        finally:
            if old_feedback is None:
                os.environ.pop("MUSIC_FEEDBACK_DIR", None)
            else:
                os.environ["MUSIC_FEEDBACK_DIR"] = old_feedback
            if old_queue is None:
                os.environ.pop("MUSIC_INGEST_QUEUE_DIR", None)
            else:
                os.environ["MUSIC_INGEST_QUEUE_DIR"] = old_queue
            (
                ingest_queue.QUEUE_ROOT,
                ingest_queue.PENDING_DIR,
                ingest_queue.PROCESSING_DIR,
                ingest_queue.DONE_DIR,
                ingest_queue.FAILED_DIR,
            ) = old_queue_paths
    return rows


def _check_ui_entrypoints() -> list[dict[str, Any]]:
    required = [
        "web/app/recommendations/page.tsx",
        "web/app/library/pending/page.tsx",
        "web/app/library/my-library/page.tsx",
        "web/components/Profile/UserProfilePanel.tsx",
        "web/lib/api.ts",
    ]
    rows = []
    for rel in required:
        path = REPO_ROOT / rel
        rows.append(_ok(f"ui:{rel}", "present") if path.exists() else _fail(f"ui:{rel}", "missing"))
    api_text = (REPO_ROOT / "web" / "lib" / "api.ts").read_text(encoding="utf-8")
    rows.append(_ok("ui_slate_feedback_api", "sendSlateFeedback") if "sendSlateFeedback" in api_text else _fail("ui_slate_feedback_api", "missing"))
    return rows


def run_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.extend(_check_context_cases())
    checks.extend(_check_catalog_gap_and_adjustments())
    checks.extend(_check_ingest_feedback_memory_and_tags())
    checks.extend(_check_ui_entrypoints())
    return {
        "checks": checks,
        "summary": {
            "passed": sum(1 for row in checks if row["status"] == "pass"),
            "warnings": sum(1 for row in checks if row["status"] == "warn"),
            "failed": sum(1 for row in checks if row["status"] == "fail"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P9-P14 lightweight smoke checks")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_checks()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for row in report["checks"]:
            marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[row["status"]]
            print(f"{marker} {row['name']}: {row.get('detail', '')}")
        print("summary:", json.dumps(report["summary"], ensure_ascii=False))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
