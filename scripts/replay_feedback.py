"""Offline replay for exposure logs and lightweight ranking-weight learning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.feedback_logger import (  # noqa: E402
    estimate_tri_anchor_weights,
    learn_tri_anchor_weights,
    load_jsonl,
)
from services.ranking_learning import learn_ranking_policy  # noqa: E402
from services.ranking_policy import (  # noqa: E402
    promote_candidate,
    rollback_policy,
    write_candidate,
)


def _default_feedback_dir() -> str:
    configured = os.getenv("MUSIC_FEEDBACK_DIR")
    if configured:
        return configured
    data_path = os.getenv("MUSIC_DATA_PATH")
    env_path = REPO_ROOT / ".env"
    if not data_path and env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MUSIC_DATA_PATH="):
                data_path = line.split("=", 1)[1].strip().strip("\"'")
                break
    return str(Path(data_path) / "feedback") if data_path else "data/feedback"


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay SoulTuner feedback logs")
    parser.add_argument("--feedback-dir", default=_default_feedback_dir(),
                        help="Directory with exposures.jsonl and events.jsonl")
    parser.add_argument("--method", choices=["v2", "legacy", "heuristic"], default="v2",
                        help="v2=strict exposure attribution + validation; legacy/heuristic retained for audits")
    parser.add_argument("--min-events", type=int, default=20,
                        help="Minimum matched labeled events required before writing learned weights")
    parser.add_argument("--per-user-min-events", type=int, default=30)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--write-candidate", action="store_true",
                        help="Write ranking_policy.candidate.json, even before manual promotion")
    parser.add_argument("--promote", action="store_true",
                        help="Promote an already-written candidate that passed validation")
    parser.add_argument("--write", action="store_true",
                        help="Compatibility shortcut: write candidate and promote only when gate passes")
    parser.add_argument("--rollback", action="store_true",
                        help="Restore ranking_policy.previous.json as the active policy")
    parser.add_argument("--force-write", action="store_true",
                        help="Legacy methods only; v2 never bypasses its validation gate")
    args = parser.parse_args()

    feedback_dir = Path(args.feedback_dir)
    if args.rollback:
        print(f"Rolled back to {rollback_policy(feedback_dir)}")
        return 0
    if args.promote and not args.write:
        print(f"Promoted {promote_candidate(feedback_dir)}")
        return 0

    exposures = load_jsonl(feedback_dir / "exposures.jsonl")
    events = load_jsonl(feedback_dir / "events.jsonl")
    if args.method == "heuristic":
        report = estimate_tri_anchor_weights(exposures, events)
    elif args.method == "legacy":
        report = learn_tri_anchor_weights(exposures, events, min_events=args.min_events)
    else:
        report = learn_ranking_policy(
            exposures,
            events,
            min_events=args.min_events,
            per_user_min_events=args.per_user_min_events,
            validation_ratio=args.validation_ratio,
        )
    report["num_exposures"] = len(exposures)
    report["num_events"] = len(events)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.method == "v2" and (args.write_candidate or args.write):
        candidate = write_candidate(report, feedback_dir)
        print(f"Wrote candidate {candidate}")
        if args.write:
            if not report.get("gate_passed"):
                print("Skipped promotion: candidate did not pass the offline validation gate.")
                return 2
            print(f"Promoted {promote_candidate(feedback_dir)}")
    elif args.method != "v2" and args.write:
        if report.get("status") == "insufficient_data" and not args.force_write:
            print("Skipped write: insufficient labeled feedback. Use --force-write to override.")
            return 2
        output = feedback_dir / "ranking_weights.legacy.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
