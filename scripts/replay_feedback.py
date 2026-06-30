"""Offline replay for exposure logs and lightweight ranking-weight learning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.feedback_logger import (  # noqa: E402
    estimate_tri_anchor_weights,
    learn_tri_anchor_weights,
    learned_weights_path,
    load_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay SoulTuner feedback logs")
    parser.add_argument("--feedback-dir", default="data/feedback", help="Directory with exposures.jsonl and events.jsonl")
    parser.add_argument("--method", choices=["learn", "heuristic"], default="learn",
                        help="learn=auditable logistic learner; heuristic=legacy event correlation")
    parser.add_argument("--min-events", type=int, default=8,
                        help="Minimum matched labeled events required before writing learned weights")
    parser.add_argument("--write", action="store_true", help="Write learned ranking_weights.json")
    parser.add_argument("--force-write", action="store_true",
                        help="Write even when the learner reports insufficient_data")
    args = parser.parse_args()

    feedback_dir = Path(args.feedback_dir)
    exposures = load_jsonl(feedback_dir / "exposures.jsonl")
    events = load_jsonl(feedback_dir / "events.jsonl")
    if args.method == "heuristic":
        report = estimate_tri_anchor_weights(exposures, events)
    else:
        report = learn_tri_anchor_weights(exposures, events, min_events=args.min_events)
    report["num_exposures"] = len(exposures)
    report["num_events"] = len(events)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write:
        if report.get("status") == "insufficient_data" and not args.force_write:
            print("Skipped write: insufficient labeled feedback. Use --force-write to override.")
            return 2
        output = learned_weights_path() if args.feedback_dir == "data/feedback" else feedback_dir / "ranking_weights.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
