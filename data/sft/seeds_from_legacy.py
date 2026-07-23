"""Convert the legacy ~600-example seed set into episode SEED specs (Phase A-3).

The legacy dataset (``data/sft/generate_planner_sft.py::data``) uses an OLD
schema (intent types like ``play_specific_song_online``/``search`` and a flat
retrieval_plan) whose ANSWERS are stale.  We keep only the INPUT queries as
seeds and discard every legacy output, so no stale answer can leak into a
training target — the current teacher re-derives current-schema decisions.

Seeds are deduplicated and excluded against the frozen eval/holdout/blind sets
(reusing the same guard as ``augment_queries``) so re-collection never leaks the
evaluation sets.

Output (jsonl, one episode per line, single-turn) matches ``collect_episodes``:
    {"episode_id": "legacy_0007", "profile": "", "memories": [], "turns": ["..."]}

Usage:
    python -m data.sft.seeds_from_legacy --out data/sft/seeds_legacy_episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_legacy_inputs() -> list[str]:
    """Return legacy input queries only — every legacy output is discarded."""
    from data.sft.generate_planner_sft import data as legacy

    inputs: list[str] = []
    for item in legacy:
        query = str((item or {}).get("input") or "").strip()
        if query:
            inputs.append(query)
    return inputs


def _eval_case_exclusions() -> set[str]:
    """Frozen eval/holdout/blind queries ONLY (NOT the seed set itself).

    ``augment_queries._load_exclusions`` also folds in the seed inputs, which is
    correct for de-duping *new* generated queries but wrong here: the legacy
    inputs ARE the seeds, so excluding against them would drop everything. We
    only need to guarantee no legacy input coincides with an evaluation case.
    """
    from data.sft.augment_queries import _normalize

    excluded: set[str] = set()
    cases_dir = PROJECT_ROOT / "tests" / "eval" / "cases"
    for path in cases_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload if isinstance(payload, list) else payload.get("cases") or payload.get("examples") or []
        for row in rows:
            if isinstance(row, dict):
                q = _normalize(row.get("query") or row.get("input") or "")
                if q:
                    excluded.add(q)
    return excluded


def build_seeds(limit: int = 0) -> tuple[list[dict], dict]:
    from data.sft.augment_queries import _normalize

    excluded = _eval_case_exclusions()
    raw = load_legacy_inputs()
    seeds: list[dict] = []
    seen: set[str] = set()
    leaked = 0
    duped = 0
    for query in raw:
        key = _normalize(query)
        if not key:
            continue
        if key in excluded:
            leaked += 1  # legacy input also lives in an eval/holdout set — drop it
            continue
        if key in seen:
            duped += 1
            continue
        seen.add(key)
        seeds.append(
            {
                "episode_id": f"legacy_{len(seeds):04d}",
                "profile": "",
                "memories": [],
                "turns": [query],
                "provenance": {"source_type": "legacy_seed_input_only"},
            }
        )
        if limit and len(seeds) >= limit:
            break
    stats = {
        "legacy_inputs": len(raw),
        "excluded_by_eval_overlap": leaked,
        "duplicates_dropped": duped,
        "seeds_written": len(seeds),
        "exclusion_set_size": len(excluded),
    }
    return seeds, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "sft" / "seeds_legacy_episodes.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    seeds, stats = build_seeds(limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for seed in seeds:
            sink.write(json.dumps(seed, ensure_ascii=False) + "\n")
    stats["out"] = str(args.out)
    print(json.dumps({"summary": stats}, ensure_ascii=False))
    return 0 if seeds else 1


if __name__ == "__main__":
    raise SystemExit(main())
