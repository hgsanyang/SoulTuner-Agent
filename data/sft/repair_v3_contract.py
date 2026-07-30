"""Build a contract-clean V3 derivative without mutating the frozen baseline.

The 2026-07-29 gate found eleven executable-contract violations:

* seven dense decisions had no acoustic query;
* four acoustic queries contained CJK text even though MuQ's text input is
  expected to be an English acoustic description.

The repairs are deliberately keyed by the immutable episode/turn identity.
They are data-review decisions, not production keyword rules. The source V3
files remain byte-for-byte frozen so every repair stays auditable and
reversible.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from schemas.planner_decision_v3 import PlannerDecisionV3


REPAIR_VERSION = "v3_contract_repairs_2026_07_29"


ACOUSTIC_QUERY_REPAIRS: dict[str, list[str]] = {
    "supp_contradiction_0029:0": [
        "A male vocalist singing in an exceptionally high soprano-like register, "
        "with clear head voice, operatic brightness, and a delicate upper range."
    ],
    "supp_contradiction_0011:0": [
        "A purely a cappella vocal ensemble with no instruments, using dense layered "
        "harmonies, vocal percussion, and a rich dramatic arrangement."
    ],
    "supp_contradiction_0010:0": [
        "A debut song by a new artist with timeless classic-pop songwriting, warm "
        "analog texture, memorable melody, and an established old-record feeling."
    ],
    "supp_contradiction_0009:0": [
        "An organic song blending distorted electric-guitar energy with intimate "
        "acoustic folk textures, without settling fully into either genre."
    ],
    "supp_contradiction_0013:0": [
        "A bittersweet love song with sorrowful emotional lyrics but a bright, "
        "playful arrangement and an unexpectedly joyful atmosphere."
    ],
    "negation_0014:0": [
        "An organic live-band performance built from real acoustic and electric "
        "instruments, natural drums, and very little synthetic electronic texture."
    ],
}


# This request compares two named albums. Graph/web metadata can serve it; a
# dense lane with no stated sonic preference has no meaningful vector query.
REMOVE_DENSE_REPAIRS = {"artist_catalog_0022:0"}


CJK_QUERY_REPLACEMENTS: dict[str, tuple[str, str]] = {
    "multi_turn_0109:2": ("Dark and凶狠 drill music", "Dark and ferocious drill music"),
    "scenario_0089:0": (
        "The melody is piercing and虐, designed for cathartic crying.",
        "The melody is piercing and emotionally devastating, designed for cathartic crying.",
    ),
    "multi_turn_0017:2": ("and缠绵", "and lingering"),
    "multi_turn_0108:1": ("a朦胧, atmospheric", "a hazy, blurred, atmospheric"),
}


EXPECTED_REPAIRS = (
    set(ACOUSTIC_QUERY_REPAIRS)
    | REMOVE_DENSE_REPAIRS
    | set(CJK_QUERY_REPLACEMENTS)
)


def _sample_key(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    return f"{meta.get('episode_id')}:{meta.get('turn_id', 0)}"


def _assistant_index(row: dict[str, Any]) -> int:
    for index, message in enumerate(row.get("messages") or []):
        if message.get("role") == "assistant":
            return index
    raise ValueError(f"missing assistant message: {_sample_key(row)}")


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return an independently validated repaired copy and whether it changed."""
    key = _sample_key(row)
    if key not in EXPECTED_REPAIRS:
        return row, False

    repaired = copy.deepcopy(row)
    assistant_index = _assistant_index(repaired)
    payload = json.loads(repaired["messages"][assistant_index]["content"])

    if key in ACOUSTIC_QUERY_REPAIRS:
        payload["acoustic_queries"] = ACOUSTIC_QUERY_REPAIRS[key]

    if key in REMOVE_DENSE_REPAIRS:
        payload["tool_names"] = [
            lane for lane in payload.get("tool_names") or [] if lane != "dense"
        ]
        payload["acoustic_queries"] = []

    if key in CJK_QUERY_REPLACEMENTS:
        old, new = CJK_QUERY_REPLACEMENTS[key]
        queries = list(payload.get("acoustic_queries") or [])
        replaced = [query.replace(old, new) for query in queries]
        if replaced == queries:
            raise ValueError(f"expected acoustic fragment not found: {key}")
        payload["acoustic_queries"] = replaced

    decision = PlannerDecisionV3.model_validate(payload)
    repaired["messages"][assistant_index]["content"] = json.dumps(
        decision.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    meta = dict(repaired.get("meta") or {})
    meta["contract_repair"] = {
        "version": REPAIR_VERSION,
        "sample_key": key,
        "review_basis": "v3_gate_2026_07_29",
    }
    repaired["meta"] = meta
    return repaired, True


def repair_file(source: Path, output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    repaired_rows: list[dict[str, Any]] = []
    repaired_keys: set[str] = set()
    for row in rows:
        repaired, changed = repair_row(row)
        repaired_rows.append(repaired)
        if changed:
            repaired_keys.add(_sample_key(row))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in repaired_rows
        ),
        encoding="utf-8",
    )
    return {
        "source": str(source),
        "output": str(output),
        "rows": len(rows),
        "repaired_keys": sorted(repaired_keys),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/sft/train_v3_chatml.jsonl"))
    parser.add_argument("--eval", dest="eval_path", type=Path, default=Path("data/sft/eval_v3_chatml.jsonl"))
    parser.add_argument("--train-out", type=Path, default=Path("data/teacher/private/v4/train_v3_repaired.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("data/teacher/private/v4/regression_v3_repaired.jsonl"))
    args = parser.parse_args()

    reports = [
        repair_file(args.train, args.train_out),
        repair_file(args.eval_path, args.eval_out),
    ]
    actual = {
        key for report in reports for key in report["repaired_keys"]
    }
    if actual != EXPECTED_REPAIRS:
        missing = sorted(EXPECTED_REPAIRS - actual)
        unexpected = sorted(actual - EXPECTED_REPAIRS)
        raise SystemExit(f"repair set mismatch: missing={missing} unexpected={unexpected}")
    print(json.dumps({"repair_version": REPAIR_VERSION, "splits": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
