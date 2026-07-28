"""Apply an explicit, reviewable judgement to the 71 quarantined V3 samples.

This script does not infer labels from keywords and does not call an LLM.  The
source file is frozen by SHA-256 and every sample key must appear in one of the
review sets below.  False clarification targets remain quarantined instead of
having a replacement target invented.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_SHA256 = "ab618590cc54cc2700d1b453c15775ddb5fa489cd6f7d1a3a634b68b7a00526d"
REVIEW_VERSION = "planner_v3_ambiguous_manual_review_2026_07_28"


WEB_RECOMMENDATION_IDS = {
    "artist_catalog_0006",
    "artist_catalog_0017",
    "artist_catalog_0027",
    "artist_catalog_0030",
    "artist_catalog_0047",
    "artist_catalog_0074",
    "profile_memory_0016",
    "timeliness_web_0003",
    "timeliness_web_0006",
    "timeliness_web_0012",
    "timeliness_web_0013",
    "timeliness_web_0022",
    "timeliness_web_0027",
    "timeliness_web_0029",
    "timeliness_web_0034",
    "timeliness_web_0035",
    "timeliness_web_0041",
}

WEB_INFORMATION_IDS = {
    "artist_catalog_0009",
    "artist_catalog_0016",
    "artist_catalog_0019",
    "artist_catalog_0043",
    "artist_catalog_0051",
    "timeliness_web_0000",
    "timeliness_web_0001",
    "timeliness_web_0002",
    "timeliness_web_0004",
    "timeliness_web_0008",
    "timeliness_web_0009",
    "timeliness_web_0010",
    "timeliness_web_0014",
    "timeliness_web_0015",
    "timeliness_web_0017",
    "timeliness_web_0019",
    "timeliness_web_0021",
    "timeliness_web_0023",
    "timeliness_web_0024",
    "timeliness_web_0026",
    "timeliness_web_0028",
    "timeliness_web_0030",
    "timeliness_web_0031",
    "timeliness_web_0032",
    "timeliness_web_0033",
    "timeliness_web_0036",
    "timeliness_web_0037",
    "timeliness_web_0038",
    "timeliness_web_0042",
    "timeliness_web_0044",
    "timeliness_web_0045",
}

# These targets over-clarify actionable music requests.  They are excluded
# until the strong teacher is called again; no replacement plan is fabricated.
FALSE_CLARIFICATION_REASONS = {
    "artist_catalog_0022": "album preference can be answered with a bounded recommendation",
    "negation_0014": "a negative timbre constraint is sufficient for acoustic retrieval",
    "supp_contradiction_0009": "rock/folk fusion is an actionable style request",
    "supp_contradiction_0010": "new artist with a nostalgic classic feel is acoustically actionable",
    "supp_contradiction_0011": "rich a-cappella arrangement is internally consistent",
    "supp_contradiction_0013": "sad lyrics over an upbeat mood is a valid contrast",
    "supp_contradiction_0029": "a male singer using soprano-like register is possible",
}

VALID_CLARIFICATION_IDS = {
    "supp_contradiction_0000",
    "supp_contradiction_0002",
    "supp_contradiction_0003",
    "supp_contradiction_0005",
    "supp_contradiction_0007",
    "supp_contradiction_0008",
    "supp_contradiction_0012",
    "supp_contradiction_0015",
    "supp_contradiction_0017",
    "supp_contradiction_0021",
    "supp_contradiction_0025",
    "supp_contradiction_0026",
    "supp_contradiction_0030",
    "supp_contradiction_0031",
    "supp_contradiction_0035",
    "supp_contradiction_0038",
}


def _load_frozen_source(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "ambiguous source changed; review decisions must be repeated "
            f"(expected {EXPECTED_SOURCE_SHA256}, got {actual_hash})"
        )
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 71:
        raise ValueError(f"expected 71 quarantined rows, got {len(rows)}")
    return rows


def _sample_key(row: dict[str, Any]) -> str:
    return f"{row.get('episode_id')}#{row.get('turn_id')}"


def _review_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    episode_id = str(row["episode_id"])
    original = dict(row["teacher_decision_v3"])
    legacy_intent = str((row.get("migration") or {}).get("legacy_intent") or "")
    revised: dict[str, Any] | None = None

    if episode_id in WEB_RECOMMENDATION_IDS:
        revised = dict(original)
        revised["request_kind"] = "recommendation"
        revised["response_mode"] = "answer"
        revised["tool_names"] = ["graph", "web"]
        revised.pop("clarification", None)
        revised["decision_summary"] = "联网补充后做曲目推荐"
        classification = "recommendation_with_web"
        reason = "the query asks the agent to choose or recommend tracks, not only report a fact"
        eligible = True
    elif episode_id in WEB_INFORMATION_IDS:
        revised = dict(original)
        revised["request_kind"] = "information"
        revised["response_mode"] = "answer"
        revised["tool_names"] = ["web"]
        revised.pop("clarification", None)
        classification = "external_information"
        reason = "the requested output is a fact, list, ranking, date, identity, or release status"
        eligible = True
    elif episode_id in VALID_CLARIFICATION_IDS:
        revised = dict(original)
        classification = "valid_clarification"
        reason = "the request has a severe contradiction or unresolved meaning with materially different answers"
        eligible = True
    elif episode_id in FALSE_CLARIFICATION_REASONS:
        classification = "false_clarification"
        reason = FALSE_CLARIFICATION_REASONS[episode_id]
        eligible = False
    else:
        raise ValueError(f"sample has no explicit review decision: {_sample_key(row)}")

    review = {
        "sample_key": _sample_key(row),
        "episode_id": episode_id,
        "turn_id": row["turn_id"],
        "current_query": row["current_query"],
        "chat_history": row.get("chat_history") or "",
        "legacy_intent": legacy_intent,
        "classification": classification,
        "reason": reason,
        "training_eligible": eligible,
        "data_purpose": "planner_v3_sft_gate_review",
        "review_version": REVIEW_VERSION,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "original_decision_v3": original,
        "revised_decision_v3": revised,
    }
    if not eligible:
        review["required_action"] = "recollect_with_strong_teacher"
        return review, None

    trainable = {
        **{key: value for key, value in row.items() if key not in {"teacher_decision_v3", "migration"}},
        "teacher_decision_v3": revised,
        "migration": {
            "ambiguous": False,
            "reason": f"resolved by {REVIEW_VERSION}",
            "legacy_intent": legacy_intent,
        },
        "training_governance": {
            "training_eligible": True,
            "data_purpose": "planner_v3_sft",
            "review_version": REVIEW_VERSION,
            "source_type": (row.get("provenance") or {}).get("source_type"),
            "source_sample_key": _sample_key(row),
        },
    }
    return review, trainable


def review(
    source: Path,
    review_output: Path,
    trainable_output: Path,
    summary_output: Path,
) -> dict[str, Any]:
    rows = _load_frozen_source(source)
    reviews: list[dict[str, Any]] = []
    trainable: list[dict[str, Any]] = []
    for row in rows:
        audit_row, resolved = _review_row(row)
        reviews.append(audit_row)
        if resolved is not None:
            trainable.append(resolved)

    expected_ids = WEB_RECOMMENDATION_IDS | WEB_INFORMATION_IDS | set(
        FALSE_CLARIFICATION_REASONS
    ) | VALID_CLARIFICATION_IDS
    actual_ids = {str(row["episode_id"]) for row in rows}
    if expected_ids != actual_ids:
        raise ValueError(
            f"review coverage mismatch; missing={sorted(actual_ids - expected_ids)}, "
            f"extra={sorted(expected_ids - actual_ids)}"
        )

    for path in (review_output, trainable_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    review_output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in reviews),
        encoding="utf-8",
    )
    trainable_output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in trainable),
        encoding="utf-8",
    )

    class_counts = Counter(item["classification"] for item in reviews)
    kind_counts = Counter(
        item["teacher_decision_v3"]["request_kind"] for item in trainable
    )
    mode_counts = Counter(
        item["teacher_decision_v3"]["response_mode"] for item in trainable
    )
    summary = {
        "review_version": REVIEW_VERSION,
        "source": "data/teacher/private/ambiguous_samples.jsonl",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "reviewed": len(reviews),
        "resolved_trainable": len(trainable),
        "still_quarantined": len(reviews) - len(trainable),
        "classification": dict(sorted(class_counts.items())),
        "request_kind": dict(sorted(kind_counts.items())),
        "response_mode": dict(sorted(mode_counts.items())),
        "review_output": str(review_output),
        "trainable_output": str(trainable_output),
    }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("data/sft/reviews/v3_ambiguous_review.jsonl"),
    )
    parser.add_argument(
        "--trainable-output",
        type=Path,
        default=Path("data/sft/reviews/v3_resolved_trainable.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/sft/reviews/v3_ambiguous_review_summary.json"),
    )
    args = parser.parse_args()
    summary = review(
        args.source,
        args.review_output,
        args.trainable_output,
        args.summary_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
