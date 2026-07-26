"""Migration gate + GBK-safe output + canonical-overall consumer equivalence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.migrate_feedback_jsonl_to_sqlite as mig
import services.feedback_store as fs


@pytest.fixture
def feedback_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_FEEDBACK_DIR", str(tmp_path))
    fs.reset_connection()
    yield tmp_path
    fs.reset_connection()


def _write(tmp_path, name, rows):
    (tmp_path / name).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_migration_succeeds_and_preserves_every_unique_record(feedback_dir, capsys):
    # 3 legacy (feedback_id only) + 2 new — none may be dropped
    _write(feedback_dir, "slate_feedback.jsonl",
           [{"feedback_id": f"l{i}", "rating": "partial", "ts": i} for i in range(3)]
           + [{"slate_feedback_id": f"n{i}", "feedback_id": f"n{i}", "rating": "off", "ts": 9 + i}
              for i in range(2)])
    rc = _run(mig)
    assert rc == 0
    assert fs.counts()["slate_feedback"] == 5


def test_migration_fails_when_a_write_is_incomplete(feedback_dir, monkeypatch):
    """The gate must be after == expected_unique, not after >= before: a partial
    write (store gains rows but not all) has to FAIL, not report success."""
    _write(feedback_dir, "slate_feedback.jsonl",
           [{"feedback_id": f"l{i}", "rating": "off", "ts": i} for i in range(5)])

    real_insert = fs.insert_slate_feedback
    calls = {"n": 0}

    def _drop_after_one(payload):
        calls["n"] += 1
        if calls["n"] > 1:      # only the first row actually lands
            return
        real_insert(payload)

    monkeypatch.setattr(fs, "insert_slate_feedback", _drop_after_one)
    rc = _run(mig)
    assert rc == 1                       # expected 5, got 1 -> must fail
    assert fs.counts()["slate_feedback"] == 1


def test_migration_output_is_ascii_only():
    """conda run on Windows re-encodes stdout as GBK and dies on any non-ASCII;
    the whole script must stay ASCII."""
    src = Path(mig.__file__).read_text(encoding="utf-8")
    # only the literals actually printed matter, but the safest guarantee is the
    # whole module being ASCII — comments included, since tracebacks echo them.
    src.encode("ascii")   # raises UnicodeEncodeError if any non-ASCII slipped in


def _run(module) -> int:
    """Invoke the migration CLI's main() with no args (write mode)."""
    import sys

    argv = sys.argv
    sys.argv = ["migrate"]
    try:
        return module.main()
    finally:
        sys.argv = argv


def test_diagnostics_count_overall_equally_for_new_and_legacy():
    """catalog + feedback diagnostics must read canonical `overall`, so a new
    {overall:off} and a legacy {rating:too_noisy} count as the same bucket."""
    from services.catalog_diagnostics import summarize_catalog_bias
    from schemas.feedback_events import slate_overall

    legacy = {"rating": "too_noisy", "reasons": []}
    new = {"overall": "off", "reasons": []}
    assert slate_overall(legacy) == slate_overall(new) == "off"

    # feed both through the catalog diagnostics counter path (catalog_rows, exposures, slate)
    report = summarize_catalog_bias([], [], [legacy, new])
    ratings = {row["label"]: row["count"] for row in report["slate_feedback"]["ratings"]}
    assert ratings.get("off") == 2


def test_memory_polarity_from_overall_matches_for_new_and_legacy():
    from services.memory_gateway import _feedback_polarity, _slate_overall_value

    legacy_overall = _slate_overall_value("too_noisy", None)
    new_overall = _slate_overall_value("", {"overall": "off"})
    assert legacy_overall == new_overall == "off"
    assert _feedback_polarity(legacy_overall, []) == _feedback_polarity(new_overall, []) == "negative"
