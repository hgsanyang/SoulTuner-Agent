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
    # The write path refuses to run while the API answers on :8501. Tests must not
    # depend on whether this dev box happens to have the backend up, so pin it
    # closed by default; the test that asserts the refusal re-opens it.
    monkeypatch.setattr(mig, "_backend_is_live", lambda: False)
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
    assert all(row["interaction_mode"] == "legacy" for row in fs.load_slate_feedback())
    assert all(row["training_eligible"] is False for row in fs.load_slate_feedback())


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


def test_migration_aborts_before_writing_when_a_row_has_no_id(feedback_dir, capsys):
    """Codex R4.7: an id-less row used to be handed to the store, which minted a
    synthetic uuid for it — so the count gate failed AFTER the store was already
    polluted, and a re-run duplicated the row. Detect it during planning and
    write nothing."""
    _write(feedback_dir, "slate_feedback.jsonl",
           [{"feedback_id": "ok-1", "rating": "fits", "ts": 1},
            {"rating": "off", "ts": 2},                       # <- no id at all
            {"feedback_id": "ok-2", "rating": "fits", "ts": 3}])
    rc = _run(mig)
    assert rc == 1
    out = capsys.readouterr().out
    assert "ABORTED BEFORE WRITING" in out
    assert "record #2" in out                    # points at the offending row
    # the decisive assertion: NOTHING was written, not even the two valid rows
    assert fs.counts()["slate_feedback"] == 0


def test_migration_refuses_to_write_while_the_backend_is_live(feedback_dir, monkeypatch, capsys):
    """A live backend appends rows mid-run, so the completeness gate compares
    against a moving target: a correct store looks broken and a torn one can look
    fine. Dry-run stays allowed (it writes nothing)."""
    _write(feedback_dir, "slate_feedback.jsonl",
           [{"feedback_id": "a", "rating": "fits", "ts": 1}])
    monkeypatch.setattr(mig, "_backend_is_live", lambda: True)

    assert _run(mig) == 1
    assert "REFUSING TO RUN" in capsys.readouterr().out
    assert fs.counts()["slate_feedback"] == 0

    # --dry-run is still allowed while the backend is up
    import sys
    argv = sys.argv
    sys.argv = ["migrate", "--dry-run"]
    try:
        assert mig.main() == 0
    finally:
        sys.argv = argv


def test_backend_probe_follows_the_configured_port(monkeypatch):
    """Hardcoding 8501 silently disabled the guard for anyone who moved the API:
    it would probe a dead port, see nothing, and migrate under a live backend."""
    monkeypatch.setenv("API_PORT", "9123")
    assert mig._backend_port() == 9123
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.setenv("BACKEND_PORT", "9124")          # accepted alias
    assert mig._backend_port() == 9124
    monkeypatch.delenv("BACKEND_PORT", raising=False)
    monkeypatch.setenv("API_PORT", "not-a-port")        # junk falls through
    assert mig._backend_port() == 8501

    probed: list[str] = []

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.setenv("API_PORT", "9125")
    monkeypatch.setattr(mig.urllib.request, "urlopen",
                        lambda url, timeout=0: (probed.append(url), _Resp())[1])
    assert mig._backend_is_live() is True
    assert probed == ["http://127.0.0.1:9125/health"]


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
