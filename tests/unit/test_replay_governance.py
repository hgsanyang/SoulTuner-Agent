"""A hand-run script must not be the loophole in a rule the API enforces.

replay_feedback.py used to read the raw JSONL, so developer-mode interactions and
pre-governance test records would have trained a ranking policy as if they were
real user feedback.
"""

from __future__ import annotations

import scripts.replay_feedback as replay


def test_replay_reads_the_training_eligible_view_by_default():
    """Not a behavioural test of the learner — a wiring test. The default path
    must call the same loaders the API route calls."""
    src = replay.__file__
    from pathlib import Path

    text = Path(src).read_text(encoding="utf-8")
    for loader in ("load_training_exposures", "load_training_events",
                   "load_training_slate_feedback"):
        assert loader in text, f"{loader} not used; governance bypassed"


def test_inspection_mode_cannot_write_or_promote(monkeypatch, capsys, tmp_path):
    """--include-ineligible exists to SHOW what governance holds back. Letting it
    also write a policy would just rename the loophole."""
    import sys

    for flag in ("--write", "--write-candidate", "--promote"):
        argv = sys.argv
        sys.argv = ["replay", "--feedback-dir", str(tmp_path),
                    "--include-ineligible", flag]
        try:
            rc = replay.main()
        finally:
            sys.argv = argv
        assert rc == 1, f"{flag} was allowed alongside --include-ineligible"
        assert "REFUSING" in capsys.readouterr().out


def test_inspection_mode_warns_that_numbers_are_diagnostic(monkeypatch, capsys, tmp_path):
    import sys

    argv = sys.argv
    sys.argv = ["replay", "--feedback-dir", str(tmp_path), "--include-ineligible"]
    try:
        replay.main()
    finally:
        sys.argv = argv
    assert "NOT training-eligible" in capsys.readouterr().out
