"""The gate has to be tested against the mistake the gate exists to prevent.

The first audit pass counted "recommendation with no acoustic_queries" as a
defect and reported 175 of them. But a named-song request that goes graph-only
and writes no acoustic query is *correct* — the number was measuring the wrong
thing, and the real count of executable defects was 7.

So the first test below is the named-song case: a validator that flags it is
broken, no matter how clean its output looks. Every other check gets a synthetic
fixture too, because a gate nobody has seen fail is a gate nobody has tested.
"""

from __future__ import annotations

import json

import pytest

from scripts.validate_sft_dataset import (
    check_contract,
    current_input,
    measure_coverage,
    measure_overlap,
    normalise,
    sample_id,
    shingles,
    validate,
)

SYSTEM = "你是音乐智能体的决策器。"


def row(assistant: dict, *, user: str = "[当前输入] 随便来点", episode="ep-1", turn=0, meta=None):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
        "meta": {"episode_id": episode, "turn_id": turn, **(meta or {})},
    }


def decision(**overrides) -> dict:
    base = {
        "request_kind": "recommendation",
        "response_mode": "answer",
        "tool_names": ["graph", "dense"],
        "hard": {"artist": [], "song": []},
        "soft": {},
        "hints": {},
        "metadata": {},
        "acoustic_queries": ["A slow ambient piece with warm pads."],
    }
    base.update(overrides)
    return base


def checks(rows, split="train"):
    findings, _ = check_contract(rows, split)
    return {f.check for f in findings}


# ---- the regression that motivated this file --------------------------------

def test_a_named_song_request_going_graph_only_is_not_a_defect():
    """"我要听 BTS 的 Dynamite" → graph, no acoustic query. Correct behaviour.
    Flagging this produced a fabricated 14% defect rate in the first pass."""
    rows = [row(decision(tool_names=["graph"], acoustic_queries=[],
                         hard={"artist": ["BTS"], "song": ["Dynamite"]}),
                user="[当前输入] 我要听 BTS 的 Dynamite")]
    assert checks(rows) == set()


def test_the_dense_lane_with_no_query_is_a_defect():
    """Same shape, one difference: dense is selected, so something has to run."""
    rows = [row(decision(tool_names=["graph", "dense"], acoustic_queries=[]))]
    assert "dense_lane_without_query" in checks(rows)


# ---- schema / parse ---------------------------------------------------------

def test_unparseable_assistant_is_hard():
    bad = {"messages": [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": "x"},
                        {"role": "assistant", "content": "{not json"}],
           "meta": {"episode_id": "ep-9", "turn_id": 0}}
    assert "assistant_not_json" in checks([bad])


def test_schema_violation_is_hard():
    rows = [row(decision(request_kind="not_a_kind"))]
    assert "schema_invalid" in checks(rows)


def test_a_clean_row_produces_nothing():
    assert checks([row(decision())]) == set()


# ---- executable contract ----------------------------------------------------

# The four rules below are enforced by PlannerDecisionV3._enforce_invariants,
# so they surface as `schema_invalid` rather than as their own check name. That
# is the point: re-implementing them in the gate would be unreachable code that
# reads like coverage. What must hold is that each is still a HARD finding.

def test_clarify_must_carry_a_question_and_run_nothing():
    silent = [row(decision(response_mode="clarify", tool_names=[], acoustic_queries=[]))]
    assert checks(silent) == {"schema_invalid"}

    busy = [row(decision(response_mode="clarify", tool_names=["graph"],
                         acoustic_queries=[], clarification="您是指哪一首？"))]
    assert checks(busy) == {"schema_invalid"}


def test_the_schema_layer_names_the_invariant_it_rejected():
    """"schema_invalid" alone is not actionable; the finding has to say which
    rule broke, or the gate just points at a file and shrugs."""
    rows = [row(decision(response_mode="clarify", tool_names=[], acoustic_queries=[]))]
    findings, _ = check_contract(rows, "train")
    assert "clarification" in findings[0].fact


def test_a_proper_clarify_row_is_clean():
    """A clarify turn runs no tools, so lane rules must not fire on it —
    otherwise every correct clarification looks like a missing-lane defect."""
    rows = [row(decision(response_mode="clarify", tool_names=[], acoustic_queries=[],
                         clarification="您是指哪一首？"))]
    assert checks(rows) == set()


def test_lane_outside_the_kind_is_a_defect():
    rows = [row(decision(request_kind="library", tool_names=["library", "dense"],
                         acoustic_queries=["x"]))]
    assert checks(rows) == {"schema_invalid"}


def test_a_recommendation_with_no_retrieval_lane_is_a_defect():
    rows = [row(decision(tool_names=[], acoustic_queries=[]))]
    assert checks(rows) == {"schema_invalid"}


def test_ingest_outside_acquisition_is_a_defect():
    rows = [row(decision(request_kind="library", tool_names=["library", "ingest"],
                         acoustic_queries=[]))]
    assert checks(rows) == {"schema_invalid"}


def test_chinese_acoustic_query_is_a_defect():
    """MuQ-MuLan's text tower is English-trained: this degrades recall, it is not
    a style nit."""
    rows = [row(decision(acoustic_queries=["一段温暖的氛围音乐"]))]
    assert "acoustic_query_not_english" in checks(rows)


def test_a_query_no_lane_consumes_is_a_defect():
    rows = [row(decision(tool_names=["graph"], acoustic_queries=["A warm ambient pad."]))]
    assert "query_with_no_dense_lane" in checks(rows)


# ---- report tier: must never block ------------------------------------------

def test_class_skew_and_entity_overlap_do_not_fail_the_gate():
    """V3 is 96% recommendation with heavy entity overlap and is still fine for a
    50-step preflight. A gate that blocked on it would have stopped work that
    was legitimately ready."""
    rows = [row(decision(hard={"artist": ["Coldplay"], "song": []})) for _ in range(5)]
    report = _validate_inline(rows, rows)
    assert report.ok
    assert report.stats["coverage"]["train"]["recommendation_share"] == 1.0
    assert report.stats["overlap"]["artists"]["share_seen"] == 1.0


def test_entity_overlap_is_reported_even_when_episodes_are_disjoint():
    """The leak episode_id cannot see: different sessions, different wording,
    same artist."""
    train = [row(decision(hard={"artist": ["Coldplay"], "song": []}), episode="tr-1")]
    ev = [row(decision(hard={"artist": ["Coldplay"], "song": []}), episode="ev-1")]
    report = _validate_inline(train, ev)
    assert report.stats["overlap"]["episode_shared"] == 0
    assert report.stats["overlap"]["artists"]["share_seen"] == 1.0


def test_disjoint_entities_report_zero_overlap():
    train = [row(decision(hard={"artist": ["Coldplay"], "song": []}), episode="tr-1")]
    ev = [row(decision(hard={"artist": ["Radiohead"], "song": []}), episode="ev-1")]
    assert _validate_inline(train, ev).stats["overlap"]["artists"]["share_seen"] == 0.0


def test_pp_per_sample_exposes_an_unmeasurable_class():
    """One sample in a class means one sample is worth 100 percentage points —
    a 3pp gate cannot be evaluated against it."""
    rows = [row(decision()) for _ in range(3)]
    rows.append(row(decision(request_kind="conversation", tool_names=[], acoustic_queries=[])))
    cov = measure_coverage(rows, check_contract(rows, "train")[1])
    assert cov["pp_per_sample"]["conversation"] == 100.0


def test_coverage_counts_multi_turn_and_traces():
    rows = [
        row(decision(), user="[对话历史]\n用户: a\n[上轮检索计划] intent=x\n[当前输入] b", turn=1),
        row(decision(), user="[当前输入] c", turn=0, meta={"trace_id": "t-1"}),
    ]
    cov = measure_coverage(rows, check_contract(rows, "train")[1])
    assert cov["multi_turn_share"] == 0.5
    assert cov["history_share"] == 0.5
    assert cov["prev_plan_share"] == 0.5
    assert cov["rows_with_execution_trace"] == 1


# ---- privacy: findings must not carry user text -----------------------------

def test_a_schema_error_does_not_echo_the_offending_value():
    """pydantic puts the rejected input in err["input"]. A free-text field can
    hold anything the user typed, so the finding must read loc/type/msg only."""
    secret = "我失恋了想听点难过的歌"
    bad = decision()
    bad["decision_summary"] = secret * 40      # exceeds max_length=200
    findings, _ = check_contract([row(bad)], "train")
    assert findings and findings[0].check == "schema_invalid"
    assert secret not in findings[0].fact


def test_findings_never_contain_the_user_query():
    secret = "我失恋了想听点难过的歌"
    rows = [row(decision(tool_names=["graph", "dense"], acoustic_queries=[]),
                user=f"[当前输入] {secret}", episode="ep-secret", turn=2)]
    findings, _ = check_contract(rows, "train")
    assert findings
    for finding in findings:
        blob = f"{finding.check}{finding.sample_id}{finding.fact}"
        assert secret not in blob
    assert findings[0].sample_id == "ep-secret:2"


# ---- helpers ----------------------------------------------------------------

def test_sample_id_falls_back_to_an_index_without_metadata():
    assert sample_id({"meta": {}}, 7) == "#7"


def test_current_input_strips_the_history_block():
    assert current_input("[对话历史]\n用户: x\n[当前输入] 换点别的") == "换点别的"


def test_normalise_is_entity_blind_for_numbers_and_quotes():
    assert normalise("播放《Lemon》 2 首") == normalise("播放《Lemon》 3 首")


def test_shingles_of_identical_text_match_completely():
    assert shingles("同一句话") == shingles("同一句话")


def _validate_inline(train_rows, eval_rows):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tr = Path(tmp) / "train.jsonl"
        ev = Path(tmp) / "eval.jsonl"
        for path, rows in ((tr, train_rows), (ev, eval_rows)):
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
        return validate(tr, ev)


def test_validate_reads_both_files_and_reports_prompt_variants():
    train = [row(decision(), episode="tr-1")]
    ev = [row(decision(), episode="ev-1")]
    report = _validate_inline(train, ev)
    assert report.ok
    assert report.stats["system_prompt_variants"] == 1
    assert report.stats["coverage"]["eval"]["rows"] == 1


def test_a_contract_violation_makes_the_report_not_ok():
    bad = [row(decision(tool_names=["graph", "dense"], acoustic_queries=[]), episode="tr-1")]
    good = [row(decision(), episode="ev-1")]
    report = _validate_inline(bad, good)
    assert not report.ok
    assert report.stats["hard_findings_by_check"]["dense_lane_without_query"] == 1


@pytest.mark.parametrize("kind,lanes", [
    ("information", ["graph"]),
    ("information", ["web"]),
    ("acquisition", ["web", "ingest"]),
    ("library", ["library"]),
    ("conversation", []),
])
def test_the_documented_lane_choices_are_all_accepted(kind, lanes):
    """If the gate rejects a decision the schema itself calls legal, the gate is
    wrong — not the data."""
    rows = [row(decision(request_kind=kind, tool_names=lanes, acoustic_queries=[]))]
    assert checks(rows) == set()
