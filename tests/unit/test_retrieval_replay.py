def test_frozen_cases_require_retrieval_plan():
    from tests.eval.evaluate_retrieval_replay import _frozen_cases

    report = {
        "cases": [
            {
                "id": "ok",
                "query": "quiet",
                "intent_type": "vector_search",
                "dialog_state": {
                    "last_complete_plan": {
                        "intent_type": "vector_search",
                        "retrieval_plan": {"use_vector": True},
                    }
                },
            },
            {"id": "missing", "query": "nothing"},
        ]
    }
    frozen = _frozen_cases(report, {"ok"})
    assert frozen == [
        {
            "id": "ok",
            "query": "quiet",
            "intent_type": "vector_search",
            "retrieval_plan": {"use_vector": True},
        }
    ]


def test_outcome_eval_exposes_sealed_cli_without_importing_agent():
    from pathlib import Path

    source = Path("tests/eval/evaluate_outcomes.py").read_text(encoding="utf-8")
    assert 'p.add_argument("--sealed"' in source
    assert '"case_hash"' in source
