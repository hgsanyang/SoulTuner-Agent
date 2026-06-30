import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.eval.calibrate_soft_judge import calibrate, load_gold
from tests.eval.soft_judge import judge_objective_soft_intent, objective_tokens, tag_hit


def test_objective_tokens_ignore_title_artist_and_explanation():
    song = {
        "title": "Energetic Party Song",
        "artist": "Sleep Band",
        "genre": "Indie/Dreamy",
        "moods": ["Peaceful"],
        "scenarios": ["Late Night"],
        "reason": "This explanation says workout party.",
        "final_response": "The system says it is energetic.",
    }

    tokens = objective_tokens(song)

    assert "indie" in tokens
    assert "dreamy" in tokens
    assert "peaceful" in tokens
    assert "late night" in tokens
    assert "energetic" not in tokens
    assert "party" not in tokens
    assert "sleep" not in tokens


def test_objective_tokens_map_instrumental_booleans():
    assert "instrumental" in objective_tokens({"instrumental": True})
    assert "vocal" in objective_tokens({"is_instrumental": False})


def test_tag_hit_does_not_treat_pop_as_kpop():
    assert not tag_hit(["pop", "r&b", "romantic"], ["k-pop"])
    assert tag_hit(["k-pop", "r&b", "romantic"], ["k-pop"])


def test_judge_objective_soft_intent_reports_confidence_and_metrics():
    decision = judge_objective_soft_intent(
        [
            {"moods": ["Energetic"], "scenarios": ["Workout"]},
            {"moods": ["Happy"], "scenarios": ["Driving"]},
        ],
        {
            "positive_any": ["energetic", "workout", "driving"],
            "negative_any": ["sleep"],
            "min_positive_ratio": 0.5,
            "max_negative_ratio": 0.25,
            "min_coverage_ratio": 0.75,
        },
    )

    assert decision.status == "pass"
    assert decision.confidence >= 0.5
    assert decision.metrics["coverage_ratio"] == 1.0
    assert decision.evidence["positive_hits"] == 2


def test_calibration_gold_set_has_high_exact_accuracy():
    gold_path = Path(__file__).resolve().parent.parent / "eval" / "judge_gold" / "objective_soft_judge_gold.json"
    report = calibrate(load_gold(gold_path))

    assert report["total"] >= 20
    assert report["exact_accuracy"] >= 0.95
    assert report["coverage"] >= 0.9
