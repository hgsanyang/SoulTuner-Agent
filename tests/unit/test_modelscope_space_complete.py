from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SPACE = Path(__file__).resolve().parents[2] / "deploy" / "modelscope_space"
sys.path.insert(0, str(SPACE))

RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "space_planner_runtime", SPACE / "planner_runtime.py"
)
assert RUNTIME_SPEC and RUNTIME_SPEC.loader
runtime = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(runtime)

RETRIEVAL_SPEC = importlib.util.spec_from_file_location(
    "space_retrieval_demo", SPACE / "retrieval_demo.py"
)
assert RETRIEVAL_SPEC and RETRIEVAL_SPEC.loader
retrieval = importlib.util.module_from_spec(RETRIEVAL_SPEC)
RETRIEVAL_SPEC.loader.exec_module(retrieval)


def test_space_card_text_has_theme_independent_contrast() -> None:
    source = (SPACE / "app.py").read_text(encoding="utf-8")
    assert ".st-card h3" in source
    assert "color: #102a20 !important" in source
    assert "color: #315647 !important" in source


def test_public_catalog_and_hybrid_retrieval() -> None:
    assert len(retrieval.load_catalog()) == 120
    query = "90年代英文摇滚，但整体要温暖一点"
    plan = runtime.safe_plan(query)
    route = runtime.compile_route(plan)
    rows = retrieval.retrieve(query, plan, route, top_k=5)
    assert len(rows) == 5
    assert route["graph_weight"] > 0
    assert route["dense_weight"] > 0


def test_subjective_acoustics_are_dense_only() -> None:
    plan = runtime.safe_plan("我希望 bass 更重、鼓声更大一些")
    assert plan["lane_policy"] == {"graph": "off", "dense": "required", "web": "off"}
