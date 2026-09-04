from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPACE = ROOT / "deploy" / "modelscope_space"


def test_modelscope_artifact_revisions_match_deployment_lock() -> None:
    lock = json.loads((SPACE / "deployment.lock.json").read_text(encoding="utf-8"))
    launcher = (SPACE / "start_amd_35b.sh").read_text(encoding="utf-8")
    audio_launcher = (SPACE / "start_space_amd.sh").read_text(encoding="utf-8")
    enrichment = (SPACE / "enrichment_runtime.py").read_text(encoding="utf-8")

    assert lock["base_model"]["revision"] in launcher
    assert lock["planner_adapter"]["revision"] in launcher
    assert lock["open_audio"]["revision"] in audio_launcher
    assert lock["source"]["revision"] in enrichment
    assert lock["retrieval"] == {
        "gpu_primary": "muq",
        "gpu_acoustic_reranker": "omar",
        "cpu_backend": "m2d",
    }
