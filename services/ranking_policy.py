"""Versioned ranking-policy storage, loading, promotion, and rollback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
from typing import Any


ACTIVE_FILE = "ranking_policy.json"
CANDIDATE_FILE = "ranking_policy.candidate.json"
PREVIOUS_FILE = "ranking_policy.previous.json"
_CACHE: dict[str, Any] = {"mtime": None, "loaded_at": 0.0, "payload": None}


def feedback_dir() -> Path:
    root = os.getenv("MUSIC_FEEDBACK_DIR")
    path = Path(root) if root else Path("data") / "feedback"
    path.mkdir(parents=True, exist_ok=True)
    return path


def policy_path(name: str = ACTIVE_FILE, directory: Path | None = None) -> Path:
    root = directory or feedback_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_candidate(payload: dict[str, Any], directory: Path | None = None) -> Path:
    path = policy_path(CANDIDATE_FILE, directory)
    _atomic_write(path, payload)
    return path


def promote_candidate(directory: Path | None = None) -> Path:
    candidate = policy_path(CANDIDATE_FILE, directory)
    if not candidate.exists():
        raise FileNotFoundError(f"Ranking candidate does not exist: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or not payload.get("gate_passed"):
        raise ValueError("Ranking candidate did not pass the offline validation gate")

    active = policy_path(ACTIVE_FILE, directory)
    previous = policy_path(PREVIOUS_FILE, directory)
    if active.exists():
        shutil.copy2(active, previous)
    promoted = dict(payload)
    promoted["status"] = "active"
    promoted["promoted_at_unix_ms"] = int(time.time() * 1000)
    _atomic_write(active, promoted)
    _CACHE.update({"mtime": None, "loaded_at": 0.0, "payload": None})
    return active


def rollback_policy(directory: Path | None = None) -> Path:
    previous = policy_path(PREVIOUS_FILE, directory)
    if not previous.exists():
        raise FileNotFoundError(f"Previous ranking policy does not exist: {previous}")
    active = policy_path(ACTIVE_FILE, directory)
    shutil.copy2(previous, active)
    _CACHE.update({"mtime": None, "loaded_at": 0.0, "payload": None})
    return active


def _validated_active_payload() -> dict[str, Any] | None:
    active = policy_path(ACTIVE_FILE)
    if not active.exists():
        return None
    mtime = active.stat().st_mtime_ns
    if _CACHE["mtime"] == mtime and _CACHE["payload"] is not None:
        return _CACHE["payload"]
    try:
        payload = json.loads(active.read_text(encoding="utf-8"))
        valid = (
            payload.get("schema_version") == 2
            and payload.get("status") == "active"
            and payload.get("gate_passed") is True
            and (payload.get("global") or {}).get("status") == "accepted"
        )
        if not valid:
            return None
    except Exception:
        return None
    _CACHE.update({"mtime": mtime, "loaded_at": time.monotonic(), "payload": payload})
    return payload


def runtime_policy_for_user(user_id: str = "local_admin") -> dict[str, Any] | None:
    payload = _validated_active_payload()
    if payload is None:
        return None
    user_model = (payload.get("users") or {}).get(user_id) or {}
    if user_model.get("status") == "accepted" and user_model.get("runtime_policy"):
        return user_model["runtime_policy"]
    return (payload.get("global") or {}).get("runtime_policy")


def apply_multipliers(
    base: dict[str, float],
    multipliers: dict[str, float] | None,
    *,
    normalise: bool,
) -> dict[str, float]:
    adjusted = {
        key: max(0.0, float(value) * float((multipliers or {}).get(key, 1.0)))
        for key, value in base.items()
    }
    if normalise:
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {key: value / total for key, value in adjusted.items()}
    return adjusted


def summarize_policy_readiness(
    *,
    num_exposures: int,
    num_events: int,
    num_slate_feedback: int,
    active: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
    min_events: int = 20,
) -> dict[str, Any]:
    """Summarize the next safe A3 action without reading private log contents."""
    labeled_signals = max(0, int(num_events)) + max(0, int(num_slate_feedback))
    min_events = max(1, int(min_events))
    active_ok = bool(
        active
        and active.get("status") == "active"
        and active.get("gate_passed") is True
        and active.get("global_status") == "accepted"
    )
    candidate_ok = bool(
        candidate
        and candidate.get("gate_passed") is True
        and candidate.get("global_status") == "accepted"
    )

    if active_ok:
        stage = "active_policy"
        next_action = "继续正常使用并收集反馈；如效果变差可 rollback 到上一版策略。"
    elif candidate_ok:
        stage = "candidate_ready"
        next_action = "候选排序策略已通过离线闸门，可以人工检查后 promote。"
    elif labeled_signals >= min_events and int(num_exposures) > 0:
        stage = "replay_ready"
        next_action = "反馈量已达到最小 replay 门槛，可以运行 ranking-policy replay 生成候选。"
    else:
        stage = "collect_feedback"
        remaining = max(0, min_events - labeled_signals)
        next_action = f"继续正常使用并提交点赞/跳过/拉黑/歌单级反馈；至少还需要约 {remaining} 条标注信号。"

    warnings: list[str] = []
    if int(num_exposures) == 0:
        warnings.append("尚无曝光日志，无法把反馈归因到具体推荐列表。")
    if int(num_events) == 0 and int(num_slate_feedback) == 0:
        warnings.append("尚无显式反馈，A3 不会生成伪策略。")
    if candidate and not candidate_ok:
        warnings.append("存在候选策略但未通过 gate，不能 promote。")

    return {
        "stage": stage,
        "next_action": next_action,
        "can_replay": stage in {"replay_ready", "candidate_ready", "active_policy"},
        "can_promote": candidate_ok,
        "has_active_policy": active_ok,
        "labeled_signals": labeled_signals,
        "min_events": min_events,
        "warnings": warnings,
    }
