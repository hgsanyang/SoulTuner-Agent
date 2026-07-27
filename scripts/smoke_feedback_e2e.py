"""End-to-end smoke: real recommendation -> exposure -> per-song + slate feedback.

Exercises the whole chain against a RUNNING backend, which the unit/integration
tests cannot: SSE streaming, the provisional exposure written before the songs
are sent, server-side backfill from the exposure record, and both feedback
endpoints landing in the canonical SQLite store.

⚠ Requires the backend image to contain the current code. If you are running the
docker container, rebuild it first — otherwise this smokes the OLD build and the
new guards will look like they are missing.

Usage:
    python -m scripts.smoke_feedback_e2e --base http://localhost:8501 \
        --query "深夜写代码，想要安静一点的"
Exit code 0 = every step passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _fail(step: str, detail: str) -> None:
    print(f"[FAIL] {step}: {detail}")
    raise SystemExit(1)


def stream_recommendation(base: str, query: str, timeout: float) -> tuple[str, list[dict]]:
    """Drive the SSE endpoint and collect exposure_id + songs."""
    exposure_id, songs = "", []
    payload = {
        "query": query,
        "user_id": "smoke_admin",
        "timezone": "Asia/Shanghai",
        "session_id": "smoke-session",
        "scene": "端到端冒烟",
        "device": "smoke",
    }
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        with client.stream("POST", f"{base}/api/recommendations/stream", json=payload) as resp:
            if resp.status_code != 200:
                _fail("stream", f"HTTP {resp.status_code}")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except Exception:
                    continue
                if event.get("exposure_id") and not exposure_id:
                    exposure_id = str(event["exposure_id"])
                if event.get("type") == "song" and isinstance(event.get("song"), dict):
                    songs.append(event["song"])
                if event.get("type") in {"complete", "done", "error"}:
                    break
    return exposure_id, songs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8501")
    parser.add_argument("--query", default="深夜写代码，想要安静一点的")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    print(f"== 1. 流式推荐 ({args.query!r}) ==")
    exposure_id, songs = stream_recommendation(args.base, args.query, args.timeout)
    if not exposure_id:
        _fail("exposure_id", "SSE 未返回 exposure_id")
    if not songs:
        _fail("songs", "没有收到歌曲")
    print(f"   exposure_id={exposure_id} songs={len(songs)}")

    print("== 2. 曝光应在发歌时就已落盘（provisional 先写） ==")
    from services.feedback_logger import lookup_exposure

    exposure = None
    for _ in range(10):
        exposure = lookup_exposure(exposure_id)
        if exposure:
            break
        time.sleep(0.5)
    if not exposure:
        _fail("exposure", "曝光未落盘（#4 预写未生效？）")
    ctx = exposure.get("context") or {}
    print(f"   provisional={exposure.get('provisional')} items={len(exposure.get('items') or [])} "
          f"local_hour={ctx.get('local_hour')} day_type={ctx.get('day_type')} scene={ctx.get('scene')!r}")
    if ctx.get("local_hour") is None:
        _fail("context", "曝光缺少收听上下文（#2 透传未生效？）")

    first = (exposure.get("items") or [])[0]
    music_id = str(first.get("music_id") or "")

    # Counted BEFORE the feedback calls: asserting "there is at least one row"
    # is what let a bug through where every slate feedback overwrote the previous
    # one, leaving the table permanently at exactly 1. Only a delta catches that.
    from services import feedback_store

    before = feedback_store.counts()

    print("== 3. 逐首语境反馈（策略字段必须由服务端回填） ==")
    with httpx.Client(timeout=30, trust_env=False) as client:
        resp = client.post(f"{args.base}/api/song-feedback", json={
            "exposure_id": exposure_id, "music_id": music_id,
            "title": first.get("title", ""), "artist": first.get("artist", ""),
            "context_fit": "off", "off_reasons": ["too_loud"], "note": "端到端冒烟",
            "timezone": "Asia/Shanghai", "session_id": "smoke-session", "scene": "端到端冒烟",
        })
    if resp.status_code != 200:
        _fail("song-feedback", f"HTTP {resp.status_code}: {resp.text[:200]}")
    print(f"   ok id={resp.json().get('song_feedback_id')}")

    print("== 4. 歌单反馈（best/worst 必须属于本组） ==")
    ids = [str(i.get("music_id")) for i in (exposure.get("items") or []) if i.get("music_id")]
    with httpx.Client(timeout=30, trust_env=False) as client:
        resp = client.post(f"{args.base}/api/slate-feedback", json={
            "exposure_id": exposure_id, "rating": "partial",
            "best_music_ids": ids[1:2], "worst_music_ids": ids[:1],
            "note": "端到端冒烟", "timezone": "Asia/Shanghai",
        })
    if resp.status_code != 200:
        _fail("slate-feedback", f"HTTP {resp.status_code}: {resp.text[:200]}")
    print("   ok")

    print("== 5. 正式存储（SQLite）应有记录 ==")
    counts = feedback_store.counts()
    print(f"   {before} -> {counts}")
    if feedback_store.get_exposure(exposure_id) is None:
        _fail("store", "SQLite 中没有该曝光（#5 镜像未生效？）")
    for table in ("song_feedback", "slate_feedback"):
        grew = counts.get(table, 0) - before.get(table, 0)
        if grew < 1:
            _fail("store", f"{table} 没有新增行（写入被覆盖或主键冲突？）")

    print("\n== 冒烟全部通过 ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
