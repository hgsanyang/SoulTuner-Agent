from __future__ import annotations

import hashlib
import json

from tools.data.merge_public_audio_bundles import merge


def _bundle(root, song_id: str, dataset: str | None) -> None:
    audio = root / "audio" / f"{song_id}.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(song_id.encode())
    row = {
        "song_id": song_id,
        "title": song_id,
        "artist": "Open Artist",
        "genres": ["Rock"],
        "audio_relpath": audio.name,
        "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
    }
    if dataset:
        row["dataset"] = dataset
    (root / "catalog.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_merge_namespaces_datasets_and_verifies_audio(tmp_path) -> None:
    base = tmp_path / "base"
    expansion = tmp_path / "expansion"
    output = tmp_path / "combined"
    _bundle(base, "sdd-1", None)
    _bundle(expansion, "fma-000002", "fma_small_balanced")

    audit = merge(base, expansion, output)

    assert audit["tracks"] == 2
    assert audit["datasets"] == {
        "fma_small_balanced": 1,
        "song_describer_full": 1,
    }
    rows = [json.loads(line) for line in (output / "catalog.jsonl").read_text().splitlines()]
    assert {row["song_id"] for row in rows} == {"sdd-1", "fma-000002"}
    assert (output / "audio" / "sdd-1.mp3").read_bytes() == b"sdd-1"
