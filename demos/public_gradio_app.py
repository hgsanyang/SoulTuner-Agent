"""Public CC-only Gradio demo for SoulTuner.

This demo intentionally avoids the private catalog and online ingestion routes.
It reads MTG-Jamendo metadata/audio from a local CC sample directory and returns
deterministic tag-matched recommendations.  It is meant for HF Space /
ModelScope exposure, not as the full local product.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_DATA_DIR = PROJECT_ROOT.parent / "data" / "mtg_sample"

QUERY_TAGS = {
    "rain": ["Rainy", "Relaxing", "Mellow", "Soft"],
    "雨": ["Rainy", "Relaxing", "Mellow", "Soft"],
    "lofi": ["Lo-Fi", "Chill", "Relaxing", "Soft"],
    "lo-fi": ["Lo-Fi", "Chill", "Relaxing", "Soft"],
    "安静": ["Calm", "Peaceful", "Soft", "Relaxing"],
    "柔": ["Soft", "Tender", "Mellow", "Relaxing"],
    "睡": ["Calm", "Peaceful", "Relaxing", "Ambient"],
    "学习": ["Background Music", "Calm", "Meditative", "Relaxing"],
    "专注": ["Background Music", "Calm", "Meditative", "Relaxing"],
    "跑步": ["Energetic", "Workout", "Sport", "Powerful"],
    "健身": ["Energetic", "Workout", "Sport", "Powerful"],
    "燃": ["Energetic", "Powerful", "Action", "Epic"],
    "happy": ["Happy", "Fun", "Upbeat", "Positive"],
    "开心": ["Happy", "Fun", "Upbeat", "Positive"],
    "sad": ["Sad", "Melancholy", "Emotional", "Sentimental"],
    "难过": ["Sad", "Melancholy", "Emotional", "Sentimental"],
    "复古": ["Retro", "Classic", "Nostalgic"],
    "电影": ["Film", "Cinematic", "Drama", "Epic"],
    "travel": ["Travel", "Adventure", "Journey"],
    "旅行": ["Travel", "Adventure", "Journey"],
}


@dataclass(frozen=True)
class DemoTrack:
    title: str
    artist: str
    audio_path: Path
    moods: tuple[str, ...]
    themes: tuple[str, ...]
    scenarios: tuple[str, ...]
    source: str

    @property
    def tags(self) -> tuple[str, ...]:
        return self.moods + self.themes + self.scenarios


def demo_data_dir() -> Path:
    configured = os.getenv("PUBLIC_DEMO_DATA_DIR", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_DEMO_DATA_DIR


def _as_tuple(values: Any) -> tuple[str, ...]:
    if isinstance(values, list):
        return tuple(str(item) for item in values if item)
    return (str(values),) if values else ()


def load_demo_tracks(root: Path | None = None, limit: int = 4000) -> list[DemoTrack]:
    root = root or demo_data_dir()
    metadata_dir = root / "metadata"
    audio_dir = root / "audio"
    tracks: list[DemoTrack] = []
    if not metadata_dir.exists() or not audio_dir.exists():
        return tracks
    for meta_path in sorted(metadata_dir.glob("*_meta.json")):
        if len(tracks) >= limit:
            break
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        base = meta_path.name.removesuffix("_meta.json")
        audio_path = audio_dir / f"{base}.mp3"
        if not audio_path.exists():
            continue
        tracks.append(
            DemoTrack(
                title=str(meta.get("musicName") or base),
                artist=str((meta.get("artist") or [["MTG-Jamendo Artist"]])[0][0]),
                audio_path=audio_path,
                moods=_as_tuple(meta.get("moods")),
                themes=_as_tuple(meta.get("themes")),
                scenarios=_as_tuple(meta.get("scenarios")),
                source=str(meta.get("source") or "mtg"),
            )
        )
    return tracks


def query_terms(query: str) -> list[str]:
    normalized = str(query or "").casefold()
    terms = [term for term in QUERY_TAGS if term in normalized]
    terms.extend(token for token in re.split(r"[\s,，。.!?;；]+", normalized) if len(token) >= 3)
    return list(dict.fromkeys(terms))


def score_track(track: DemoTrack, query: str) -> tuple[float, list[str]]:
    terms = query_terms(query)
    tags = [tag.casefold() for tag in track.tags]
    title_artist = f"{track.title} {track.artist}".casefold()
    score = 0.0
    reasons: list[str] = []
    for term in terms:
        mapped = QUERY_TAGS.get(term, [term])
        for wanted in mapped:
            wanted_norm = str(wanted).casefold()
            if any(wanted_norm in tag or tag in wanted_norm for tag in tags):
                score += 2.0
                reasons.append(wanted)
                break
        if term in title_artist:
            score += 1.0
            reasons.append(term)
    if not reasons and tags:
        score += 0.1
        reasons.append(tags[0])
    return score, list(dict.fromkeys(reasons))[:4]


def recommend_demo(query: str, tracks: list[DemoTrack], limit: int = 5) -> list[tuple[DemoTrack, float, list[str]]]:
    scored = [(track, *score_track(track, query)) for track in tracks]
    scored.sort(key=lambda row: (-row[1], row[0].title.casefold(), row[0].artist.casefold()))
    return [row for row in scored[: max(1, limit)] if row[1] > 0]


def format_recommendations(query: str, tracks: list[DemoTrack], limit: int = 5) -> tuple[str, list[str]]:
    rows = recommend_demo(query, tracks, limit=limit)
    if not rows:
        return "没有找到可展示的 CC 曲目。请确认 PUBLIC_DEMO_DATA_DIR 指向 MTG-Jamendo 样本。", []
    lines = [
        "这是公开 demo 的 CC-only 推荐结果，只使用 MTG-Jamendo 样本，不读取私人曲库。",
        "",
    ]
    audio_paths: list[str] = []
    for index, (track, score, reasons) in enumerate(rows, start=1):
        tag_text = ", ".join(reasons) if reasons else "catalog match"
        lines.append(f"{index}. **{track.title}** - {track.artist}")
        lines.append(f"   匹配信号：{tag_text} | source={track.source} | score={score:.1f}")
        audio_paths.append(str(track.audio_path))
    return "\n".join(lines), audio_paths


def build_app():
    import gradio as gr

    tracks = load_demo_tracks()

    def _respond(message: str, history: list[tuple[str, str]] | None = None):
        text, audio_paths = format_recommendations(message, tracks, limit=5)
        first_audio = audio_paths[0] if audio_paths else None
        second_audio = audio_paths[1] if len(audio_paths) > 1 else None
        third_audio = audio_paths[2] if len(audio_paths) > 2 else None
        return text, first_audio, second_audio, third_audio

    with gr.Blocks(title="SoulTuner Public Demo") as demo:
        gr.Markdown("# SoulTuner Public Demo\nCC-only MTG-Jamendo catalog. Public demo mode: no download, no ingestion, no private library.")
        query = gr.Textbox(label="Describe your listening moment", placeholder="雨天通勤，柔软一点，不要太吵")
        button = gr.Button("Recommend")
        output = gr.Markdown()
        with gr.Row():
            audio1 = gr.Audio(label="Track 1", interactive=False)
            audio2 = gr.Audio(label="Track 2", interactive=False)
            audio3 = gr.Audio(label="Track 3", interactive=False)
        button.click(_respond, inputs=[query], outputs=[output, audio1, audio2, audio3])
        query.submit(_respond, inputs=[query], outputs=[output, audio1, audio2, audio3])
    return demo


if __name__ == "__main__":
    os.environ.setdefault("PUBLIC_DEMO_MODE", "1")
    build_app().launch()
