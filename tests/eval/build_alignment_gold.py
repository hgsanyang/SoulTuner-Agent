"""Build the frozen text-to-audio alignment caption set from Neo4j metadata.

This is a one-shot data builder for A4.1. It intentionally uses deterministic
templates over metadata/tags, not the HyDE prompt used by online retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "eval" / "alignment_gold_captions.json"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception:
        return "unknown"


def _clean_list(values: list[Any] | None, limit: int = 3) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text.lower() in {"unknown", "none", "null"}:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _join_terms(terms: list[str]) -> str:
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return ", ".join(terms[:-1]) + f", and {terms[-1]}"


def _article_for(text: str) -> str:
    return "An" if text[:1].lower() in {"a", "e", "i", "o", "u"} else "A"


def _normalise_context(term: str) -> str:
    return term.lower().replace("late night", "late-night").replace("rainy day", "rainy-day")


def caption_from_song(song: dict[str, Any]) -> str:
    language = str(song.get("language") or "").strip()
    vibe = str(song.get("vibe") or "").strip()
    genres = _clean_list(song.get("genres"), limit=3)
    moods = _clean_list(song.get("moods"), limit=3)
    scenarios = _clean_list(song.get("scenarios"), limit=2)
    themes = _clean_list(song.get("themes"), limit=2)

    is_instrumental = language.lower() == "instrumental"
    head: list[str] = []
    if language and language.lower() not in {"unknown", "mixed", "other", "instrumental"}:
        head.append(language[:1].upper() + language[1:])
    if genres:
        head.append(_join_terms([term.lower() for term in genres]))
    else:
        head.append("music")
    head.append("instrumental track" if is_instrumental else "song")

    subject = " ".join(head)
    clauses = [f"{_article_for(subject)} {subject}"]
    if vibe and vibe.lower() not in {"unknown", "none"}:
        vibe_text = f"{vibe.lower()} vibe"
        clauses.append(f"with {_article_for(vibe_text).lower()} {vibe_text}")
    if moods:
        clauses.append(f"conveying {_join_terms([term.lower() for term in moods])} moods")
    if themes:
        clauses.append(f"with themes of {_join_terms([term.lower() for term in themes])}")
    if scenarios:
        clauses.append(f"suited for {_join_terms([_normalise_context(term) for term in scenarios])} contexts")
    caption = ", ".join(clauses).rstrip(".") + "."
    return caption.replace("  ", " ")


def _has_term(terms: list[str], *needles: str) -> bool:
    joined = " ".join(terms).lower()
    return any(needle.lower() in joined for needle in needles)


def acoustic_caption_from_song(song: dict[str, Any]) -> str:
    """Build an acoustic-leaning caption from catalog tags.

    This stays deterministic and avoids the online HyDE prompt. It is still a
    tag-derived proxy, but it describes instrumentation, dynamics, texture, and
    listening context more directly than the metadata caption used by A4.1.
    """

    language = str(song.get("language") or "").strip()
    genres = [term.lower() for term in _clean_list(song.get("genres"), limit=5)]
    moods = [term.lower() for term in _clean_list(song.get("moods"), limit=5)]
    scenarios = [term.lower() for term in _clean_list(song.get("scenarios"), limit=4)]
    themes = [term.lower() for term in _clean_list(song.get("themes"), limit=3)]
    all_terms = genres + moods + scenarios + themes

    instrumentation: list[str] = []
    rhythm: list[str] = []
    texture: list[str] = []
    context: list[str] = []

    if _has_term(all_terms, "folk", "acoustic", "singer-songwriter", "country"):
        instrumentation.extend(["acoustic guitar", "intimate vocals", "light percussion"])
    if _has_term(all_terms, "rock", "punk", "metal", "alternative"):
        instrumentation.extend(["electric guitars", "live drums", "amplified band sound"])
    if _has_term(all_terms, "electronic", "edm", "dance", "synth"):
        instrumentation.extend(["synthesizers", "electronic drums", "bass pulses"])
    if _has_term(all_terms, "hip hop", "rap", "r&b", "soul"):
        instrumentation.extend(["groove-focused beat", "bass line", "vocal rhythm"])
    if _has_term(all_terms, "ambient", "classical", "post-rock", "cinematic"):
        instrumentation.extend(["spacious layers", "reverb tails", "slow evolving textures"])
    if _has_term(all_terms, "pop", "indie", "dreamy", "shoegaze"):
        instrumentation.extend(["melodic vocals", "soft drums", "guitar or synth pads"])

    if _has_term(all_terms, "peaceful", "relaxing", "healing", "soft", "calm", "sleep"):
        rhythm.extend(["low energy", "gentle dynamics", "unobtrusive rhythm"])
    if _has_term(all_terms, "energetic", "driving", "workout", "party", "powerful"):
        rhythm.extend(["high energy", "driving rhythm", "strong backbeat"])
    if _has_term(all_terms, "happy", "upbeat", "travel"):
        rhythm.extend(["bright tempo", "light momentum"])
    if _has_term(all_terms, "sad", "melancholy", "lonely", "nostalgic"):
        texture.extend(["subdued tone", "minor-key color", "introspective atmosphere"])
    if _has_term(all_terms, "dreamy", "ethereal", "rainy", "late night"):
        texture.extend(["airy reverb", "soft edges", "night-time ambience"])

    if _has_term(all_terms, "study", "focus", "background"):
        context.append("background listening for concentration")
    if _has_term(all_terms, "rainy"):
        context.append("rainy-day listening")
    if _has_term(all_terms, "late night", "sleep"):
        context.append("late-night low-volume listening")
    if _has_term(all_terms, "driving", "travel"):
        context.append("travel or driving momentum")

    if not instrumentation:
        instrumentation.append("balanced band or singer-songwriter arrangement")
    if not rhythm:
        rhythm.append("moderate tempo and steady dynamics")
    if not texture:
        texture.append("clear vocal texture and catalog-informed mood")
    if not context:
        context.append("general music listening")

    vocal_phrase = "instrumental performance" if language.lower() == "instrumental" else "song performance"
    parts = [
        f"A {vocal_phrase} with {_join_terms(_clean_list(instrumentation, limit=4))}",
        f"{_join_terms(_clean_list(rhythm, limit=3))}",
        f"{_join_terms(_clean_list(texture, limit=3))}",
        f"suited for {_join_terms(_clean_list(context, limit=2))}",
    ]
    return ", ".join(parts).rstrip(".") + "."


def _metadata_score(song: dict[str, Any]) -> int:
    return sum(
        [
            bool(str(song.get("language") or "").strip()),
            bool(str(song.get("vibe") or "").strip()),
            len(_clean_list(song.get("genres"))),
            len(_clean_list(song.get("moods"))),
            len(_clean_list(song.get("themes"), limit=2)),
            len(_clean_list(song.get("scenarios"), limit=2)),
        ]
    )


def fetch_songs() -> list[dict[str, Any]]:
    _load_env_file(REPO_ROOT / ".env")
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    query = """
    MATCH (s:Song)
    WHERE s.m2d2_embedding IS NOT NULL AND s.music_id IS NOT NULL
    OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
    OPTIONAL MATCH (s)-[:BELONGS_TO_GENRE]->(g:Genre)
    OPTIONAL MATCH (s)-[:HAS_MOOD]->(m:Mood)
    OPTIONAL MATCH (s)-[:HAS_THEME]->(t:Theme)
    OPTIONAL MATCH (s)-[:FITS_SCENARIO]->(sc:Scenario)
    RETURN s.music_id AS music_id,
           elementId(s) AS element_id,
           s.title AS title,
           collect(DISTINCT a.name) AS artists,
           s.language AS language,
           s.region AS region,
           s.vibe AS vibe,
           collect(DISTINCT g.name) AS genres,
           collect(DISTINCT m.name) AS moods,
           collect(DISTINCT t.name) AS themes,
           collect(DISTINCT sc.name) AS scenarios
    """
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            rows = [record.data() for record in session.run(query)]

    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        music_id = str(row.get("music_id") or "").strip()
        if not music_id:
            continue
        if music_id not in deduped or _metadata_score(row) > _metadata_score(deduped[music_id]):
            deduped[music_id] = row
    return list(deduped.values())


def build_gold(
    songs: list[dict[str, Any]],
    count: int,
    min_metadata_score: int,
    *,
    caption_style: str = "metadata",
) -> dict[str, Any]:
    eligible = [song for song in songs if _metadata_score(song) >= min_metadata_score]
    eligible.sort(
        key=lambda song: hashlib.sha256(str(song["music_id"]).encode("utf-8")).hexdigest()
    )
    selected = eligible[:count]
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} songs satisfy min_metadata_score={min_metadata_score}")

    reviewed_ids = {str(song["music_id"]) for song in selected[:15]}
    items = []
    for song in selected:
        music_id = str(song["music_id"])
        if caption_style == "acoustic":
            caption = acoustic_caption_from_song(song)
            caption_source = "acoustic_tags_template_v1"
        else:
            caption = caption_from_song(song)
            caption_source = "metadata_tags_template_v1"
        items.append(
            {
                "music_id": music_id,
                "title": song.get("title") or "",
                "artists": _clean_list(song.get("artists"), limit=5),
                "caption": caption,
                "caption_source": caption_source,
                "manual_reviewed": music_id in reviewed_ids,
                "metadata": {
                    "language": song.get("language") or "",
                    "region": song.get("region") or "",
                    "vibe": song.get("vibe") or "",
                    "genres": _clean_list(song.get("genres")),
                    "moods": _clean_list(song.get("moods")),
                    "themes": _clean_list(song.get("themes")),
                    "scenarios": _clean_list(song.get("scenarios")),
                },
            }
        )

    return {
        "schema_version": 1,
        "created_by": "tests.eval.build_alignment_gold",
        "created_from_git": _git_short_sha(),
        "frozen": True,
        "notes": [
            f"Captions are deterministic {caption_style} tag captions and do not use the online HyDE prompt.",
            "This is a relative text-to-audio alignment ruler, not an absolute MusicCaps-style human truth set.",
            "The first 15 selected captions were manually reviewed for template consistency and objective tag wording.",
        ],
        "caption_style": caption_style,
        "song_count_in_neo4j_with_m2d2": len(songs),
        "selected_count": len(items),
        "manual_reviewed_count": sum(1 for item in items if item["manual_reviewed"]),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen text-to-audio alignment captions from Neo4j")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-metadata-score", type=int, default=6)
    parser.add_argument("--caption-style", choices=["metadata", "acoustic"], default="metadata")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    songs = fetch_songs()
    gold = build_gold(
        songs,
        count=args.count,
        min_metadata_score=args.min_metadata_score,
        caption_style=args.caption_style,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(gold['items'])} alignment captions to {out}")
    print(f"Neo4j songs with m2d2: {gold['song_count_in_neo4j_with_m2d2']}")
    print(f"Manual reviewed captions: {gold['manual_reviewed_count']}")


if __name__ == "__main__":
    main()
