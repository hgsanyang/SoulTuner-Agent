"""Build entity-disjoint V4 seeds for later strong-teacher annotation.

The output deliberately contains user inputs and provenance only.  It never
creates planner decisions or other teacher labels.

Neo4j is an adapter at the edge: ``build_sealed_seeds`` accepts any candidate
loader with the same signature, so all selection and leakage checks can be
tested without a database.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

Candidate = Mapping[str, Any]
CandidateLoader = Callable[[set[str], set[str], int], Iterable[Candidate]]

_CURRENT_INPUT = re.compile(r"(?:\[当前输入\]|\[current input\])\s*(.*)", re.I | re.S)
_BOOK_TITLE = re.compile(r"《[^》]+》")
_NUMBER = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[\W_]+", re.UNICODE)

_ARTIST_QUERIES = (
    "我第一次听说{artist}，想从最能代表他们创作气质的作品开始认识。",
    "如果只用几首歌介绍{artist}，请按风格层次而不是热度给我安排。",
    "想了解{artist}的音乐脉络，本地没有合适作品时也请选对检索路径。",
    "请从{artist}较有辨识度的作品切入，不要只给最热门的那几首。",
    "我对{artist}还不熟，想听一组能体现其声音变化的入门歌单。",
    "围绕{artist}做一次短歌单探索，兼顾代表作和容易被忽略的作品。",
)

_SONG_QUERIES = (
    "帮我找{artist}的《{song}》，先确认是原版，别混进翻唱或现场录音。",
    "以{artist}的《{song}》为起点，推荐几首创作气质相近但不重复的歌。",
    "我想听《{song}》，演唱者是{artist}；本地没有时请选择合适的外部路径。",
    "先核对{artist}的《{song}》版本，再给我几首适合接着听的作品。",
    "围绕《{song}》做一组延伸推荐，保留{artist}这首歌的核心氛围。",
    "请找到{artist}的《{song}》，并说明应从本地曲库还是联网候选获取。",
)


def canonical_entity(value: Any) -> str:
    """Stable identity used by both the Neo4j adapter and the pure logic."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return _SPACE.sub(" ", text).strip()


def _as_names(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _assistant_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = row.get("teacher_decision_v3")
    if isinstance(direct, Mapping):
        return direct
    for message in row.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        try:
            parsed = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def row_input(row: Mapping[str, Any]) -> str:
    direct = str(row.get("current_query") or "").strip()
    if direct:
        return direct
    for message in row.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        match = _CURRENT_INPUT.search(content)
        return (match.group(1) if match else content).strip()
    return ""


def row_episode_id(row: Mapping[str, Any]) -> str:
    meta = row.get("meta")
    if isinstance(meta, Mapping) and meta.get("episode_id"):
        return str(meta["episode_id"]).strip()
    return str(row.get("episode_id") or "").strip()


def row_entities(row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    decision = _assistant_payload(row)
    hard = decision.get("hard") or decision.get("hard_constraints") or {}
    if not isinstance(hard, Mapping):
        return [], []
    return _as_names(hard.get("artist")), _as_names(hard.get("song"))


def entity_blind_template(
    text: str,
    *,
    artists: Sequence[str] = (),
    songs: Sequence[str] = (),
) -> str:
    """Collapse named entities before comparing query forms."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _BOOK_TITLE.sub(" song ", normalized)
    replacements = [
        *(("artist", name) for name in artists),
        *(("song", name) for name in songs),
    ]
    for marker, name in sorted(replacements, key=lambda item: len(item[1]), reverse=True):
        entity = canonical_entity(name)
        if entity:
            normalized = re.sub(re.escape(entity), f" {marker} ", normalized, flags=re.I)
    normalized = _NUMBER.sub(" number ", normalized)
    return _PUNCTUATION.sub("", normalized)


def char_ngrams(text: str, n: int = 5) -> set[str]:
    normalized = _PUNCTUATION.sub(
        "",
        unicodedata.normalize("NFKC", text).casefold(),
    )
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def ngram_jaccard(left: str, right: str, n: int = 5) -> float:
    a = char_ngrams(left, n)
    b = char_ngrams(right, n)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def training_snapshot(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    artists: set[str] = set()
    songs: set[str] = set()
    inputs: set[str] = set()
    episodes: set[str] = set()
    templates: set[str] = set()
    input_list: list[str] = []

    for row in rows:
        text = row_input(row)
        row_artists, row_songs = row_entities(row)
        artists.update(filter(None, (canonical_entity(name) for name in row_artists)))
        songs.update(filter(None, (canonical_entity(name) for name in row_songs)))
        episode = row_episode_id(row)
        if episode:
            episodes.add(episode)
        if not text:
            continue
        canonical_input = canonical_entity(text)
        inputs.add(canonical_input)
        input_list.append(text)
        templates.add(
            entity_blind_template(text, artists=row_artists, songs=row_songs)
        )

    return {
        "artists": artists,
        "songs": songs,
        "inputs": inputs,
        "episodes": episodes,
        "templates": templates,
        "input_list": input_list,
    }


def load_neo4j_candidates(
    excluded_artists: set[str],
    excluded_songs: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Default candidate adapter. Importing this module never connects to Neo4j."""

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    except ImportError:
        pass
    from retrieval.neo4j_client import get_neo4j_client

    query = """
    MATCH (s:Song)
    OPTIONAL MATCH (s)-[:PERFORMED_BY]->(a:Artist)
    WITH s, coalesce(head(collect(a.name)), s.artist, '') AS artist
    WHERE trim(coalesce(s.title, '')) <> ''
      AND trim(artist) <> ''
      AND NOT (toLower(trim(artist)) IN $excluded_artists)
      AND NOT (toLower(trim(s.title)) IN $excluded_songs)
    RETURN s.music_id AS music_id, s.title AS song, artist
    ORDER BY artist, song
    LIMIT $limit
    """
    rows = get_neo4j_client().execute_query(
        query,
        {
            "excluded_artists": sorted(excluded_artists),
            "excluded_songs": sorted(excluded_songs),
            "limit": max(limit, 1),
        },
    )
    return [dict(row) for row in rows]


def _candidate_identity(candidate: Candidate) -> tuple[str, str]:
    return (
        canonical_entity(candidate.get("artist")),
        canonical_entity(candidate.get("song") or candidate.get("title")),
    )


def _candidate_queries(candidate: Candidate) -> Iterable[tuple[str, str]]:
    artist = str(candidate.get("artist") or "").strip()
    song = str(candidate.get("song") or candidate.get("title") or "").strip()
    if not artist or not song:
        return
    for template in _ARTIST_QUERIES:
        yield "artist", template.format(artist=artist)
    for template in _SONG_QUERIES:
        yield "song", template.format(artist=artist, song=song)


def measure_disjointness(
    train_rows: Sequence[Mapping[str, Any]],
    sealed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    snapshot = training_snapshot(train_rows)
    sealed_episodes: set[str] = set()
    sealed_inputs: set[str] = set()
    sealed_artists: set[str] = set()
    sealed_songs: set[str] = set()
    sealed_templates: set[str] = set()
    max_jaccard = 0.0

    for row in sealed_rows:
        text = row_input(row)
        entity = row.get("entity") or {}
        artist = str(entity.get("artist") or "") if isinstance(entity, Mapping) else ""
        song = str(entity.get("song") or "") if isinstance(entity, Mapping) else ""
        episode = row_episode_id(row)
        if episode:
            sealed_episodes.add(episode)
        if text:
            sealed_inputs.add(canonical_entity(text))
            sealed_templates.add(
                entity_blind_template(text, artists=[artist], songs=[song])
            )
            for train_input in snapshot["input_list"]:
                max_jaccard = max(max_jaccard, ngram_jaccard(text, train_input))
        if artist:
            sealed_artists.add(canonical_entity(artist))
        if song:
            sealed_songs.add(canonical_entity(song))

    return {
        "shared_episodes": len(snapshot["episodes"] & sealed_episodes),
        "shared_inputs": len(snapshot["inputs"] & sealed_inputs),
        "shared_artists": len(snapshot["artists"] & sealed_artists),
        "shared_songs": len(snapshot["songs"] & sealed_songs),
        "shared_templates": len(snapshot["templates"] & sealed_templates),
        "max_near_dupe_jaccard": round(max_jaccard, 6),
    }


def assert_sealed(metrics: Mapping[str, Any], max_jaccard: float = 0.60) -> None:
    for field in (
        "shared_episodes",
        "shared_inputs",
        "shared_artists",
        "shared_songs",
        "shared_templates",
    ):
        if int(metrics.get(field) or 0) != 0:
            raise ValueError(f"sealed split rejected: {field}={metrics[field]}")
    if float(metrics.get("max_near_dupe_jaccard") or 0.0) > max_jaccard:
        raise ValueError(
            "sealed split rejected: "
            f"max_near_dupe_jaccard={metrics['max_near_dupe_jaccard']} > {max_jaccard}"
        )


def build_sealed_seeds(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_loader: CandidateLoader = load_neo4j_candidates,
    target_count: int = 400,
    max_jaccard: float = 0.60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_count < 1:
        raise ValueError("target_count must be positive")
    snapshot = training_snapshot(train_rows)
    candidate_limit = max(math.ceil(target_count / 2) * 8, target_count)
    raw_candidates = candidate_loader(
        set(snapshot["artists"]),
        set(snapshot["songs"]),
        candidate_limit,
    )
    candidates = sorted(
        (dict(candidate) for candidate in raw_candidates),
        key=lambda candidate: (*_candidate_identity(candidate), str(candidate.get("music_id") or "")),
    )

    seeds: list[dict[str, Any]] = []
    seen_entities: set[tuple[str, str]] = set()
    seen_inputs: set[str] = set()
    for candidate in candidates:
        artist_key, song_key = _candidate_identity(candidate)
        if not artist_key or not song_key:
            continue
        if artist_key in snapshot["artists"] or song_key in snapshot["songs"]:
            continue
        identity = (artist_key, song_key)
        if identity in seen_entities:
            continue
        seen_entities.add(identity)

        artist = str(candidate.get("artist") or "").strip()
        song = str(candidate.get("song") or candidate.get("title") or "").strip()
        accepted_for_entity = 0
        for seed_kind, query in _candidate_queries(candidate):
            query_key = canonical_entity(query)
            template = entity_blind_template(query, artists=[artist], songs=[song])
            if query_key in snapshot["inputs"] or query_key in seen_inputs:
                continue
            if template in snapshot["templates"]:
                continue
            if any(
                ngram_jaccard(query, train_input) > max_jaccard
                for train_input in snapshot["input_list"]
            ):
                continue

            index = len(seeds) + 1
            seeds.append(
                {
                    "episode_id": f"sealed_{index:05d}",
                    "turn_id": 0,
                    "current_query": query,
                    "chat_history": "",
                    "previous_plan": "",
                    "profile_snapshot": "",
                    "retrieved_memories": [],
                    "entity": {
                        "artist": artist,
                        "song": song,
                        "music_id": candidate.get("music_id"),
                    },
                    "seed_kind": seed_kind,
                    "annotation_status": "pending_strong_teacher",
                    "provenance": {
                        "seed_source": "neo4j_unseen_entity",
                        "teacher_output_present": False,
                    },
                }
            )
            seen_inputs.add(query_key)
            accepted_for_entity += 1
            if len(seeds) >= target_count or accepted_for_entity >= 2:
                break
        if len(seeds) >= target_count:
            break

    if len(seeds) != target_count:
        raise ValueError(
            f"only {len(seeds)} sealed seeds passed the gate; requested {target_count}"
        )
    metrics = measure_disjointness(train_rows, seeds)
    assert_sealed(metrics, max_jaccard=max_jaccard)
    return seeds, metrics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target-count", type=int, default=400)
    parser.add_argument("--max-jaccard", type=float, default=0.60)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    seeds, metrics = build_sealed_seeds(
        train_rows,
        target_count=args.target_count,
        max_jaccard=args.max_jaccard,
    )
    write_jsonl(args.output, seeds)
    report = {
        "rows": len(seeds),
        "annotation_status": "pending_strong_teacher",
        "teacher_outputs": 0,
        "measured": metrics,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
