"""Live Neo4j smoke test for profile and developer-sandbox isolation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.eval.runtime_fingerprint import capture_runtime_fingerprint, fingerprint_changes


async def verify() -> dict:
    from config.settings import settings
    from retrieval.neo4j_client import get_neo4j_client
    from services.memory_gateway import MemoryGateway, Neo4jPreferenceAdapter

    settings.eval_disable_side_effects = False
    client = get_neo4j_client()
    songs = client.execute_query(
        "MATCH (s:Song) WHERE coalesce(s.title, '') <> '' "
        "RETURN s.title AS title, coalesce(s.artist, '') AS artist ORDER BY s.title LIMIT 2"
    )
    if len(songs) < 2:
        raise RuntimeError("Need at least two songs for the isolation smoke test")

    suffix = uuid.uuid4().hex[:10]
    profile_a = f"__isolation_profile_a_{suffix}"
    profile_b = f"__isolation_profile_b_{suffix}"
    developer_a = f"__dev__:{profile_a}"
    developer_b = f"__dev__:{profile_b}"
    users = [profile_a, profile_b, developer_a, developer_b]
    before = capture_runtime_fingerprint()
    gateway = MemoryGateway(
        primary=Neo4jPreferenceAdapter(),
        episodic_adapters=[],
        enable_graphzep_sidecar=False,
    )

    try:
        await gateway.remember_event(
            event_type="like",
            title=songs[0]["title"],
            artist=songs[0]["artist"],
            user_id=profile_a,
        )
        await gateway.remember_event(
            event_type="dislike",
            title=songs[1]["title"],
            artist=songs[1]["artist"],
            user_id=profile_b,
        )
        await gateway.remember_event(
            event_type="dislike",
            title=songs[0]["title"],
            artist=songs[0]["artist"],
            user_id=developer_a,
        )
        await gateway.remember_event(
            event_type="like",
            title=songs[1]["title"],
            artist=songs[1]["artist"],
            user_id=developer_b,
        )
        rows = client.execute_query(
            "MATCH (u:User) WHERE u.id IN $users "
            "OPTIONAL MATCH (u)-[r]->(s:Song) "
            "RETURN u.id AS user_id, type(r) AS relation, s.title AS title "
            "ORDER BY user_id, relation, title",
            {"users": users},
        )
        by_user = {user_id: [] for user_id in users}
        for row in rows:
            if row.get("relation"):
                by_user.setdefault(row["user_id"], []).append(
                    {"relation": row["relation"], "title": row.get("title")}
                )
        expected = {
            profile_a: {("LIKES", songs[0]["title"])},
            profile_b: {("DISLIKES", songs[1]["title"])},
            developer_a: {("DISLIKES", songs[0]["title"])},
            developer_b: {("LIKES", songs[1]["title"])},
        }
        actual = {
            user_id: {(row["relation"], row["title"]) for row in by_user[user_id]}
            for user_id in users
        }
        isolated = actual == expected
    finally:
        client.execute_query(
            "MATCH (u:User) WHERE u.id IN $users DETACH DELETE u",
            {"users": users},
        )

    after = capture_runtime_fingerprint()
    changes = fingerprint_changes(before, after)
    return {
        "isolated": isolated,
        "personal_and_developer_spaces_isolated": isolated,
        "restored_after_cleanup": not changes,
        "state_changes_after_cleanup": changes,
        "relations": by_user,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Verify live per-user memory isolation")
    parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="soultuner-phase0-feedback-") as temp_dir:
        os.environ["MUSIC_FEEDBACK_DIR"] = str(Path(temp_dir))
        result = asyncio.run(verify())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not all(
        result[key]
        for key in (
            "isolated",
            "personal_and_developer_spaces_isolated",
            "restored_after_cleanup",
        )
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
