"""Identity for joining recalled candidates to catalog-derived metadata."""
import json
from typing import Any, Mapping


def candidate_identity(song: Mapping[str, Any]) -> dict[str, str]:
    music_id = str(song.get("music_id") or "")
    title = str(song.get("title") or "")
    artist = str(song.get("artist") or "")
    key = json.dumps(["id", music_id] if music_id else ["pair", title, artist],
                     ensure_ascii=False, separators=(",", ":"))
    return {"key": key, "music_id": music_id, "title": title, "artist": artist}


# Ambiguous matches are omitted instead of attaching another recording's data.
IDENTITY_MATCH = """
UNWIND $identities AS identity
MATCH (s:Song)
WHERE (identity.music_id <> '' AND toString(s.music_id) = identity.music_id)
   OR (identity.music_id = '' AND identity.title <> '' AND identity.artist <> ''
       AND s.title = identity.title
       AND (s.artist = identity.artist OR EXISTS {
           MATCH (s)-[:PERFORMED_BY]->(a:Artist {name: identity.artist})
       }))
WITH identity, collect(DISTINCT s) AS matches
WHERE size(matches) = 1
WITH identity, matches[0] AS s
"""
