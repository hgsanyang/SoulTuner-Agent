from scripts.p11_backfill_catalog_metadata import (
    artist_string,
    backfill_language_relationships,
    backfill_release_year_from_knowledge_cards,
    mark_unplayable_stubs,
    normalize_artist_name,
    normalize_title,
    release_year_from_metadata,
)


def test_artist_string_accepts_netease_pairs():
    assert artist_string([["周传雄", 0], ["小刚", 1]]) == "周传雄、小刚"


def test_normalize_artist_name_keeps_cjk_and_dedup_separator():
    assert normalize_artist_name(" 周杰伦 / Jay Chou ") == "周杰伦、jaychou"


def test_normalize_title_removes_version_noise():
    assert normalize_title("Running Up That Hill (Live Remaster)") == "runningupthathill"


def test_release_year_prefers_explicit_year():
    assert release_year_from_metadata({"release_year": "2000", "publishTime": 1}) == 2000


def test_release_year_can_use_publish_time_ms():
    assert release_year_from_metadata({"publishTime": 977673600000}) == 2000


class _FakeClient:
    def __init__(self):
        self.calls = []

    def execute_query(self, cypher, params=None):
        self.calls.append(cypher)
        if "RETURN count(s) AS n" in cypher or "RETURN count(DISTINCT s) AS n" in cypher:
            return [{"n": 3}]
        return []


def test_backfill_language_relationships_is_dry_run_safe():
    client = _FakeClient()

    assert backfill_language_relationships(client, dry_run=True) == 3
    assert len(client.calls) == 1


def test_backfill_language_relationships_uses_existing_language_property():
    client = _FakeClient()

    assert backfill_language_relationships(client, dry_run=False) == 3
    assert len(client.calls) == 2
    assert "MERGE (lang:Language" in client.calls[1]
    assert "properties(s)['language']" in client.calls[1]


def test_mark_unplayable_stubs_is_dry_run_safe():
    client = _FakeClient()

    assert mark_unplayable_stubs(client, dry_run=True) == 3
    assert len(client.calls) == 1


def test_mark_unplayable_stubs_marks_missing_audio_without_deleting():
    client = _FakeClient()

    assert mark_unplayable_stubs(client, dry_run=False) == 3
    assert len(client.calls) == 2
    assert "SET s.unplayable_stub = true" in client.calls[1]
    assert "DELETE" not in client.calls[1].upper()


def test_backfill_release_year_from_knowledge_cards_is_dry_run_safe():
    client = _FakeClient()

    assert backfill_release_year_from_knowledge_cards(client, dry_run=True) == 3
    assert len(client.calls) == 1
    assert "KnowledgeCard" in client.calls[0]


def test_backfill_release_year_from_knowledge_cards_uses_source_backed_cards():
    client = _FakeClient()

    assert backfill_release_year_from_knowledge_cards(client, dry_run=False) == 3
    assert len(client.calls) == 2
    assert "HAS_KNOWLEDGE" in client.calls[1]
    assert "release_year_source = 'knowledge_card'" in client.calls[1]
