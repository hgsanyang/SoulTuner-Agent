from __future__ import annotations

import pytest

from data.sft.build_v4_sealed_seeds import (
    assert_sealed,
    build_sealed_seeds,
    entity_blind_template,
    measure_disjointness,
    ngram_jaccard,
    training_snapshot,
)


def _train_row(
    episode_id: str,
    query: str,
    *,
    artist: str = "",
    song: str = "",
) -> dict:
    return {
        "episode_id": episode_id,
        "turn_id": 0,
        "current_query": query,
        "teacher_decision_v3": {
            "hard": {
                "artist": [artist] if artist else [],
                "song": [song] if song else [],
            }
        },
    }


def _seed(
    episode_id: str,
    query: str,
    *,
    artist: str,
    song: str,
) -> dict:
    return {
        "episode_id": episode_id,
        "turn_id": 0,
        "current_query": query,
        "entity": {"artist": artist, "song": song},
    }


def test_build_uses_injected_loader_and_never_creates_teacher_output():
    train = [
        _train_row(
            "train_1",
            "给我推荐周杰伦的《晴天》",
            artist="周杰伦",
            song="晴天",
        )
    ]
    captured: dict = {}

    def fake_loader(excluded_artists, excluded_songs, limit):
        captured.update(
            artists=excluded_artists,
            songs=excluded_songs,
            limit=limit,
        )
        return [
            {"music_id": "seen-artist", "artist": "周杰伦", "song": "新歌"},
            {"music_id": "seen-song", "artist": "Unknown Band", "song": "晴天"},
            {"music_id": "new", "artist": "Pink Floyd", "song": "Echoes"},
        ]

    rows, metrics = build_sealed_seeds(
        train,
        candidate_loader=fake_loader,
        target_count=2,
    )

    assert captured["artists"] == {"周杰伦"}
    assert captured["songs"] == {"晴天"}
    assert captured["limit"] >= 2
    assert len(rows) == 2
    assert all(row["episode_id"].startswith("sealed_") for row in rows)
    assert all(row["entity"]["artist"] == "Pink Floyd" for row in rows)
    assert all(row["annotation_status"] == "pending_strong_teacher" for row in rows)
    assert all(row["provenance"]["teacher_output_present"] is False for row in rows)
    assert all("teacher_decision_v3" not in row for row in rows)
    assert all("messages" not in row for row in rows)
    assert metrics["shared_artists"] == 0
    assert metrics["shared_songs"] == 0
    assert metrics["shared_templates"] == 0
    assert metrics["shared_inputs"] == 0


def test_chatml_training_rows_are_understood_without_infrastructure():
    row = {
        "messages": [
            {
                "role": "user",
                "content": "[画像] demo\n[当前输入] 请播放 The Cure 的《Lovesong》",
            },
            {
                "role": "assistant",
                "content": (
                    '{"request_kind":"recommendation","response_mode":"answer",'
                    '"tool_names":["graph"],'
                    '"hard":{"artist":["The Cure"],"song":["Lovesong"]}}'
                ),
            },
        ],
        "meta": {"episode_id": "train_chatml", "turn_id": 0},
    }
    snapshot = training_snapshot([row])

    assert snapshot["artists"] == {"the cure"}
    assert snapshot["songs"] == {"lovesong"}
    assert snapshot["episodes"] == {"train_chatml"}
    assert snapshot["inputs"] == {"请播放 the cure 的《lovesong》"}


def test_entity_blind_template_detects_same_form_with_different_entities():
    train = [
        _train_row(
            "train_1",
            "我第一次听说周杰伦，想从最能代表他们创作气质的作品开始认识。",
            artist="周杰伦",
        )
    ]
    sealed = [
        _seed(
            "sealed_00001",
            "我第一次听说Pink Floyd，想从最能代表他们创作气质的作品开始认识。",
            artist="Pink Floyd",
            song="Echoes",
        )
    ]

    metrics = measure_disjointness(train, sealed)

    assert metrics["shared_artists"] == 0
    assert metrics["shared_inputs"] == 0
    assert metrics["shared_templates"] == 1
    with pytest.raises(ValueError, match="shared_templates"):
        assert_sealed(metrics)


@pytest.mark.parametrize(
    ("field", "sealed"),
    [
        (
            "shared_episodes",
            _seed("train_1", "完全不同的请求", artist="New Artist", song="New Song"),
        ),
        (
            "shared_artists",
            _seed("sealed_1", "换一种说法", artist="周杰伦", song="New Song"),
        ),
        (
            "shared_songs",
            _seed("sealed_1", "另一种说法", artist="New Artist", song="晴天"),
        ),
        (
            "shared_inputs",
            _seed(
                "sealed_1",
                "给我推荐周杰伦的《晴天》",
                artist="New Artist",
                song="New Song",
            ),
        ),
    ],
)
def test_each_overlap_dimension_is_reported(field, sealed):
    train = [
        _train_row(
            "train_1",
            "给我推荐周杰伦的《晴天》",
            artist="周杰伦",
            song="晴天",
        )
    ]
    metrics = measure_disjointness(train, [sealed])

    assert metrics[field] == 1


def test_five_gram_jaccard_is_measured_and_gated():
    train = [_train_row("train_1", "请给我一些安静柔和的雨天音乐")]
    sealed = [
        _seed(
            "sealed_1",
            "请给我一些安静柔和的雨天音乐",
            artist="New Artist",
            song="New Song",
        )
    ]

    metrics = measure_disjointness(train, sealed)

    assert ngram_jaccard("abcdef", "abcdef") == 1.0
    assert metrics["max_near_dupe_jaccard"] == 1.0
    jaccard_only = {
        **metrics,
        "shared_inputs": 0,
        "shared_templates": 0,
    }
    with pytest.raises(ValueError, match="max_near_dupe_jaccard"):
        assert_sealed(jaccard_only, max_jaccard=0.60)


def test_generator_rejects_when_not_enough_candidates_survive():
    train = [_train_row("train_1", "播放周杰伦的歌", artist="周杰伦")]

    def only_seen(_artists, _songs, _limit):
        return [{"music_id": "1", "artist": "周杰伦", "song": "新歌"}]

    with pytest.raises(ValueError, match="only 0 sealed seeds"):
        build_sealed_seeds(
            train,
            candidate_loader=only_seen,
            target_count=1,
        )


def test_template_normalization_masks_artist_and_song_names():
    left = entity_blind_template(
        "播放 The Cure 的《Lovesong》",
        artists=["The Cure"],
        songs=["Lovesong"],
    )
    right = entity_blind_template(
        "播放 椅子乐团 的《巴黎德州》",
        artists=["椅子乐团"],
        songs=["巴黎德州"],
    )

    assert left == right
