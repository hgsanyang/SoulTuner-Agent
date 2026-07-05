from services.tag_policy import MAX_TAGS_PER_FIELD, clean_tag_payload, clean_tag_values


def test_clean_tag_values_dedupes_caps_and_ignores_unknowns():
    values = [
        " Indie ",
        "indie",
        "Unknown",
        "未知",
        "Folk",
        "Rock",
        "Pop",
        "Dream Pop",
        "Electronic",
    ]

    cleaned = clean_tag_values(values)

    assert cleaned == ["Indie", "Folk", "Rock", "Pop", "Dream Pop"]
    assert len(cleaned) == MAX_TAGS_PER_FIELD


def test_clean_tag_values_does_not_force_minimum_count():
    assert clean_tag_values(["Healing"]) == ["Healing"]
    assert clean_tag_values([]) == []


def test_clean_tag_payload_only_returns_supported_fields():
    cleaned = clean_tag_payload(
        {
            "genres": ["Folk", "Folk"],
            "moods": ["Soft", "unknown"],
            "themes": [],
            "scenarios": ["Rainy Day"],
            "tempo": ["120bpm"],
        }
    )

    assert cleaned == {
        "genres": ["Folk"],
        "moods": ["Soft"],
        "themes": [],
        "scenarios": ["Rainy Day"],
    }

