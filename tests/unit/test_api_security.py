from pathlib import Path

import pytest

from api.security import HTTPException, reject_shared_safe_action, safe_resolve_child, safe_static_url_to_path
from config.settings import settings


def test_safe_resolve_child_allows_nested_relative_path(tmp_path: Path):
    resolved = safe_resolve_child(tmp_path, "album/song.mp3")
    assert resolved == (tmp_path / "album" / "song.mp3").resolve()


@pytest.mark.parametrize("bad_path", ["../secret.txt", "/tmp/secret.txt", r"..\\secret.txt", "~/.ssh/id_rsa"])
def test_safe_resolve_child_rejects_escape_paths(tmp_path: Path, bad_path: str):
    with pytest.raises(ValueError):
        safe_resolve_child(tmp_path, bad_path)


def test_safe_static_url_to_path_maps_only_known_prefix(tmp_path: Path):
    assert safe_static_url_to_path("/static/audio/a.mp3", "/static/audio/", tmp_path) == (
        tmp_path / "a.mp3"
    ).resolve()
    assert safe_static_url_to_path("/static/covers/a.jpg", "/static/audio/", tmp_path) is None


def test_reject_shared_safe_action_blocks_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "public_demo_mode", True)
    with pytest.raises(HTTPException) as exc:
        reject_shared_safe_action("delete song")
    assert exc.value.status_code == 403


def test_reject_shared_safe_action_allows_local_mode(monkeypatch):
    monkeypatch.setattr(settings, "public_demo_mode", False)
    reject_shared_safe_action("delete song")
