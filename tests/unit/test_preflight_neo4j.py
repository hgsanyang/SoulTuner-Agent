"""Community must never be pointed at an Enterprise `block` volume.

The repo default is Community (correct for the empty volume a fresh clone gets).
This check is the only thing stopping that default from opening someone's
existing Enterprise volume, where the failure surfaces as a generic startup crash.
"""

from __future__ import annotations

import pytest

import scripts.preflight_neo4j as pf


@pytest.fixture
def env(monkeypatch):
    """Neutralise every external probe; each test re-arms only what it asserts on."""
    monkeypatch.setattr(pf, "_read_env", lambda path: {})
    monkeypatch.setattr(pf, "_store_format_via_http", lambda env: "")
    monkeypatch.setattr(pf, "_previous_container_image", lambda: "")
    monkeypatch.setattr(pf, "_volume_exists", lambda: False)
    monkeypatch.setattr(pf, "_docker_available", lambda: True)
    monkeypatch.delenv("NEO4J_IMAGE", raising=False)
    monkeypatch.delenv("NEO4J_ACCEPT_LICENSE_AGREEMENT", raising=False)
    return monkeypatch


def test_community_on_a_block_volume_is_refused(env):
    env.setattr(pf, "_read_env", lambda p: {"NEO4J_IMAGE": "neo4j:2026.03.1-community"})
    env.setattr(pf, "_store_format_via_http", lambda e: "block-block-1.1")
    ok, lines = pf.check()
    assert ok is False
    body = "\n".join(lines)
    assert "REFUSING TO START" in body
    assert "NEO4J_MIGRATION.md" in body      # tells them where the fix lives


def test_community_on_an_aligned_volume_is_fine(env):
    env.setattr(pf, "_read_env", lambda p: {"NEO4J_IMAGE": "neo4j:2026.03.1-community"})
    env.setattr(pf, "_store_format_via_http", lambda e: "record-aligned-1.1")
    ok, _ = pf.check()
    assert ok is True


def test_community_is_refused_when_the_format_cannot_be_proven(env):
    """Neo4j down + the last container was Enterprise: we cannot read the store,
    but that combination only arises from an existing Enterprise volume. Refusing
    on 'unknown' is the point — an optimistic guess corrupts nothing but wastes an
    afternoon on a crash loop that reads like a broken image."""
    env.setattr(pf, "_read_env", lambda p: {"NEO4J_IMAGE": "neo4j:2026.03.1-community"})
    env.setattr(pf, "_previous_container_image", lambda: "neo4j:2026-enterprise")
    ok, lines = pf.check()
    assert ok is False
    assert "cannot prove" in "\n".join(lines)


def test_orphaned_volume_after_compose_down_is_refused(env):
    """The fail-open Codex found. `docker compose down` deletes the container but
    KEEPS the named volume (only `down -v` removes it), so the state is: Neo4j
    unreachable, NO container, volume still full of block-format data. The old
    code read "no container" as "no evidence of Enterprise" and let Community
    through — into a startup crash. A volume we cannot read must stop us."""
    env.setattr(pf, "_read_env", lambda p: {"NEO4J_IMAGE": "neo4j:2026.03.1-community"})
    env.setattr(pf, "_previous_container_image", lambda: "")     # container deleted
    env.setattr(pf, "_volume_exists", lambda: True)              # data still there
    env.setattr(pf, "_store_format_via_http", lambda e: "")      # Neo4j is down
    ok, lines = pf.check()
    assert ok is False
    body = "\n".join(lines)
    assert "REFUSING TO START" in body
    # the message has to explain WHY a missing container proves nothing
    assert "KEEPS the volume" in body


def test_fresh_install_with_no_volume_passes(env):
    """A brand-new clone has no volume and no container: Community is correct and
    must not be blocked. This is the ONLY unknown-format state that may pass."""
    env.setattr(pf, "_read_env", lambda p: {"NEO4J_IMAGE": "neo4j:2026.03.1-community"})
    ok, lines = pf.check()
    assert ok is True
    assert "fresh install" in "\n".join(lines)


def test_docker_unavailable_is_reported_as_not_checked_not_as_clean(env):
    """"docker says there is no volume" and "we could not ask docker" collapse to
    the same falsy value downstream. Only the first means fresh install; the
    second must not be reported as a clean bill of health."""
    env.setattr(pf, "_read_env", lambda p: {"NEO4J_IMAGE": "neo4j:2026.03.1-community"})
    env.setattr(pf, "_docker_available", lambda: False)
    ok, lines = pf.check()
    body = "\n".join(lines)
    assert "NOT CHECKED" in body
    assert "fresh install" not in body
    assert ok is True      # no docker means no `compose up` to protect


def test_enterprise_without_a_licence_choice_is_refused(env):
    """The container would print the licence and exit; say so before `up` runs.
    The repo must not pick yes-vs-eval on the operator's behalf."""
    env.setattr(pf, "_read_env", lambda p: {"NEO4J_IMAGE": "neo4j:2026-enterprise"})
    ok, lines = pf.check()
    assert ok is False
    body = "\n".join(lines)
    assert "yes" in body and "eval" in body


@pytest.mark.parametrize("value", ["yes", "eval"])
def test_enterprise_with_either_documented_licence_value_passes(env, value):
    env.setattr(pf, "_read_env", lambda p: {
        "NEO4J_IMAGE": "neo4j:2026-enterprise",
        "NEO4J_ACCEPT_LICENSE_AGREEMENT": value,
    })
    env.setattr(pf, "_store_format_via_http", lambda e: "block-block-1.1")
    ok, _ = pf.check()
    assert ok is True      # Enterprise opens block AND aligned


def test_repo_default_is_community():
    """The shipped default must suit a fresh clone (empty volume, no licence to
    accept), not this machine's Enterprise volume."""
    assert "community" in pf._compose_default_image().lower()


def test_output_is_ascii_only():
    """conda run on Windows re-encodes stdout as GBK; a non-ASCII char kills the
    preflight and takes `soultuner.ps1 up` down with it."""
    from pathlib import Path

    Path(pf.__file__).read_text(encoding="utf-8").encode("ascii")
