"""ToolPlan v1 schema freeze guard.

Once ToolPlan is a promotion candidate, its version, tool whitelist, and
per-tool argument surface are a security-and-compat contract. This guard
fails loudly if any of them drift, forcing a deliberate version bump and
review rather than an accidental change.
"""

from schemas.tool_plan import (
    TOOL_ARGUMENT_MODELS,
    TOOL_PLAN_VERSION,
    ToolName,
    ToolPlan,
)


def test_tool_plan_version_is_frozen():
    assert TOOL_PLAN_VERSION == "1.0"
    assert ToolPlan.model_fields["version"].default == "1.0"


def test_tool_whitelist_is_frozen():
    # Adding/removing a tool changes the agent's capability surface — must be
    # an intentional, reviewed change (update this set + bump the version).
    assert {name.value for name in ToolName} == {
        "retrieve_memory",
        "search_graph",
        "search_audio",
        "inspect_catalog_gap",
        "search_external_music",
        "resolve_playable_tracks",
        "commit_memory_delta",
    }


def test_every_tool_has_a_registered_argument_model():
    # No tool may execute without a strict (extra="forbid") argument schema.
    assert set(TOOL_ARGUMENT_MODELS.keys()) == set(ToolName)
    for model in TOOL_ARGUMENT_MODELS.values():
        assert model.model_config.get("extra") == "forbid"


def test_bounded_execution_limits_are_frozen():
    # max_replans is capped at 1 by schema; the plan cannot request unbounded retries.
    field = ToolPlan.model_fields["max_replans"]
    assert field.default == 1
    metadata = {type(m).__name__: m for m in field.metadata}
    # pydantic stores Le(le=1) / Ge(ge=0) constraints in field metadata
    assert any(getattr(m, "le", None) == 1 for m in field.metadata), metadata
