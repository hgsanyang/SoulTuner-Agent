from retrieval.hybrid_retrieval import local_recall_tools_from_plan


def _plan(*names):
    return {
        "_tool_plan": {
            "tool_calls": [
                {"id": f"call_{index}", "name": name, "arguments": {}}
                for index, name in enumerate(names)
            ]
        }
    }


def test_shadow_mode_preserves_dual_recall():
    assert local_recall_tools_from_plan(_plan("search_graph"), execution_enabled=False) == (
        True,
        True,
        False,
    )


def test_active_tool_plan_can_select_graph_only():
    assert local_recall_tools_from_plan(_plan("search_graph"), execution_enabled=True) == (
        True,
        False,
        True,
    )


def test_active_tool_plan_can_select_audio_only_or_both():
    assert local_recall_tools_from_plan(_plan("search_audio"), execution_enabled=True) == (
        False,
        True,
        True,
    )
    assert local_recall_tools_from_plan(
        _plan("search_graph", "search_audio"), execution_enabled=True
    ) == (True, True, True)
