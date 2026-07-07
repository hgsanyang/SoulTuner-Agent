from schemas.query_plan import RetrievalPlan


def test_retrieval_plan_promotes_single_acoustic_query_to_variant_list():
    plan = RetrievalPlan(vector_acoustic_query="soft piano and warm guitar")

    assert plan.vector_acoustic_queries == ["soft piano and warm guitar"]


def test_retrieval_plan_uses_first_variant_as_primary_and_caps():
    plan = RetrievalPlan(
        vector_acoustic_queries=["a", "b", "a", "c", "d", "e"],
    )

    assert plan.vector_acoustic_query == "a"
    assert plan.vector_acoustic_queries == ["a", "b", "c", "d"]
