"""Capture model roles synchronously before any request work is scheduled."""

from services.recommendation_execution import RecommendationExecution, RecommendationModels


def build_recommendation_execution(
    *, web_search_enabled: bool, provider: str | None = None
) -> RecommendationExecution:
    from config.settings import settings
    from llms.multi_llm import (
        get_chat_model,
        get_conversation_chat_model,
        get_explain_chat_model,
        get_intent_chat_model,
    )

    # No await between the factories: all four roles are captured for this turn.
    # Do not fall back to another request's global client if construction fails.
    models = RecommendationModels(
        main=get_chat_model(
            provider=provider or settings.llm_default_provider or "dashscope",
            model_name=None if provider else settings.llm_default_model,
        ),
        intent=get_intent_chat_model(),
        conversation=get_conversation_chat_model(),
        explain=get_explain_chat_model(),
    )
    return RecommendationExecution(web_search_enabled=web_search_enabled, models=models)
