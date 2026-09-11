# Retired integrations — 2026-09-11

The current distribution no longer includes the Tavily/SearXNG/standalone web-search
aggregator, GraphZep service and client, or the old all-services launcher.
Do not restore their Docker services or credentials as default dependencies.

Web discovery uses the model API's native search capability. The remaining music
provider lookup resolves track metadata/playback; it is not a retired general web
search engine. API failures must not silently activate an old search engine.
Knowledge enrichment also uses native API search; legacy snippet fallback is retired.

Structured local memory and Neo4j remain supported. Historical field/argument names
containing `graphzep` may remain for response and persisted-state compatibility;
they do not load or start the retired service. A legacy `graphzep` backend setting
is ignored with a warning.

The maintainer keeps a private archive outside the repository, containing the
removed working files and a verified Git bundle at the pre-retirement commit.
It must not be published: old history can contain sensitive configuration.
Ignored retired paths prevent accidental re-addition during normal staging.

This retirement removes files from the current tree, not from earlier Git commits.
Historical secret remediation and any history rewrite are separate decisions.

Native API reference: https://help.aliyun.com/en/model-studio/qwen-api-via-openai-responses
and https://help.aliyun.com/zh/model-studio/web-search
