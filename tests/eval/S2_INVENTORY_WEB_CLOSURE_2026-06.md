# S2 Inventory-aware Web Closure (2026-06-21)

## Reproduction

Every DashScope process explicitly loads the project-private `.env`; the key is never written to this report or command history.

```powershell
$line = Get-Content .env | Where-Object { $_ -match '^\s*DASHSCOPE_API_KEY\s*=' } | Select-Object -First 1
$env:DASHSCOPE_API_KEY = ($line -replace '^\s*DASHSCOPE_API_KEY\s*=\s*', '').Trim().Trim('"').Trim("'")
python -m tests.eval.evaluate_outcomes --split holdout --planner-temperature 0 --quiet
```

- Git under test: `1edd377dc291` (`dirty=True`; S2 implementation not committed yet)
- Planner: `dashscope / qwen3.7-plus`, temperature `0`
- Unit tests: `109 passed`
- Targeted regression: `4/4 (100%)`
- Frozen holdout: `17/20 (85%)`, up from the accepted R1.5 baseline `16/20 (80%)`
- Local reports: `outcome_eval_dashscope_20260621_164222.json` (targeted) and `outcome_eval_dashscope_20260621_165153.json` (holdout)

## Recovered Cases

| Case | Before | After | Mechanism |
|---|---|---|---|
| `holdout_artist_01_jay_not_love_song` | fail | pass | Empty hybrid inventory now uses the unified web fallback |
| `holdout_artist_02_eason_exclude_ten_years` | fail | pass | Literal `soft_intent.avoid` is applied to web results |
| `holdout_song_01_qingtian_not_jay` | fail | pass | Missing exact title triggers a cover-oriented query; excluded artist stays excluded |
| `dev_context_04_same_artist_different_mood` | fail | pass | Multi-turn artist constraints use the same inventory decision, independent of intent label |

## Remaining Holdout Failures

- `holdout_language_02_korean_not_dance`: no Korean inventory satisfies the language threshold.
- `holdout_timeliness_01_new_but_not_chart`: intent analysis degraded.
- `holdout_fallback_02_private_memory_reference`: missing reliable private-memory evidence degraded safely.

S2 returns `retrieval_meta` with local inventory count, final source (`local` or `web`), degradation state/reason, result count, and the number removed by explicit avoid terms.
