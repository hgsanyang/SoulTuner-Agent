# S4 English Mirror Outcome Report

Date: 2026-06-22  
Branch: `codex/r1-retrieval-refactor`  
Base git before commit: `e70fa818e852` (`dirty=True` because S4 cases were under validation)  
Effective planner: `dashscope / qwen3.7-plus`, temperature `0`  
Evaluation profile: `evaluate_outcomes --fast`

## Added Cases

S4 added 10 English natural-language outcome cases:

- dev: 6 cases covering artist, song title, language, soft mood, scenario, and negation.
- holdout: 4 frozen cases covering negation, song title, instrumental/focus, and multi-turn context.

This changes the splits to:

- dev: `50 → 56`
- holdout: `20 → 24`

## Results

| Split | Total | English mirror | Existing non-English |
|---|---:|---:|---:|
| dev | 50/56 | 4/6 | 46/50 |
| holdout | 22/24 | 4/4 | 18/20 |
| combined | 72/80 | 8/10 (80.0%) | 64/70 (91.4%) |

Detailed result files:

- `tests/eval/results/outcome_eval_dashscope_20260622_012559.json` (dev)
- `tests/eval/results/outcome_eval_dashscope_20260622_013207.json` (holdout)

## English Failures

| Case | Failing check | Main cause |
|---|---|---|
| `dev_en_mood_01_uplift` | `objective_soft_judge`: negative 50%, required <= 40% | Soft ranking / tag-quality issue |
| `dev_en_negative_01_chinese_not_sad` | `objective_soft_judge`: negative 45%, required <= 35% | Same "not sad" soft-ranking issue already visible in Chinese |

No English mirror failure points to entity linking, language-slot parsing, or hard-constraint interpretation. The hard English cases all passed, including Jay Chou, JJ Lin/Jiangnan, Eason Chan with exclusion, Taylor Swift/Love Story, instrumental focus, Japanese language, and multi-turn "same vibe but Chinese".

## Gate

The combined English-vs-non-English gap is `91.4% - 80.0% = 11.4pp`, below the S4 trigger threshold of roughly 15pp. Holdout English is `4/4`.

Decision: **do not trigger A2 yet**. Language normalization remains valuable for product polish, but the current measurable failures are better explained by:

1. Korean/Cantonese inventory coverage.
2. Soft-intent ranking and tag quality.
3. Web fallback tail latency.

Next quality work should target those causes before a broader multilingual schema migration.
