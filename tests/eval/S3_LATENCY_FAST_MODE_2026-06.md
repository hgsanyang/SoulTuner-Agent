# S3 Latency Fast-Mode Report

Date: 2026-06-22  
Branch: `codex/r1-retrieval-refactor`  
Base git before commit: `807d69b0c3b1` (`dirty=True` because S3 changes were under validation)  
Effective planner: `dashscope / qwen3.7-plus`, temperature `0`  
Evaluation profile: `evaluate_outcomes --fast --timing`

## What Changed

- GraphZep now uses a short request timeout plus a 5-minute unavailable-state cache, so an offline memory service does not add repeated multi-second waits.
- Explanation fast-mode skips the explanation LLM during eval/fast runtime and returns deterministic lightweight copy.
- Planner results are cached by query plus profile/chat/previous-plan context hash.
- HuggingFace model loading is forced to local/offline cache by default.
- Local processes explicitly load the project-private `.env` DashScope key before settings are built; the secret is not printed or tracked.

## Latency Comparison

| Split | Profile | Pass rate | End-to-end p50 | End-to-end p95 | GraphZep p50 | Intent p50 | Retrieval p50 | Web fallback p50 | Explanation p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev S1 baseline | normal | 42/50 | 29.88s | 42.52s | 1.51s | 6.85s | 0.65s | 6.87s | 17.41s |
| dev S3 fast | fast | 44/50 | 8.88s | 31.84s | 0.00s | 6.06s | 0.67s | 6.05s | 0.00s |
| holdout S2 gate | normal | 17/20 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| holdout S3 fast | fast | 19/20 | 10.97s | 49.10s | 0.00s | 6.21s | 0.61s | 11.77s | 0.00s |

Detailed result files:

- `tests/eval/results/outcome_eval_dashscope_20260622_010202.json` (dev)
- `tests/eval/results/outcome_eval_dashscope_20260622_010740.json` (holdout)

## Gate

- ✅ dev quality did not regress relative to the accepted S2 state: `44/50`.
- ✅ holdout did not regress and improved from `17/20 (85%)` to `19/20 (95%)`.
- ✅ p50 latency dropped materially on dev: `29.88s → 8.88s`.
- ✅ explanation p50 dropped to zero in fast-mode: `17.41s → 0.00s`.
- ✅ GraphZep offline p50 dropped to near-zero after the first circuit-breaker probe.

## Remaining Latency

S3 removes the largest deterministic waste, but it does not solve all latency:

- Planner remains the dominant local path cost, around 6 seconds p50.
- Web fallback tail latency is still high and volatile, especially on holdout p95.
- First query in a process can still pay cold-load cost for M2D-CLAP/BM25.

Next work should measure English coverage in S4, then return to Planner distillation/cache hit-rate and web fallback tail control.
