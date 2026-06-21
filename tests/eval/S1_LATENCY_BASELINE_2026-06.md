# S1 Latency Baseline (2026-06-21)

## Reproduction

```powershell
$line = Get-Content .env | Where-Object { $_ -match '^\s*DASHSCOPE_API_KEY\s*=' } | Select-Object -First 1
$env:DASHSCOPE_API_KEY = ($line -replace '^\s*DASHSCOPE_API_KEY\s*=\s*', '').Trim().Trim('"').Trim("'")
python -m tests.eval.evaluate_outcomes --split dev --planner-temperature 0 --timing --quiet
```

- Git: `73c4a9629856` (`dirty=True` because timing instrumentation was under test)
- Planner: `dashscope / qwen3.7-plus`, temperature `0`
- Cases: dev `50`
- Local report: `tests/eval/results/outcome_eval_dashscope_20260621_162226.json`
- Outcome result: `42/50 (84%)`; this run establishes latency, not a new quality claim. The accepted R1.5 quality gate remains dev `44/50`, holdout `16/20`.

## Baseline

| Stage | Count | p50 | p95 | Mean |
|---|---:|---:|---:|---:|
| End to end | 50 | 29.88s | 42.52s | 29.62s |
| GraphZep | 50 | 1.51s | 2.45s | 1.52s |
| Intent Planner | 50 | 6.85s | 17.43s | 8.67s |
| Retrieval total | 46 | 0.65s | 1.95s | 0.92s |
| Graph recall | 46 | 0.19s | 0.39s | 0.21s |
| Dense recall | 46 | 0.27s | 0.41s | 0.49s |
| BM25 recall | 46 | 0.01s | 1.53s | 0.22s |
| Personalized recall | 46 | 0.04s | 0.08s | 0.05s |
| Cold-start recall | 46 | 0.02s | 0.02s | 0.02s |
| Fusion + hard filter | 46 | 0.01s | 0.01s | 0.01s |
| Ranking | 46 | 0.29s | 0.63s | 0.29s |
| Netease web fallback | 14 | 6.87s | 34.87s | 10.88s |
| Explanation | 50 | 17.41s | 28.89s | 15.52s |

The five local recall paths run in parallel and are not the main latency problem. The first B1 pass should prioritize explanation fast-mode, Planner result caching, and web-fallback tail latency. GraphZep is measurable but smaller than the two LLM stages while healthy; its existing 5-minute offline circuit breaker remains important when the service is down.

A one-case cold-start smoke reached about 39 seconds because M2D-CLAP and the BM25 directory loaded for the first time. The 50-case table above is the comparable warm-process baseline for S3.
