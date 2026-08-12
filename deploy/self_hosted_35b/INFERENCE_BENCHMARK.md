# Verified AMD inference profile

The SoulTuner V4.2 adapter was validated on one AMD MI308X (192 GB HBM) with
the BF16 Qwen3.6-35B-A3B base, vLLM through ms-swift, thinking disabled,
`temperature=0`, a 4096-token model limit, prefix caching, and at most 16
sequences. The endpoint remained private and authenticated.

The standard runtime loaded model weights in 34.18 seconds, used 74.8 GiB for
model loading, and reserved 93.21 GiB for a 1,895,219-token KV cache. These are
engine log values; overall process/device memory also includes the runtime,
graphs, allocator reservations, and other workspace overhead.

The benchmark contains 15 public representative requests per concurrency
level. It does not contain training, regression, or sealed-evaluation rows.
All 60 requests in each sweep produced valid JSON, passed the deterministic
Planner guard, and compiled to a safe production plan.

| Runtime | Concurrency | p50 | p95 | Requests/s | Completion tokens/s |
|---|---:|---:|---:|---:|---:|
| Standard vLLM | 1 | 3.94 s | 4.56 s | 0.258 | 71.2 |
| Standard vLLM | 4 | 5.64 s | 6.35 s | 0.669 | 184.5 |
| Standard vLLM | 8 | 7.25 s | 9.00 s | 0.959 | 270.5 |
| Standard vLLM | 16 | 8.21 s | 9.07 s | 1.640 | 454.7 |
| vLLM + AITER | 1 | 4.07 s | 4.38 s | 0.259 | 72.8 |
| vLLM + AITER | 4 | 5.97 s | 6.43 s | 0.670 | 189.3 |
| vLLM + AITER | 8 | 7.04 s | 8.81 s | 0.966 | 270.0 |
| vLLM + AITER | 16 | 8.40 s | 8.97 s | 1.671 | 474.5 |

The measured AITER request-throughput change was only about 0.1% to 1.9%, and
the installed stack reported fallback paths for sampler and paged attention.
Standard vLLM therefore remains the portable default. AITER is an opt-in
runtime experiment, not a claimed universal speedup.

The production integration smoke test also exercised the real
`IntentPlanner -> PlannerDecisionV5 -> deterministic guard -> MusicQueryPlan`
path for five public request types: acoustic, affective, catalog entity/date,
reference-track follow-up, and fresh external information. All five compiled
successfully; unsafe or missing Web permission was replaced by the bounded
deterministic plan.

Latency and throughput are measurements of this one hardware/software profile,
not service-level guarantees. Re-run `run_benchmark_sweep.sh` after changing
the base revision, adapter revision, vLLM, ROCm, AITER, quantization, or GPU.
