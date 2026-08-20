# AMD Creation Space deployment checklist

Use this checklist only after ModelScope assigns an AMD MI308X resource and a
matching ROCm image to the Creation Space. A CUDA image or an NVIDIA resource
is not evidence of an AMD deployment.

## Preserve the CPU fallback

- [ ] Record the currently working CPU Space revision and settings
- [ ] Keep the deterministic demo available until the GPU version passes all
      checks
- [ ] Do not replace the only healthy public demo during initial GPU bring-up

## Resource and image identity

- [ ] Resource selector explicitly shows AMD MI308X with 192 GB HBM
- [ ] Selected image is a platform-provided ROCm image, not a CUDA image
- [ ] `/dev/kfd` and `/dev/dri` are available
- [ ] `torch.cuda.is_available()` is true and `torch.version.hip` is non-empty
- [ ] Device name and HBM capacity are recorded without exposing host secrets

## Required configuration

```text
SOULTUNER_REQUIRE_ROCM=1
SOULTUNER_MODEL_PROFILE=soultuner-v4.2-35b
SOULTUNER_BASE_MODEL_ID=Qwen/Qwen3.6-35B-A3B
SOULTUNER_ADAPTER_MODEL_ID=hgsanyang/SoulTuner-Planner-V4.2-35B-LoRA
SOULTUNER_MODEL_CACHE=<persistent ModelScope path>
SOULTUNER_INFER_BACKEND=vllm
```

- [ ] Base and adapter revisions are immutable and pinned
- [ ] Adapter SHA-256 is configured as
      `9a3d2cb5bc2eee3dfc9f7c76c5350509d075aad11b61ddee3b9af2ad90ac272e`
- [ ] Model cache uses persistent storage
- [ ] Any registry credential is stored only in ModelScope secret management
- [ ] No secret is present in code, README files, ordinary variables,
      screenshots, or logs

## Fail-closed startup

- [ ] Startup refuses to continue without ROCm/HIP
- [ ] Base and adapter downloads fail loudly
- [ ] Adapter digest mismatch stops startup
- [ ] vLLM endpoint must become healthy before Gradio starts
- [ ] Startup cannot silently fall back to CPU while displaying the 35B profile
- [ ] `/v1/models` returns `soultuner-planner-v4.2-35b`

Validated baseline for comparison: BF16 LoRA, thinking disabled,
`temperature=0`, `max_model_len=4096`, `gpu_memory_utilization=0.90`,
`max_num_seqs=16`, and prefix caching enabled. It is a measured starting point,
not a universal service-level objective.

## Functional acceptance

- [ ] Acoustic-preference request parses, passes the guard, and compiles
- [ ] Affective-preference request parses, passes the guard, and compiles
- [ ] Catalog entity/date request parses, passes the guard, and compiles
- [ ] Reference-track follow-up parses, passes the guard, and compiles
- [ ] Fresh external-information request parses, passes the guard, and compiles
- [ ] Invalid model output is rejected or replaced by the bounded deterministic
      plan
- [ ] Logs contain no raw private prompts, sealed inputs, user privacy data,
      private paths, or tokens

## Performance acceptance

- [ ] Run 15 public representative requests at concurrency 1, 4, 8, and 16
- [ ] Record model load time, HBM use, KV-cache capacity, p50, p95, throughput,
      valid JSON rate, guard pass rate, and compile rate
- [ ] Compare with `INFERENCE_BENCHMARK.md`
- [ ] Re-run the sweep after changing model revisions, ROCm, vLLM, AITER,
      quantization, or GPU type

## Publish and review

- [ ] User reviewed the resource/image identity and acceptance results
- [ ] User explicitly approved switching the public default to 35B
- [ ] CPU fallback remains visible or is documented clearly
- [ ] Public anonymous Space visit succeeds
- [ ] Only after all checks pass, user explicitly approves formal platform
      review submission
