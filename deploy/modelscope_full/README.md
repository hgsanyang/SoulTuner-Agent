# ModelScope full-experience deployment candidate

This directory prepares the existing Next.js + FastAPI application for a
single **public** Studio port without replacing the working Gradio Space.

## Safety gate

The current `hgsanyang/SoulTuner-Agent` Space is a verified Gradio + MI308X
deployment.  ModelScope hardware entitlement can differ by SDK type, and the
anonymous hardware API does not prove that the same account can select MI308X
for a Docker Studio.  Do not switch the live Space from Gradio to Docker until
the signed-in settings page offers both:

- `AMD Instinct MI308X / 192G`
- `ubuntu22.04-rocm7.2.1-py312-torch2.10.0-modelscope1.36.3` (or an explicitly
  validated successor)

Use a copied test Space for the first Docker/full-UI deployment.  The existing
Gradio application remains the rollback target.

## Process layout

Only the Next.js listener is public:

| Process | Address | Purpose |
|---|---|---|
| Next.js | `0.0.0.0:7860` | public UI; proxies `/api` and `/static` |
| FastAPI | `127.0.0.1:8501` | recommendation, flywheel, Range-capable media |
| vLLM/ms-swift | `127.0.0.1:8000` | SoulTuner V4.2 35B planner |
| Neo4j | external private URI | graph and durable catalog |

Studio does not provide a second public port and should not be treated as a
Docker Compose host.  Neo4j is therefore not started inside this entrypoint.
For a public demo, restore a reviewed graph snapshot into a private external
Neo4j service; for a production deployment, use a managed/dedicated Neo4j.

The full entrypoint sets `SOULTUNER_DUAL_ROLE_MODELS=1`.  ms-swift receives the
named multi-LoRA argument
`--adapters soultuner-v4.2-35b=<adapter_dir>` and exposes two model IDs from the
same vLLM process: `qwen3.6-35b-a3b` for natural-language conversation and
`soultuner-v4.2-35b` for Planner calls.  The startup health gate requires both
IDs in `/v1/models`.  Leaving the flag unset keeps the already verified
single-LoRA Gradio command unchanged.

## Persistent layout

All rebuild-sensitive assets live below `/mnt/workspace/soultuner`:

```text
/mnt/workspace/soultuner/
  model_cache/       # Qwen 35B + LoRA
  hf/                # MuQ and Hugging Face cache
  data/              # catalog, audio, metadata, feedback/ingest state
  logs/              # planner/backend logs
```

Canonical licensed audio and its manifest should still be versioned in a
separate ModelScope dataset repository.  Startup materializes that dataset into
`data/mtg_sample`, verifies every SHA-256 and per-track licence, and idempotently
routes missing tracks through the normal Neo4j + MuQ/M2D/OMAR ingest worker
before loading the 35B endpoint.  The Studio Git repository contains code and
version pins only; the audio remains in
[`hgsanyang/SoulTuner-Open-Audio-Demo`](https://modelscope.cn/datasets/hgsanyang/SoulTuner-Open-Audio-Demo).

## Build and start

Inside a validated ROCm image, install Python dependencies and build the web
artifact once during the image build:

```bash
python -m pip install -r requirements.txt
bash deploy/modelscope_full/build_frontend.sh
```

Configure `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` as Studio secrets or
private variables.  Then explicitly enable the experimental full entrypoint:

```bash
SOULTUNER_ENABLE_FULL_SPACE=1 bash deploy/modelscope_full/start_full_space.sh
```

The script refuses to run without ROCm, a built Next.js artifact, and explicit
Neo4j credentials.  It never retrains the 35B adapter and never runs sealed 500.
