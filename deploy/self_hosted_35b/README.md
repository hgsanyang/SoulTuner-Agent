# Self-host the SoulTuner V4.2 35B Planner

This package connects the fine-tuned SoulTuner Planner to the complete
SoulTuner Agent. It is not a separate recommendation product: LangGraph,
Neo4j, dense audio retrieval, controlled web discovery, long-term memory,
feedback, ranking, and the Next.js frontend remain in the main application.

The model proposes a compact `PlannerDecisionV5`. `planner_guard.py` validates
that proposal, enforces hard lane boundaries, and deterministically compiles it
into the existing `MusicQueryPlan` / `ToolPlan` interface before any retrieval
tool can run.

```text
SoulTuner frontend and LangGraph Agent
             |
             +-- hosted general-model API
             |
             `-- OpenAI-compatible SoulTuner Planner endpoint
                          |
                          `-- Qwen3.6-35B-A3B + SoulTuner PEFT adapter
```

## What runs where

| Profile | Model location | Client GPU requirement |
|---|---|---|
| Hosted API | provider endpoint | none |
| SoulTuner V4.2 35B | private GPU server or managed GPU workspace | none; only HTTPS access is required |
| Deterministic demo | application process | none |

The verified BF16 base files occupy about 72 GB before KV cache and runtime
workspace. Allow at least about 96 GB usable GPU memory; a 192 GB accelerator is
comfortable. A 12 GB RTX 4070 can run the application client but cannot host
this BF16 model. Quantized variants require a fresh accuracy and safety
evaluation and are not the default release path.

## Publish only the adapter

Do not copy the 72 GB base model into the SoulTuner repository. Publish a clean
PEFT adapter and let each inference host download the official base separately:

```text
SoulTuner-Planner-V4.2-35B-LoRA/
|-- adapter_model.safetensors
|-- adapter_config.json
|-- README.md
|-- LICENSE
|-- NOTICE
`-- SHA256SUMS
```

GitHub contains code, documentation, model cards, and registry links—not model
weights. A Hugging Face-compatible registry is the portable default for the
international community; regional registries can mirror the same immutable
adapter revision from their platform-specific deployment directory.

Create a release copy without changing the private checkpoint:

```bash
python prepare_adapter_release.py \
  --checkpoint /private/checkpoint-450 \
  --output /private/SoulTuner-Planner-V4.2-35B-LoRA \
  --base-model Qwen/Qwen3.6-35B-A3B
```

Add `MODEL_CARD_TEMPLATE.md` as `README.md`, `NOTICE_TEMPLATE` as `NOTICE`,
and the selected license. Before making the repository public, confirm every
training-data source permits adapter publication and run the fail-closed audit:

```bash
python audit_public_adapter_repo.py /private/SoulTuner-Planner-V4.2-35B-LoRA \
  --expected-adapter-sha256 9a3d2cb5bc2eee3dfc9f7c76c5350509d075aad11b61ddee3b9af2ad90ac272e \
  --expected-adapter-size 90018600
```

The audit rejects optimizer/resume state, private paths, evaluation rows,
unknown files, an incorrect base identity, and a mismatched adapter digest.
Complete the full registry procedure in
[`MODELSCOPE_UPLOAD_CHECKLIST.md`](MODELSCOPE_UPLOAD_CHECKLIST.md) before any
visibility change.

## Download model assets

Use local directories when the assets are already present:

```bash
export SOULTUNER_BASE_MODEL=/models/qwen3.6-35b-a3b
export SOULTUNER_ADAPTER=/models/soultuner-v4.2-adapter
```

For a Hugging Face-compatible registry:

```bash
python -m pip install -U "huggingface_hub[hf_xet]"
export SOULTUNER_ADAPTER_REPO=YOUR_ORG/SoulTuner-Planner-V4.2-35B-LoRA
export SOULTUNER_BASE_REVISION=PINNED_BASE_COMMIT
export SOULTUNER_ADAPTER_REVISION=PINNED_ADAPTER_COMMIT
bash download_huggingface_assets.sh
```

Production deployments must pin immutable revisions and verify `SHA256SUMS`.
The platform-specific directory documents optional regional download mirrors.

## Start an authenticated endpoint

Install a ROCm- or CUDA-compatible PyTorch environment, `ms-swift`, and vLLM,
then run:

```bash
export SOULTUNER_BASE_MODEL=/models/qwen3.6-35b-a3b
export SOULTUNER_ADAPTER=/models/soultuner-v4.2-adapter
export SOULTUNER_SERVE_API_KEY="$(openssl rand -hex 32)"
bash start_35b_endpoint.sh
```

The default bind address is `127.0.0.1:8000`. The script refuses an
unauthenticated `0.0.0.0` endpoint. Use TLS termination, a VPN, a private
network, or an SSH tunnel—never expose an unauthenticated inference port.

The validated defaults are BF16 LoRA, vLLM, thinking disabled,
`temperature=0`, `max_model_len=4096`, `gpu_memory_utilization=0.90`,
`max_num_seqs=16`, and prefix caching enabled. The corresponding environment
variables are documented in `start_35b_endpoint.sh`.

## Connect the complete SoulTuner Agent

The standalone Gradio file in this folder is only a small endpoint diagnostic.
For the real product, configure the main application and keep every retrieval,
memory, ranking, and frontend service unchanged:

```env
INTENT_LLM_PROVIDER=vllm
INTENT_LLM_MODEL=soultuner-planner-v4.2-35b
INTENT_PLANNER_CONTRACT=v42
VLLM_BASE_URL=https://planner.example.com/v1
VLLM_API_KEY=replace-with-the-endpoint-secret
```

Then start the normal Docker Compose application. To return to a hosted model,
change the intent provider/model/contract settings; no business-logic code
changes are required.

From the repository root, the standard application command remains:

```bash
docker compose up -d --build
```

The Planner endpoint can run on another GPU host; Compose only needs the URL
and secret above. Keeping the model service separate also lets the web, Agent,
Neo4j, Qdrant, and memory services scale or restart independently.

## Measure before tuning

`benchmark_endpoint.py` uses five public representative requests. Reports
contain anonymous case IDs, aggregate contract/guard rates, latency, throughput,
and token counts—never private regression/sealed rows or raw model responses.

```bash
export SOULTUNER_PLANNER_API_KEY="$SOULTUNER_SERVE_API_KEY"
python benchmark_endpoint.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model soultuner-planner-v4.2-35b \
  --repeat 3 --concurrency 4 \
  --json benchmark-results/vllm-c4.json
```

Use `run_benchmark_sweep.sh` for concurrency 1/4/8/16. Change one runtime
variable at a time. AITER, SGLang, FP8, or 4-bit variants become defaults only
after the same contract, guard, latency, throughput, and memory checks pass.
The measured MI308X profile and its limits are recorded in
[`INFERENCE_BENCHMARK.md`](INFERENCE_BENCHMARK.md).

For a ModelScope AMD Creation Space, use
[`DEPLOYMENT_CHECKLIST.md`](DEPLOYMENT_CHECKLIST.md) to preserve the CPU
fallback, verify ROCm identity, and gate the final review submission.

## Aggregate Planner evaluation

These metrics use one fixed evaluator and measure the SoulTuner structured
planning contract, not general model ability. Private evaluation rows are not
redistributed.

| Metric | Regression 412 | Canonical sealed 500 |
|---|---:|---:|
| Schema valid | 99.51% | 99.40% |
| Compilable | 99.51% | 99.20% |
| Intent / route | 99.03% | 95.60% |
| Lane policy satisfaction | 91.75% | 86.80% |
| Lane role exact match | 78.88% | 76.20% |
| Required lane recall | - | 89.98% |
| HyDE present when Dense | 98.88% | 84.44% |

`Lane policy satisfaction` verifies required lanes are enabled and prohibited
lanes are disabled. `Lane role exact match` additionally requires Graph, Dense,
and Web to match `required / optional / off` exactly.

## Privacy boundary

Do not place private music libraries, user memory, raw feedback, training rows,
regression/sealed prompts, optimizer state, resume files, absolute private
paths, or credentials in an image or public model repository. Only authorized,
reviewed, and versioned examples may enter the data flywheel.
