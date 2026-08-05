# Planner distillation harness

This directory contains the public contracts, validators and launch scripts for
distilling SoulTuner's LLM planner. Private conversations and frozen training
JSONL files are intentionally excluded from Git.

## Output contract

The student emits one strict `PlannerDecisionV3` object. A deterministic
compiler turns it into executable `ToolPlan 1.1` calls for graph, dense audio,
web discovery, library reads and reversible ingest previews. Tool observations
remain input context; they are never mixed into the assistant target as a
second JSON protocol.

## Data gates

A formal release must include three immutable splits:

- `train`: teacher-reviewed planning trajectories;
- `regression`: continuity checks for known behaviours;
- `sealed`: independently reviewed, entity- and template-disjoint cases that
  are never used for training or checkpoint selection. Its recommendation
  slice also covers single-turn requests, multi-turn inheritance, memory/current
  conflicts, necessary clarification and over-clarification traps.

Every row carries provenance. `MANIFEST.json` records row counts, SHA-256
digests, the generator commit and measured split overlap. Formal training stops
before using a GPU if the manifest, row contract or fingerprints do not match.

The local private corpus can be rebuilt with:

```bash
python -m data.sft.build_v4_release
python -m data.sft.verify_frozen_manifest \
  --manifest data/teacher/private/v4/frozen-v4.0.0/MANIFEST.json
```

## 9B / 35B comparison

The reference comparison uses `Qwen/Qwen3.5-9B` and
`Qwen/Qwen3.6-35B-A3B` with the same data, seed and LoRA settings. Runs are
sequential so they do not distort each other's memory use or throughput.

```bash
MANIFEST_FILE=<manifest> \
TRAIN_FILE=<train.jsonl> \
VAL_FILE=<regression.jsonl> \
SEALED_FILE=<sealed.jsonl> \
bash data/sft/run_planner_contrast.sh
```

This command performs environment and 50-step preflights only. Set
`RUN_FULL=1` only after both preflights pass. The harness verifies installed
ms-swift flags, AMD ROCm availability, clean code provenance, fresh logs,
non-zero finite loss, adapter artifacts, complete predictions and strict V3
schema compliance.

After training, `check_planner_release.py` applies the thresholds frozen before
training, while `compare_planner_scores.py` compares 9B and 35B overall and by
request kind. `benchmark_planner_endpoint.py` measures schema validity and p50 /
p95 latency against an OpenAI-compatible local endpoint. Planner quality,
end-to-end recommendation quality and served latency remain separate gates.

## Historical files

Earlier V2/V3 datasets and review artifacts remain for reproducibility. They
are not formal V4 training inputs unless a new manifest explicitly names and
fingerprints them.
