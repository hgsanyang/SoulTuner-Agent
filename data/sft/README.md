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

## Which prompt a sealed score is measured under

The frozen splits do not agree on the system prompt. `train` and `regression`
carry the full 662-character `STUDENT_SYSTEM_PROMPT_V3`; the frozen `sealed`
split carries a 77-character one, because `build_v4_contract_curriculum`
inherits the prompt from `train_rows[0]` while `collect_v4_sealed_teacher`
hardcodes its own. Each side is internally uniform, which is what an accident
looks like rather than a designed contrast.

It is not a cosmetic difference. Measured on the 9B run, same adapter, same 500
questions, same gold, prompt swapped:

| | frozen 77-char | canonical 662-char |
|---|---|---|
| JSON parse rate | 0.0% | 100.0% |
| `schema_valid` | 0.0 | 1.0 |
| `lane_authority_violations` | 500 | 0 |

Under the short prompt the model emits an invented schema — that prompt never
names the contract fields, and it appears in none of the 8000 training rows. So
a score taken there measures the prompt mismatch, not the student.

**Rules:**

- The **canonical release evaluation** runs on sealed derived by
  `derive_canonical_prompt_eval.py`, which replaces only the system message and
  carries gold, row order, `meta` and `lineage` through unchanged.
- The **frozen sealed bytes are never rewritten**; SHA-256 stays
  `325008e104f0502b7aec196bf553c7593b5bb297070f84d6863364629e13bbba`.
- The short-prompt form is a `prompt_contract_stress` observation reported
  alongside, and is **not** a model-quality release gate.
- The derivation report records source/target/prompt SHA-256, the row count, and
  explicit `gold_unchanged` / `row_order_preserved` findings.
- Formal post-train evaluation goes through `run_planner_eval.sh`; it locks
  the exact base model and greedy decoding, refuses reused result paths,
  verifies the manifest against all three frozen split bytes, enforces one
  prediction per input before scoring, gates on the canonical derivative, and
  records the frozen short-prompt form separately as `prompt_contract_stress`.
  `evaluation_identity.json` and each inference-contract report retain the
  code, adapter, input and prediction fingerprints needed to audit the score.

## Metrics that can be undefined rather than zero

Two report fields exist because a metric read 0.0 where it had no meaning:

- `lane_f1` is `null` with `lane_f1_status: not_applicable` for a category whose
  gold never asks for a lane. `conversation` is such a category: every gold row
  correctly wants no tools, so correct rows contribute nothing to tp/fp/fn and
  the score is decided entirely by whichever row is wrong. Read
  `tool_set_exact_match` there instead — it was 100/101 where the F1 said 0.0.
- `clarification_precision` and `clarification_recall` carry separate supports
  (`..._support`), because precision rests on how many cases were *predicted*
  and recall on how many were in the *gold*. Below
  `MIN_OPERATIONAL_SUPPORT` a metric is reported but not enforced — that floor
  is an operating decision, **not** a claim that N rows resolve 3pp.

## 9B / 35B comparison

The reference comparison uses `Qwen/Qwen3.5-9B` and
`Qwen/Qwen3.6-35B-A3B` with the same data, seed and LoRA settings. Runs are
sequential so they do not distort each other's memory use or throughput.
`run_planner_v4.sh` accepts `MODEL_9B` / `MODEL_35B` overrides so a cloud
window can bind a SHA-verified local-disk copy instead of cold-reading the
persistent model cache; the default remains the ModelScope model id.
`NUM_TRAIN_EPOCHS` is likewise explicit and is included in the preflight
fingerprint (default `3`), so selecting two epochs does not require editing the
launcher after the comparison has been reviewed.

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
