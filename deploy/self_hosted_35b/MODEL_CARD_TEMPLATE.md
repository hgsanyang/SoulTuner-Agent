---
base_model: Qwen/Qwen3.6-35B-A3B
library_name: peft
license: apache-2.0
language:
  - zh
  - en
tags:
  - lora
  - music-recommendation
  - planner
---

# SoulTuner Planner V4.2 35B LoRA

SoulTuner Planner is a domain adapter for producing guarded `PlannerDecisionV5`
candidates used by the SoulTuner music recommendation agent.

## Base model

- `Qwen/Qwen3.6-35B-A3B`
- Load the base model and this PEFT adapter separately.

## Intended use

- Music retrieval planning across Graph, Dense, and optional Web lanes.
- The raw model output must pass the repository's deterministic Planner guard
  before it is executed.

## Evaluation

Add the released aggregate Regression 412 and canonical sealed 500 metrics here.
Do not publish private evaluation rows.

## Loading

See `deploy/self_hosted_35b/README.md` in SoulTuner-Agent for download,
endpoint startup, environment variables, and hardware sizing.

## License and attribution

Include Apache-2.0 `LICENSE` and a `NOTICE` describing the Qwen base model and
the SoulTuner adapter. Confirm that every training-data source permits this
release before publishing.
