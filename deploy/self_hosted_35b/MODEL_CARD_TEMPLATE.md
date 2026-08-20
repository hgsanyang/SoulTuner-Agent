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

SoulTuner Planner V4.2 is a domain LoRA adapter for producing guarded
`PlannerDecisionV5` candidates in the SoulTuner music recommendation agent.
It plans Graph, Dense, and optional Web retrieval; it does not replace the
retrieval engines, memory system, ranker, or frontend.

## Base model

- `Qwen/Qwen3.6-35B-A3B`
- The base weights are not included in this repository.
- Load the official base and this PEFT adapter separately.

## Intended use

- Music-retrieval planning from Chinese or English user requests.
- Producing structured evidence, lane roles, and retrieval hints.
- Serving behind the deterministic guard in SoulTuner-Agent.

Raw model output must not directly invoke retrieval tools. The public
deployment validates the schema and task type, enforces required lanes, and
compiles execution weights deterministically.

## Aggregate evaluation

| Metric | Regression 412 | Canonical sealed 500 |
|---|---:|---:|
| Schema valid | 99.51% | 99.40% |
| Compilable | 99.51% | 99.20% |
| Intent / route | 99.03% | 95.60% |
| Lane policy satisfaction | 91.75% | 86.80% |
| Lane role exact match | 78.88% | 76.20% |
| Required lane recall | — | 89.98% |
| HyDE present when Dense | 98.88% | 84.44% |

The private evaluation rows are not redistributed. These results measure the
SoulTuner Planner contract and are not a general-purpose model ranking.

## Training and provenance

- Training target: 2 total epochs
- Best checkpoint identity: `checkpoint-450`
- Training code commit: `7c543bb6f66ffba8fcc25dfd74ee157a1e424c55`
- Frozen manifest SHA-256:
  `4ebaeeadcc843389efdbeb66cdebc2aef6014680f76074a621e6d2d9283c228c`
- Verified inference environment: AMD MI308X 192 GB HBM
- Training inputs were project-authored. Accepted targets were either generated
  or reviewed through Qoder `qmodel_38max`, or migrated and validated by the
  deterministic SoulTuner V4-to-V5 pipeline. Raw teacher outputs and training
  rows are not redistributed.

## Loading and deployment

Use the self-hosted instructions in:

<https://github.com/hgsanyang/SoulTuner-Agent/tree/main/deploy/self_hosted_35b>

The application supports one profile switch between Qwen3.7 Plus and the
self-hosted SoulTuner endpoint. An RTX 4070 can run the client application but
cannot host the BF16 35B base model.

## Data and privacy

This model repository contains no private training rows, user memory, music
audio, regression/sealed prompts, optimizer state, scheduler state, RNG state,
or access tokens.

## License

The adapter is prepared for Apache-2.0 release. Before changing a release
candidate to public, the publisher must confirm that every training-data source
permits publication of the adapter. The Qwen base model is distributed
separately under its own Apache-2.0 license and attribution.
