# Outcome Eval

This harness evaluates whether the returned songs satisfy the user's music intent.
It is not an intent-label accuracy test.

## Splits

- `smoke`: the original 12-case fast regression set.
- `dev`: 57 cases for day-to-day iteration, including 6 English mirror cases
  and one clarification case.
- `holdout`: 34 frozen cases, including English mirrors, multi-turn context,
  negative constraints, and soft-intent cases. Do not tune
  directly against this set.
- `context_dev`: 52 Chinese context-matching cases, written against this
  catalog and bucketed by context goal plus specificity.
- `context_holdout`: 16 frozen context-matching cases. Use only as a milestone
  regression check.
- `context_all`: context_dev + context_holdout, for explicit milestone checks.
- `dev_easy` / `dev_hard` and `holdout_easy` / `holdout_hard`: derived views
  over the same frozen JSON files. `hard` currently covers negation,
  soft-intent, scenario, catalog-gap/web-fallback, and multi-turn cases.
- `all`: dev + holdout, for explicit milestone checks only.

Run:

```powershell
python -m tests.eval.evaluate_outcomes --split smoke
python -m tests.eval.evaluate_outcomes --split dev
python -m tests.eval.evaluate_outcomes --split holdout
python -m tests.eval.evaluate_outcomes --split holdout_hard
python -m tests.eval.evaluate_outcomes --split context_dev --fast --case-timeout 75
python -m tests.eval.calibrate_soft_judge --min-accuracy 0.95
```

Reports are written to `tests/eval/results/` and include git sha, branch, dirty
state, effective model config, Planner temperature, and key non-secret settings.
Each case now includes `intent_status` and `ranking_status`, and the aggregate
report includes `by_dimension` so failures can be traced to intent planning
versus ranking/retrieval quality.
Use `--require-no-failures` or `--min-decided-pass-rate 0.95` when running a
full-stack quality gate in a local/remote environment that has Neo4j and model
credentials.
Outcome eval sets `EVAL_DISABLE_SIDE_EFFECTS=True` internally so it measures the
recommendation path without writing preference extraction, MemoryGateway sidecar
persistence, or profile-refresh side effects.

Add `--timing` to include per-case stage timings and aggregate p50/p95 latency:

```powershell
python -m tests.eval.evaluate_outcomes --split dev --planner-temperature 0 --timing
python -m tests.eval.evaluate_outcomes --split dev --planner-temperature 0 --fast --timing --case-timeout 45
```

The timing report covers MemoryGateway/episodic memory, intent planning, each recall source,
fusion/filter, ranking, web fallback, explanation, Agent total, and end to end.
`--case-timeout` is useful for slow dev profiling: a stuck case is marked
`TIMEOUT`, the run continues, and the JSON report includes `slow_cases`.

MemoryGateway feedback semantics have a separate deterministic ruler:

```powershell
python -m tests.eval.evaluate_memory
```

It currently covers six slate-feedback mappings, including noisy/sad/quiet,
over-familiar, niche discovery, and seed-closeness feedback. It does not call
Neo4j, GraphZep, Mem0, or LLMs.

A3 ranking policy readiness is exposed through the API and a lightweight smoke
script. It is not a quality ruler; it answers the operational question "what is
safe to do next with the collected feedback?"

```powershell
python scripts/p7_smoke.py
python scripts/p7_smoke.py --api-base http://localhost:8501
```

The smoke check verifies that public-demo safety guards, path validation,
ranking-policy readiness, dense-backend configuration, optional calibration
configuration, and selected API endpoints are wired correctly without calling an
LLM or consuming outcome-eval budget.

### Context matching ruler

The `context_*` splits are the non-saturated Chinese ruler for the current
product direction: "understand this moment and pick fitting songs from a liked
library."  They borrow the TalkPlayData taxonomy shape, but not its data or
target songs.

Each case carries:

- `goal_category`: one of `audio_attribute`, `lyrics_theme`, `emotion`,
  `scenario`, `artist_entity`, `song_entity`, `language_region`, `era_style`,
  `negative_refinement`, `interaction_refinement`, or `clarification`.
- `specificity`: `LL`, `HL`, `LH`, or `HH`, used to spot whether broad or highly
  constrained requests are weaker.

Reports include pass/fail buckets by `goal_category` and `specificity`. This
split is expected to expose failures; do not tune against `context_holdout`.

## Text-To-Audio Alignment Eval

The recommendation-facing comparison is the frozen bilingual attribute ruler:

```powershell
python -m tests.eval.evaluate_alignment_attribute --k 10
```

It evaluates 24 Chinese/English language, genre, mood, and scenario queries
against catalog labels and reports P@10 for MuQ-MuLan and M2D-CLAP on the same
corpus. It is deterministic and does not call an LLM. Use this ruler together
with outcome dev/holdout when changing the dense text-to-music backend.
When experimenting with `MUSIC_DENSE_QUERY_VARIANTS=1` or
`MUSIC_ALIGNMENT_CALIBRATION_PATH`, compare this attribute ruler before and
after the change, then confirm the end-to-end outcome eval does not regress.

CLaMP3 can be included as an optional offline bake-off backend after its 768d
vectors have been backfilled into Neo4j. It is not part of the default online
retrieval path:

```powershell
$env:CLAMP3_REPO_DIR = "C:\path\to\clamp3"
python data/pipeline/backfill_clamp3_embeddings.py --limit 100
python -m tests.eval.evaluate_alignment_attribute --k 10 --include-clamp3
```

If `clamp3_embedding` is absent, the report keeps the M2D/MuQ baseline and marks
CLaMP3 as `missing_corpus`.

To train and validate the reversible text/audio gap calibration:

```powershell
python scripts/train_alignment_calibration.py --backend both --k 10 --output data/alignment_calibration.json
python -m tests.eval.evaluate_alignment_attribute --k 10 --calibration-path data/alignment_calibration.json
```

The current centroid-bias calibration is deliberately conservative: it performs
a train split shrink search and may choose `shrink=0` when validation would not
improve. In that case the calibration file is a safe no-op and should not be
enabled in production unless a later frozen validation report shows a gain.

The next, stronger but still reversible, option is a text-side linear adapter
trained from frozen `(caption, audio vector)` pairs:

```powershell
python scripts/train_alignment_adapter.py --backend muq --caption-style acoustic --output data/alignment_adapter.json
python -m tests.eval.evaluate_alignment_attribute --k 10 --adapter-path data/alignment_adapter.json
```

`MUSIC_ALIGNMENT_ADAPTER_PATH` applies the adapter to query text vectors before
Neo4j KNN and before the tri-anchor semantic rerank. Stored audio vectors remain
unchanged. Missing files, unknown backends, and dimension mismatches are no-op,
so rollout is reversible. Treat the adapter as accepted only if attribute P@10
and outcome/context eval do not regress.

`--caption-style metadata` preserves the original A4.1 tag-sentence captions.
`--caption-style acoustic` deterministically rewrites the same frozen tags into
instrumentation/dynamics/texture/context captions, which is usually a better
fit for MuQ-style text-to-audio training. It still must pass frozen validation;
do not enable an adapter just because training completed.

`evaluate_alignment` isolates M2D-CLAP text-to-audio alignment from the
end-to-end Agent. It uses a frozen metadata/tag caption set and does not call
the Planner, HyDE, or any LLM during evaluation.

Build or refresh the frozen captions only as an explicit milestone action:

```powershell
python -m tests.eval.build_alignment_gold --count 100
```

Then evaluate M2D-CLAP text captions against the full Neo4j audio-vector corpus:

```powershell
python -m tests.eval.evaluate_alignment
```

Reports include git sha, dirty state, M2D-CLAP checkpoint path, corpus size,
Recall@1/5/10, MRR, and per-caption ranks. Treat this exact-song caption task as
a diagnostic only: captions are not unique song identities, so low recall must
not be used alone to choose between MuQ and M2D. Attribute P@10 plus end-to-end
outcomes are the acceptance signals.

## DST And Clarification Checks

A7 adds explicit session-local dialogue state and whitelisted PlanDelta
operations. Established follow-ups update state deterministically; full-plan
generation is retained only for first turns, topic resets, and fallback.
Outcome cases may provide an
initial `dialog_state` next to `chat_history`; the harness passes it to the
agent and records the returned `dialog_state` plus `dialog_delta`.

Useful checks:

- `expected_clarification: true`: a clarification question is the expected
  result, and it should include `clarification_options` for future UI chips.
- `dialog_state_contains`: asserts a nested state path contains a value, for
  example `{"hard_constraints.language": "Chinese"}`.
- `dialog_delta_contains`: asserts the current turn delta, for example
  `{"followup": true, "inherited": "soft_intent.vibe"}`.

## Soft-Intent Judge

`objective_soft_judge` is an early, non-cyclic heuristic for soft intents. It
only reads objective song attributes (`genre`, `genres`, `moods`, `scenarios`,
`language`, `region`, `instrumental`, `is_instrumental`) and never reads the
system-generated explanation.

Use it conservatively:

- Calibrate against a small human-labeled gold set before enabling it broadly.
- Keep low-confidence or underspecified cases in `manual_review`.
- Prefer it for coarse objective tags such as calm/energetic/sleep/commute, not
  for subtle taste statements such as "like Friday after work".
- The calibration seed lives in
  `tests/eval/judge_gold/objective_soft_judge_gold.json`; it covers pass/fail/skip
  examples and should be extended whenever a new soft-intent pattern is promoted
  from `manual_review`.

## Discipline

- Keep Planner temperature at `0` for reproducibility unless you are explicitly
  testing stochastic behavior.
- Add user-intent cases, not cases that merely match the current implementation.
- Track per-category pass rates, not only the aggregate pass rate.
- Holdout cases should contain hard boundaries: negation, mixed language,
  self-reference, vague taste, conflicting constraints, and multi-turn context.
- A future LLM judge must only see the raw query plus objective song attributes
  such as title, artist, language, genre, moods, scenarios, and instrumental
  flags. It must not see the system-generated explanation.
