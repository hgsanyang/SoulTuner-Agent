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
- `all`: dev + holdout, for explicit milestone checks only.

Run:

```powershell
python -m tests.eval.evaluate_outcomes --split smoke
python -m tests.eval.evaluate_outcomes --split dev
python -m tests.eval.evaluate_outcomes --split holdout
python -m tests.eval.calibrate_soft_judge --min-accuracy 0.95
```

Reports are written to `tests/eval/results/` and include git sha, branch, dirty
state, effective model config, Planner temperature, and key non-secret settings.
Outcome eval sets `EVAL_DISABLE_SIDE_EFFECTS=True` internally so it measures the
recommendation path without writing preference extraction, GraphZep persistence,
or profile-refresh side effects.

Add `--timing` to include per-case stage timings and aggregate p50/p95 latency:

```powershell
python -m tests.eval.evaluate_outcomes --split dev --planner-temperature 0 --timing
python -m tests.eval.evaluate_outcomes --split dev --planner-temperature 0 --fast --timing --case-timeout 45
```

The timing report covers GraphZep, intent planning, each recall source,
fusion/filter, ranking, web fallback, explanation, Agent total, and end to end.
`--case-timeout` is useful for slow dev profiling: a stuck case is marked
`TIMEOUT`, the run continues, and the JSON report includes `slow_cases`.

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

To train and validate the reversible text/audio gap calibration:

```powershell
python scripts/train_alignment_calibration.py --backend both --k 10 --output data/alignment_calibration.json
python -m tests.eval.evaluate_alignment_attribute --k 10 --calibration-path data/alignment_calibration.json
```

The current centroid-bias calibration is deliberately conservative: it performs
a train split shrink search and may choose `shrink=0` when validation would not
improve. In that case the calibration file is a safe no-op and should not be
enabled in production unless a later frozen validation report shows a gain.

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
