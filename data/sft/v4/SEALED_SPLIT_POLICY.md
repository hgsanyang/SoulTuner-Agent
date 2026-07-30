# V4 sealed split policy

Interface draft. **No V4 data exists yet** — this file defines what a build must
satisfy, not what any build currently does.

## Why a third split

V3 has two: `train` (1289) and `eval` (226). The eval split is genuinely clean on
every leakage check that `episode_id` can see — zero shared episodes, zero exact
duplicates, zero shared entity-blind templates. It is still unable to answer the
question a release gate needs answered.

64 of its 86 artists appear in train. 81.5% of its artist mentions and 78.4% of
its song mentions are names the model saw during training. So a high score on it
is consistent with two very different models: one that learned to plan, and one
that learned which lane each familiar artist takes. **The split cannot tell them
apart**, and no threshold placed on it can either.

This is not a mislabelled leak — the sessions really are disjoint and the eval
remains a valid *regression* set. It is a missing measurement. The fix is a third
split whose entities the model has never seen.

```
train        learning
regression   the existing 226 rows, kept — "did this change break what worked"
sealed       new, entity-disjoint — "did it learn to plan, or to recognise"
```

## Disjointness required of `sealed`

All five, measured and recorded in `MANIFEST.schema.json → sealed_policy.measured`:

| Property | Threshold | V3 eval for comparison |
| --- | --- | --- |
| shared `episode_id` | 0 | 0 ✅ |
| shared exact user input | 0 | 0 ✅ |
| shared entity-blind template form | 0 | 0 ✅ |
| **shared artist / song / album** | **0** | 64 artists / 18 songs ❌ |
| max char 5-gram Jaccard vs any train row | ≤ 0.60 | 0.84 ❌ |

Episode ids are prefixed `sealed_` so a cross-split join cannot happen by
accident, and the seed pool must be a different batch of real user queries from
the one train drew on — otherwise the same phrasings reappear under new ids.

### The cost, stated plainly

Entity-disjoint means most sealed tracks will not be in the local catalogue. That
is deliberate. It turns the split into a test of the behaviour that actually
matters on unseen input: **does the planner route an unknown artist to the web
lane, or does it force a graph lookup that returns nothing?** V3 cannot ask this
question of any of its 226 rows.

## Size

300–500 rows, allocated so that every scored dimension has enough samples to
measure the gate applied to it.

| Dimension | Rows | pp per sample |
| --- | ---: | ---: |
| schema validity | all 400 | — |
| request_kind | all 400 | 0.25 |
| tool choice (lane F1) | all 400 | 0.25 |
| arguments (hard ×2 / soft / hints) | all 400 | 0.25 |
| clarification precision + recall | 80 (40 positive / 40 negative) | 1.25 |
| multi-turn inheritance | 80 | 1.25 |
| memory vs current request | 60 | 1.67 |
| failure recovery | 60 | 1.67 |

The floor of 60 comes straight from the V3 audit: its eval has **1** row each for
`library`, `acquisition` and `conversation`, where one sample moves the score by
100 percentage points. A 3pp gate against that is not a strict standard, it is an
unreadable instrument. At 60 rows one sample is 1.67pp, so a 3pp gate is about two
samples wide — tight, but at least measurable. Any class the release gate names
individually should be at 100.

## Clarification: the split that has to include negatives

All 13 clarification rows in V3 train and all 3 in eval are the same trope — a
self-contradictory request ("纯器乐但人声突出", "冷门到没人听过但 KTV 人人会唱").
A model trained on that learns to detect an oxymoron, not to recognise ambiguity.

The sealed set therefore scores clarification on **both** directions, with the
negative half deliberately larger in the training data (1 : 1.5):

**Should clarify** — genuinely underdetermined:
- missing a required slot ("帮我做个歌单" — how long? for whom? what for?)
- referent could be several things with different answers
- a constraint that cannot be satisfied as stated *and* has no reasonable default

**Should NOT clarify** — resolvable, and answering is correct:
- referent recoverable from history or the previous plan ("再来点这种")
- profile conflicts with the current request → **current request wins**, no question
- constraints in tension but rankable ("要新歌，也要我熟悉的") → answer with a trade-off
- vague but with an obvious default ("来点好听的")

Precision and recall are reported separately, never combined into F1. A model that
never clarifies scores perfect precision, and an F1 would hide that.

## Failure recovery requires real observations

`observation_origin: teacher_authored` is **forbidden** for
`trajectory_kind: failure_recovery` (enforced by the manifest schema's conditional
rule). A teacher inventing a plausible failure produces a plausible-looking
recovery from a failure mode the system does not actually have, and the model
learns to handle imaginary problems.

These rows must come from an executed run: either product traffic
(`real_execution`) or deterministic fault injection through the production
orchestrator (`harness_execution`). Both require a `trace_id`; harness rows also
record `execution_environment=controlled_harness`. This distinction prevents a
controlled timeout from being misreported as a user-facing production incident.

Note this needs a message shape V3 does not have (tool / observation roles). Keep
it in a separate file from the single-shot planner rows so the loss masks do not
interfere.

## Handling

- The sealed split is **not** read by any training or hyper-parameter selection
  step. Its only consumer is the release gate.
- Scored at most once per candidate build. Repeated scoring against the same
  sealed set turns it into a validation set, and the disjointness stops meaning
  anything after enough attempts.
- When it has been used enough times to influence decisions, retire it and build
  a new one. Record the retirement in the manifest rather than quietly reusing it.
- MuQ/HyDE text-to-music quality stays on its own ruler
  (`tests/eval/evaluate_alignment_attribute`) and is never folded into the planner
  score. A model that writes beautiful acoustic descriptions but picks the wrong
  lane must not be able to average its way to a pass.
