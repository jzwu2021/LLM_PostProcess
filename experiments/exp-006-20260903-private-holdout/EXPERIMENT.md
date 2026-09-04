# Experiment exp-006-20260903: private, objectively-scored holdout

Builds the evaluation set that exp-005 concluded was missing: one that can
separate capability from answer style, and that no arm's training data can leak
an answer into.

## The honest limitation, stated first

**This set is not fully clean for exp-004, and cannot be made so by me.** The same
author wrote the 100 v0.4 mechanisms and selected the subjects here. Topic-level
contamination is therefore present by construction.

What this set does eliminate is *answer-level* contamination, which is the route
that actually inflates scores:

| contamination route | eliminated? | how |
| --- | --- | --- |
| answer-level: evaluator knows the expected answer text | **yes** | there is no expected prose; ground truth is a number computed by a reference function, or a hidden unit test |
| topic-level: evaluator chose subjects the training data covers | no | declared, and bounded by including out-of-scope items |

A model cannot benefit from having memorised a mechanism's wording, because no
wording is scored. It must produce a correct number or working code.

To bound the topic route, 14 of 63 items cover quantities that **no v0.4
mechanism addresses** and are flagged `in_v04_scope: false`: speculative decoding
break-even, Little's law, MIG partitioning, Amdahl's bound, and one code task.
Scores can be reported for the in-scope and out-of-scope partitions separately,
and a fine-tune that only improves in-scope is showing topic transfer rather than
capability.

## Composition

| | |
| --- | --- |
| items | 63 |
| numeric | 55 |
| code | 8 |
| outside every v0.4 mechanism | 14 |
| seed | 20260903 |
| sha256 | `dfd600d74c5d4b357823883010b40710043795202376a9304a38a1fcd0013cab` |

Numeric quantities: KV bytes per token (6), KV per request in GiB (5), weight GiB
per device (5), paged blocks (5), all-reduce bytes per token (4), pipeline bubble
fraction (4), effective batch (4), nearest-rank percentile index (4), speculative
decoding break-even (4), Little's law (3), MIG slices (3), Amdahl bound (3),
availability product (3), read amplification (2).

Parameters are drawn from a fixed seed, not chosen by hand, so the specific
numbers were not selected to suit any model.

## Scoring

- **Numeric**: the model must end with `ANSWER: <number>`. Scored against the
  reference value within a stated relative tolerance, exact for integer
  quantities.
- **Code**: the model returns one Python function. Scored by executing hidden
  tests it never sees.

Neither scorer looks at prose, so a terse model and a verbose model are treated
identically. This is the property exp-005 lacked.

## Ground-truth validation

`selftest_holdout.py` runs before any model is scored and checks:

1. every numeric answer is reproduced by an **independent recomputation** parsed
   from the question text, not from the builder's helper functions;
2. every code item's hidden tests **pass against a correct reference solution**,
   so a model that solves the stated task cannot be failed by a broken test.

Result: `SELFTEST_PASS`.

The self-test earned its place immediately. It flagged four `nearest_rank_index`
items as mismatched. Investigation showed the stored answers were correct and the
*checker* was wrong: its positional number extraction was picking up `1` from
"indexed from 1." and "1-based" alongside the real operands. Fixed by locating
both operands with semantic markers rather than by position. Had the self-test not
existed, the same fragile parsing could equally have produced wrong stored
answers, and every model would have been ranked against them.

## Results

### Two runs: the first measured the generation cap, not capability

The first run used `max_tokens=1024`. It produced base 57.1%, exp-002 63.5%,
exp-004 61.9%, and those numbers are void:

| arm | truncated | correct among finished | correct among truncated |
| --- | --- | --- | --- |
| base | 49 of 63 | 14 of 14 (100%) | 22 of 49 |
| exp-004 | 33 of 63 | 29 of 30 (97%) | 10 of 33 |

Models that finished were essentially always right. The cap was measuring whether
a model could finish inside 1024 tokens while doing chain-of-thought arithmetic.
Worse, the truncation rate differed per arm (49 / 43 / 33), so the harness
systematically favoured the least verbose model, which happened to be exp-004.

This is `generation_cap_truncation`, a mechanism in the v0.4 corpus, reproduced by
its own author while evaluating it. The capped results are kept in
`results_cap1024/` as evidence of the defect.

Re-run at `max_tokens=12288`, `max_model_len=16384`.

### Clean results (63 items, greedy, identical settings)

| arm | correct | accuracy | 95% Wilson interval | truncated |
| --- | --- | --- | --- | --- |
| base | 59 | 0.937 | [0.848, 0.975] | 13 |
| exp002_step75 | 61 | **0.968** | [0.891, 0.991] | 9 |
| exp004_step170 | 60 | 0.952 | [0.869, 0.984] | 9 |

| arm | numeric | code | in-scope (49) | out-of-scope (14) |
| --- | --- | --- | --- | --- |
| base | 0.927 | 1.000 | 0.939 | 0.929 |
| exp002_step75 | 0.982 | 0.875 | 0.959 | 1.000 |
| exp004_step170 | 0.945 | 1.000 | 0.939 | 1.000 |

### Paired comparisons (McNemar, exact, on identical items)

| comparison | disagreements | exact two-sided p | verdict |
| --- | --- | --- | --- |
| base vs exp002_step75 | 6 | 0.6875 | not distinguishable |
| base vs exp004_step170 | 7 | 1.0000 | not distinguishable |
| exp002_step75 vs exp004_step170 | 5 | 1.0000 | not distinguishable |

**No pair is distinguishable.** The arms disagree on 5 to 7 items out of 63, split
almost evenly in both directions.

## The set cannot answer the question it was built for

The base model scores 93.7%. That leaves **4 items** of headroom in the whole set.
A perfect model could gain 6.3 points over base, and any real difference between
two fine-tunes of the same base is far smaller than that.

Detecting a 95%-versus-90% difference at 80% power needs roughly **434 items per
arm**. This set has 63.

So the correct reading is not "the fine-tunes did not help". It is: **this
instrument cannot resolve differences of the size that plausibly exist here.** The
items are too easy for a 9B model that already does this arithmetic competently.

## What was learned anyway

1. **exp-005's benchmark ranking does not reproduce under objective scoring.**
   There, exp-002 led exp-004 with a tight interval on 8 of 10 categories. Here,
   with style removed from the measurement, the two are indistinguishable. That is
   consistent with the exp-005 gap having been style agreement rather than
   capability, which was the hypothesis stated there and could not be tested.
2. **Both fine-tunes score 100% on the out-of-scope partition** against base's
   92.9%, on 14 items. That is the direction a genuine transfer effect would take,
   but 14 items and a single-item base deficit cannot support the claim.
3. **Neither fine-tune broke anything.** No arm regressed materially anywhere.

## Status

Provisional and underpowered. No capability ranking is supported by these data.

## What a useful next version needs

- **Harder items.** The current numeric tasks are single-formula substitutions. A
  discriminating set needs multi-step problems where an intermediate error
  propagates, and items with a distractor that a plausible wrong model would take.
- **More items.** Several hundred per arm, not 63.
- **An author who did not write the training corpus.** Topic-level contamination
  remains unaddressed and I cannot fix it myself.

