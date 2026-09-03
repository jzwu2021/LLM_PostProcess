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

## Status

Built and hash-pinned. **No model has been run against it yet**, and the hash is
committed before any results exist so the set cannot be quietly adjusted to suit
an outcome.

## Next

Export the exp-002 step-75 and exp-004 step-170 FSDP checkpoints to HF format,
generate under identical decoding settings for base, exp-002 and exp-004, and
score. Report in-scope and out-of-scope partitions separately.
