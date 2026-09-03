# Experiment exp-005-20260901: cross-evaluation of exp-002 and exp-004 checkpoints

Compares the exp-002 and exp-004 fine-tunes against the base model, on every
evaluation set available, with the bias of each cell declared.

## Why the design is a matrix and not a single number

No unbiased evaluation set exists for this comparison:

| set | bias toward exp-002 | bias toward exp-004 |
| --- | --- | --- |
| `teacher_b_native_heldout` (100) | style-native | none |
| `aie_v04_heldout` (300) | none | style-native |
| `benchmark.jsonl` (499) | **contaminated by authoring** | 38/41 topics covered, items clean |

exp-002's own record states its repair data "must not be evaluated on the public
benchmark that informed their design". exp-003 established that exp-004 has no
item or answer overlap with the benchmark but covers 38 of its 41 topics.

So every arm runs on every set and the whole matrix is reported. A model winning
on its own set is a sanity check. The informative cells are off-diagonal.

## Harness validation

The standalone evaluator reproduced both original training-time numbers exactly:

- exp-002 step-75 on its own held-out set: **2.379644**, matching exp-002's log;
- exp-004 step-170 on its own held-out set: 1.706882 on the same 100-example
  slice used during training, matching exp-004's log.

## Correction made during this experiment

The first run evaluated the first 100 benchmark records. The benchmark is ordered
by category, so that slice was Knowledge 49 + Concept Understanding 50 +
Calculation 1, and **seven of ten categories were never evaluated**. The reported
benchmark column was a knowledge-and-concept result mislabelled as a benchmark
result.

Re-run over all 499 records and all 300 exp-004 held-out records. Base benchmark
loss moved from 2.733 on the truncated slice to 3.094 on the full set, so the
unevaluated categories were the harder ones and the original slice flattered every
arm. Both result sets are kept: `results_first100/` and `results_full/`.

## Results

### Full matrix (masked loss)

| model | exp002_heldout (100) | exp004_heldout (300) | benchmark (499) |
| --- | --- | --- | --- |
| base | 2.738953 | 3.099224 | 3.094407 |
| exp002_step75 | **2.379644** | 2.795483 | **2.616479** |
| exp004_step170 | 2.586446 | **1.721704** | 2.668712 |

### Change from base

| set | exp002_step75 | exp004_step170 |
| --- | --- | --- |
| exp002_heldout | −0.359 (home) | −0.153 |
| exp004_heldout | −0.304 | −1.378 (home) |
| benchmark | −0.478 (contaminated) | −0.426 |

**Both fine-tunes improved on all three sets relative to base, including each
other's held-out set.** Neither run produced a measurable capability regression in
the other's direction. That is the clearest result here and it is symmetric.

### Benchmark by category, with paired 95% intervals

| category | n | base | exp002 | exp004 | diff (e4−e2) | 95% interval | call |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Architecture Comparison | 50 | 3.8014 | 3.3076 | 3.4143 | +0.1067 | [+0.0845, +0.1288] | exp002 |
| Calculation | 50 | 1.2570 | 0.9657 | 0.9613 | −0.0044 | [−0.0248, +0.0160] | **tie** |
| Code | 50 | 2.6986 | 2.3250 | 2.3249 | −0.0001 | [−0.0130, +0.0128] | **tie** |
| Concept Understanding | 50 | 2.6486 | 2.0550 | 2.1104 | +0.0553 | [+0.0258, +0.0849] | exp002 |
| Knowledge | 49 | 2.8466 | 2.3735 | 2.3945 | +0.0210 | [−0.0101, +0.0520] | **tie** |
| Long-form Technical Analysis | 50 | 3.4850 | 2.9766 | 3.0349 | +0.0583 | [+0.0375, +0.0792] | exp002 |
| Performance Analysis | 50 | 3.4653 | 2.9111 | 3.0250 | +0.1138 | [+0.1002, +0.1275] | exp002 |
| Reasoning | 50 | 3.5773 | 3.2241 | 3.2769 | +0.0528 | [+0.0414, +0.0642] | exp002 |
| System Design | 50 | 3.6021 | 2.9726 | 3.0396 | +0.0670 | [+0.0436, +0.0904] | exp002 |
| Troubleshooting | 50 | 3.5572 | 3.0486 | 3.0998 | +0.0513 | [+0.0381, +0.0644] | exp002 |

exp-002 is ahead on seven categories, and three are ties. exp-004 is not ahead on
any category at this sample size.

The three ties are **Calculation, Code and Knowledge**. Calculation and Code are
the two categories whose benchmark verifiers are objective (`numeric_tolerance`
and `unit_test`) rather than rubric-based, so they are the categories where answer
style matters least. That the two models are indistinguishable exactly there, and
separated everywhere the verifier is a style-sensitive rubric, is consistent with
the whole benchmark column measuring style agreement rather than capability. It is
a hypothesis this experiment cannot test, not a conclusion.

### Paired comparison across whole sets

| set | n | mean diff (e4 − e2) | 95% interval | items where exp004 lower | verdict |
| --- | --- | --- | --- | --- | --- |
| exp002_heldout | 100 | +0.2068 | [+0.1850, +0.2286] | 1 of 100 | exp002 lower |
| exp004_heldout | 300 | −1.0738 | [−1.1064, −1.0412] | 300 of 300 | exp004 lower |
| benchmark | 499 | +0.0522 | [+0.0449, +0.0596] | 126 of 499 | exp002 lower |

The home-set results are near-total: exp-004 is lower on 300 of 300 of its own
items, exp-002 on 99 of 100 of its own. That is what fitting a distinctive answer
style looks like, and it is why these two columns carry no capability information.

## Conclusions

1. **Neither fine-tune damaged the other's domain.** Both improved on all three
   sets against base. This is the one clean, symmetric finding.
2. **exp-002 has lower loss on the benchmark**, by 0.052 with a tight interval.
   This cannot be read as better capability: the benchmark is contaminated for
   exp-002 by authoring, and the advantage disappears on exactly the categories
   with objective verifiers.
3. **exp-004's 45% home improvement is not evidence of superiority.** The v0.4
   corpus is uniform by construction, so it is easy to fit. A large home gain
   measures corpus regularity.
4. **The comparison remains undecided on capability.** Loss compares agreement
   with a target text, and the two models were trained toward different answer
   styles.

## What would settle it

Generation against style-independent verifiers, on a set that neither arm's
authoring process saw. The benchmark's `numeric_tolerance` (50) and `unit_test`
(45) items are the right kind of instrument but the wrong provenance for exp-002.
A newly authored, hash-pinned private holdout is required, which is the same gate
exp-002's own record demanded and which still does not exist.

## Status

Provisional. Loss-only. No capability claim is made or supported.
