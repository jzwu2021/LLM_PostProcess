# Experiment exp-004-20260901: fine-tune Qwen3.5-9B on the v0.4 AI-infra corpus

First training run against `research/ai-infra-expert/corpus_v04`, the 3000-record
mechanism-based corpus built to replace the v0.3 corpus whose 5399 rows contained
only 522 distinct questions.

## Gate that was deliberately skipped

Domain-expert review of the corpus content was **not performed**. The corpus is
verified for uniqueness, structure and contamination, and not for technical
correctness. Every record carries `review_status: needs_domain_expert_review`.

This run therefore tests whether the pipeline produces a trainable, measurable
result. It cannot support a claim that the model learned correct AI-infrastructure
knowledge, because nobody has established that the corpus contains any.

## Design decisions, and why

Three of these are taken directly from mechanisms the corpus itself teaches. The
corpus was used as a checklist against its own training run.

### Held-out split removes whole mechanisms, not random rows

A random row split would place the same mechanism in both halves. Held-out loss
would then fall because the model memorised content present on both sides, which
is the `heldout_shares_repetition` failure. The split holds out one mechanism per
topic:

| | |
| --- | --- |
| train | 2700 records, 90 mechanisms |
| held out | 300 records, 10 mechanisms |
| shared mechanisms | 0 |
| shared questions | 0 |

Held-out mechanisms: `cache_transfer_budget`, `checkpoint_vs_artifact_size`,
`dependency_availability_ceiling`, `kv_quant_quality_cost`, `max_len_overreservation`,
`sampling_destroys_tail`, `step_count_vs_coverage`, `tool_call_trust_boundary`,
`tp_allreduce_cost`, `utilisation_target_conflict`.

Held-out loss here therefore measures transfer to **unseen mechanisms**, which is
a harder and more informative quantity than a random split would give.

### The training file is shuffled

The trainer consumes records by index with no shuffling:
`index = ((step-1) * world * grad_accum + rank * grad_accum + micro) % len(records)`.
An unshuffled corpus grouped by topic would train the early steps on serving and
the late steps on reliability. This is the `data_ordering_reachability` failure
that made the exp-002 repair records unreachable. The split script shuffles with a
fixed seed.

### Sequence length verified against measured token lengths

`training_sequence_truncation` says a sequence length below the data length
teaches truncated answers. Measured with the actual tokenizer on the training
split:

| statistic | tokens |
| --- | --- |
| min | 252 |
| p50 | 348 |
| p90 | 414 |
| p99 | 439 |
| max | 510 |

At `--seq-len 768`, **0 of 2700 records are truncated**. The value was verified
rather than inherited from exp-002.

### Step count covers whole epochs

`step_count_vs_coverage` says a step count is not a coverage figure. Records per
optimiser step is `world 8 x grad_accum 4 = 32`, so one epoch is
`ceil(2700 / 32) = 85` steps. `--max-steps 170` is exactly two epochs, so every
record is seen twice. exp-002's 75 steps covered less than one epoch of its
corpus, which is why its repair records were never consumed.

## Configuration

| parameter | value | basis |
| --- | --- | --- |
| model | Qwen3.5-9B | unchanged |
| method | FSDP masked SFT, 8x A30 | unchanged |
| train | `data/aie_v04_train.jsonl` (2700) | mechanism-disjoint split |
| eval | `data/aie_v04_heldout.jsonl` (300), 100 examples | mechanism-disjoint |
| max-steps | 170 | 2 epochs |
| save-steps | 34 | 5 checkpoints |
| seq-len | 768 | measured max 510 |
| grad-accum | 4 | unchanged from exp-002 |
| lr | 2e-6 | unchanged from exp-002, so data is the only changed variable |

Learning rate is deliberately held at the exp-002 value. Changing the corpus and
the learning rate together would make any difference unattributable.

## Run history

### Attempt 1 — killed by SIGHUP at step ~33, no checkpoint

Started 11:13:36, died 11:26:19 after roughly 33 of 170 steps. Log preserved as
`artifacts/attempt1_sighup.out`.

The cause was not a training fault. Loss was falling normally (3.157 → 2.634 at
step 30) and device memory was stable at 22.3 of 24 GB. The run received
`signal: 1`:

```
Received 1 death signal, shutting down workers
torch.distributed.elastic.multiprocessing.api.SignalException: Process 1292490 got signal: 1
```

The wrapper script was launched under `nohup`, which sets the wrapper itself to
ignore SIGHUP, but `torchrun` spawns an elastic agent and eight workers that
remained in the launching shell's session. When that session was torn down, the
signal reached the process group and the agent shut the workers down.

Nothing was salvageable: the first checkpoint was due at step 34 and the run died
just before it, so 13 minutes of eight-GPU time produced no artifact. Workers also
did not respond to SIGKILL for several minutes, which is the "wedged on GPU"
teardown path; the devices did release cleanly afterwards.

**Fix:** launch with `setsid` so the whole job runs in its own session with no
controlling terminal. Verified after relaunch: worker session id 1297443 against
shell session id 1297331, TTY `?`.

**Lesson to carry:** `nohup` protects the process it wraps, not a process tree
that creates its own group. For any multi-process launcher the detachment has to
be at session level. A cheaper mitigation would also have helped: the first
checkpoint at step 34 was too late to protect a 13-minute investment.

### Attempt 2 — relaunched detached

Started 11:31. Same configuration, same dataset hashes, `setsid`-detached.

## Results

Pending. To be filled from `artifacts/train_aie_v04.log`.

## What this run can and cannot support

Can support:
- whether the corpus trains without truncation, divergence or memory failure;
- whether held-out loss on unseen mechanisms moves.

Cannot support:
- any claim about domain capability, because the corpus is unreviewed;
- any comparison against exp-002, whose corpus, split policy and coverage all differ;
- any benchmark result described as generalisation, since exp-003 established
  that 38 of 41 benchmark topics correspond to training mechanisms.

## Status

Provisional.
