import json, os

EXP = "experiments/2026-08-17-teacher-b-corpus-review"
START, N = 1740, 10
OUT = os.path.join(EXP, "results", "train-batch-0175.jsonl")

rows = [json.loads(l) for l in open("research/ai-infra-expert/corpus/train.jsonl")]
sl = rows[START:START + N]

A = {}

A["corpus-01919"] = """Scope: this is a *design* answer for a weight-only quantization (WOQ) cost study. Nothing below is MEASURED on your fleet; every number is an ESTIMATE with its derivation shown.

Assumptions (write these into the test plan, and fail the study if any is violated)
- One frozen BF16 baseline checkpoint; WOQ variants derive from that exact checkpoint hash.
- Weight-only means activations stay BF16/FP16; only weight tensors are stored at 4- or 8-bit with per-group scales (typical group size 64 or 128). KV cache precision is a *separate* knob and must be held fixed, otherwise you are measuring two changes.
- Serving stack, kernel versions, scheduler settings, max_num_seqs, max_model_len, and prompt set are pinned and recorded.

Falsifiable hypothesis
H1: "At equal p95 end-to-end latency SLO, INT4-WOQ raises sustainable output tokens/s per GPU by >=1.5x versus BF16, while task quality drops by <=1.0 point absolute on the held-out eval."
H1 is rejected if either the throughput gain is <1.5x or the quality drop exceeds 1.0 point with 95% CI excluding zero-drop.

Mechanism, so you know where the win can and cannot come from
- WOQ shrinks *weight bytes moved per token*. In the decode phase, small-batch decoding is weight-bandwidth bound, so time per token scales roughly with weight bytes / HBM bandwidth. ESTIMATE: a 9B model at BF16 is ~18 GB of weights; at INT4 with group scales it is ~5.3 GB (9e9 * 0.5 B/param = 4.5 GB plus ~0.8 GB of scales/zeros at group 128 in FP16). Derivation is byte counting only, no kernel efficiency assumed.
- That predicts a decode-side ceiling of ~3.4x (18/5.3) *only* in the memory-bound, batch-1 limit. As batch size grows, GEMMs become compute bound and the dequantize step adds arithmetic, so the realized gain collapses. This is why single-stream benchmarks systematically overstate WOQ value.
- Prefill is compute bound; expect little or negative TTFT gain, because dequantization is extra work.

Fair-comparison protocol
1. Fix the request trace (same prompts, same output-length distribution, same arrival process). Replay it, do not generate fresh sampling.
2. Sweep concurrency for both arms and build a latency-throughput curve. Compare arms *at equal SLO attainment*, not at equal batch size. Comparing at fixed batch size is the most common way these studies lie.
3. Report TTFT p50/p95, TPOT p50/p95, output tok/s per GPU, and GPU-seconds per 1k output tokens (the actual cost unit).
4. Quality: run the same eval suite with greedy decoding and a fixed seed; report per-task scores with bootstrap 95% CIs. Add a long-context and a code/math slice, since WOQ damage is not uniform.
5. Memory: record steady-state weight bytes, and how much of the freed HBM you actually converted into KV cache blocks. If you do not raise KV capacity, you did not spend the savings.

Confounders to control explicitly
- Kernel maturity: an INT4 path may fall back to a slow dequant-GEMM for some shapes or tensor-parallel degrees. Log which kernel each layer selected.
- Calibration set leakage: calibration data must be disjoint from the eval set; otherwise quality is inflated.
- Group size and outlier handling change both quality and speed; they are part of the arm definition, not free parameters to tune after seeing results.
- Autotuning/warmup: discard the first N requests, and report steady state only.

Evidence to collect before any rollout decision
- Latency-throughput curves for both arms, raw traces retained.
- Quality table with CIs and per-slice breakdown.
- Kernel selection logs and HBM occupancy snapshots.
- A cost calculation in GPU-hours per million output tokens, derived from measured throughput, not from the byte-count ESTIMATE above.

Rollback gates
- Roll back if p95 TTFT or p95 TPOT regresses beyond the SLO at the target concurrency.
- Roll back if any eval slice drops more than 1.0 point absolute, or if refusal/format-compliance rates shift by more than 2 points.
- Roll back if measured cost per million output tokens does not improve by at least 25% at equal SLO; below that margin the operational risk of a second numeric path is not worth it.

Uncertainty I will not paper over: whether your specific kernels hit the memory-bound regime at your production concurrency is unknown until measured. The byte-count argument sets an upper bound, nothing more."""

A["corpus-01920"] = """Treat "WOQ is cheaper" as a diagnosis problem, not a procurement decision. The failure mode I see most often is a team that already shipped INT4 and cannot explain why cost per token barely moved. Here is how to design the comparison so it can actually catch that.

Assumptions to pin first
- Same base checkpoint (record the hash), same tokenizer, same serving engine build, same TP/PP layout.
- Weight-only: activations and KV cache stay at the baseline precision. If you also switched KV to FP8, stop; that is a confounded experiment and must be split into two arms.
- The workload trace is captured from production, not synthesized, and is replayed identically to both arms.

Falsifiable hypothesis
H1: "The observed cost-per-token gap between BF16 and INT4-WOQ is explained by decode-phase weight bandwidth; therefore the gap shrinks monotonically as concurrency rises and approaches zero once decode GEMMs become compute bound."
If measured gain is instead *flat* across concurrency, H1 is wrong and the difference is coming from somewhere else (memory headroom enabling larger KV, scheduler behaviour, or a kernel regression in the baseline).

Controlled experiment
- Arms: A = BF16 baseline, B = INT4-WOQ, identical everything else. Optionally C = INT8-WOQ as a middle point; a monotone A>C>B ordering in decode speed is a useful sanity check, and its absence is itself a finding.
- Sweep concurrency across at least 5 points spanning memory-bound to compute-bound. For each point, run to steady state, discard warmup, and record TTFT p50/p95, TPOT p50/p95, throughput, queue depth, and GPU utilization.
- Force KV cache capacity to be *equal* in one sub-experiment (so you isolate the bandwidth effect) and *unequal/native* in another (so you see the practical benefit of freed HBM). Reporting only the second one is how teams accidentally attribute a batching win to quantization.

What each outcome tells you
- Large gain at low concurrency that vanishes at high concurrency: consistent with H1; the win is real only if you actually serve at low concurrency.
- Gain that persists at high concurrency: suspect the extra KV blocks are doing the work, or the BF16 arm is misconfigured. Check preemption/recompute counters before celebrating.
- No gain anywhere: likely a dequant-GEMM fallback for your shapes. Inspect kernel selection logs and per-layer timings.

Quality regression checks that must run in the same experiment
- Fixed-seed greedy eval on a held-out set disjoint from the calibration set, with bootstrap 95% CIs.
- Separate slices for long context, structured output, and math/code; WOQ damage concentrates in outlier-heavy layers and shows up unevenly.
- Format-compliance and refusal-rate deltas, because those break downstream parsers before eval scores look bad.

Numbers, labeled honestly
- ESTIMATE: weight bytes drop from ~2 B/param to ~0.5 B/param plus per-group scales; at group 128 in FP16 that is ~0.06 B/param of overhead, so roughly a 3.4x reduction in weight bytes. Derivation: pure byte counting.
- ESTIMATE: therefore decode-phase upper bound speedup ~3.4x at batch 1, and this bound is not attainable in practice because dequant adds arithmetic and kernels are less tuned.
- Everything else must be MEASURED; do not quote vendor numbers as if they were your fleet.

Rollback gates
- Any eval slice down >1.0 point absolute with CI excluding zero, or format-compliance down >2 points.
- p95 TPOT above SLO at target concurrency.
- Cost per million output tokens not improved by >=25% at equal SLO attainment.
- Any increase in preemption or recompute events that is not explained and reproducible.

Retain raw traces and kernel logs; a cost claim without the underlying latency-throughput curve is not reviewable."""

A["corpus-01921"] = """Framing this as performance analysis: the question "does WOQ lower serving cost" is only answerable as a curve, never as a single number. A single-point comparison is the defect to design out.

Cost unit
Use GPU-seconds per 1,000 output tokens at a fixed SLO. Tokens/s alone hides the latency price you paid, and $/hour hides hardware differences. Everything below feeds that unit.

Assumptions
- Identical checkpoint hash, engine build, TP degree, max_model_len, scheduler policy, and replayed request trace.
- Weight-only quantization; activation and KV precision held at baseline. KV precision changes get their own arm.
- Steady state only; warmup requests excluded and the exclusion rule recorded.

Falsifiable hypothesis
H1: "INT4-WOQ reduces GPU-seconds per 1k output tokens by >=30% at the p95 TPOT SLO, and the reduction is largest at low concurrency."
Rejected if the measured reduction at the SLO-satisfying operating point is <30%, or if the concurrency trend is flat/inverted (which would mean the mechanism is not weight bandwidth).

Roofline reasoning that bounds what you can expect
- Decode at small batch is weight-bandwidth bound: time/token ~= weight_bytes / HBM_BW. ESTIMATE for a 9B model: BF16 weights ~18 GB; INT4 with group-128 FP16 scales ~5.3 GB. Ratio ~3.4x. Derivation: parameter count times bytes per parameter, plus scale overhead. No kernel efficiency assumed.
- ESTIMATE, illustrative only: on a device with ~933 GB/s HBM bandwidth, an 18 GB weight sweep implies ~19 ms/token at batch 1 at 100% bandwidth efficiency; real efficiency is typically well below that, so treat it as a floor on latency, not a prediction.
- Prefill and large-batch decode are compute bound; dequantization adds FLOPs and often *costs* time there. Expect TTFT to be flat or worse.
- Therefore the honest expectation is: big win at low concurrency, shrinking win as you batch, possible loss in prefill-heavy traffic.

Measurement protocol
1. Sweep concurrency over >=5 points per arm; at each point collect TTFT p50/p95, TPOT p50/p95, output tok/s, GPU util, HBM occupancy, KV block utilization, preemption count.
2. Plot latency vs throughput per arm; read off throughput at the SLO. Compare arms *only* at that intersection.
3. Repeat each point at least 3 times with different trace seeds; report medians and spreads. Single runs are not evidence.
4. Split prefill-dominant and decode-dominant traces; report separately. A blended number conceals a prefill regression.
5. Record per-layer kernel selection and timings so a null result can be attributed to a fallback rather than to the method.

Quality arm, run concurrently
Greedy, fixed seed, held-out eval disjoint from the calibration set; bootstrap 95% CIs; slices for long context, structured output, code, and math. Report absolute deltas, not relative.

Confounders
Calibration-set contamination; group size and outlier handling silently differing between builds; autotune cache warm in one arm and cold in the other; KV capacity differing because of freed HBM (isolate with an equal-KV sub-experiment); CPU-side tokenization or logging overhead dominating at low concurrency.

Evidence required for sign-off
Latency-throughput curves with repeats, the equal-KV isolation run, kernel selection logs, quality table with CIs, and the derived GPU-seconds per 1k output tokens for both arms.

Rollback gates
Revert if p95 TPOT or p95 TTFT breaches SLO at target concurrency; if any eval slice regresses >1.0 point absolute with CI excluding zero; if cost per 1k output tokens improves by <25%; or if variance across repeats exceeds the claimed effect size, which means the study is underpowered rather than positive."""

A["corpus-01922"] = """A fair WOQ comparison is mostly an exercise in refusing to let two variables move at once. Below is the design, with the trap listed next to each control.

Arm definition (freeze before any run)
- Baseline A: BF16 weights, BF16 activations, baseline KV precision.
- Candidate B: INT4 weight-only, same activations, same KV precision, group size and outlier policy explicitly recorded as part of the arm identity.
- Trap: teams often ship "INT4 + FP8 KV" as one change. That is two experiments; the KV change usually explains most of the throughput delta and none of the quality delta gets attributed correctly.

Falsifiable hypothesis
H1: "At matched p95 latency SLO and matched KV cache capacity, INT4-WOQ improves output tokens/s per GPU by >=1.5x with <=1.0 point absolute quality loss on held-out evals."
Two rejection paths, both pre-registered: throughput below 1.5x, or quality loss above 1.0 point with 95% CI excluding zero.

Why matched KV capacity matters
Freed HBM lets the scheduler hold more sequences, which raises throughput independently of any kernel speedup. If you compare at native (unequal) KV capacity only, you cannot separate "quantization is fast" from "we got more batching". Run both: an equal-KV arm to isolate the bandwidth mechanism, and a native-KV arm to quantify the deployable benefit. Report them as separate claims.

Workload and statistics
- Replay a captured production trace; hold the arrival process and output-length distribution fixed.
- >=5 concurrency points per arm, >=3 repeats per point, warmup discarded by a fixed rule.
- Compare at the SLO intersection of the latency-throughput curve, never at fixed batch size.
- Report medians plus spread; if run-to-run spread is comparable to the effect, the result is "underpowered", not "positive".

Quality evaluation
Greedy decoding, fixed seed, held-out set strictly disjoint from calibration data. Slice by long context, structured/JSON output, code, math, and safety/refusal behaviour. Bootstrap 95% CIs, absolute deltas. Also diff token-level outputs on a sample to catch degenerate repetition that aggregate scores miss.

Numbers, labeled
- ESTIMATE: weight footprint for a 9B model goes from ~18 GB (2 B/param) to ~5.3 GB (0.5 B/param + ~0.06 B/param of FP16 group-128 scales), a ~3.4x reduction. Derivation: byte counting.
- ESTIMATE: that 3.4x is an upper bound on decode-phase speedup in the memory-bound limit and is not achievable once dequant arithmetic and kernel inefficiency are included.
- All latency, throughput, and quality figures in the report must be MEASURED and traceable to a raw log.

Confounders checklist
Kernel fallbacks for unsupported shapes or TP degrees; autotune cache state; calibration leakage; differing tokenizer or prompt templates; logging/telemetry overhead asymmetry; thermal or clock drift across long runs (record clocks).

Evidence required
Frozen arm manifests with checkpoint hashes, latency-throughput curves with repeats, equal-KV and native-KV results reported separately, kernel selection logs, quality tables with CIs, sampled output diffs, and the final GPU-seconds per 1k output tokens.

Rollback gates
Breach of p95 TTFT/TPOT SLO at target concurrency; any eval slice down >1.0 point absolute with CI excluding zero; format-compliance down >2 points; cost improvement <25%; or unexplained preemption/recompute increases. Any single gate trips a revert to arm A, which stays deployable and warm for the duration of the study."""

A["corpus-01923"] = """Diagnostic posture: assume the WOQ deployment will produce an ambiguous result, and build the comparison so the ambiguity is resolvable afterwards from logs you already collected.

Preconditions
- Checkpoint hash identical across arms; quantized weights derived from that exact checkpoint, not from a re-download.
- Weight-only; activations and KV precision frozen at baseline values. Record them explicitly in the run manifest so a later reader cannot assume.
- Same engine build, TP/PP layout, scheduler config, max_num_seqs, max_model_len, and replayed trace.

Falsifiable hypothesis
H1: "Any throughput advantage of INT4-WOQ at production concurrency is attributable to increased KV cache capacity rather than to faster decode GEMMs."
This is deliberately the *skeptical* hypothesis. It is rejected if the equal-KV sub-experiment still shows >=1.3x decode throughput at matched SLO. Pre-registering the skeptical direction is what keeps the study from becoming a press release.

Experiment matrix
- A: BF16, native KV capacity.
- B: INT4-WOQ, native KV capacity (deployable configuration).
- B': INT4-WOQ, KV capacity clamped to A's block count (mechanism isolation).
- Optional C: INT8-WOQ as an ordering check.
Run all at >=5 concurrency points, >=3 repeats, steady state only.

Signals to capture per run
TTFT p50/p95, TPOT p50/p95, output tok/s, GPU util, achieved HBM bandwidth if available, KV block occupancy, preemption/recompute counts, per-layer kernel names and timings, and clock/thermal state. The kernel names matter: a null result with a dequant-GEMM fallback in the logs is a tooling finding, not a verdict on quantization.

Failure modes this design catches
- Silent kernel fallback for odd hidden sizes or TP degrees.
- Throughput "win" that is entirely batching, revealed by B' collapsing to A.
- Quality loss concentrated in a slice (long context, JSON, math) that a blended average hides.
- Calibration leakage inflating eval scores; prevented by disjointness check on the calibration and eval sets, verified by hashing.
- Prefill regression masked by a decode-heavy trace; prevented by reporting prefill-dominant and decode-dominant traces separately.

Quantities, labeled
- ESTIMATE: ~3.4x weight-byte reduction for a 9B model going BF16 to INT4 with group-128 FP16 scales (18 GB to ~5.3 GB). Derivation: parameters times bytes per parameter plus scale overhead.
- ESTIMATE: freed HBM of ~12.7 GB converts to additional KV blocks; the exact sequence-count gain depends on layers, KV heads, head dim, and block size, so compute it from your config rather than assuming a multiplier.
- Everything reported to stakeholders as a benefit must be MEASURED.

Evidence required before rollout
Run manifests with hashes, the A/B/B' comparison at matched SLO, kernel logs, quality tables with bootstrap CIs and per-slice breakdown, calibration/eval disjointness proof, and cost expressed as GPU-seconds per 1k output tokens.

Rollback gates
Revert to A if p95 TPOT or TTFT breaches SLO; if any quality slice drops >1.0 point absolute with CI excluding zero; if format compliance drops >2 points; if measured cost improvement is <25%; or if B' shows the mechanism is purely batching and the batching gain can be obtained more safely by tuning KV settings on the BF16 arm. That last case is a real outcome and should be allowed to win."""

A["corpus-01924"] = """Performance-analysis view, organised around where the time actually goes.

Phase decomposition first
Split every request into prefill and decode and measure them separately. WOQ acts almost entirely on decode-phase weight traffic. If your trace is prefill-dominant (short outputs, long prompts), the achievable win is small by construction, and no amount of tuning changes that. Establish the phase mix before running arms; it determines whether the study is worth doing.

Assumptions
Identical checkpoint hash and engine build; weight-only quantization with activations and KV precision frozen; pinned scheduler settings; replayed production trace; steady-state windows only.

Falsifiable hypothesis
H1: "Decode-phase time per token scales with weight bytes moved; hence TPOT_INT4 / TPOT_BF16 approaches the weight-byte ratio at batch 1 and rises toward 1.0 as batch grows."
Rejected if the batch-1 TPOT ratio is far from the byte ratio (indicating kernel inefficiency or a fallback) or if the ratio does not trend toward 1.0 with batch (indicating the mechanism is not bandwidth).

Byte accounting, explicitly ESTIMATE
- BF16 9B weights: 9e9 params x 2 B = 18 GB.
- INT4 group-128 with FP16 scale and zero per group: 0.5 B/param + ~0.03 B/param scale + ~0.03 B/param zero ~= 0.56 B/param, so ~5.0-5.3 GB.
- Ratio ~3.2-3.6x. Derivation is arithmetic on parameter counts only; it assumes nothing about kernels and is an upper bound on decode speedup.
- Do not report this ratio as a throughput result. It is a hypothesis generator.

Instrumentation plan
- Per-phase timers, not just end-to-end latency.
- Achieved bandwidth and achieved FLOP/s per phase where the profiler supports it; a decode phase running far below peak bandwidth in the INT4 arm points at dequant overhead or a fallback kernel.
- Per-layer kernel selection dumps for both arms, diffed.
- KV block occupancy and preemption counters, so batching effects can be subtracted.

Comparison rules
- Build latency-throughput curves; compare at the SLO intersection.
- Run an equal-KV-capacity variant so the batching contribution is separable.
- >=3 repeats per point; report spread. Claim nothing whose effect size is smaller than the spread.
- Report GPU-seconds per 1k output tokens as the headline, with tokens/s and latency percentiles as supporting detail.

Quality control arm
Fixed-seed greedy eval on a held-out set hashed and verified disjoint from calibration data; per-slice results with bootstrap 95% CIs; explicit checks on structured-output compliance and repetition/degeneration rates.

Confounders
Prefill/decode mix drift between arms; autotune cache warmth; clock throttling on long runs; telemetry overhead; group size differences smuggled in by a library default change; calibration leakage.

Evidence required
Phase-split timings, kernel diffs, curves with repeats, equal-KV isolation, quality tables with CIs, and a written derivation for every ESTIMATE that appears in the summary.

Rollback gates
p95 TTFT or TPOT beyond SLO at target concurrency; any eval slice down >1.0 point absolute with CI excluding zero; degeneration rate up measurably on sampled diffs; measured cost improvement <25%; or profiler evidence that the INT4 arm is running a fallback path, in which case the correct action is to fix tooling and re-run rather than to accept or reject the method."""

A["corpus-01925"] = """Design brief: a WOQ evaluation that a skeptical reviewer would accept.

1. Freeze the comparand
Both arms start from one checkpoint hash. The quantized artifact records: bit width, group size, symmetric/asymmetric, outlier or mixed-precision policy, calibration dataset hash and size, and the library version that produced it. If any of these change, it is a new arm with a new ID, not a tweak.

2. Hold the second-order knobs constant
Activation precision, KV cache precision, block size, max_num_seqs, max_model_len, chunked-prefill setting, speculative decoding on/off, and TP/PP layout are all pinned and recorded. Weight-only means weight-only; a KV precision change belongs to a different study.

3. Pre-register the hypothesis
H1: "At the production p95 latency SLO, INT4-WOQ delivers >=1.5x output tokens/s per GPU with <=1.0 point absolute degradation on every held-out eval slice."
Rejection is defined before data collection: miss the throughput bar, or breach the quality bar on any slice with a 95% CI excluding zero.

4. Build curves, not points
Sweep concurrency across the memory-bound to compute-bound range, >=5 points, >=3 repeats each, steady state only. Read throughput at the SLO. Fixed-batch-size comparisons are inadmissible because the two arms have different optimal batch sizes.

5. Separate the two sources of gain
Run INT4 twice: once with KV capacity clamped to the BF16 arm's block count (isolates kernel/bandwidth effect), once with native capacity (shows deployable benefit). If the clamped run shows no gain, the honest conclusion is "we bought KV headroom", and the cheaper alternative of tuning KV on BF16 must be evaluated before shipping a second numeric path.

6. Quality with teeth
Greedy, fixed seed, held-out set proven disjoint from calibration by hash comparison. Slices: long context, structured output, code, math, multilingual if relevant, and safety/refusal behaviour. Bootstrap 95% CIs, absolute deltas, plus sampled side-by-side output diffs to catch degeneration that aggregates hide.

7. Numbers with provenance
ESTIMATE: 9B BF16 weights ~18 GB; INT4 group-128 with FP16 scales ~5.0-5.3 GB; ratio ~3.2-3.6x, derived purely by byte counting and valid only as an upper bound on decode speedup in the memory-bound limit. MEASURED values replace it in the final report; the estimate stays in the appendix as the prior that was tested.

8. Confounders logged, not assumed away
Kernel fallbacks (dump per-layer kernel names and diff the arms), autotune warmth, clock/thermal drift, telemetry overhead, prefill/decode mix drift, and library default changes between builds.

9. Evidence package for sign-off
Arm manifests with hashes, raw traces, curves with repeats, clamped-KV isolation, kernel diffs, quality tables with CIs, sampled diffs, and cost in GPU-seconds per 1k output tokens.

10. Rollback gates
Any SLO breach at target concurrency; any slice regression >1.0 point absolute with CI excluding zero; format compliance down >2 points; cost improvement <25%; or an unexplained increase in preemption/recompute. The BF16 arm stays warm and routable for the entire canary so rollback is a routing change, not a redeploy."""

A["corpus-01926"] = """Start from the complaint you will actually receive: "we quantized and the bill did not move." A comparison design that cannot diagnose that outcome is not worth running.

Root-cause hypotheses to keep alive from the start
- RC1: workload is prefill-dominant, so decode-side weight savings barely matter.
- RC2: production concurrency is high enough that decode GEMMs are compute bound, and dequant overhead cancels the bandwidth saving.
- RC3: the INT4 path silently falls back to a slow kernel for your hidden size or TP degree.
- RC4: the throughput gain exists but was never converted into cost because instance count and autoscaling policy did not change.
- RC5: the gain exists but SLO tightening forced lower concurrency, eating it.

Falsifiable hypothesis
H1: "Cost per million output tokens falls by >=25% after INT4-WOQ at unchanged SLO attainment."
Each rejection routes to a specific RC above, which is why the instrumentation below is mandatory rather than optional.

Instrumentation that discriminates between RC1-RC5
- Phase split (prefill vs decode tokens and time) per arm -> tests RC1.
- Concurrency sweep with TPOT ratio plotted against batch size -> tests RC2.
- Per-layer kernel name/timing dumps, diffed between arms -> tests RC3.
- Fleet-level accounting: instances, GPU-hours, and tokens served over the same window -> tests RC4.
- SLO attainment and admitted concurrency before/after -> tests RC5.

Arm hygiene
Identical checkpoint hash; weight-only, with activation and KV precision frozen and recorded; identical engine build, scheduler config, and replayed trace; quantization parameters (bits, group size, outlier policy, calibration hash) recorded as part of the arm identity. Also run an INT4 arm with KV capacity clamped to baseline, so batching gains are separable from kernel gains.

Statistics
>=5 concurrency points, >=3 repeats, steady state only, compare at the SLO intersection of the latency-throughput curve. Report medians and spread; an effect smaller than the spread is not an effect.

Quality gate running in parallel
Greedy, fixed seed, held-out eval hashed and verified disjoint from calibration data. Per-slice scores (long context, structured output, code, math, safety) with bootstrap 95% CIs, plus sampled output diffs for degeneration.

Numbers
ESTIMATE: weight bytes 18 GB -> ~5.0-5.3 GB for a 9B model at INT4 group-128 with FP16 scales, ~3.2-3.6x, derived by byte counting; an upper bound on decode speedup only, never a throughput claim. All cost figures reported to finance must be MEASURED from fleet accounting over a matched window, not extrapolated from a benchmark.

Evidence required
Phase-split metrics, kernel diffs, curves with repeats, clamped-KV run, fleet GPU-hour accounting, quality tables with CIs, and the calibration/eval disjointness proof.

Rollback gates
SLO breach at target concurrency; any slice down >1.0 point absolute with CI excluding zero; format compliance down >2 points; measured fleet cost improvement <25%; or kernel logs showing a fallback path, which means re-run after fixing tooling rather than concluding anything about the method. Keep the BF16 arm warm and routable throughout so rollback is a routing flip."""

A["corpus-01927"] = """The measurable claim is cost, so define cost first and let every metric hang off it.

Definition
Cost = GPU-seconds per 1,000 output tokens at the p95 latency SLO, converted to currency only at the last step. This makes the comparison invariant to instance pricing and forces the latency constraint into the number, which raw tokens/s does not.

Assumptions
One checkpoint hash for both arms; weight-only quantization with activation and KV precision frozen; identical engine build, TP layout, scheduler config; replayed production trace with fixed arrival process and output-length distribution; steady-state windows with a fixed warmup-exclusion rule.

Falsifiable hypothesis
H1: "INT4-WOQ reduces GPU-seconds per 1k output tokens by >=30% at the p95 TPOT SLO, and the reduction shrinks monotonically with concurrency."
Rejected on either clause: insufficient reduction at the SLO point, or a non-monotone/flat concurrency trend, which would falsify the bandwidth mechanism and require re-attribution.

Analytical prior, explicitly ESTIMATE
- BF16 9B weights: 18 GB (9e9 x 2 B). INT4 group-128 with FP16 scale+zero: ~0.56 B/param -> ~5.0 GB. Ratio ~3.6x. Derivation: byte counting, no kernel model.
- Memory-bound decode implies time/token proportional to weight bytes, so 3.6x is the batch-1 ceiling. ESTIMATE of realized gain: materially lower, because dequantization adds arithmetic and INT4 kernels are usually less tuned than BF16 GEMMs. I will not put a specific realized number here without measurement; doing so would be inventing a platform fact.

Measurement design
1. Phase-split timing (prefill vs decode) so a prefill-dominant workload cannot masquerade as a WOQ failure.
2. Concurrency sweep, >=5 points, >=3 repeats, steady state; build latency-throughput curves per arm.
3. Compare at the SLO intersection only.
4. Clamped-KV variant of the INT4 arm to isolate kernel effect from batching effect; report both numbers separately with different names so they cannot be conflated in a slide.
5. Per-layer kernel selection dumps, diffed, attached to the report.
6. Fleet-level GPU-hour accounting over a matched production window as the confirmatory measurement; benchmark results alone do not establish cost.

Quality, run in the same study
Fixed-seed greedy evaluation on a held-out set hash-verified disjoint from the calibration data. Per-slice results with bootstrap 95% CIs: long context, structured output, code, math, safety/refusal. Sampled output diffs for repetition and degeneration. Report absolute point deltas.

Confounders to log
Prefill/decode mix drift; autotune cache warmth; clock and thermal drift across long runs; telemetry overhead asymmetry; library default changes to group size or outlier handling; calibration leakage; unequal KV capacity.

Evidence required
Arm manifests with hashes and quantization parameters, raw latency traces, curves with repeats, clamped-KV isolation, kernel diffs, quality tables with CIs, fleet accounting, and written derivations for every ESTIMATE quoted in the summary.

Rollback gates
p95 TTFT or TPOT beyond SLO at target concurrency; any eval slice down >1.0 point absolute with 95% CI excluding zero; structured-output compliance down >2 points; measured cost improvement <25%; unexplained preemption/recompute increase; or kernel-fallback evidence. The BF16 arm remains warm and routable for the whole canary."""

A["corpus-01928"] = """A defensible WOQ comparison has four separable questions. Answer them in order; collapsing them is where these studies go wrong.

Q1. Does the quantized model produce acceptable outputs?
Fixed-seed greedy decoding on a held-out set whose hash is verified disjoint from the calibration data. Slice by long context, structured/JSON output, code, math, and safety/refusal. Bootstrap 95% CIs on absolute deltas. Add sampled side-by-side output diffs, because aggregate scores can stay flat while repetition or truncation behaviour degrades. Acceptance: no slice down more than 1.0 point absolute with CI excluding zero, and structured-output compliance within 2 points.

Q2. Is it faster on the hardware you actually own?
Phase-split timing (prefill vs decode), concurrency sweep of >=5 points with >=3 repeats, steady state only, latency-throughput curves per arm, comparison at the SLO intersection. Attach per-layer kernel selection dumps; an unexpected null result with a dequant-GEMM fallback in the logs is a tooling defect, not a verdict on quantization.

Q3. Is the speed attributable to quantization or to batching?
Run INT4 twice: KV capacity clamped to the BF16 arm's block count, and native capacity. The clamped run isolates the bandwidth/kernel effect; the native run shows deployable benefit. If the clamped run is flat, the win is KV headroom, and you must compare against the cheaper alternative of tuning KV settings on BF16 before adopting a second numeric path.

Q4. Does it lower the bill?
Cost is GPU-seconds per 1k output tokens at the SLO, confirmed by fleet-level GPU-hour and token accounting over a matched production window. Benchmarks propose; fleet accounting disposes. If instance count or autoscaling policy never changed, throughput gains did not become savings.

Falsifiable hypothesis tying it together
H1: "INT4-WOQ passes Q1, yields >=1.5x throughput at the SLO in Q2, retains >=1.3x in the clamped-KV run of Q3, and reduces fleet cost per million output tokens by >=25% in Q4."
Any clause failing rejects H1, and the failing clause names the next investigation.

Frozen arm identity
Checkpoint hash; bits, group size, symmetry, outlier policy, calibration dataset hash; engine build; TP/PP layout; activation and KV precision; scheduler settings; chunked prefill and speculative decoding state. Changing any field creates a new arm ID.

Numbers, labeled
ESTIMATE: 9B weights 18 GB at BF16 versus ~5.0-5.3 GB at INT4 group-128 with FP16 scales, ~3.2-3.6x; derived by byte counting; valid only as an upper bound on decode speedup at batch 1. All figures in the decision memo must be MEASURED with a pointer to the raw log.

Confounders
Prefill/decode mix drift, autotune warmth, thermal/clock drift, telemetry asymmetry, calibration leakage, silent library default changes, unequal KV capacity.

Rollback gates
SLO breach at target concurrency; any Q1 acceptance criterion violated; clamped-KV gain below 1.3x combined with an untested BF16 KV-tuning alternative; fleet cost improvement <25%; unexplained preemption/recompute increase. Keep BF16 warm and routable so rollback is a routing change; pre-write the revert runbook and rehearse it once before the canary starts."""

recs = []
for r in sl:
    sid = r["id"]
    msgs = {m["role"]: m["content"] for m in r["messages"]}
    recs.append({
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": msgs["user"],
        "source_assistant": msgs["assistant"],
        "corrected_answer": A[sid],
        "quality_dimensions": {
            "technical_correctness": 3,
            "instruction_coverage": 2,
            "operational_safety": 3,
        },
        "risks": [
            "source_assistant is a grading rubric, not an answer; training on it teaches meta-commentary instead of engineering reasoning",
            "no mechanism given for why weight-only quantization helps or fails, so a model could claim gains in compute-bound regimes",
            "no separation between KV-cache capacity gains and kernel/bandwidth gains, the most common attribution error in quantization studies",
            "no ESTIMATE vs MEASURED labeling, inviting fabricated platform-specific numbers",
        ],
        "evidence_required": [
            "checkpoint hash and full quantization arm manifest (bits, group size, symmetry, outlier policy, calibration dataset hash)",
            "latency-throughput curves per arm with >=5 concurrency points and >=3 repeats, compared at the SLO intersection",
            "clamped-KV-capacity run isolating kernel/bandwidth effect from batching effect",
            "per-layer kernel selection and timing dumps diffed between arms to detect fallback paths",
            "held-out quality table with bootstrap 95% CIs, per-slice, plus proof of calibration/eval disjointness",
            "fleet-level GPU-hour and token accounting over a matched production window",
        ],
        "confidence": 0.62,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for rec in recs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs))
