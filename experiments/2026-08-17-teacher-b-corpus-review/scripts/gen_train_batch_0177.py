import json, os

EXP = '/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review'
CORPUS = '/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl'
START, N = 1760, 10
OUT = os.path.join(EXP, 'results', 'train-batch-0177.jsonl')

corpus = []
with open(CORPUS, encoding='utf-8') as f:
    for i, l in enumerate(f):
        if START <= i < START + N:
            corpus.append(json.loads(l))
assert len(corpus) == N

PREAMBLE = """
Shared assumptions I am making explicit (change these and the whole comparison changes):
- "Weight-only quantization" (WOQ) means weights stored/loaded at reduced precision and
  dequantized into a higher-precision accumulate path. Activations and KV cache stay at the
  baseline precision. If the change ships bundled with FP8 KV or activation quantization it is
  two experiments, not one, and the KV change will absorb most of the throughput delta while
  none of the quality delta gets attributed correctly. Split them.
- The mechanism is bytes moved per decode step. WOQ helps only where weight-load bandwidth is
  the binding constraint, i.e. small-batch decode. Prefill is compute bound and dequantization
  adds arithmetic there, so TTFT is flat or slightly worse. As batch grows, decode GEMMs become
  compute bound too and the advantage decays toward 1.0x. Any claim of a uniform speedup is
  wrong on its face.
- Byte-count bound, ESTIMATE, derivation inline: a 9B-parameter model at BF16 is
  9e9 x 2 B = 18 GB of weights; INT4 with group-128 FP16 scale plus zero-point is about
  0.5 + ~0.06 B/param, i.e. ~5.0-5.3 GB. Ratio ~3.2-3.6x. This is an upper bound on batch-1
  decode step time only, obtained by pure byte counting with no kernel efficiency model, no
  attention cost and no framework overhead. It is not a throughput result and must never be
  quoted as one.
- Run the quantized arm TWICE. Arm Q-clamped has KV cache block count pinned to the baseline's,
  isolating the kernel/bandwidth effect. Arm Q-native uses the freed HBM for more KV blocks,
  showing the deployable benefit. If Q-clamped is flat and Q-native is fast, the honest finding
  is "we bought KV headroom", and tuning KV settings on the BF16 arm must be priced as the
  cheaper alternative before shipping a second numeric path.
- Cost unit: GPU-seconds per 1,000 output tokens at a fixed p95 SLO, converted to currency last.
  Raw tokens/s hides the latency price; $/hr hides hardware differences.
- Compare arms at the SLO intersection of each arm's own latency-throughput curve, never at a
  fixed batch size. The arms have different optimal batch sizes; fixed-batch comparison is the
  most common way these studies lie.
- Arm identity is frozen and hashed before any measurement: checkpoint hash, bit width, group
  size, symmetric/asymmetric, outlier or mixed-precision policy, calibration set hash, library
  and engine build, TP/PP layout, activation precision, KV precision, block size, max_num_seqs,
  max_model_len, chunked-prefill and speculative-decoding state.

Measurements: >=5 concurrency points spanning below and above the SLO knee, >=3 repeats per
point, steady state with a fixed warmup-exclusion rule, reporting p50/p95 TTFT, p50/p95 TPOT,
output tokens/s, achieved concurrency, preemption and recompute counts, and HBM high-water mark.
Quality: greedy decoding, fixed seed, sliced by long context, structured/JSON output, code, math
and safety/refusal, with bootstrap 95% CIs on absolute deltas plus sampled side-by-side output
diffs, because aggregate scores can stay flat while repetition or truncation degrades.

Confounders to control and report: silent kernel fallback to a dequant-then-GEMM path (dump
per-layer kernel names and timings and diff the arms; a null result with a fallback in the log
is a tooling defect, not a verdict); calibration/eval leakage (prove disjoint by hash);
autotune cache warmth; clock and thermal drift on long runs; asymmetric telemetry overhead;
prefill/decode mix drift between runs; library defaults silently changing group size.

Rollback gates, pre-committed: p95 TTFT or TPOT beyond SLO at target concurrency; any eval slice
down more than 1.0 point absolute with a 95% CI excluding zero; structured-output compliance
down more than 2 points; measured cost improvement under 25%; unexplained preemption or
recompute increase; any kernel-fallback evidence (fix tooling and re-run rather than concluding
anything). Keep the BF16 arm warm and routable throughout so rollback is a routing flip rather
than a redeploy, and rehearse the revert once before the canary."""

EVID = [
    "Frozen arm-identity manifest (checkpoint hash, bit width, group size, calibration set hash, engine build, parallelism and scheduler settings) for both arms",
    "Per-layer kernel name and timing dumps from both arms, diffed to prove no silent dequant-GEMM fallback",
    "Latency-throughput curves (>=5 concurrency points, >=3 repeats) for BF16, Q-clamped and Q-native arms with p95 TTFT/TPOT",
    "Quality evaluation with per-slice absolute deltas and bootstrap 95% CIs under greedy decoding with a fixed seed, plus sampled side-by-side output diffs",
    "Hash proof that the calibration set and the evaluation sets are disjoint",
    "GPU-seconds per 1,000 output tokens at the fixed p95 SLO for each arm, plus the matched-window production GPU-hour and token accounting used to confirm the saving",
]

RISKS = [
    "Reporting a speedup measured at fixed batch size rather than at each arm's SLO-constrained operating point, overstating the benefit",
    "Attributing throughput gained from extra KV cache capacity to the quantized kernels because no capacity-clamped arm was run",
    "Silent fallback to a dequantize-then-GEMM kernel producing a null or negative result that is read as a verdict on quantization itself",
    "Quality regression concentrated in long-context, structured-output or safety slices while the aggregate score stays flat",
    "Shipping weight-only quantization bundled with KV or activation precision changes so neither effect is attributable",
    "Benchmark gains never converting into invoice savings because instance count and autoscaling policy did not change",
]

STANCES = [
 ("Failure-mode-first: enumerate the ways this study can return a confidently wrong verdict, and design each one out before collecting a single number",
  """Before designing the measurement I design the post-mortem. There are six ways this comparison
returns a confident wrong answer, and each gets a specific countermeasure written into the plan.
(F1) Fixed-batch comparison: the quantized arm looks fast at a batch the BF16 arm was never
tuned for. Countermeasure: sweep and compare at each arm's own SLO intersection. (F2) Capacity
confound: freed HBM becomes KV blocks and batching does the work. Countermeasure: the Q-clamped
arm. (F3) Kernel fallback: the fast path is not actually used. Countermeasure: per-layer kernel
name diff, treated as a gate on whether the run counts at all. (F4) Quality blind spot: the
harness measures what quantization does not break. Countermeasure: pre-declared slices with
absolute deltas and CIs, plus manual output diffs. (F5) Leakage: the calibration set overlaps
eval, inflating quality. Countermeasure: hash-level disjointness proof. (F6) Underpowered run:
the effect is smaller than run-to-run spread and gets reported as positive. Countermeasure:
declare the minimum detectable effect from pilot variance before the main run and report
"underpowered" as a legitimate outcome.""",
  """Every distinct failure mode above has a named countermeasure and a named artifact; if any
countermeasure is missing at analysis time, the corresponding conclusion is withheld rather than
weakened."""),

 ("Traffic-shape-first: the production request mix decides whether this study is worth running at all",
  """The first artifact is not a benchmark, it is a week of production request traces: input and
output length distributions, arrival concurrency, prefix reuse rate, and the resulting share of
GPU time spent in prefill versus decode. WOQ acts only on decode weight loading, so the trace
sets the ceiling before any kernel runs. If decode is 40% of GPU time, even a perfect 3.4x
decode-step improvement gives at most about 1.33x end to end (ESTIMATE, Amdahl applied to the
byte-count bound: 1/(0.6 + 0.4/3.4)); at 80% decode it is about 2.1x (ESTIMATE, same derivation).
Benchmark prompt mixes must be resampled from that trace, not chosen for convenience, and the
report must state the decode fraction of the mix it used, because a synthetic long-output mix
manufactures a win that production will never see.""",
  """Go/no-go rule: if the trace-derived ceiling is below the 25% cost-improvement gate, the study
is declined with that arithmetic as the reason, and effort moves to prefix caching or chunked
prefill instead."""),

 ("Accuracy-budget-first: negotiate and freeze the acceptable quality loss with the product owner before any performance number exists",
  """Cost work goes wrong when the quality bar is set after the speedup is known. So the first
deliverable is a signed accuracy budget: per-slice maximum tolerable absolute degradation
(e.g. code and structured output 0.5 points, general chat 1.0 point, safety/refusal 0.0 with a
one-sided test), fixed before measurement and versioned in the repo. The hypothesis is then
stated as a joint condition rather than a speed claim, and quality is evaluated and gated first;
performance numbers are not even computed for an arm that has already failed its accuracy
budget. This ordering removes the strongest bias in these studies, which is re-reading an
inconvenient quality regression as acceptable once a large throughput number is on the slide.""",
  """The budget also names who may amend it and by what process; an amendment made after seeing
results invalidates the pre-registration and forces a re-run."""),

 ("Power-first: size the experiment from pilot variance so that a null result is informative rather than merely quiet",
  """I run a short pilot on the BF16 arm alone to estimate run-to-run variance of p95 TPOT and of
each eval slice, then compute the minimum detectable effect for the planned number of repeats.
If the design cannot resolve a 25% cost difference at the SLO point with the repeats we can
afford, that is discovered before the main run and the design changes (more repeats, tighter
warmup exclusion, dedicated unshared hosts) rather than after. Every reported delta carries a CI
and the pre-declared MDE; effects smaller than the MDE are labelled "underpowered", never
"no difference" and never "small win".""",
  """This makes the negative branch publishable: a properly powered null is a real finding that
saves the team a second numeric path in production."""),

 ("Reconciliation-first: treat the offline benchmark and a shadow-traffic replay as two instruments that must agree before either is believed",
  """Offline benchmarks and production disagree routinely, so I plan both and make the disagreement
the object of study. Arm A is the controlled synthetic sweep; arm B is a shadow replay of real
traffic at real arrival timing against an isolated replica of each configuration. If the two
instruments give cost-unit answers within the CI, the result is believed. If they diverge, the
divergence is diagnosed before any decision: usual causes are arrival burstiness and scheduler
queueing that synthetic closed-loop harnesses erase, prefix-cache hit rates absent from the
synthetic mix, and output-length distribution mismatch.""",
  """Shadow replay also produces the only quality signal that matters operationally: response diffs
on real prompts, reviewed on a sampled basis by the owning team rather than scored only by an
automated metric."""),

 ("Memory-ledger-first: account for every HBM byte in both arms, because the freed memory is the real product of this change",
  """I build an explicit per-GPU HBM ledger for each arm: weights, optimizer-free inference
residuals, activation workspace and fragmentation, CUDA graph pools, framework overhead, and KV
blocks as the residual. For the 9B example the weights line moves from ~18 GB to ~5.0-5.3 GB
(ESTIMATE, byte counting as above), and the ledger shows exactly how many extra KV blocks that
buys at the configured block size and sequence length. Framing the change as a memory ledger
rather than a speed knob makes the central question unavoidable: is the goal more concurrent
sequences, longer supported context, fewer replicas, or a smaller SKU? Each answer implies a
different experiment and a different cost unit.""",
  """The ledger is also the falsification tool: if measured HBM high-water marks do not match the
ledger within a few percent, the arm is misconfigured and its performance numbers are void."""),

 ("Attribution-first: the per-layer kernel profile is the primary artifact and the end-to-end number is secondary",
  """End-to-end deltas are an aggregate of many effects, so I collect per-layer kernel names,
occupancy and achieved memory bandwidth for both arms first, and only then look at the
end-to-end curves. The prediction is specific and falsifiable at layer granularity: in decode,
the quantized linear layers show reduced bytes read and reduced duration while attention kernels
are unchanged; if attention time moves, something other than WOQ changed. If quantized layers
show a dequantize kernel followed by a BF16 GEMM, the fast path was never engaged and no
end-to-end number from that run is interpretable.""",
  """This ordering also localises partial wins: mixed-precision outlier handling often leaves some
projections in BF16, and the profile says precisely which, turning a disappointing aggregate into
an actionable coverage gap."""),

 ("Capacity-planning-first: the study is only decision-relevant if it changes the fleet plan, so evaluate it against the fleet decision it would alter",
  """I write down the fleet decision first: current replica count, the SKU, the headroom policy,
the autoscaling trigger, and the peak-hour concurrency the fleet must absorb. The study then
answers one question, does this change let us serve the same peak with fewer replicas or on a
cheaper SKU at the same SLO. Benchmarks are translated into required replica count at peak, not
into tokens/s. A 30% throughput gain that does not cross a replica boundary, or that is absorbed
by a headroom policy nobody updates, produces zero savings and should not ship a second numeric
path into production for nothing.""",
  """Closure requires the matched-window fleet accounting: GPU-hours and served tokens before and
after over comparable traffic. Absent a change in instance-hours, the claimed saving is a
projection, not a result."""),

 ("Reproducibility-first: design so that a skeptical third party can rerun the comparison from the artifacts alone and get the same verdict",
  """The deliverable is a re-runnable bundle, not a slide. It contains both arm manifests with
hashes, the exact container images and library versions, the prompt corpus and its hash, the
sweep driver with its seeds, raw per-request latency records rather than only summaries, the
analysis notebook that turns raw records into the reported numbers, and the environment capture
(driver, firmware, clock and power caps, NUMA and topology). Anyone re-running it must reproduce
the reported cost unit within the stated CI on comparable hardware; if they cannot, the result is
withdrawn rather than defended.""",
  """This constraint quietly fixes several other problems: it forces raw-record retention, forbids
hand-edited numbers, and makes unreported configuration drift between arms visible as a diff
between two manifests."""),

 ("Value-of-information-first: price the study itself and stop as soon as the decision is determined",
  """The study costs engineering time, GPU hours and, if shipped, the permanent maintenance burden
of a second numeric path (extra CI matrix, extra regression surface, extra incident modes). So I
price it against the saving it could unlock, and structure it as staged gates that can terminate
early: stage 0, trace-derived ceiling from the decode fraction; stage 1, quality on the strictest
slices only; stage 2, Q-clamped kernel isolation; stage 3, Q-native SLO sweep and cost unit;
stage 4, canary and fleet accounting. Failing any stage stops the study and the remaining stages
are never paid for.""",
  """Alternatives are priced on the same cost unit at stage 0 (prefix caching, chunked-prefill
tuning, KV block tuning on BF16, speculative decoding, replica right-sizing, a smaller model);
if a cheaper option plausibly reaches the same 25% gate, it runs first."""),
]

recs = []
for c, (stance, body, tail) in zip(corpus, STANCES):
    msgs = {m['role']: m['content'] for m in c['messages']}
    ans = ("Analytical stance under test: %s.\n"
           "Falsifiable hypothesis, pre-registered: at matched output quality within the declared "
           "per-slice budget, the INT4 weight-only arm reduces GPU-seconds per 1,000 output tokens "
           "at the fixed p95 SLO by at least 25%% relative to BF16, and at least half of that "
           "reduction survives in the KV-capacity-clamped arm. Failing either half falsifies the "
           "cost claim as stated.\n\n%s\n%s\n%s\n" % (stance, body.strip(), PREAMBLE.strip(), tail.strip()))
    recs.append({
        "source_id": c['id'],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": msgs['user'],
        "source_assistant": msgs['assistant'],
        "corrected_answer": ans,
        "quality_dimensions": {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3},
        "risks": list(RISKS),
        "evidence_required": list(EVID),
        "confidence": 0.62,
    })

with open(OUT, 'w', encoding='utf-8') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('wrote', OUT, len(recs))
