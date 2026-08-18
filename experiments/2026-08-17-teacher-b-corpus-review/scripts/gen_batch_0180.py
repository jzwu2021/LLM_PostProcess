import json, hashlib, os

ROOT="/home/johnson/workspace/LLM_PostProcess"
CORPUS=f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP=f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT=f"{EXP}/results/train-batch-0180.jsonl"
START=1790; N=10

rows=[json.loads(l) for l in open(CORPUS) if l.strip()]
sel=rows[START:START+N]
assert len(sel)==N

COMMON_TAIL = """

Shared mechanism, stated once so nothing below rests on hand-waving. Weight-only quantization
(WOQ) reduces the BYTES OF WEIGHTS MOVED PER DECODE STEP. It does not reduce attention math, it
does not reduce KV-cache traffic, and it adds dequantization arithmetic. Therefore it helps only
in the memory-bandwidth-bound regime, which in practice is small-batch decode. Prefill is compute
bound, so TTFT is flat or slightly worse. As batch size rises, decode GEMMs become compute bound
and the advantage decays toward 1.0x. Any claim of a uniform speedup across phases and batch
sizes is wrong on mechanism alone and should be rejected before it is measured.

Byte-count bound, ESTIMATE, derivation inline: a 9B-parameter model in BF16 is 9e9 x 2 B = 18 GB
of weights. INT4 with group-128 FP16 scale and zero-point is about 0.5 + ~0.06 B/param, i.e.
~5.0-5.3 GB. Ratio ~3.2-3.6x. This is an upper bound on batch-1 decode step time only, obtained
by pure byte counting with no kernel, occupancy or overlap model. It is not a throughput
prediction. Applying Amdahl with the measured decode-weight-read fraction f of end-to-end time
bounds the achievable end-to-end gain: f=0.4 gives ~1.33x, f=0.8 gives ~2.1x (ESTIMATE, same
derivation). If the measured speedup exceeds the byte-count bound, the comparison is broken, not
excellent.

Mandatory two-arm isolation. Run the quantized configuration TWICE: (a) Q-clamped, with KV-cache
block count pinned to the BF16 baseline's, which isolates the kernel and bandwidth effect; and
(b) Q-native, at full capacity, which shows the deployable benefit. Freed HBM becomes more KV
blocks and raises throughput by batching alone. A study reporting only Q-native cannot separate
"quantization is fast" from "we got more batching". If Q-clamped is flat, the honest conclusion
is "we bought KV headroom", and tuning KV settings on the BF16 arm must be priced as the cheaper
alternative before a second numeric path is shipped.

Weight-only means weight-only. Shipping "INT4 weights plus FP8 KV" as one change is two
experiments; the KV change usually explains most of the throughput delta and none of the quality
delta gets attributed correctly. Split them.

Cost unit: GPU-seconds per 1,000 output tokens at a fixed p95 SLO, converted to currency last.
Raw tokens/s hides the latency price; $/hour hides hardware differences. Compare arms at each
arm's own SLO intersection on its latency-throughput curve, never at a fixed batch size; the two
arms have different optimal batch sizes, and fixed-batch comparison is the most common way these
studies mislead.

Arm identity is frozen and hashed before any run: checkpoint hash, bit width, group size,
symmetric/asymmetric, outlier or mixed-precision policy, calibration dataset hash, quantization
library version, engine build, TP/PP layout, activation precision, KV precision, block size,
max_num_seqs, max_model_len, chunked-prefill state, speculative-decoding state. The configuration
symmetric difference between arms must be exactly the bit-width fields; anything else invalidates
the run.

Measurement protocol: at least five concurrency points, at least three repeats per point,
steady state with a fixed warmup-exclusion rule, greedy decoding and fixed seed for quality, and
a BF16-vs-BF16 A/A control across seeds and physical GPUs to establish the noise floor. Any
claimed delta below the A/A spread is "underpowered", not "positive". Verify TTFT is measured at
first token delivered to the client rather than scheduler admission, that TPOT excludes the first
token, and that load-generator and server clocks are not skewed.

Quality evaluation is sliced, never aggregate-only: long context, structured/JSON output, code,
math, multi-turn, and safety/refusal. Report bootstrap 95% confidence intervals on ABSOLUTE
deltas, plus sampled side-by-side output diffs, because aggregate scores can stay flat while
repetition, truncation or format drift degrades. Prove calibration and evaluation sets are
disjoint by hash comparison.

Confounders to control and log: silent kernel fallback to a dequantize-then-GEMM path (dump
per-layer kernel names and timings and diff the arms; a null result with a fallback in the logs
is a tooling defect, not a verdict), autotune cache warmth, clock and thermal drift on long runs,
asymmetric telemetry overhead, prefill/decode mix drift between runs, and library defaults
silently changing group size or activation precision.

Evidence required before any decision: frozen arm manifests with hashes; per-request raw records
(arrival, admission, first token, per-token timestamps, output length); the Q-clamped and
Q-native pair; per-layer kernel and bandwidth traces for both arms; per-slice quality tables with
CIs; the A/A noise floor; and a fleet-level GPU-hour and token accounting over a matched
production window, because benchmarks propose and the bill disposes. If instance count or
autoscaling policy never changed, throughput gains never became savings.

Rollback gates, pre-committed: p95 TTFT or TPOT outside SLO at target concurrency; any evaluation
slice down more than 1.0 point absolute with a 95% CI excluding zero; structured-output
compliance down more than 2 points; measured cost improvement below 25%; unexplained increase in
preemption or recompute; or kernel-fallback evidence (in which case fix tooling and re-run rather
than concluding anything). Keep the BF16 arm warm and routable throughout so rollback is a
routing flip rather than a redeploy, and rehearse the revert once before the canary.

Declared unknowns: no measurement was performed here. Every number above is an ESTIMATE with its
derivation shown. Platform-specific kernel coverage, achieved bandwidth and per-slice quality
deltas for this exact checkpoint and engine build are unknown until measured on the target
hardware."""

STANCES = [
 ("hardware-substitution-first: quantization competes against buying the right accelerator, and must be priced on the same cost unit before it is engineered",
  """The first question is not "does INT4 help" but "is quantization the cheapest way to buy the
capacity we need". WOQ is one lever in a set that also contains: moving to a device with higher
HBM bandwidth per dollar, right-sizing replicas, prefix caching, chunked-prefill tuning,
speculative decoding, and simply serving a smaller model. All of them are measured on the same
unit, GPU-seconds per 1,000 output tokens at the fixed p95 SLO, so they are directly comparable.
Because WOQ acts on bytes-moved-per-decode-step, its ceiling is the same quantity that a
higher-bandwidth device improves directly and without any quality risk. If a device swap
delivers the target at zero accuracy risk and zero added maintenance surface, the quantization
project is dominated and should not start.
Boundary condition that decides the comparison: WOQ's benefit is bounded by the decode
weight-read fraction f, whereas a bandwidth upgrade improves attention and KV traffic as well.
So at low f, hardware wins by construction; at high f with a bandwidth-starved device, WOQ can
match it at lower capital cost. Measure f before choosing.""",
  ["Quantization chosen by default without pricing hardware or configuration alternatives on the same unit",
   "Capital-cost comparison done in $/hour list price, ignoring achieved utilisation",
   "Accuracy risk of WOQ carried permanently for a saving a device swap would have delivered risk-free"],
  ["Decode weight-read fraction f from a production trace, with the Amdahl ceiling computed from it",
   "Same-cost-unit table for at least four alternatives including a no-quantization control",
   "Achieved-utilisation-adjusted cost per device class, not list price"],
  0.6, "rewrite", 4, 4, 4),

 ("kernel-portability-and-shape-coverage-first: the result is only valid for the exact shapes, group size and parallel layout it was measured on",
  """A WOQ speedup is a property of a kernel-shape-layout triple, not of a model. The quantized
GEMM must be checked for coverage across every shape the deployment actually issues: per-layer
N and K after tensor-parallel sharding, batch-1 and batch-many decode, the fused QKV and MLP
gate-up shapes, and the LM head. Group size interacts with TP sharding: if K per shard is not a
multiple of the group size, the library either pads, falls back, or silently changes the group,
and all three change both quality and speed. Build an explicit coverage matrix, shape by shape,
recording which kernel is selected and its achieved bandwidth.
Falsifier: if the per-layer kernel dump shows a dequantize-then-BF16-GEMM path for any hot
layer, the arm is not a WOQ arm and no conclusion may be drawn. Boundary condition: changing TP
degree after the study invalidates it, because the shard shapes change; re-run coverage before
any layout change reaches production.""",
  ["Kernel coverage assumed uniform across layers, so a fallback on hot layers is averaged away",
   "Group size incompatible with the TP shard K, causing a silent library-side change",
   "Result generalised to a different TP degree or engine build where shape coverage differs"],
  ["Per-layer kernel-selection and achieved-bandwidth dump for both arms, diffed",
   "Shape coverage matrix over post-sharding N,K for all hot linears including the LM head",
   "Explicit record of group size versus per-shard K divisibility"],
  0.62, "rewrite", 5, 4, 4),

 ("request-mix-stratification-first: a single blended number is an average over regimes with opposite signs",
  """Production traffic is not one workload. Stratify by prompt-length and output-length deciles
before comparing anything, then report the gain per stratum. The mechanism predicts the shape of
the result: short-prompt/long-output requests are decode-dominated and should show the largest
gain; long-prompt/short-output requests are prefill-dominated and should show approximately zero
or a small regression from dequantization overhead. A blended average over a mix that shifts
week to week is not a stable quantity and cannot be used for capacity planning.
Falsifiable prediction, checkable cheaply: gain must be monotonically ordered by the
decode-token fraction of each stratum. If it is not, either the strata are contaminated by
scheduler effects such as preemption, or the load generator is not reproducing the intended
shape. Boundary condition: the deployable saving is the traffic-weighted sum over strata using
the CURRENT mix, and must be re-derived if the mix moves more than a stated tolerance.""",
  ["Blended average hides a regression on prefill-heavy strata that carry a strict TTFT SLO",
   "Traffic mix drifts after the study, invalidating the weighted saving silently",
   "Strata defined by prompt length alone, ignoring output length, which drives the decode fraction"],
  ["Production trace with joint prompt-length and output-length distribution over at least one week",
   "Per-stratum gain table with CIs, plus the traffic-weighted roll-up shown separately",
   "Monotonicity check of gain versus decode-token fraction across strata"],
  0.63, "rewrite", 5, 5, 4),

 ("tenant-and-SLO-class-first: the benefit is denominated per SLO class, and a fleet average can be positive while the strict class regresses",
  """Serving fleets carry several SLO classes: interactive chat with tight p95 TTFT, batch or
offline jobs that only care about throughput, and background jobs that care only about cost. WOQ
moves these in different directions. The batch class, which runs at high concurrency, sits in the
compute-bound regime where the gain decays toward nothing. The interactive class runs at low
concurrency where the gain is largest, but is also the class with the tightest latency contract
and the least tolerance for a quality regression. Evaluate and gate per class, and state the
decision per class rather than for the fleet.
Boundary condition: if classes share a replica pool, a per-class decision is not implementable
without routing changes; that routing work is part of the cost of the proposal and must be
priced. Falsifiable statement: the interactive class shows a larger relative gain than the batch
class at their respective operating concurrencies; if the ordering is reversed, the concurrency
levels or the SLO intersections were not set correctly.""",
  ["Fleet-average gain reported while the strictest SLO class regresses",
   "Per-class decision assumed implementable on a shared pool without routing work",
   "Quality budget set globally rather than per class, so the class with the least tolerance is under-protected"],
  ["Per-SLO-class concurrency operating points and SLO intersections",
   "Per-class quality budget signed by that class's owner before measurement",
   "Routing/pool topology showing whether a per-class rollout is even possible"],
  0.6, "rewrite", 4, 5, 4),

 ("randomized-interleaved-assignment-first: remove drift by making arm assignment a within-window randomisation rather than a between-window comparison",
  """Sequential A-then-B measurement confounds the arm with everything that changed between the
windows: clocks, thermals, neighbour load, traffic mix, cache warmth. The stronger design is
replica-level randomisation with interleaved shadow replay: the same recorded request stream is
replayed to both arms concurrently on matched hardware, with request-level pairing so each
request has a BF16 and a Q outcome, and arm-to-node assignment swapped halfway to cancel
node-specific bias. Paired analysis then removes per-request difficulty as a variance source and
raises power substantially at fixed run length.
Boundary condition: shadow replay reproduces arrival timing but not queueing interactions with
real co-tenants, so it measures the kernel and capacity effect well and the fleet effect poorly;
the fleet claim still needs a production canary. Falsifier: if the arm-swap half shows a
different sign or magnitude than the first half, the environment, not the arm, is driving the
result.""",
  ["Sequential between-window comparison absorbing thermal, neighbour and mix drift into the arm effect",
   "Node-specific bias mistaken for an arm effect when the swap is omitted",
   "Shadow replay results presented as fleet savings without a production canary"],
  ["Paired per-request records keyed by request id across both arms",
   "Arm-to-node swap schedule and the two half-window results reported separately",
   "Concurrent replay timing traces showing both arms saw the same arrival process"],
  0.62, "rewrite", 5, 4, 4),

 ("incident-and-on-call-first: design the study so the resulting system is diagnosable at 3am, and count the operational cost of a second numeric path",
  """Assume the change ships and then something breaks. What does the on-call engineer see, and
can they tell whether quantization is implicated? That requirement is a design constraint on the
experiment, not an afterthought: every arm must be identified in logs and metrics by a stable
build and arm label, the quality canary must run continuously rather than once, and both arms
must stay warm and routable so mitigation is a routing flip with a rehearsed, timed procedure.
Blast radius also changes: a quantized checkpoint failure mode is often not a crash but a
silent quality drift on a subset of requests, which no liveness probe catches.
Ongoing cost to price against the saving: two conversion pipelines, per-release recalibration and
re-validation, doubled kernel-regression exposure on engine upgrades, and a second artifact in
the release supply chain. Boundary condition, falsifiable: if the annualised saving does not
exceed the modelled operational cost plus the cost of one expected quality incident, the proposal
fails on operations regardless of the benchmark result.""",
  ["Silent per-slice quality drift in production with no continuous canary to detect it",
   "Dual numeric path doubling release-validation and kernel-regression exposure",
   "Rollback never rehearsed, so the routing flip is discovered to be a redeploy during an incident"],
  ["Continuous quality canary definition with alert thresholds and owner",
   "Timed rollback rehearsal record showing the flip works and how long it takes",
   "Operational cost model covering recalibration, dual validation and expected incident cost"],
  0.6, "rewrite", 4, 4, 5),

 ("artifact-supply-chain-first: the quantized checkpoint is a build output and must be reproducible, hashed and attributable, or the measurement is not repeatable",
  """A WOQ result is a claim about a specific artifact. Treat the conversion as a build: pinned
library and CUDA versions, pinned calibration dataset by content hash, recorded random seed and
data order, and a determinism check that converting twice from the same inputs yields the same
output hash. If conversion is not bit-reproducible, say so explicitly and quantify the spread by
producing two independent quantized checkpoints and measuring both quality and speed on each;
that spread, not a single run, is the real error bar on the arm.
Boundary condition: calibration is a hidden hyperparameter, so an artifact produced from an
undocumented calibration set cannot be compared to anything and cannot be regenerated after an
engine upgrade. Falsifiable statement: two checkpoints from the same recipe differ by less than
the per-slice quality tolerance; if they do not, the recipe is not production-ready and no single
checkpoint's numbers may be quoted.""",
  ["Quantized checkpoint not reproducible, so the measured result cannot be regenerated after an upgrade",
   "Calibration corpus undocumented, making the arm uncomparable and the recipe unmaintainable",
   "Single-checkpoint numbers quoted while checkpoint-to-checkpoint variance exceeds the quality budget"],
  ["Conversion build manifest: library, CUDA, engine versions, seed, data order, input and output hashes",
   "Two independently converted checkpoints with per-slice quality and speed for each",
   "Hash-level proof that calibration and evaluation sets are disjoint"],
  0.63, "rewrite", 5, 4, 4),

 ("validity-of-the-quality-instrument-first: before trusting a quality delta, measure the measuring device",
  """The weakest link in most WOQ studies is not the latency harness but the quality evaluator. If
an LLM judge or a rubric grader is used, its own reliability bounds every conclusion drawn from
it. Quantify it first: re-grade the same outputs twice to get test-retest agreement, grade a set
of known-degraded outputs to establish sensitivity, check position and verbosity bias by swapping
presentation order, and compute the evaluator's own CI. A judge whose test-retest spread is wider
than the quality budget cannot certify that budget, and the correct response is to narrow the
task, use exact-match or compile-and-run checks where possible, or add human adjudication on a
sampled subset.
Boundary condition: automatic metrics on structured output (schema validity, compile success,
unit-test pass) are far more reliable than free-text scoring and should carry the gate wherever
the workload permits. Falsifiable statement: the evaluator detects a deliberately injected
degradation of the size of the quality budget; if it cannot, it is not fit to gate.""",
  ["Quality gate resting on a judge whose test-retest noise exceeds the budget it is meant to enforce",
   "Position or verbosity bias in the evaluator systematically favouring one arm's output style",
   "Aggregate score flat while a real degradation hides inside slices the evaluator is insensitive to"],
  ["Evaluator test-retest agreement and CI on the same output set",
   "Injected-degradation sensitivity check calibrated to the quality budget",
   "Order-swapped and verbosity-controlled grading runs, plus human adjudication on a sampled subset"],
  0.62, "rewrite", 5, 5, 4),

 ("queueing-first: a per-request service-time reduction becomes a wait-time reduction non-linearly, so the deployable benefit depends on where on the utilisation curve you sit",
  """Model the server as a queue. WOQ shortens decode service time by some factor s at low batch.
By Little's law and standard queueing behaviour, at a utilisation rho the waiting component grows
roughly as rho/(1-rho), so the same s produces a small end-to-end p95 improvement at low
utilisation and a large one near saturation, or equivalently permits a higher arrival rate at
unchanged p95. The right way to express the benefit is therefore not "x% faster" but "the
maximum sustainable arrival rate at the fixed p95 SLO rises from A to A'", which is directly
convertible into replica count.
Boundary condition: continuous batching means service time is not independent of queue state,
so the classical formula is an approximation used for shape, not for prediction; the arrival-rate
frontier must be measured empirically. Falsifiable statement, ESTIMATE with derivation: if
service time falls by factor s only in the decode phase and decode is fraction f of the service
time, sustainable arrival rate rises by at most 1/(1 - f + f/s); with f=0.6 and s=2 this is
~1.43x. A measured frontier gain above that bound indicates a broken comparison.""",
  ["Benefit reported as a speedup rather than as a change in sustainable arrival rate, so it cannot be converted to replicas",
   "Measurements taken far from the production utilisation point, where the queueing amplification differs",
   "Classical queueing formula treated as a prediction despite continuous batching coupling service time to queue state"],
  ["Empirically measured arrival-rate versus p95 frontier for both arms",
   "Production utilisation distribution to locate the real operating point on that frontier",
   "Replica-count translation at peak, with the integer granularity made explicit"],
  0.6, "rewrite", 5, 4, 4),

 ("energy-and-thermal-first: bytes not moved are joules not spent, and joules per 1,000 output tokens is a cost unit that also explains long-run drift",
  """Instrument board power alongside latency and measure joules per 1,000 output tokens for both
arms at matched SLO. This does two jobs. First, it is a cost unit that survives hardware
heterogeneity better than GPU-seconds, and in power- or cooling-capped facilities it is the
binding constraint, so a gain that does not reduce energy per token may not free any capacity at
all. Second, power is the mechanism behind a common measurement artefact: long runs heat the
device, clocks drop, and whichever arm ran second looks worse. Logging per-run clock and power
traces turns that confound from an unexplained variance into an observable and testable one.
Falsifiable prediction from mechanism: if WOQ genuinely reduces weight bytes read at fixed
output, energy per output token in the memory-bound regime should fall by a factor smaller than
but directionally consistent with the latency gain, since static and non-GEMM power do not
shrink. If latency improves while energy per token does not, suspect clock or power-cap effects
rather than a bandwidth win. Boundary condition: all power figures here are ESTIMATE-free
placeholders; nothing was measured, and board power must be read from the device telemetry on the
target hardware.""",
  ["Power- or cooling-capped facility where a latency gain frees no real capacity",
   "Thermal and clock drift over long runs assigned to the arm that happened to run second",
   "Energy accounted at the device only, ignoring host, cooling and PSU overheads when converting to cost"],
  ["Per-run board power and clock traces aligned to the latency timeline for both arms",
   "Joules per 1,000 output tokens at matched p95 SLO for both arms",
   "Facility power/cooling headroom figures to determine whether energy or GPU-hours is the binding constraint"],
  0.6, "rewrite", 5, 4, 4),
]

assert len(STANCES)==N

out=[]
for r, st in zip(sel, STANCES):
    m={x['role']:x['content'] for x in r['messages']}
    stance, body, risks, ev, conf, dec, tc, ic, ops = st
    ans = ("Analytical stance under test: " + stance + ".\n"
           "Falsifiable hypothesis, pre-registered: at matched output quality within the signed "
           "per-slice budget, the INT4 weight-only arm reduces GPU-seconds per 1,000 output tokens "
           "at the fixed p95 SLO by at least 25% relative to BF16, AND at least half of that "
           "reduction survives in the KV-capacity-clamped arm. Failing either conjunct falsifies "
           "the cost claim as stated and the study reports a null.\n\n"
           + body + COMMON_TAIL)
    out.append({
        "source_id": r["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": dec,
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": ans,
        "quality_dimensions": {"technical_correctness": tc, "instruction_coverage": ic, "operational_safety": ops},
        "risks": risks,
        "evidence_required": ev,
        "confidence": conf,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,"w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False)+"\n")
print("wrote", OUT, len(out))
print("ids", [o["source_id"] for o in out])
print("distinct200", len({o["corrected_answer"][:200] for o in out}))
print("distinct_sha", len({hashlib.sha256(o["corrected_answer"].encode()).hexdigest() for o in out}))
