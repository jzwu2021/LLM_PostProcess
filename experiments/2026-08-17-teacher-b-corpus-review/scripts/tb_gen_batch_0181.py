import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0181.jsonl"
START, END = 1800, 1810

MECH = """Shared mechanism, stated once so that nothing below rests on hand-waving. Weight-only
quantization (WOQ) compresses the stored weight tensors and dequantizes them inside the GEMM
epilogue. It therefore reduces BYTES OF WEIGHTS READ PER DECODE STEP and nothing else. It does not
shrink the KV cache, it does not reduce attention FLOPs, and it adds dequantization work on the
compute path. The consequence is a hard structural boundary: WOQ can only pay in the
memory-bandwidth-bound regime, which in practice means small-batch, long-decode traffic. Prefill is
compute bound, so time-to-first-token (TTFT) is flat or mildly worse. As batch size grows, decode
GEMMs cross into the compute-bound regime and the speedup decays monotonically toward 1.0x. Any
report of a single uniform "Nx faster" number across a mixed workload is measuring an arrival
pattern, not a kernel."""

FAIR = """Fair-comparison invariants, all held byte-identical across arms: same model checkpoint
lineage, same tokenizer, same prompt set and sampling seeds, same server build and flags except the
quantization method, same GPU SKU/driver/clock policy, same max-batch and scheduling parameters,
same KV dtype and same max context. Only the weight numeric format varies. Arms are interleaved in
time (A/B/A/B) rather than run back to back, so thermal drift, background tenants and driver state
cannot be aliased into the treatment effect."""

ROLLBACK = """Rollback gate, pre-committed before any measurement is taken: revert to the BF16 arm
if any of (a) the paired quality delta breaches the signed per-slice budget on any slice, (b) p95
end-to-end latency at the production concurrency regresses, (c) the observed GPU-seconds saving is
smaller than the pre-registered threshold, or (d) any arm shows a kernel fallback, numerical
instability or NaN in a 24-hour soak. Rollback is a config flip to the untouched BF16 deployment,
not a rebuild; the BF16 replicas stay warm for the whole trial."""

STANCES = [
    (
        "quality-budget-first: the accuracy contract is signed before any speed number is allowed to exist",
        """Hypothesis, pre-registered and falsifiable: with the per-slice quality budget signed in
advance, the INT4 weight-only arm holds every slice inside budget AND reduces GPU-seconds per 1,000
output tokens at the fixed p95 SLO by at least 20% relative to BF16. If any slice breaches budget,
the cost result is discarded regardless of how good it looks; that is the falsification condition.

The failure mode this stance defends against is retro-fitting the quality bar to the speedup that
was obtained. So the bar is written first: task-level pass rates on the production slice mix, each
slice with its own tolerance, each tolerance derived from what the downstream consumer can absorb,
not from what the quantizer happens to deliver. Aggregate scores are explicitly banned as the
decision statistic, because averaging hides the tail slices where WOQ actually breaks: rare tokens,
long-context recall, code and numeric reasoning, and any non-English slice whose activations were
under-represented in the calibration set.

Boundary condition that decides this stance: WOQ error is not uniform across the weight
distribution. Group-wise scaling bounds the per-group dynamic range, so accuracy loss concentrates
where outlier channels live. That is why a single average metric is the wrong instrument and why
per-slice paired testing is mandatory.

Statistics: paired per-prompt comparison, bootstrap confidence intervals over prompts, and a
pre-declared minimum detectable effect. Report intervals, never point estimates.

Sizing arithmetic, all ESTIMATE, derivation shown so a reviewer can falsify it: for a 9B-parameter
model, BF16 weights occupy 9e9 x 2 B = 18 GB; INT4 with group-wise scales occupies roughly
9e9 x 0.5 B = 4.5 GB plus about 3-6% scale/zero-point overhead, so about 4.6-4.8 GB. The freed
13.2-13.4 GB is available for KV cache, which is a capacity effect, not a latency effect. The
decode-step ceiling is 18/4.7 = 3.8x on weight reads only (ESTIMATE); realized speedup is that
ceiling multiplied by the weight-read fraction f of the step, and f must be MEASURED per profile,
never assumed.""",
    ),
    (
        "confound-isolation-first: the KV-capacity side effect must be clamped or the study measures the wrong thing",
        """Hypothesis, pre-registered and falsifiable: at least half of the INT4 arm's throughput gain
survives when KV-cache capacity is clamped to the BF16 arm's value. If the gain collapses by more
than half under the clamp, the claim "quantization made decoding faster" is falsified and replaced
by "quantization bought concurrency", which is a different engineering conclusion with different
scaling behaviour.

This is the single most common measurement error in WOQ studies. Freeing weight memory lets the
scheduler admit more concurrent sequences, which raises aggregate throughput even if per-step
decode latency is unchanged. Two mechanisms, one number. So the experiment runs three arms, not
two: BF16 baseline, INT4 unclamped, and INT4 with gpu_memory_utilization and max-num-seqs pinned so
that the admitted KV blocks match the BF16 arm exactly. The difference between the two INT4 arms is
the capacity effect; the difference between clamped-INT4 and BF16 is the bandwidth effect.

Second confound, controlled explicitly: scheduler and batching policy. Chunked prefill, prefix
caching and continuous batching all change the prefill/decode mix, which changes f, which changes
the WOQ payoff. These are frozen identically across arms and their hit rates are logged per arm;
a prefix-cache hit-rate difference greater than 2 percentage points invalidates the pair.

Third confound: request arrival pattern. Fixed-QPS closed-loop and open-loop Poisson give different
queueing tails. Both are run; only same-generator comparisons are reported.

Numbers, all ESTIMATE with derivation: for the same 9B model, freeing about 13 GB at roughly 0.14
MB per token of KV (GQA, 8 KV heads x 128 dim x 40 layers x 2 tensors x 2 B, MEASURED per build
before use) admits on the order of 90,000 additional cached tokens, which at 2,000-token sequences
is about 45 extra concurrent slots. That is a capacity number, and it is exactly the quantity the
clamped arm removes from the comparison.""",
    ),
    (
        "kernel-support-first: the numeric format is a claim about which GEMM you will actually execute",
        """Hypothesis, pre-registered and falsifiable: every INT4-arm decode step dispatches to a
fused low-precision kernel with zero fallback to a dequantize-then-BF16-GEMM path, and under that
condition the arm meets its GPU-seconds target. Observing any fallback dispatch falsifies the
configuration, not the idea; the run is void and must be reconfigured before any cost claim is made.

The reason this stance exists: WOQ savings are realized by a kernel, and kernels are not universal.
Support depends jointly on the quantization scheme (group size, symmetric vs asymmetric, act-order
or not), the serving engine's kernel library and version, the GPU architecture, and the shape of
the GEMM at the batch sizes actually served. A scheme with excellent published numbers on one
architecture can silently fall back on another, at which point the arm is strictly slower than BF16
because it pays dequantization on top of full-precision math.

Boundary condition: many low-precision kernels are tuned for small-M decode GEMMs and lose to
BF16/FP16 tensor-core paths as M grows. So the sweep over batch size is not optional decoration; it
is where the crossover point lives, and the crossover point is the deployment decision.

Verification procedure, executed before the benchmark, not after: capture the dispatched kernel
names with a profiler trace for both a decode step and a prefill step in each arm; assert the
expected fused symbol appears and no dequant-GEMM pair appears; record engine version, kernel
library version, driver and architecture in the run manifest. Repeat after any dependency bump,
because a patch-level engine upgrade can silently change dispatch.

Estimate, labelled: if fallback occurs, expect the INT4 arm to land roughly 5-15% SLOWER than BF16
at small batch (ESTIMATE, derived from adding a dequantization pass over the same 18 GB of
materialized weights plus unchanged GEMM cost). This is a falsifiable prediction and a cheap first
smoke test.""",
    ),
    (
        "calibration-provenance-first: the calibration set is a hidden training input and must be governed like one",
        """Hypothesis, pre-registered and falsifiable: quality on held-out slices whose distribution is
absent from the calibration set degrades no more than on slices that are represented. If the
unrepresented slices degrade materially more, the claim "INT4 preserves quality" is falsified and
downgraded to "INT4 preserves quality on the calibration distribution", which is a much weaker and
much more honest statement.

Data-driven WOQ methods fit scaling and clipping decisions to activation statistics gathered from a
calibration corpus. That corpus is an input to the model's final weights. Treating it as an
implementation detail is how teams ship a quantized model that quietly lost a language, a domain or
a long-context behaviour. Governance therefore mirrors training-data governance: the calibration
set is versioned, hashed, licence-checked, and pinned in the run manifest; it is disjoint from the
evaluation set by construction and the disjointness is verified by exact and near-duplicate match,
not asserted.

Contamination check that makes the result falsifiable: hold out a slice family entirely from
calibration, evaluate on it, and compare its degradation to a matched represented family. Report
both. If the gap is inside noise, the method generalizes; if not, the calibration set is the
binding constraint and expanding it is the cheapest next intervention.

Also controlled: calibration sample count and sequence length. Both change the fitted statistics.
Both are logged. A re-run with a different seed but identical count is executed to separate
calibration variance from method effect.

Estimate, labelled: calibration on 256-512 sequences of 2,048 tokens costs on the order of minutes
of single-GPU time for a 9B model (ESTIMATE, derived from one forward pass per sequence at
production prefill throughput). That cost is negligible against the study, so under-calibrating to
save time is never the right trade.""",
    ),
    (
        "unit-economics-first: latency percentages are not money, and only one denominator settles the argument",
        """Hypothesis, pre-registered and falsifiable: the INT4 arm reduces fully loaded cost per 1,000
output tokens at the fixed p95 SLO by at least 20%, where cost includes accelerator time, host
overhead, and the amortized engineering and validation effort over a 12-month horizon. A kernel
speedup that does not move this number falsifies the business case even if the kernel result is
real.

The stance exists because the request that started this study was about serving cost, and cost is
not latency. The correct denominator is GPU-seconds per 1,000 output tokens at a fixed SLO, because
that is the quantity that converts directly to instance-hours. Reporting tokens/s at unconstrained
latency is the classic way to claim a win that the SLO forbids you to collect: the throughput
optimum usually sits at a concurrency where p95 is already out of contract.

Procedure: for each arm, sweep concurrency, find the highest sustainable load whose p95 meets the
SLO, and read GPU-seconds per 1,000 output tokens at that point. Compare arms only at their
respective SLO-feasible points. Report the full frontier so the reader can see how fragile the
comparison is to the SLO choice.

Amortization, made explicit: quantization adds a recurring cost every time the base model changes,
because calibration, kernel validation and quality re-certification must be repeated. A team
shipping monthly model updates pays that tax monthly.

Numbers, all ESTIMATE with derivation shown: suppose BF16 needs 1.00 GPU-second per 1,000 output
tokens at SLO and INT4 needs 0.72. At a list price of about 2.0 USD per GPU-hour that is
(1.00-0.72)/3600 x 2.0 = 1.6e-4 USD saved per 1,000 output tokens. At 1e10 output tokens per month
the saving is about 1,600 USD/month. If re-certification costs 40 engineer-hours per model refresh
at 150 USD/hour, that is 6,000 USD per refresh, so monthly refreshes make the project
cash-negative. Every input here is an ESTIMATE and must be replaced with MEASURED throughput and
the organization's real rates before any decision.""",
    ),
    (
        "tail-behaviour-first: the mean is a marketing number and the p99 is the operational one",
        """Hypothesis, pre-registered and falsifiable: the INT4 arm's p99 inter-token latency under
production-shaped bursty arrivals is no worse than the BF16 arm's, at equal admitted concurrency.
A mean-throughput win accompanied by a p99 regression falsifies the deployment case, because the
user-visible contract is written on the tail.

Mechanism behind the tail risk: WOQ shifts where the step spends its time and changes memory access
patterns. Under bursty arrivals the scheduler mixes prefill chunks and decode steps within the same
iteration; because WOQ helps decode and not prefill, the variance of per-iteration time can rise
even when the mean falls. Larger admitted concurrency, made possible by freed KV memory, further
lengthens the queue and fattens the tail. Two effects, opposite signs, invisible in a mean.

Measurement design: open-loop Poisson arrivals at the target rate plus a burst generator, at least
30 minutes per arm per point, interleaved arms, warmup discarded. Record the full distribution of
TTFT and inter-token latency, report p50/p95/p99 and the number of SLO violations, not averages.
Preemption and recompute events are counted per arm; a difference in preemption rate is itself a
finding and must be reported rather than smoothed away.

Falsifiability aid: predict, before the run, the direction of the p99 change under the clamped-KV
arm (expected: neutral to slightly better, because bandwidth pressure per step drops without adding
queue depth) and under the unclamped arm (expected: worse, because queue depth rises). If the
observed signs contradict both predictions, the scheduler configuration differs across arms and the
run is void.

Estimate, labelled: with the decode step's weight-read component cut about 3.8x (ESTIMATE, from the
18 GB to 4.7 GB weight footprint above) and a MEASURED weight-read fraction f, per-step decode time
scales as (1-f) + f/3.8. At f = 0.7 that is 0.48, so a roughly 2.1x per-step improvement is the
ceiling; at f = 0.3 it is 0.78, only 1.3x. f must be measured, and the tail must be measured at the
same operating point.""",
    ),
    (
        "null-result-first: the study is designed so that finding nothing is a publishable, actionable outcome",
        """Hypothesis, pre-registered and falsifiable: INT4 weight-only quantization delivers no
statistically detectable GPU-seconds reduction at the fixed p95 SLO for this workload, where "no
detectable" is defined against a minimum detectable effect fixed in advance. The study is powered to
reject this null if a real 20% effect exists; failing to reject it is a complete answer and closes
the project cleanly.

Framing the null as the primary hypothesis is a deliberate defence against the incentive to keep
tuning until the desired number appears. The analysis plan, the arms, the slice list, the tolerance
budget, the stopping rule and the number of repetitions are all written down and hashed before the
first measurement. Any deviation is reported as a deviation.

Why a null is plausible here and worth pre-empting: if the production traffic is prefill-heavy
(retrieval-augmented, long prompts, short answers), f is small, decode is a minority of the step
budget, and WOQ's ceiling is close to 1.0x by construction. Measuring the prefill/decode token
ratio in production is therefore the cheapest experiment in the whole plan and should be run first;
it can falsify the project in an afternoon for zero GPU cost.

Stopping rule: fixed number of repetitions, no peeking-driven extension. If the confidence interval
for the cost delta contains zero and excludes the minimum detectable effect, the result is a null
and the recommendation is to spend the effort on batching, prefix caching or right-sizing instead.

Estimate, labelled: with per-arm run-to-run variation of about 3% (ESTIMATE, to be replaced by the
MEASURED standard deviation of five baseline repetitions), detecting a 20% effect with paired
comparison needs only a handful of repetitions; detecting a 5% effect needs many more. That
arithmetic is why the minimum detectable effect is declared before, not after.""",
    ),
    (
        "reversibility-first: the arm that cannot be un-shipped in one config flip is not eligible for the trial",
        """Hypothesis, pre-registered and falsifiable: the INT4 arm can be withdrawn from production
traffic and fully replaced by BF16 within a bounded, measured drain-and-flip window, with zero
request loss, and this is demonstrated by an actual rehearsal before the cost experiment begins. If
the rehearsal cannot meet the window, the arm is disqualified irrespective of its performance.

Operational safety is the reason this stance leads. A quantized deployment changes model weights,
which means it changes outputs. Regressions can be subtle, slice-specific and delayed, surfacing as
downstream quality complaints days later. The only acceptable posture is that reverting is cheap,
fast, rehearsed and does not depend on the very system under test.

Concrete requirements: BF16 replicas stay resident and warm for the entire trial, so rollback is a
routing weight change rather than a cold start; the quantized artifact is a separate immutable
versioned artifact with its own hash, never an in-place mutation of the BF16 one; traffic ramps
1% -> 5% -> 25% -> 50% with a bake period at each step and automatic halt on any guard breach; per-arm
output samples are logged with the arm label so that a later quality complaint can be attributed
without guesswork.

Guards that trigger automatic rollback without human deliberation: p95 SLO breach, quality-canary
pass-rate drop beyond the signed budget, any NaN or kernel fallback, error-rate rise, or preemption
rate above the baseline band.

Estimate, labelled: keeping BF16 replicas warm alongside the trial costs roughly one extra replica
set for the trial duration; at a two-week trial and a four-replica set at about 2.0 USD/GPU-hour
that is 4 x 336 x 2.0 = 2,688 USD (ESTIMATE, list price, no committed-use discount). That is the
explicit price of reversibility and it belongs in the project's cost line, not hidden.""",
    ),
    (
        "workload-profile-first: measure f in production before spending a single GPU-hour on quantization",
        """Hypothesis, pre-registered and falsifiable: the production traffic's decode weight-read
fraction f, measured on the live prefill/decode token mix and batch-size distribution, is high
enough that the analytic ceiling (1-f) + f/3.8 predicts at least a 20% step-time reduction. If the
measured f predicts less, the project is falsified before any quantization work starts.

This stance inverts the usual order: instead of quantizing and then discovering the payoff, it
derives the payoff analytically from a cheap production measurement and only then decides whether to
build. The required inputs are all obtainable from existing serving telemetry: the ratio of prompt
tokens to generated tokens, the distribution of concurrent sequences per iteration, and the fraction
of iteration time spent in decode versus prefill. None of them need a quantized build.

Boundary conditions that make or break the case, stated as testable predicates: (a) median batch
size at peak must sit in the bandwidth-bound region, verified by showing that step time is roughly
flat in batch size there; (b) generated-token share must be large enough that decode dominates the
GPU-second budget; (c) prefix-cache hit rate must not be so high that prefill is already nearly
free, which would shift the mix again. Each predicate is checked against telemetry and each has a
numeric threshold written down first.

If the predicates fail, the recommended alternatives are ranked on the same GPU-seconds denominator:
raise batch efficiency via chunked prefill tuning, raise prefix-cache hit rate, right-size replicas
to the diurnal curve, or serve a smaller model. Each is cheaper and carries no accuracy risk.

Estimate, labelled: for a workload with 4,000 prompt tokens and 200 generated tokens per request,
decode is a minority of total GPU time and f at the iteration level is small (ESTIMATE, derived from
the token ratio and the compute-bound nature of prefill); the analytic ceiling then sits well under
20% and the null should be expected. Replace with MEASURED per-phase timing before deciding.""",
    ),
    (
        "auditability-first: a cost claim that cannot be independently re-run by a second engineer does not count",
        """Hypothesis, pre-registered and falsifiable: an independent engineer, given only the published
manifest, can re-run both arms and reproduce the reported GPU-seconds-per-1,000-output-tokens delta
within the pre-declared run-to-run band. If the reproduction lands outside that band, the original
result is falsified as reported, regardless of whether the underlying effect is real.

The stance targets the actual failure mode of internal benchmark reports: the number is true of one
machine, one afternoon, one engine build and one undocumented flag, and nobody can tell which. The
remedy is mechanical. Every arm publishes an immutable manifest containing: model artifact hash,
quantized artifact hash, quantization method, group size and symmetry, calibration corpus hash and
sample count, engine and kernel library versions, driver and firmware versions, GPU SKU and clock
policy, container image digest, full server flag set, load-generator version and its seed, arrival
process and rate, prompt set hash, and the raw per-request latency traces rather than only the
summaries.

Raw traces matter because summaries cannot be re-analyzed. With traces, a reviewer can recompute
percentiles, re-slice by prompt length, and check whether a claimed win came from a handful of
outlier requests.

Deviation log: any manual retune, restart, or excluded run is recorded with a reason. Silent
exclusion of "bad runs" is the single most common way a false positive survives review.

Estimate, labelled: storing full per-request traces for a two-week trial at roughly 200 bytes per
request and about 1e7 requests is about 2 GB (ESTIMATE, derived from request volume times record
size), which is negligible against the value of being able to re-derive every number in the report.
The audit cost is not the storage; it is the discipline of writing the manifest before the run
rather than reconstructing it afterwards.""",
    ),
]


def build(stance, body):
    return (
        f"Analytical stance under test: {stance}.\n\n"
        f"{body}\n\n"
        f"{MECH}\n\n"
        f"{FAIR}\n\n"
        f"{ROLLBACK}"
    )


def main():
    corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
    seg = corpus[START:END]
    assert len(seg) == len(STANCES), (len(seg), len(STANCES))
    out = []
    for row, (stance, body) in zip(seg, STANCES):
        m = {x["role"]: x["content"] for x in row["messages"]}
        rec = {
            "source_id": row["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": m["user"],
            "source_assistant": m["assistant"],
            "corrected_answer": build(stance, body),
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 3,
            },
            "risks": [
                "Source assistant text is a grading rubric ('answer should ...'), not an answer; training on it teaches meta-commentary instead of engineering reasoning.",
                "Rubric omits the KV-capacity confound, so a naive study attributes concurrency gains to decode speedup.",
                "Rubric does not require kernel-dispatch verification; a silent dequant fallback can make the quantized arm slower than baseline.",
                "No rollback threshold or reversibility rehearsal is specified, so a quality regression has no pre-committed exit.",
                "Calibration set is named but not governed (no hash, no evaluation-set disjointness check), risking contaminated quality claims.",
            ],
            "evidence_required": [
                "MEASURED production prefill/decode token ratio and decode weight-read fraction f from profiler traces.",
                "MEASURED GPU-seconds per 1,000 output tokens at the fixed p95 SLO for BF16, INT4-unclamped and INT4-KV-clamped arms.",
                "Profiler kernel-dispatch trace proving the fused low-precision kernel is used with zero fallback in every arm.",
                "Per-slice paired quality results with bootstrap confidence intervals against the pre-signed tolerance budget.",
                "Run manifest with model/quantized-artifact/calibration-corpus hashes, engine, kernel-library, driver and container digests, plus raw per-request latency traces.",
            ],
            "confidence": 0.62,
        }
        out.append(rec)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("WROTE", OUT, len(out))
    print("IDS", [r["source_id"] for r in out])


if __name__ == "__main__":
    main()
