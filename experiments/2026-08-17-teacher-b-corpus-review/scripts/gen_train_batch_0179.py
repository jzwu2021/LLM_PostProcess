import json, os

EXP = '/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review'
CORPUS = '/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl'
START, N = 1780, 10
OUT = os.path.join(EXP, 'results', 'train-batch-0179.jsonl')

corpus = []
with open(CORPUS, encoding='utf-8') as f:
    for i, l in enumerate(f):
        if START <= i < START + N:
            corpus.append(json.loads(l))
assert len(corpus) == N

PREAMBLE = """
Shared assumptions I am making explicit (change these and the whole comparison changes):
- "Weight-only quantization" (WOQ) means weights stored and loaded at reduced precision and
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
 ("Calibration-provenance-first: the calibration corpus is a hidden hyperparameter and must be treated as a versioned, audited input",
  """Most WOQ result variance that teams blame on kernels is actually calibration variance. So the
calibration set gets the same treatment as a training set: documented source, size, sequence
length distribution, domain mix, hash, and an explicit statement of how it relates to production
traffic. The design runs at minimum two calibration variants (a production-sampled one and a
generic corpus one) through the identical quantization recipe, because the spread between them
is the honest error bar on any single quantized checkpoint. If the two variants differ by more
than the per-slice accuracy budget, no single quantized arm is representative and the study must
report the range rather than one lucky checkpoint. Calibration/eval disjointness is proven by
hash, not asserted, and the calibration set is stored alongside the checkpoint because a
checkpoint without its calibration provenance is not reproducible.""",
  """Falsifier at this layer: if two independently calibrated checkpoints from the same recipe
disagree beyond the accuracy budget, the claim "INT4 costs us X quality" is unsupported
regardless of how good the performance numbers look."""),

 ("Tail-first: the decision is owned by p99 behaviour and preemption, not by the mean",
  """Cost at a fixed SLO is a tail statistic, so I design around the tail from the start. Means and
p50 are reported only as context. The instrumented quantities are p95 and p99 TTFT and TPOT,
inter-token gap distributions rather than averaged TPOT, preemption and recompute counts, queue
wait separated from execution time, and the frequency of scheduler-induced stalls when KV blocks
run short. The specific concern with the Q-native arm is that larger admitted batches improve
mean throughput while lengthening the tail: more sequences share the same decode step, so the
slowest request waits behind more work, and KV pressure raises preemption. Aggregate tokens/s
can rise while the arm becomes unshippable under an existing p99 commitment.""",
  """Reporting rule: any arm whose p99 TPOT regresses is disqualified even if its mean throughput
and its p95 both improve, and the report must carry the inter-token gap histogram, not just
scalar percentiles, so a bimodal stall pattern cannot hide inside a percentile."""),

 ("Multi-tenant-first: measure on a host that shares what production shares, because isolation is the most common source of unreproducible wins",
  """A benchmark run on a quiet, exclusively owned node measures a machine that does not exist in
production. The design therefore names the sharing regime explicitly and measures both: an
isolated arm to establish the clean mechanism, and a co-tenanted arm reproducing the real
neighbours (other replicas on the same host, PCIe and NVLink contention, host CPU contention for
tokenization and sampling, shared network for weight loading and telemetry, cgroup limits, NUMA
placement). WOQ shifts the bottleneck away from HBM bandwidth, which means the co-tenanted arm
can expose a different limiting resource entirely: host-side sampling overhead, Python scheduler
loop, or PCIe traffic that was previously hidden behind weight loading.""",
  """Falsifiable prediction with a real decision attached: if the isolated speedup does not survive
co-tenancy within its CI, the fleet win is not real, and the actionable finding is the newly
exposed bottleneck rather than a verdict on quantization."""),

 ("Ops-burden-first: cost includes the operational surface the second numeric path adds, not just GPU-seconds",
  """The cost unit is written as total cost of ownership, and GPU-seconds is only its largest term.
The other terms are enumerated and estimated before the study: a second CI matrix and its
regression surface, quantization time on every model refresh, a second set of accuracy baselines
to maintain, incident modes that only appear on the quantized path (kernel/driver version
coupling, engine build pinning), on-call knowledge that must be doubled, and rollback rehearsal
maintenance. These are labelled ESTIMATE with a stated derivation (engineer-days per quarter x
loaded rate) and carried through to the go/no-go arithmetic. A 25% GPU-seconds saving on a small
fleet is routinely erased by these terms; on a large fleet it is negligible against them.""",
  """This makes fleet size the decisive variable and it is stated up front: the study asks whether
the saving times the fleet exceeds the recurring ops burden, and if the fleet is below that
break-even the study is declined before any GPU time is spent."""),

 ("Falsification-symmetry-first: give the null hypothesis the same engineering effort as the alternative",
  """The most common defect in these studies is asymmetric effort: the quantized arm is tuned,
retried and debugged, while the baseline is whatever the last deploy left behind. So the design
imposes symmetry as a procedural rule. Both arms get the same tuning budget in engineer-hours,
the same sweep of scheduler parameters, the same autotune warmth, the same number of retries
after anomalies, and anomaly-driven re-runs are permitted only if the same trigger would have
been applied to the other arm. Tuning actions per arm are logged so the asymmetry is auditable
after the fact rather than remembered charitably.""",
  """The pre-registered null is stated as the default outcome: absent evidence meeting the gate,
the conclusion is "no shippable difference", and the burden of proof sits entirely on the
quantized arm."""),

 ("Model-family-scope-first: state the generalization boundary, because a result on one architecture is not a result on the next",
  """A WOQ verdict is scoped to a specific architecture, size and precision layout, and the report
must say so on its first line. The mechanisms that break generalization are named: MoE models
move a different weight-byte profile per token so the byte-count bound does not transfer; GQA or
MQA changes the KV-to-weight byte ratio and therefore how much of decode time WOQ can touch;
very large models are more tolerant of 4-bit than small ones, so a 9B result must not be quoted
for a 2B; unusual activation-outlier distributions decide whether mixed-precision handling is
required at all. The plan therefore fixes one model and states explicitly which future models
would need a re-run and which would not, with the reason.""",
  """The falsifiable form: for any model claimed to be covered by this result, the decode fraction
of GPU time and the weight-bytes-per-token must be within a stated tolerance of the tested model,
and that check is cheap enough to run before every reuse of the conclusion."""),

 ("Data-pipeline-first: the numbers are only as trustworthy as the record-keeping that produced them",
  """Before any arm runs, I fix how raw records become reported numbers: per-request records with
arrival, first-token and completion timestamps, token counts, arm id, run id and configuration
hash, written append-only; a single analysis script that is the only permitted path from records
to figures; a declared warmup-exclusion rule and outlier policy fixed before data collection; and
a rule that no number appears in the report unless it can be regenerated from raw records by that
script. Summary-only telemetry is rejected, because aggregation choices (which requests count,
how percentiles are computed across repeats, whether failed requests are dropped) can move the
headline number by more than the effect being measured.""",
  """Validity check before trusting anything: an A/A run through the whole pipeline must return a
delta whose CI contains zero. If A/A shows a difference, the apparatus is measuring itself and
every A/B number from that pipeline is void until fixed."""),

 ("Deployment-mechanics-first: a comparison that ignores load, warmup and rollout dynamics understates the real cost of switching",
  """Steady-state serving cost is not the whole cost. The plan measures the transition surface as
well: checkpoint conversion time and storage, model load time into HBM from the artifact store,
CUDA graph capture and autotune warmup duration before the replica reaches steady state, and the
resulting time-to-ready that autoscaling depends on. A quantized artifact is smaller on disk,
which usually helps load time, but dequant-aware kernels can lengthen warmup and autotune. If a
replica takes materially longer to become serve-ready, the autoscaler must be re-tuned and the
headroom policy changed, and that headroom is a permanent cost that partially cancels the
steady-state gain.""",
  """Explicitly measured and reported: time-to-first-healthy-request per arm across >=3 cold starts,
plus the headroom change implied for the autoscaling policy, so the fleet arithmetic uses the
policy that would actually run rather than the current one."""),

 ("Cross-checking-first: derive the expected result from an independent model before measuring, and treat disagreement as the finding",
  """I write down a predicted decode-step time for each arm from an independent analytic model
before running anything: weight bytes per step divided by achieved HBM bandwidth, plus attention
bytes for the configured context length, plus a stated fixed per-step overhead. For the 9B
example at batch 1 this yields an ESTIMATE of roughly 18 GB / achieved-bandwidth for BF16 and
~5.0-5.3 GB / achieved-bandwidth for INT4, using the measured achieved bandwidth of the specific
GPU (not its datasheet peak) as the divisor; the ratio is the same 3.2-3.6x byte bound quoted
above and is an upper bound, not a prediction of end-to-end throughput. Measurement then either
lands near the prediction or does not, and the gap is the informative quantity: a measured
speedup far below the byte bound points to kernel efficiency, dequant overhead or a fallback,
while a measured speedup at or above it points to an uncontrolled confound such as extra KV
capacity or a differently tuned baseline.""",
  """This makes the study self-checking: the number that gets reported is not just the delta but
the ratio of measured to predicted, and any arm whose measured result exceeds its own analytic
upper bound is treated as evidence of a broken comparison rather than an unusually good one."""),

 ("Stakeholder-decision-first: write the one-page decision memo before the study, and design only the measurements that could change its verdict",
  """I draft the final memo first, with the numbers blank: the recommendation options (ship
fleet-wide, ship for a specific traffic class only, decline, defer pending a cheaper lever), the
threshold that selects each option, the named owner who decides, and the review date. Then the
experiment is designed backwards from those blanks, and any measurement that cannot move the
recommendation between options is cut from the plan. This removes the most expensive failure in
these studies, which is collecting a large and impressive set of numbers that leaves the decision
exactly where it started, and it forces the thresholds (25% cost gate, per-slice accuracy budget,
p99 tail constraint, ops-burden break-even) to be agreed while they are still cheap to argue
about.""",
  """The memo also pre-writes the negative outcome and its follow-up: if the gate is missed, the
named alternatives (prefix caching, chunked-prefill tuning, KV block tuning on BF16, replica
right-sizing, a smaller model) are already prioritised, so a null result ends in a next action
rather than in a re-litigation of the measurement."""),
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
