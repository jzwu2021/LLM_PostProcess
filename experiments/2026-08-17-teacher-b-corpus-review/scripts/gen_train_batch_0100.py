import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0100.jsonl")
START, N = 990, 10  # 0-based corpus index of first record

COMMON_HEAD = """Evaluation plan for a mixed short-prompt / long-generation LLM serving workload.

0. Scope and non-claims
This measures serving-system behavior only. Nothing here is evidence about model quality.
No number transfers across engine build, model size, quantization, sequence-length regime
or GPU SKU without a re-run.

1. Assumptions that bound every number
- Fixed weights, quantization, engine commit, and launch flags; record all four.
- Clocks pinned (nvidia-smi -lgc), persistence mode on, MIG off, so DVFS/thermal drift
  cannot be mistaken for a treatment effect.
- Closed-loop client is NOT used for latency claims; open-loop Poisson arrivals only,
  because a closed-loop harness self-throttles and hides queueing collapse.
- Load generator is off the serving host, and its own CPU is never the bottleneck
  (verify: generator CPU < 60% at peak).

2. Metric definitions (fixed before any run; ambiguity here invalidates comparison)
- TTFT = arrival timestamp at server ingress -> first output token flushed to socket.
  Ingress, not client send, so client-side network noise is separable.
- TPOT = (E2E - TTFT) / (output_tokens - 1); undefined and excluded when output_tokens < 2.
- Queueing delay = ingress -> first prefill scheduler admission. Reported separately;
  it is the only metric that isolates admission from compute.
- Throughput reported as three numbers: request/s, prefill tok/s, decode tok/s. A single
  aggregate tok/s is not falsifiable because prefill and decode have different roofline
  regimes (prefill compute-bound, decode memory-bandwidth-bound at small batch).
- P99 computed per stratum from raw per-request records, never from pre-averaged windows,
  and never averaged across strata (Simpson's paradox risk).

3. Workload control
Stratify by (input_len, output_len): S=(256, 64), M=(1k, 256), L=(4k, 2k), plus a
long-tail stratum XL=(16k, 2k) if the deployment allows it. Mixture ratio is FIXED per
run (e.g. 60/25/12/3) and drawn from a seeded generator; report the seed. Same prompt
set across arms, hash-verified, so tokenizer-length drift cannot differ between arms.

4. Procedure
- Warmup: discard the first 120 s and the first 200 requests; report both cut points.
- 5 independent repeats per arm, fresh process each time (KV-cache fragmentation and
  allocator state are path-dependent). Randomize arm order across repeats to break
  drift confounding.
- Sweep offered load from 20% to 130% of measured saturation in >=6 steps; the knee is
  the interesting region and a single load point cannot show it.
- Report medians of per-repeat P99 plus the min/max across repeats. Do not pool repeats
  into one P99: that hides run-to-run variance.
"""

COMMON_TAIL = """
7. Confounders to instrument explicitly
- Prefix-cache / radix-cache hit rate: a higher hit rate in one arm makes TTFT look
  better for reasons unrelated to the treatment. Report hit rate per arm; if it differs
  by >2 percentage points, the comparison is void.
- Continuous-batching batch-size distribution over time, not just its mean.
- Preemption / recompute events, and swapped-out sequences.
- KV-cache utilization and any allocator OOM-retry path.
- Tokenizer version, sampling params, and EOS behavior (early stopping shortens
  generations and inflates apparent decode throughput).
- Host-side: NUMA placement, CPU governor, and network RTT to the generator.

8. Telemetry to capture
Per-request trace records (arrival, admit, first-token, completion, in/out token counts,
cache-hit flag, preemption count) written to disk, not just histograms. Plus 1 Hz
nvidia-smi / DCGM sampling for SM utilization, memory-bandwidth utilization, power and
clock throttle reasons. Engine-level scheduler counters at 1 Hz.

9. Rollback / stop gates (decide before the run)
- Abort the arm if any GPU reports a thermal or power throttle reason during the
  measured window: the sample is invalid, not merely noisy.
- Abort if error rate > 0.5% or if any request is dropped by admission control unless
  dropping is the treatment.
- Promote only if the pre-registered primary metric improves beyond the noise band
  (defined as 2x the max across-repeat spread observed in the control arm) AND no
  stratum regresses on P99 E2E by more than 5%.
- Rollback within one deploy cycle on: P99 E2E regression > 10% for any stratum,
  starvation (long-generation completion rate drop), or OOM/preemption rate increase.

10. Honest limits
Five repeats bound run-to-run variance only coarsely; P99 on a few thousand requests has
wide confidence intervals, so quote the interval (e.g. binomial/bootstrap CI on the 99th
percentile) rather than a bare number. Any effect smaller than that interval is not a
result.
"""

# Per-item distinct hypothesis + controlled experiment blocks.
BLOCKS = [
    # 1091
    """
5. Falsifiable hypothesis (primary)
H1: Capping the per-iteration prefill token budget (chunked prefill) at 2048 tokens
reduces P99 TTFT of the S stratum by >=25% at 90% of saturation load, while reducing
aggregate decode throughput by <5%.
Null: the P99 TTFT change is within the control-arm noise band, or decode throughput
falls >=5%.
This is falsifiable because both quantities are measured on the same runs and the
decision rule is fixed in advance.

6. Controlled experiment
Two arms, identical in everything except the prefill chunk budget (unbounded vs 2048).
Same seed, same prompt set, same load ladder, 5 repeats, interleaved arm order.
Mechanism being tested: long prefills monopolize a scheduler iteration and push queued
short requests behind a head-of-line block; chunking bounds that blocking interval at the
cost of extra kernel-launch and re-attention overhead.
Boundary conditions: the effect should shrink toward zero as the L/XL stratum share goes
to 0 (no long prefills to block on), and should reverse into a throughput loss when the
chunk size is small enough that prefill GEMMs stop being compute-efficient. Run a third
arm at chunk=512 specifically to look for that reversal; if the reversal does not appear,
the proposed mechanism is wrong and the result should not be generalized.
""",
    # 1092
    """
5. Falsifiable hypothesis (primary)
H1: The observed P99 E2E latency spikes are caused by KV-cache exhaustion triggering
sequence preemption/recompute, not by network or client effects. Concretely: >=80% of
requests above the P99 E2E threshold overlap in time with a nonzero preemption counter
delta, and raising gpu_memory_utilization so that peak KV occupancy stays below 85%
removes >=70% of those spikes.
Null: spike timestamps are uncorrelated with preemption events (or the correlation
survives the memory change), which would redirect the diagnosis to admission/queueing or
the host network path.

6. Controlled experiment
Arm A: current configuration. Arm B: same, with KV budget raised (or max_num_seqs lowered)
so measured peak KV occupancy < 85%. Everything else pinned, 5 repeats, interleaved.
Diagnostic joins per-request traces against the 1 Hz preemption counter, so the claim is
tested on overlap, not on aggregate means.
Boundary conditions: if spikes persist in Arm B at unchanged rate, the KV hypothesis is
dead and the next candidates are (i) ingress queueing under bursty arrivals, (ii) CPU-side
tokenization/detokenization stalls, (iii) a noisy neighbor on the host. Explicitly rule
out the client by comparing server-ingress TTFT against client-observed TTFT: if the gap
carries the spike, the problem is off-GPU.
""",
    # 1093
    """
5. Falsifiable hypothesis (primary)
H1: In the decode phase the service is memory-bandwidth-bound, not compute-bound. Test:
across the batch-size sweep, decode tok/s scales sub-linearly with SM utilization and
tracks achieved DRAM bandwidth; specifically, measured decode tok/s stays within 20% of
the roofline estimate tok/s ~= (achieved_BW) / (bytes_read_per_token), where
bytes_read_per_token ~= weight_bytes/batch + KV_bytes_per_token_per_seq.
Null: decode tok/s is >20% below that bound at every batch size, which implies the
bottleneck is elsewhere (kernel launch overhead, scheduler CPU time, or sampling).

6. Controlled experiment
Single configuration, batch size swept by controlling offered load in >=6 steps with
max_num_seqs fixed high enough not to bind. Capture DCGM DRAM-bandwidth utilization and
SM occupancy per step. Compute the roofline prediction independently from model config
(hidden size, layers, dtype, KV head count) before looking at measurements, and record
the prediction first so the comparison cannot be fitted after the fact.
Boundary conditions: the bandwidth-bound regime must weaken as batch grows (weights are
amortized) and must not hold at all during prefill, which should sit near the compute
roofline. If prefill also looks bandwidth-bound, the instrumentation or the byte
accounting is wrong; fix that before drawing any tuning conclusion.
""",
    # 1094
    """
5. Falsifiable hypothesis (primary)
H1: Disaggregating prefill and decode onto separate replica pools reduces P99 TTFT for
the S stratum by >=30% at 90% saturation without lowering aggregate decode tok/s by more
than 10%, provided the KV transfer cost per request stays under 15% of that request's
TTFT budget.
Null: either the TTFT gain is inside the noise band, or KV transfer cost exceeds the 15%
budget, in which case colocated continuous batching remains the correct design.

6. Controlled experiment
Arm A: colocated engine. Arm B: prefill pool + decode pool with the same total GPU count
and the same model/quantization, KV handed over via the interconnect. Report the KV
transfer time as a first-class measured metric (bytes moved and achieved link bandwidth),
because the entire design rests on it.
Mechanism: separation removes prefill/decode interference inside a scheduler iteration,
at the price of an explicit cross-node KV movement and a second queue.
Boundary conditions: the gain should grow with the share of long prefills and shrink to
zero (then negative) as prompts get short, since transfer overhead is then amortized over
nothing. Also test an unbalanced pool split (e.g. 1:3 vs 1:1) to show the result is a
property of disaggregation and not of accidentally right-sized pools.
Do not claim any interconnect capability (RDMA/GDR path, achieved GB/s) that was not
measured on this cluster; assert only the numbers the run produced.
""",
    # 1095
    """
5. Falsifiable hypothesis (primary)
H1: The tail latency is dominated by admission queueing rather than by execution. Test:
at 100% of saturation, queueing delay accounts for >=60% of P99 E2E for the S stratum,
and reducing max_num_seqs (shrinking the in-flight set) shifts time from execution into
queue without improving P99 E2E by more than the noise band.
Null: queueing is <60% of P99 E2E, which relocates the problem to per-iteration execution
time and makes batching/kernel-level work the right lever.

6. Controlled experiment
One configuration swept across the load ladder, with queueing delay measured directly at
ingress->admit. Then two arms at fixed 100% load with max_num_seqs at 1x and 0.5x.
Because Little's law ties in-flight count, arrival rate and residence time, an intervention
that only moves time between queue and execution while leaving E2E flat is the signature
of a saturated server: it tells you to add capacity or shed load, not to tune the scheduler.
Boundary conditions: below ~70% load the queueing share must fall sharply; if it does not,
arrivals are more bursty than the assumed Poisson process and the load model itself must
be re-derived from production traces before any tuning conclusion is drawn.
""",
    # 1096
    """
5. Falsifiable hypothesis (primary)
H1: Enabling prefix caching improves P99 TTFT by >=20% for this workload only because the
synthetic prompt set shares a system preamble; with the shared preamble randomized per
request the improvement drops below 5%.
Null: the improvement survives preamble randomization, which would mean the gain comes
from genuine intra-request reuse and generalizes.
This hypothesis is deliberately aimed at invalidating our own benchmark rather than at
confirming a win.

6. Controlled experiment
Four arms: (cache on/off) x (shared preamble / randomized preamble), same seed, same load
ladder, 5 repeats. Prefix-cache hit rate is reported per arm and is the mediating variable;
if hit rate does not move between preamble conditions, the manipulation failed and the
result is uninterpretable.
Boundary conditions: the effect must be confined to TTFT and to the prefill token/s
counter; decode tok/s should be unchanged. Any decode change indicates a confound (e.g.
freed KV blocks changing the batch-size distribution) that must be explained before the
TTFT number is quoted.
Operational note: prefix caching across tenants is also a data-isolation question. Do not
enable cross-request reuse in a shared deployment without an explicit isolation review.
""",
    # 1097
    """
5. Falsifiable hypothesis (primary)
H1: Increasing tensor-parallel degree from TP=2 to TP=4 lowers P99 TTFT on the L stratum
by >=25% but lowers decode tok/s per GPU by >=15%, because decode is latency-bound on
per-layer all-reduce while prefill benefits from the extra compute.
Null: either the TTFT gain is within noise, or per-GPU decode throughput does not degrade
by 15%, which would make the interference model wrong.

6. Controlled experiment
Arms: TP=2 and TP=4 at equal total GPU count (so replica count halves), identical model
and quantization, identical load ladder, 5 repeats. Report per-GPU decode tok/s, not
per-replica, so the arms are comparable.
Instrument collective cost directly: per-iteration all-reduce time from engine counters,
plus an independent NCCL bandwidth/latency measurement at the message sizes actually used
(decode all-reduces are small and latency-dominated; prefill ones are large and
bandwidth-dominated). Record the measured intra-node topology from nvidia-smi topo -m
rather than assuming a link type.
Boundary conditions: if the small-message collective latency measured out-of-band does not
account for most of the decode regression, the explanation is not the collective and the
result must not be generalized to other TP degrees or other topologies. Confirm that no
arm silently crossed a NUMA or PCIe-host-bridge boundary, which would change the mechanism
entirely.
""",
    # 1098
    """
5. Falsifiable hypothesis (primary)
H1: The intermittent throughput collapse is a fragmentation effect: it appears only after
the process has served >N requests with heterogeneous sequence lengths, and it disappears
after a process restart with identical config and load.
Null: the collapse reproduces immediately from a cold start at the same offered load,
which makes it a steady-state capacity problem rather than a state-accumulation bug.

6. Controlled experiment
Arm A: long soak, single process, >=6 hours at 85% saturation with the mixed stratum
distribution. Arm B: same total request count split across processes restarted every 30
minutes. Compare throughput and P99 E2E as functions of wall-clock and of cumulative
requests served. Track KV block allocator statistics (free-block count, largest free run,
preemption/recompute counters) at 1 Hz.
Because the treatment is only "restart", a difference cannot be attributed to load.
Boundary conditions: if degradation tracks wall-clock rather than cumulative requests,
suspect thermal drift or a host-level leak instead; check clock throttle reasons and host
RSS. If neither pattern holds, do not ship a periodic-restart mitigation, since it would
add availability risk without an established mechanism.
Safety: a periodic-restart mitigation, if ever adopted, must be drain-then-restart with a
health gate, never a hard kill of in-flight requests.
""",
    # 1099
    """
5. Falsifiable hypothesis (primary)
H1: Reported P99 latency is dominated by measurement artifacts of a closed-loop harness:
switching from closed-loop with fixed concurrency to open-loop Poisson arrivals at the
same mean throughput raises measured P99 E2E by >=40% at loads above 90% saturation.
Null: the two harnesses agree within the noise band, which would mean the closed-loop
numbers already reflect queueing and prior results stand.

6. Controlled experiment
Arms: closed-loop (concurrency tuned to hit throughput X) vs open-loop Poisson (arrival
rate tuned to the same X), identical server config, same seed and prompt set, 5 repeats.
Coordinated omission is the mechanism: a closed-loop client cannot issue the request that
would have queued, so it silently deletes the worst samples. Verify by checking that
open-loop inter-arrival gaps match the intended distribution (KS test) and that the
closed-loop arm's effective arrival rate dips exactly when server latency rises.
Boundary conditions: the discrepancy must vanish at low load (<50% saturation) where
queues are empty. If it does not vanish, the harness difference is not coordinated
omission and something else differs between the two clients; find it before publishing
either number.
Consequence if H1 holds: all previously reported P99 figures from the closed-loop harness
must be relabeled as lower bounds, not corrected by a fudge factor.
""",
    # 1100
    """
5. Falsifiable hypothesis (primary)
H1: A length-aware admission policy that routes requests declaring output_len <= 128 to a
short queue with a reserved share of scheduler slots reduces P99 TTFT of the S stratum by
>=30% at 90% saturation, without increasing P99 E2E of the L/XL strata by more than 10%
and without reducing their completion rate.
Null: any of those three gates fails, in which case the policy is rejected regardless of
how good the S-stratum number looks.

6. Controlled experiment
Arms: FCFS baseline vs length-aware reservation, identical everything else, 5 repeats,
interleaved order, full load ladder. Three pre-registered gates as above; per-stratum
metrics only, never pooled.
Explicit starvation instrumentation: max queue wait and completion rate for L/XL, plus
the fraction of L/XL requests exceeding their SLO. Starvation of the heaviest users is
the known failure mode of any priority scheme and must be measured, not argued away.
Adversarial arm: clients declaring output_len=64 but generating 2000 tokens. Declared
length is untrusted input; if the policy degrades under mis-declaration, it needs a
runtime demotion rule (reclassify once a request exceeds its declared budget) before it
can ship.
Boundary conditions: the benefit must shrink as the workload becomes homogeneous in
length; if a gain persists on a single-stratum workload, the measured effect is not
length-awareness and the causal story is wrong.
Rollback: revert on any SLO breach for L/XL or any drop in their completion rate.
"""
]

RISKS = [
    "source_assistant is a grading rubric, not an answer; supervising on it teaches meta-commentary instead of engineering content",
    "no metric definitions in the source, so TTFT/TPOT/P99 are not comparable across runs or arms",
    "closed-loop load generation causes coordinated omission and silently understates tail latency",
    "pooling P99 across strata or across repeats hides both Simpson's paradox and run-to-run variance",
]

EVIDENCE = [
    "engine commit, launch flags, quantization and pinned clock settings for every arm",
    "per-stratum TTFT/TPOT/queueing/E2E percentiles with sample counts and across-repeat spread",
    "prefix-cache hit rate, preemption counts and KV occupancy per arm to rule out confounds",
    "raw per-request trace records retained, not just aggregated histograms",
]

EXTRA_RISK = {
    0: "chunked prefill adds re-attention overhead; a too-small chunk can regress decode throughput",
    1: "attributing spikes to KV pressure without joining against preemption timestamps is a correlation error",
    2: "roofline byte accounting is easy to get wrong (KV dtype, GQA head count), which fakes a bottleneck",
    3: "disaggregation claims often assume an interconnect capability that was never measured on the cluster",
    4: "if arrivals are burstier than Poisson, the whole load model and every derived percentile is wrong",
    5: "shared system preamble in synthetic prompts inflates prefix-cache wins that do not exist in production",
    6: "changing TP degree also changes replica count and NUMA placement; unmeasured, that confounds the result",
    7: "periodic restart as a mitigation adds availability risk if it is not drain-gated",
    8: "prior P99 numbers from a closed-loop harness may need relabeling as lower bounds, not rescaling",
    9: "client-declared output length is untrusted input; the policy fails silently without runtime demotion",
}

EXTRA_EV = {
    0: "chunk-size sweep including a small-chunk arm that should show the predicted throughput reversal",
    1: "time-joined per-request outliers against 1 Hz preemption counters, plus client-vs-ingress TTFT gap",
    2: "pre-registered roofline prediction recorded before measurement, with DCGM DRAM bandwidth traces",
    3: "measured KV transfer bytes and achieved link bandwidth per request, plus nvidia-smi topo -m output",
    4: "ingress->admit queueing delay share of P99 at each load step, and production inter-arrival trace",
    5: "prefix-cache hit rate under shared vs randomized preamble, proving the manipulation worked",
    6: "out-of-band NCCL small- and large-message latency/bandwidth at the sizes the engine actually uses",
    7: "KV allocator free-block statistics over a 6h soak vs periodic-restart arm",
    8: "KS test on open-loop inter-arrival gaps and closed-loop effective arrival-rate dips",
    9: "per-stratum starvation metrics and results of the mis-declared max_tokens adversarial arm",
}

QD = {
    0: (3, 2, 3), 1: (3, 2, 3), 2: (3, 2, 3), 3: (3, 2, 3), 4: (3, 2, 3),
    5: (3, 2, 3), 6: (3, 2, 3), 7: (3, 2, 3), 8: (3, 2, 3), 9: (3, 2, 3),
}
CONF = [0.72, 0.70, 0.71, 0.68, 0.72, 0.71, 0.67, 0.69, 0.73, 0.70]

rows = []
with open(os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")) as f:
    for i, line in enumerate(f):
        if i < START:
            continue
        if i >= START + N:
            break
        d = json.loads(line)
        m = {x["role"]: x["content"] for x in d["messages"]}
        k = i - START
        tc, ic, os_ = QD[k]
        rows.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": m["user"],
            "source_assistant": m["assistant"],
            "corrected_answer": COMMON_HEAD + BLOCKS[k] + COMMON_TAIL,
            "quality_dimensions": {
                "technical_correctness": tc,
                "instruction_coverage": ic,
                "operational_safety": os_,
            },
            "risks": RISKS + [EXTRA_RISK[k]],
            "evidence_required": EVIDENCE + [EXTRA_EV[k]],
            "confidence": CONF[k],
        })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(rows), "records:", rows[0]["source_id"], "->", rows[-1]["source_id"])
