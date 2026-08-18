import json, os

BASE = "/home/johnson/workspace/LLM_PostProcess"
SRC = f"{BASE}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{BASE}/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0221.jsonl"

rows = [json.loads(l) for l in open(SRC)]
sl = rows[2200:2210]

COMMON = """Common frame (applies to every stance below).
Assumptions (must be restated by the answering engineer, not inherited silently):
A1. Single node, 8 GPUs, NVLink/NVSwitch intra-node; inter-node paths only appear where a stance says so explicitly.
A2. Decode-dominant, latency-sensitive serving: the SLO is TTFT p95 and TPOT (inter-token latency) p95, never a mean.
A3. Model weights fit in aggregate HBM with at least 20% KV-cache headroom at target concurrency.
A4. Exactly one variable moves per arm: no simultaneous change of quantization, batching policy, or speculative decoding.
Mechanism, stated plainly:
- Tensor parallelism (TP) shards every layer's GEMMs. Each transformer block needs two all-reduces (after attention out-projection and after the MLP down-projection), so decode carries L * 2 * allreduce_latency of synchronous cost, where L is the layer count. TP is latency-additive in collectives but capacity- and bandwidth-multiplying: per-GPU weight bytes and per-GPU KV bytes both fall by the TP degree.
- Pipeline parallelism (PP) shards layers into stages. Per token it adds only (PP-1) small point-to-point hidden-state sends, which are cheap, but a single request serializes through all stages and the bubble fraction is (PP-1)/(micro_batches + PP-1). At low concurrency there are too few in-flight micro-batches to fill the pipeline, so PP loses badly on single-request latency.
Boundary conditions that flip the answer:
- B1. On NVLink-class fabric, small-message all-reduce latency is in the single-digit microseconds and TP up to 8 is normally latency-viable. Over PCIe-only or across nodes on RoCE/IB the same collective's latency floor rises roughly an order of magnitude and TP stops paying past TP=2 (ESTIMATE; derivation: decode all-reduce payload is hidden_size * dtype_bytes per token per layer, which is small, so the collective is latency-bound rather than bandwidth-bound and the per-hop latency floor dominates).
- B2. If the model cannot fit on one GPU, sharding is mandatory and the question reduces to which axis, not whether.
- B3. Under high, steady concurrency the PP bubble amortizes and PP becomes competitive on throughput per GPU while still losing on single-request latency.
Default recommendation: use TP inside the node up to the point where collective cost stops being repaid by reduced per-GPU memory traffic; use PP only to cross a node boundary or to fit a model TP alone cannot fit. PP is not a latency optimization.
Measurement and evidence policy: every number below that was not produced by a run on this hardware is labelled ESTIMATE and carries its derivation. Only values read out of named benchmark artifacts may be labelled MEASURED. This review reports no MEASURED values, because no benchmark was executed for it."""

CRITIQUE = """Critique of the source item: the prompt is a legitimate infrastructure question and does ask for assumptions, a falsifiable hypothesis, measurements, confounders and rollback criteria, but the corpus pair is degenerate - the assistant turn contains only a rubric describing what an answer should contain, not an answer. There is therefore no substantive content to keep, and the item is rewritten into a complete response that supplies the mechanism, the boundary conditions that flip the recommendation, an explicit falsifiable hypothesis, a single-variable controlled experiment, the evidence artifacts required to adjudicate it, and a rollback gate. Every quantitative claim is labelled ESTIMATE and carries its derivation; no value here is MEASURED, because no benchmark run was performed for this review. This output is provisional teacher-B review material, not expert gold, and it is not evidence about any model's domain capability."""

BASE_RISKS = "Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning."
TAIL_RISK = "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."

STANCES = [
 (210,
  "CPU-side launch overhead and CUDA graph capture can dominate the decode step at small batch, so a TP-versus-PP comparison run without graph capture may be measuring the Python and launch path rather than the parallelism layout.",
  "At decode with small batch the per-layer GEMMs are tiny and the GPU is not the bottleneck; the host must issue a long sequence of kernel launches and collective calls each step. TP multiplies the number of launched collectives per step by the layer count, which is precisely the traffic that CUDA graph capture removes by replaying a pre-recorded launch sequence. PP issues fewer collectives but adds cross-stage scheduling that may itself be host-driven. Comparing an eager-mode TP arm against an eager-mode PP arm therefore conflates layout cost with launch cost, and the ranking can invert once graphs are enabled.",
  "H210: enabling CUDA graph capture on both arms reduces TPOT p95 by an amount that differs between the TP and PP arms by more than run-to-run dispersion, and the layout ranking at batch size 1 is not preserved between the eager and graph-captured configurations (ESTIMATE; derivation: launch overhead scales with the number of launched operations per step, which grows with layer count under TP but not under PP, so graph capture removes an asymmetric amount of host time from the two arms).",
  "Run each layout twice, eager and graph-captured, with everything else pinned; capture a host-side profile per run showing kernel launch count and host gap time per decode step; confirm graph capture actually engaged rather than silently falling back.",
  "If graph capture silently fails or falls back on either arm, discard that arm's numbers entirely and re-run; do not ship a layout decision derived from a comparison in which only one arm was graph-captured.",
  [BASE_RISKS,
   "Host launch overhead is attributed to the parallelism layout, inverting the conclusion at small batch.",
   "Graph capture failing back to eager mode silently, so one arm is measured under a different execution path than intended.",
   "Profiling only GPU kernel time, which hides the host gaps that actually set decode latency at batch size 1.",
   TAIL_RISK],
  ["Host-side profile per arm showing kernel launch count, host gap time and total step wall time.",
   "Explicit log line or counter proving CUDA graph capture engaged, per arm and per configuration.",
   "TPOT p50/p95/p99 for four cells: TP-eager, TP-graph, PP-eager, PP-graph, at least three repeats each.",
   "Batch-size sweep showing where the launch-bound regime ends and the compute-bound regime begins."]),

 (211,
  "Attention head count divisibility is a hard constraint on TP degree, and a sweep that quietly pads or reshapes heads to make a TP degree legal has changed the model, not just the layout.",
  "Tensor parallelism for attention splits along the head dimension, so TP degree must divide the number of key/value heads under grouped-query attention, not merely the number of query heads. When it does not divide, frameworks either refuse, replicate KV heads across ranks, or pad. Replication silently increases per-GPU KV bytes and undoes the memory saving that motivated TP; padding changes the arithmetic. Either way the arm at that TP degree is no longer the same model configuration as its neighbours, so the sweep is not a clean single-variable sweep and any inflection at that point is an artifact.",
  "H211: at TP degrees that do not evenly divide the KV head count, measured per-GPU KV bytes exceed total_kv_bytes/TP by a margin larger than allocator overhead, revealing KV-head replication rather than a genuine sharding win (ESTIMATE; derivation: if a framework replicates KV heads to satisfy divisibility, per-GPU KV bytes scale with ceil(kv_heads/TP)*TP/kv_heads rather than with 1/TP, which is a computable, checkable ratio).",
  "For each TP degree in the sweep, dump the model's per-rank head assignment and per-GPU KV allocation at fixed concurrency; compare measured per-GPU KV bytes against the ideal 1/TP prediction; restrict the reported sweep to TP degrees that divide the KV head count evenly.",
  "If any TP degree in the sweep shows KV replication, drop that point from the comparison rather than reporting it; if the target deployment requires that degree, the rollback is to the nearest dividing degree with its memory footprint re-measured.",
  [BASE_RISKS,
   "TP degree treated as a free continuous knob when it is constrained by KV head count under grouped-query attention.",
   "Silent KV-head replication presented as a memory saving that did not occur.",
   "An inflection point in the sweep caused by a configuration change being read as a fabric or collective effect.",
   TAIL_RISK],
  ["Per-rank head assignment dump for every TP degree in the sweep.",
   "Measured per-GPU KV bytes at fixed concurrency versus the ideal 1/TP prediction, per TP degree.",
   "Model configuration showing query head count, key/value head count and head dimension.",
   "Framework log lines showing whether replication or padding was applied at each degree."]),

 (212,
  "Chunked prefill and the prefill/decode mix change which layout wins, because prefill is compute- and bandwidth-bound while decode is latency-bound, and a single blended benchmark reports an average of two opposite regimes.",
  "Prefill runs large GEMMs over the whole prompt and is throughput-bound; TP helps it almost linearly because the work per GPU falls and the collectives are amortized over large tensors. Decode runs one token at a time and is latency-bound; TP's per-layer collectives are pure overhead relative to the tiny amount of compute. If prefill and decode are interleaved in the same engine, the observed latency is a weighted blend whose weights depend on the input/output length distribution of the trace. Change the trace's prompt-to-generation ratio and the layout ranking can move without anything about the layout changing.",
  "H212: holding the layout fixed and varying only the trace's prompt-to-generation token ratio moves the TP-versus-PP latency gap by more than run-to-run dispersion, demonstrating that the comparison is sensitive to workload shape and not a property of the layouts alone (ESTIMATE; derivation: prefill cost scales with prompt tokens and decode cost with generated tokens, and TP's benefit has opposite sign in the two phases, so the blended metric is a weighted sum whose weights are set by the ratio).",
  "Run both layouts against at least three traces with deliberately different prompt-to-generation ratios, holding concurrency, chunked-prefill chunk size and scheduler policy fixed; report TTFT and TPOT separately, never blended into one latency number.",
  "If the layout ranking is not stable across the three traces, do not ship a general recommendation: pin the decision to the production trace's measured ratio and set an alert on that ratio drifting outside the tested range.",
  [BASE_RISKS,
   "Prefill and decode blended into a single latency metric, averaging two regimes with opposite sensitivity to TP.",
   "Benchmark trace shape unrepresentative of production, making the ranking unreproducible in the real workload.",
   "Chunked-prefill chunk size left at different defaults per arm, changing the interleaving granularity.",
   TAIL_RISK],
  ["Prompt-length and generation-length distributions for every trace used, not just the means.",
   "TTFT and TPOT reported separately per trace and per layout, p50/p95/p99.",
   "Chunked-prefill configuration dump per arm proving identical chunk size and scheduling policy.",
   "Production trace statistics to justify which of the tested ratios is the operative one."]),

 (213,
  "Thermal and power capping make later runs systematically slower than earlier ones, so an A/B comparison run back-to-back without clock control can attribute a drift artifact to the layout tested second.",
  "Sustained GPU load raises die temperature and board power draw; once the card hits its power or thermal limit the clocks drop and stay down until the load relents. A benchmark that runs arm A then arm B on the same devices measures arm B under lower clocks. The effect is systematic, not random, so repeating the runs in the same order does not average it out, and confidence intervals computed over same-order repeats will be misleadingly tight.",
  "H213: interleaving the arm order (A,B,B,A) rather than running them sequentially changes the measured TP-versus-PP latency gap by more than the within-arm dispersion, indicating a thermal or power drift component in the original result (ESTIMATE; derivation: clock throttling is a monotone function of accumulated thermal load within a session, so it aliases perfectly onto run order and can only be separated by randomizing or counterbalancing that order).",
  "Counterbalance run order, insert a fixed cooldown between arms, and log SM clock, memory clock, board power and any throttle reason counter at fixed sampling interval throughout every run; discard any run whose clock trace deviates from the session baseline beyond a pre-declared band.",
  "If clock traces show throttling in any accepted run, discard the entire comparison and re-run with locked clocks or longer cooldowns; do not correct throttled numbers analytically after the fact.",
  [BASE_RISKS,
   "Run order aliased onto thermal drift, producing a systematic bias that repeats do not remove.",
   "Clock and power telemetry not captured, so throttling is invisible in the result artifact.",
   "Other tenants or background processes on the same node perturbing power headroom between arms.",
   TAIL_RISK],
  ["Per-run time series of SM clock, memory clock, board power and throttle reason counters.",
   "Run order log proving counterbalancing, with cooldown intervals recorded.",
   "Baseline idle and warm-up clock traces for the session, to define the acceptance band.",
   "Node exclusivity evidence showing no competing workload during the comparison window."]),

 (214,
  "Speculative decoding interacts with the layout: the draft model's placement and the verification step's batch shape change the collective pattern, so enabling speculation on one arm silently breaks the single-variable rule.",
  "Speculative decoding replaces one-token decode steps with propose-then-verify rounds. Verification processes several candidate tokens at once, which enlarges the per-step tensors and pushes the collective from the small-message latency-bound regime toward the bandwidth-bound regime, changing TP's cost profile. The draft model must also live somewhere: colocated on the same GPUs it competes for HBM and SMs; placed on separate GPUs it changes the effective GPU count per replica. Under PP the draft and verify phases interact with pipeline stage boundaries differently again.",
  "H214: enabling speculative decoding changes the sign or magnitude of the TP-versus-PP latency gap by more than run-to-run dispersion, and the acceptance rate differs between the two layouts by less than dispersion, isolating the effect to execution rather than to draft quality (ESTIMATE; derivation: verification batches several tokens per step, which multiplies collective payload by the speculation length and moves the message size across the latency-to-bandwidth crossover, while draft quality is a property of the draft model and should be layout-invariant).",
  "Measure four cells: TP and PP, each with speculation off and on, holding draft model, speculation length and acceptance threshold fixed; record acceptance rate per cell to confirm draft behaviour did not change; report TPOT per accepted token, not per step.",
  "If acceptance rate differs between layouts beyond dispersion, the arms are not comparable and the speculation result must be discarded; roll back to the non-speculative comparison for the layout decision.",
  [BASE_RISKS,
   "Speculation enabled on only one arm, violating the single-variable discipline the prompt demands.",
   "Latency reported per step rather than per accepted token, which flatters speculation regardless of layout.",
   "Draft model placement changing the effective GPU count per replica, so the two arms use different resources.",
   TAIL_RISK],
  ["Acceptance rate and speculation length per cell, with the draft model identity and version recorded.",
   "TPOT per accepted token, p50/p95/p99, for all four cells, at least three repeats each.",
   "Draft model placement and per-GPU memory accounting proving identical resource footprints across arms.",
   "Collective message-size histogram per cell to show whether verification moved the crossover."]),

 (215,
  "Quantization changes the collective payload dtype and the arithmetic intensity, so a layout comparison at one precision does not transfer to another and must be re-run rather than extrapolated.",
  "Weight-only quantization shrinks weight bytes but leaves activations, and therefore all-reduce payloads, at the higher precision; activation quantization shrinks the collective payload too. TP's cost is dominated by activation-sized collectives, so weight-only schemes reduce memory pressure without reducing collective cost, while activation quantization reduces both. Dequantization kernels also add per-layer work that scales with TP shard count in ways that differ by kernel implementation. PP is largely indifferent to the weight scheme but is affected by any change in per-stage step time.",
  "H215: moving from a weight-only quantization scheme to one that also quantizes activations reduces TP arm TPOT p95 by more than dispersion while leaving the PP arm's TPOT p95 unchanged within dispersion, isolating the improvement to collective payload rather than to compute (ESTIMATE; derivation: TP's per-layer all-reduce payload is proportional to activation dtype bytes, so halving activation width halves collective bytes, whereas PP's per-token point-to-point traffic is already negligible and cannot improve materially).",
  "Run both layouts under at least two quantization schemes with identical calibration data and identical kernel backends; verify output quality parity with a fixed evaluation set before comparing latency, so a faster but degraded configuration is not mistaken for a win.",
  "If output quality under the more aggressive scheme fails the pre-declared parity threshold, discard its latency numbers and roll back to the scheme that passed; latency gains behind a quality regression are not shippable.",
  [BASE_RISKS,
   "Quantization scheme changed together with the layout, confounding two variables.",
   "Latency improvement reported without a quality parity check, so a degraded model looks like a win.",
   "Dequantization kernel backend differing between arms, changing per-layer overhead independently of layout.",
   TAIL_RISK],
  ["Quantization scheme, calibration dataset and kernel backend recorded per arm.",
   "Fixed-evaluation-set quality scores per configuration, with a pre-declared parity threshold.",
   "Collective payload byte counts per layer per scheme, to confirm the mechanism.",
   "TPOT and TTFT p50/p95/p99 for every layout-by-scheme cell, at least three repeats."]),

 (216,
  "Continuous batching admission policy is a confounder of first order: the layout that admits more requests per step will look better on throughput and worse on tail latency purely because of scheduler settings, not parallelism.",
  "In continuous batching the scheduler decides each step which waiting requests to admit given remaining KV blocks and a maximum batched token budget. TP and PP present different per-GPU memory and different step times to that scheduler, so with identical scheduler parameters the two arms naturally settle at different in-flight batch sizes. Latency and throughput then differ because the operating points differ, not because one layout is intrinsically faster. Comparing at a fixed offered load without pinning the achieved batch size compares two different operating points.",
  "H216: pinning the maximum in-flight batch size and the maximum batched token budget to identical achieved values on both arms shrinks the TP-versus-PP tail latency gap by more than dispersion, showing that a substantial fraction of the original gap was an operating-point difference rather than a layout difference (ESTIMATE; derivation: tail latency in continuous batching rises with queueing delay, which is set by admitted batch size and step time, both of which are scheduler-mediated rather than layout-intrinsic).",
  "Instrument achieved in-flight batch size distribution per arm; re-run with the scheduler constrained so both arms hold the same mean and variance of in-flight batch size; report throughput and tail latency at the matched operating point as well as at the free-running one.",
  "If the arms cannot be brought to a matched operating point without starving one of them, report the comparison as inconclusive rather than picking a winner, and roll back to the incumbent layout.",
  [BASE_RISKS,
   "Scheduler operating point allowed to differ between arms, so the comparison measures admission policy rather than layout.",
   "Throughput and tail latency reported at different achieved batch sizes and presented as comparable.",
   "Queueing delay excluded from the reported latency, hiding the dominant tail component.",
   TAIL_RISK],
  ["Achieved in-flight batch size distribution per arm, sampled over the run, not just its mean.",
   "Queueing delay separated from execution time in the latency breakdown.",
   "Scheduler configuration dump per arm proving identical admission parameters.",
   "Throughput and TPOT p95/p99 at both the free-running and the matched operating points."]),

 (217,
  "Cross-node placement turns TP's collectives into fabric traffic and makes RDMA transport configuration part of the layout decision, so a single-node result cannot be extrapolated to a multi-node deployment.",
  "Once a TP group spans nodes, every per-layer all-reduce traverses the NIC path rather than NVLink. Whether that path uses GPUDirect RDMA, whether the NIC and GPU share a PCIe switch, whether RoCE is configured with lossless flow control, and how many queue pairs NCCL opens all become first-order latency terms. PP is far more tolerant of a node boundary because it sends one small hidden-state tensor per stage transition per token rather than two collectives per layer. This is the boundary condition that most reliably flips the default recommendation.",
  "H217: placing the TP group across a node boundary raises decode TPOT p95 by a factor larger than the same move applied to a PP stage boundary, by more than run-to-run dispersion (ESTIMATE; derivation: TP emits 2*L latency-bound collectives per token across the boundary while PP emits at most PP-1 point-to-point sends per token, so with L in the tens the count of boundary-crossing synchronous operations differs by more than an order of magnitude).",
  "Measure both layouts single-node and split across two nodes, with GPUDirect RDMA explicitly verified enabled or disabled and logged; capture NIC counters and NCCL topology detection per arm; hold model, trace and scheduler fixed.",
  "If GDR is not actually active on the multi-node arm, fix the topology or disable the multi-node plan; do not ship a cross-node TP layout whose collectives are staging through host memory.",
  [BASE_RISKS,
   "Single-node results extrapolated to multi-node without re-measuring the collective path.",
   "GPUDirect RDMA assumed active when the NIC and GPU are not under a common PCIe switch, silently staging through host memory.",
   "RoCE flow control misconfiguration producing pause storms or drops that appear as sporadic tail latency.",
   TAIL_RISK],
  ["NCCL init logs showing detected topology, transport per rank pair and whether GDR is in use.",
   "PCIe topology dump proving NIC-to-GPU affinity, plus NIC error, pause and retransmission counters per run.",
   "TPOT p95 for four cells: TP single-node, TP cross-node, PP single-node, PP cross-node.",
   "Fabric configuration evidence for RoCE priority flow control and congestion control settings."]),

 (218,
  "Disaggregated prefill and decode, as in Mooncake- or Dynamo-style architectures, dissolves the single-layout question: prefill and decode can take different parallelism degrees, and the real cost moves into KV transfer between the two pools.",
  "If prefill and decode run in separate pools, each can be sized for its own regime: prefill favours a layout that maximizes compute throughput, decode favours one that minimizes per-token synchronous cost. The coupling is the KV cache produced by prefill and consumed by decode, which must move across the fabric or be reconstructed. That transfer is a bulk, bandwidth-bound operation whose cost scales with prompt length, layer count and KV head count, and it lands directly on TTFT. The decision therefore stops being TP versus PP and becomes whether the KV transfer cost is smaller than the efficiency gained by specializing each pool.",
  "H218: in a disaggregated configuration, end-to-end TTFT p95 is lower than the best colocated configuration only when measured KV transfer time is below the prefill-phase efficiency gain, and the crossover occurs at a prompt length that can be located by sweeping prompt length alone (ESTIMATE; derivation: KV transfer bytes scale linearly with prompt length while the specialization gain in prefill is roughly proportional to prefill compute, so the two curves have different slopes in prompt length and must cross).",
  "Sweep prompt length across the production range in a disaggregated deployment and in the best colocated baseline, instrumenting KV transfer bytes and transfer wall time separately from prefill and decode compute; hold model and trace generation logic fixed.",
  "If measured KV transfer time exceeds the specialization gain anywhere in the production prompt-length distribution's bulk, do not ship disaggregation; roll back to the colocated layout and record the crossover point for re-evaluation after fabric upgrades.",
  [BASE_RISKS,
   "Disaggregation adopted on architectural appeal without measuring the KV transfer cost it introduces.",
   "KV transfer time folded into prefill time, hiding the term that decides the trade-off.",
   "Prompt-length distribution assumed rather than measured, so the crossover is evaluated at the wrong operating point.",
   TAIL_RISK],
  ["KV transfer bytes and wall time per request, separated from prefill and decode compute time.",
   "Prompt-length distribution from production traffic, including the tail, not just the mean.",
   "TTFT p50/p95/p99 versus prompt length for disaggregated and colocated configurations.",
   "Fabric bandwidth measurement on the actual transfer path, to check the transfer is near line rate rather than protocol-bound."]),

 (219,
  "The decision must be stated as a cost-per-served-token under an SLO constraint, not as a latency number, otherwise a layout that meets the SLO while wasting half the fleet can win the benchmark and lose the budget.",
  "Latency alone is not decidable: TP=8 may hit the tightest TPOT target while serving few concurrent requests per GPU, and a layout with slightly worse tail latency may serve enough more concurrency to halve the fleet. The correct objective is the minimum GPU-seconds per served token subject to the p95 SLO being met, evaluated at the production arrival process. That reframing also makes the rollback gate concrete, since it converts an aesthetic preference into a budget comparison against the incumbent.",
  "H219: ranking the layouts by GPU-seconds per served token subject to meeting the TPOT p95 SLO produces a different winner than ranking by TPOT p95 alone, for at least one point in the production concurrency range (ESTIMATE; derivation: the two rankings optimize different objectives, and they coincide only when the latency-optimal layout is also the one with the highest SLO-feasible concurrency per GPU, which is not guaranteed since TP trades concurrency headroom for per-token latency).",
  "For each layout, find the maximum concurrency at which the p95 SLO still holds, then compute GPU-seconds per served token at that concurrency; compare layouts on that metric and on latency separately, using the production arrival process rather than a fixed-rate synthetic load.",
  "If the cost-optimal layout fails the SLO at the p99 level even while passing at p95, do not ship it; roll back to the incumbent and re-open the comparison only with an explicitly agreed p99 target.",
  [BASE_RISKS,
   "Layout chosen on latency alone, ignoring the concurrency headroom that determines fleet size and cost.",
   "Maximum SLO-feasible concurrency never measured, so cost per token cannot be computed at all.",
   "Synthetic fixed-rate load substituted for the production arrival process, which changes queueing behaviour and therefore the feasible concurrency.",
   TAIL_RISK],
  ["Maximum SLO-feasible concurrency per layout, established by a load sweep rather than assumed.",
   "GPU-seconds per served token at that concurrency, with the GPU count and utilization accounting shown.",
   "Production arrival-process statistics including burstiness, not just mean request rate.",
   "p95 and p99 TPOT at the chosen operating point for every layout considered."]),
]

recs = []
for r, st in zip(sl, STANCES):
    n, thesis, mech, hyp, exp, gate, risks, ev = st
    m = r["messages"]
    u = [x["content"] for x in m if x["role"] == "user"][0]
    a = [x["content"] for x in m if x["role"] == "assistant"][0]
    ans = (
        f"Analytical stance under test: Stance {n} - {thesis}\n\n"
        f"{COMMON}\n\n"
        f"{mech}\n"
        f"Falsifiable hypothesis H{n}: {hyp}\n"
        f"Controlled experiment: {exp}\n"
        f"Rollback gate: {gate}\n\n"
        f"{CRITIQUE}"
    )
    recs.append({
        "source_id": r["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": ans,
        "quality_dimensions": {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2},
        "risks": risks,
        "evidence_required": ev,
        "confidence": 0.78,
    })

with open(OUT, "w") as f:
    for x in recs:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(recs))
