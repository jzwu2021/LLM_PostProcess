import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT = f"{EXP}/results/train-batch-0219.jsonl"
START = 2180
N = 10

FRAME = """
Common frame (applies to every stance below).
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
Measurement and evidence policy: every number below that was not produced by a run on this hardware is labelled ESTIMATE and carries its derivation. Only values read out of named benchmark artifacts may be labelled MEASURED. This review reports no MEASURED values, because no benchmark was executed for it.
"""

CRITIQUE = """Critique of the source item: the prompt is a legitimate infrastructure question and does ask for assumptions, a falsifiable hypothesis, measurements, confounders and rollback criteria, but the corpus pair is degenerate - the assistant turn contains only a rubric describing what an answer should contain, not an answer. There is therefore no substantive content to keep, and the item is rewritten into a complete response that supplies the mechanism, the boundary conditions that flip the recommendation, an explicit falsifiable hypothesis, a single-variable controlled experiment, the evidence artifacts required to adjudicate it, and a rollback gate. Every quantitative claim is labelled ESTIMATE and carries its derivation; no value here is MEASURED, because no benchmark run was performed for this review. This output is provisional teacher-B review material, not expert gold, and it is not evidence about any model's domain capability."""

STANCES = [
 (190,
  "Speculative decoding changes the unit of work from one token per step to a variable-length accepted block, so a layout comparison run without it measures a different machine than the one in production.",
  "A draft-and-verify loop replaces the per-token decode step with a draft phase of k tokens plus one batched verification forward pass. The verification pass is prefill-shaped, not decode-shaped: it is compute-bound over k positions rather than memory-bound over one. That shift favours tensor parallelism, because TP splits the GEMM work that verification actually spends its time in, while pipeline parallelism gains little since the verification pass still traverses every stage once. The acceptance rate also couples to the layout indirectly: a quantized or differently-sharded target model can change the accept-length distribution even when greedy outputs are nominally identical, because numerics differ.",
  "H190: enabling speculative decoding at a fixed draft length narrows the TPOT gap between TP and PP by less than it narrows the absolute TPOT of either, so the ordering of the two layouts is preserved while the magnitude of the gap shrinks (ESTIMATE; derivation: the per-step collective count under TP scales with layers, not with accepted tokens, so amortizing one step over multiple accepted tokens divides the TP collective overhead by the mean accept length, which compresses but does not invert the gap).",
  "Controlled experiment: hold the target model, draft model, draft length and request trace fixed; run each layout with speculation off and on; record mean accept length and its distribution alongside TPOT, and confirm the accept-length distributions across layouts overlap within dispersion before comparing latency at all.",
  "Rollback gate: if the accept-length distribution differs across layouts by more than run-to-run dispersion, the latency comparison is confounded and must be discarded rather than reported; revert to the non-speculative arm as the reference and re-measure.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "Comparing layouts with speculative decoding enabled on one arm and disabled on the other silently changes the workload shape and invalidates the result.",
   "Acceptance rate is treated as a property of the draft model alone, when sharding and precision changes perturb target-model numerics and therefore accept length.",
   "Verification passes are prefill-shaped, so a decode-only capacity model under-provisions compute and the layout choice inherits that error.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Mean and full distribution of accepted draft length per layout, per concurrency level, across at least three repeated runs.",
   "TPOT and TTFT p50/p95/p99 for each layout with speculation both disabled and enabled, same request trace.",
   "Device-timeline attribution separating draft-phase, verification-phase, collective and idle time.",
   "Greedy-decode exact-match parity between speculative and non-speculative arms, to confirm speculation is lossless as configured."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.78),

 (191,
  "Chunked prefill and the scheduler's admission policy decide how prefill and decode contend for the same GPU, so the layout comparison is really a comparison of two schedulers unless the policy is pinned.",
  "In a continuous-batching server, arriving prefills are interleaved with in-flight decodes. Without chunking, one long prefill occupies the device for its full duration and every co-resident decode stalls, appearing as a TPOT spike. Chunked prefill splits that prefill into token-budgeted slices so decodes interleave, trading a slightly worse TTFT for a much tighter TPOT tail. This interacts with layout: under pipeline parallelism a chunk boundary also becomes a stage-scheduling boundary, and the bubble structure changes with chunk size; under tensor parallelism chunk size mainly changes GEMM efficiency because small chunks under-fill the shards.",
  "H191: at fixed chunk token budget, the TPOT p99 penalty from prefill interference is larger under pipeline parallelism than under tensor parallelism, because a chunk must traverse all stages before the next decode micro-batch can be issued (ESTIMATE; derivation: PP serializes a chunk across stages so the interference window is stage-count times the per-stage chunk time, while TP executes the chunk once with all ranks participating, giving a single interference window).",
  "Controlled experiment: fix the request trace with a bimodal prompt-length distribution, sweep chunk token budget across a small grid for each layout, and record TPOT p99 and TTFT p95 jointly; the comparison is only valid at the chunk budget that is Pareto-optimal for each layout, not at one shared budget.",
  "Rollback gate: if enabling chunking regresses TTFT p95 beyond the SLO while improving TPOT p99, revert to the previous scheduler configuration and re-open the trade-off explicitly with the SLO owner rather than choosing unilaterally.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "Scheduler admission and chunking policy is left unpinned, so a layout comparison silently becomes a scheduler comparison.",
   "Prefill-decode interference is invisible in mean latency and only shows in the TPOT tail, which the source never asks to be reported.",
   "Optimal chunk budget differs per layout, so imposing one shared budget systematically penalises whichever layout it suits less.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Scheduler configuration dump per arm: chunk token budget, max running requests, max batched tokens, preemption policy.",
   "TPOT p99 and TTFT p95 jointly per layout across a chunk-budget sweep, with a fixed bimodal prompt-length trace.",
   "Per-step timeline showing prefill-chunk and decode-batch interleaving, to confirm interference windows match the mechanism claimed.",
   "Preemption and recompute counters, since a saturated scheduler can hide interference as request restarts."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.77),

 (192,
  "CUDA graph capture and kernel launch overhead dominate small-shape decode steps, so an uncaptured baseline measures the host, not the layout.",
  "A decode step issues on the order of several kernels per layer. At small batch and small hidden dimension each kernel's device time can fall below the host-side launch cost, making the step launch-bound. Graph capture replays the whole step as one submission and removes that overhead. This matters asymmetrically across layouts: tensor parallelism inserts collectives that must be captured together with compute or the graph breaks, and pipeline parallelism introduces cross-stage dependencies that may prevent capturing a full step at all. A comparison where one arm is captured and the other is not attributes host overhead to the parallelism axis.",
  "H192: with graph capture disabled on both arms, the measured TP-versus-PP TPOT gap shrinks relative to the captured comparison, because launch overhead is a layout-independent constant added to both arms (ESTIMATE; derivation: adding an approximately equal additive term to both arms leaves the absolute difference unchanged while increasing both denominators, so any ratio-form gap compresses; if the measured gap instead widens, the overhead is not layout-independent and the mechanism claim is refuted).",
  "Controlled experiment: run each layout twice, capture on and capture off, with everything else pinned; report the four cells rather than two, and verify with a host-side profile that the uncaptured arms are actually launch-bound at the tested batch size before drawing any conclusion.",
  "Rollback gate: if graph capture cannot be enabled on one layout because of dynamic shapes or collective placement, that layout must be reported as capture-incapable rather than compared against a captured arm; revert to the uncaptured-both configuration for any published number.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "Host-side kernel launch overhead can exceed device time at decode shapes, so an uncaptured baseline measures CPU scheduling rather than the parallelism axis.",
   "Graph capture support differs by layout, and comparing a captured arm against an uncaptured one manufactures a spurious gap.",
   "Dynamic shapes silently invalidate a captured graph and force a fallback path, producing bimodal latency that a mean hides.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Per-arm record of whether CUDA graph capture was active, the captured batch-size buckets, and the count of fallback (uncaptured) steps.",
   "Host-side and device-side profile for one decode step per arm, showing launch gaps versus kernel duration.",
   "Four-cell TPOT matrix: {TP, PP} x {capture on, capture off}, with dispersion across repeated runs.",
   "Distribution, not mean, of per-step latency, to expose bimodality from capture fallback."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.76),

 (193,
  "Weight and KV movement across the storage and host path - GPUDirect Storage, GPUDirect RDMA, pinned-memory staging - sets cold-start and elastic-scaling latency, which a steady-state layout benchmark never observes.",
  "Steady-state TPOT says nothing about how long a replica takes to become servable. Loading sharded weights touches the filesystem, the host page cache, PCIe and finally HBM. GDS lets the DMA engine move file pages straight into device memory and bypass a host bounce buffer; without it, every byte is copied through pinned host memory and the effective bandwidth is bounded by the weaker of the storage read path and the host copy. Layout matters because tensor parallelism reads a different shard per rank, so eight ranks issue eight concurrent read streams, while pipeline parallelism reads contiguous layer ranges per stage. The two access patterns stress the storage backend differently.",
  "H193: at fixed total weight bytes, tensor-parallel load time is more sensitive to storage random-read performance than pipeline-parallel load time, because per-rank shards interleave at tensor granularity while stages read contiguous layer ranges (ESTIMATE; derivation: shard layout determines request size and locality at the filesystem, and smaller, more scattered reads degrade faster on any backend whose IOPS ceiling binds before its sequential-bandwidth ceiling).",
  "Controlled experiment: measure time-to-first-servable-token from a cold page cache for each layout, with GDS enabled and disabled, dropping caches between runs; instrument storage-side read size distribution and queue depth so the mechanism claim can be checked directly rather than inferred from wall time.",
  "Rollback gate: if enabling GDS does not reduce cold-start time beyond run-to-run dispersion, disable it and keep the simpler host-staged path, since an unmeasured optimization adds a failure mode without a demonstrated benefit.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "Steady-state latency is reported as if it characterised the service, while cold-start and rescale latency are what determine behaviour during an incident or autoscale event.",
   "Page cache state is not controlled, so a second run reads from RAM and reports a load time the production cold path will never achieve.",
   "GDR and GDS availability depend on driver, filesystem and topology, and a silent fallback to host staging looks like a slow disk rather than a misconfiguration.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Cold-cache time-to-first-servable-token per layout, with caches explicitly dropped, repeated at least three times.",
   "Explicit confirmation that the GDS path is active rather than silently falling back, plus the storage-side read size and queue-depth distribution.",
   "PCIe and NVLink counters during load, to identify which link is the binding constraint.",
   "Steady-state TPOT measured after warm-up on the same replicas, so cold and warm numbers are never conflated."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.75),

 (194,
  "NUMA placement, PCIe topology and NIC affinity determine whether the nominal fabric bandwidth is reachable at all, so a layout comparison on a mis-affinitised host measures the host wiring.",
  "GPUs, NICs and CPU sockets sit on a physical tree. A collective whose participants span a socket boundary traverses the inter-socket link, and a GPU whose traffic egresses through a NIC attached to the other socket pays that crossing on every message. GPUDirect RDMA only avoids the host bounce when the GPU and the NIC sit under the same PCIe root complex; otherwise the driver falls back to a staged copy without raising an error. Tensor parallelism, which issues many small latency-bound collectives per step, is far more exposed to an extra hop than pipeline parallelism, which issues few point-to-point sends.",
  "H194: pinning each rank's process to the socket local to its GPU and its NIC reduces TP decode TPOT p95 by more than it reduces PP decode TPOT p95, because TP's per-step collective count multiplies any per-message affinity penalty (ESTIMATE; derivation: the penalty is per message and TP issues roughly 2L messages per step against PP's PP-1, so the same per-message cost is amplified by the larger message count).",
  "Controlled experiment: capture the topology matrix and NIC-to-GPU affinity before any run; execute each layout under correct affinity and under a deliberately inverted affinity; report both, so the affinity effect size is measured rather than assumed to be zero.",
  "Rollback gate: if the topology cannot be confirmed - for example the GPU-to-NIC mapping is not stable across reboots - no cross-node number may be published; revert to intra-node-only claims until affinity is pinned and verified.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "Nominal link bandwidth is assumed reachable, when NUMA and root-complex placement can silently halve it.",
   "GPUDirect RDMA falls back to host staging without an error when GPU and NIC are not under the same root complex, so the failure presents as a slow fabric.",
   "Process-to-socket pinning is usually left to the launcher default and therefore varies between the two arms being compared.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Full topology matrix (GPU-GPU link types, GPU-NIC affinity, PCIe root complex membership) captured per run.",
   "Per-rank CPU affinity mask and memory-binding policy as actually applied, read back from the running processes.",
   "Confirmation that the RDMA path is in use rather than host-staged fallback, from transport-level counters.",
   "TPOT p95 per layout under correct and inverted affinity, with dispersion, so the affinity effect size is quantified."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.77),

 (195,
  "A mixture-of-experts model adds expert parallelism as a third axis whose all-to-all cost is data-dependent, so a two-way tensor-versus-pipeline framing is incomplete for that model class.",
  "In an MoE layer the router sends each token to a small number of experts. If experts are sharded across devices, every layer performs two all-to-all exchanges, dispatch and combine, whose message sizes depend on how the router happened to distribute this batch's tokens. That makes the collective cost load-dependent rather than shape-determined, unlike the all-reduce in dense tensor parallelism whose size is fixed by hidden dimension. Capacity factor and token dropping add a second data-dependent term: a hot expert either drops tokens, changing outputs, or stalls the step, changing latency. Expert parallelism composes with, rather than replaces, tensor and pipeline parallelism.",
  "H195: for an MoE model at fixed total device count, decode TPOT p99 dispersion under expert parallelism exceeds that of an equivalently sized dense model under tensor parallelism, because the all-to-all payload varies with the router's per-batch token distribution (ESTIMATE; derivation: dense all-reduce payload is a constant function of hidden size and batch, while all-to-all payload is a function of a data-dependent routing histogram, so the variance of the latter is strictly greater than the near-zero variance of the former).",
  "Controlled experiment: log the per-layer expert-assignment histogram alongside per-step latency for the same trace under two expert-parallel degrees; correlate step latency against the max-to-mean expert load ratio for that step; a mechanism claim is supported only if the correlation is present and disappears in the dense control arm.",
  "Rollback gate: if token dropping is active at the target capacity factor, latency numbers are not comparable to a non-dropping arm because the two arms compute different functions; revert to a capacity factor with zero measured drops before publishing any comparison.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "The two-axis framing omits expert parallelism entirely, which is the dominant axis for MoE serving.",
   "All-to-all cost is data-dependent through the router, so a single-trace measurement does not generalise to a different prompt mix.",
   "Token dropping at a low capacity factor changes model outputs, turning a latency comparison into a comparison of two different functions.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Per-layer expert-assignment histograms and max-to-mean expert load ratio per step, aligned in time with step latency.",
   "Measured token-drop rate at the configured capacity factor, required to be zero for any published latency comparison.",
   "All-to-all payload size distribution and its collective time, separated from dense compute in the device timeline.",
   "Dense control arm at matched active-parameter count, to isolate the routing-induced variance from model-size effects."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.74),

 (196,
  "On RoCE the lossless fabric is maintained by PFC and ECN feedback loops, so a cross-node layout comparison is partly a measurement of the congestion-control configuration.",
  "RoCE relies on a link-level pause mechanism and an end-to-end rate-control loop to keep queues shallow without dropping packets. Misconfigured priority-flow-control watermarks cause pause storms that propagate backwards through the fabric and appear as multi-millisecond latency spikes unrelated to the parallelism axis; an ECN marking threshold that is too high lets queues build before rate reduction begins, which inflates tail latency specifically for the many-small-message pattern that tensor parallelism generates. Because the effect is fabric-wide, a noisy neighbour on a shared switch can move the measurement without touching the host under test.",
  "H196: cross-node TP decode TPOT p99 correlates with switch-reported ECN marking and pause counters during the run, while PP decode TPOT p99 does not, because TP's per-step small-message all-reduce is latency-bound and therefore sensitive to queueing that bulk transfers absorb (ESTIMATE; derivation: latency-bound traffic pays the full queueing delay per message and issues many messages per step, whereas a few larger point-to-point sends amortize the same delay over more bytes).",
  "Controlled experiment: collect switch and NIC congestion counters at fine granularity for the exact run window, run each layout during both a quiet fabric period and a period with a controlled background load, and report the four cells; only a layout ordering stable across both fabric conditions may be published.",
  "Rollback gate: if pause or congestion counters are non-zero during a measurement window, that window is discarded rather than interpreted; revert to the last configuration with clean counters and re-measure before making any claim.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "Cross-node results are treated as a property of the layout when they are partly a property of the fabric's congestion-control configuration.",
   "A noisy neighbour on a shared switch can shift results without any change on the host under test, and nothing in the source requires checking for it.",
   "PFC misconfiguration produces rare, large latency spikes that a mean hides entirely and that only tail percentiles expose.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Switch and NIC congestion counters (pause frames sent and received, ECN marks, discards) for the exact measurement window.",
   "Documented PFC priority mapping, buffer watermarks and ECN marking thresholds for every switch on the path.",
   "TPOT p99 per layout under quiet fabric and under controlled background load, with dispersion across repeats.",
   "Fabric topology and confirmation of which links are shared with other tenants during the run."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.76),

 (197,
  "The attention kernel backend and KV layout are chosen per configuration, so a layout comparison can silently compare two different attention implementations.",
  "Serving stacks dispatch attention to one of several kernels depending on head count, head dimension, sequence length, paging scheme and available hardware features. Tensor parallelism divides attention heads across ranks, so a TP degree that leaves too few heads per rank can push the dispatcher onto a slower fallback kernel; pipeline parallelism leaves head count per device unchanged. The KV cache page size and block layout interact with the same dispatch, and grouped-query attention changes how many KV heads survive sharding. The result is that changing the parallelism degree can change the kernel, and the measured difference then reflects kernel quality rather than the parallelism mechanism.",
  "H197: at TP degrees where heads-per-rank falls below the fast kernel's threshold, the measured attention kernel time per token rises discontinuously rather than continuing the smooth trend predicted by pure work division (ESTIMATE; derivation: kernel dispatch is a step function of shape, so crossing a dispatch threshold produces a discontinuity, whereas ideal work division would predict a continuous inverse relationship with TP degree).",
  "Controlled experiment: sweep TP degree across the full range and record, for each point, both the selected attention kernel name and its measured device time per token; the hypothesis is refuted if the kernel name is constant across the sweep and the time still jumps.",
  "Rollback gate: if the two arms resolve to different attention kernels, the comparison is reported as kernel-confounded and not as a layout result; revert to a configuration where both arms dispatch to the same kernel, or report the kernel difference explicitly as the finding.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "Changing parallelism degree can change the dispatched attention kernel, so the measured difference may be kernel quality rather than the parallelism mechanism.",
   "Heads-per-rank after tensor sharding is a hard constraint that the source never mentions, and grouped-query attention tightens it further.",
   "KV page size and block layout interact with kernel dispatch, so a cache-configuration change can masquerade as a layout effect.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Selected attention kernel name and version per arm, read from the runtime rather than assumed from configuration.",
   "Heads-per-rank and KV-heads-per-rank at each tested parallelism degree, with the model's head configuration recorded.",
   "Attention kernel device time per token across a TP-degree sweep, to expose any dispatch discontinuity.",
   "KV cache page size, block layout and measured fragmentation at the target operating point."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.75),

 (198,
  "Power capping, clock throttling and thermal drift make measured latency a function of run duration and rack conditions, so a short benchmark reports the machine's best case rather than its serving case.",
  "Sustained dense GEMM work drives GPUs toward their power and thermal limits, after which clocks drop and every kernel takes longer. The onset is gradual and duration-dependent, so a sixty-second run and a sixty-minute run can produce materially different numbers from identical software. Layouts differ in their exposure: tensor parallelism keeps all devices busy on the same step and synchronises them, so the slowest throttled device sets the step time for the whole group, whereas pipeline parallelism's stages are more loosely coupled per token but equally exposed over a full pipeline. Ambient temperature and neighbouring rack load are outside the experiment yet inside the measurement.",
  "H198: under a sustained run long enough to reach thermal steady state, TP step time degrades by more than the mean per-device clock reduction, because the synchronous collective makes the group step time track the slowest device rather than the average one (ESTIMATE; derivation: a synchronised group's step time is the maximum over devices, and the maximum of a set of throttled clocks degrades at least as fast as their mean, strictly faster when the throttling is heterogeneous across devices).",
  "Controlled experiment: run each layout to thermal steady state rather than for a fixed short window, log per-device clock, power draw and throttle reason at fixed intervals, and report latency separately for the warm-up phase and the steady-state phase instead of averaging across both.",
  "Rollback gate: if any device reports a thermal or power throttle reason during the steady-state window, the run is labelled thermally constrained and excluded from cross-layout comparison; revert to a lower-power configuration or a cooler window and re-measure before publishing.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "Short benchmark windows report pre-throttle clocks that the production service will not sustain.",
   "Synchronous collectives make the slowest throttled device set the group step time, so heterogeneous throttling penalises tensor parallelism disproportionately.",
   "Ambient and neighbouring-rack conditions vary between the two arms if they are not run in the same thermal window.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Per-device clock, power draw, temperature and throttle-reason samples at fixed intervals for the whole run.",
   "Latency reported separately for warm-up and thermal steady-state phases, never averaged across both.",
   "Configured power cap and persistence-mode settings per device, read back from the driver.",
   "Run start and end timestamps plus ambient conditions, so the two arms can be shown to share a thermal window."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}, 0.78),

 (199,
  "The decision is a cost-per-served-token decision under an SLO constraint, so a latency-only comparison answers a question nobody is funding.",
  "The operational question is not which layout is faster but which layout meets the SLO at the lowest cost per token at forecast load, including the replica count needed for headroom and failure tolerance. Tensor parallelism inside a node reduces per-GPU memory pressure and therefore raises the concurrency reachable before the KV cache saturates, which raises tokens per second per GPU at the SLO; pipeline parallelism lets a model span nodes but adds bubble waste that must be paid for. Cost also includes the operational surface: more axes means more failure modes and longer recovery, and recovery time consumes error budget just as latency does.",
  "H199: at the concurrency where each layout just meets the TPOT p95 SLO, the ranking by cost per served token differs from the ranking by single-request latency for at least one realistic load level (ESTIMATE; derivation: single-request latency is measured at concurrency one where pipeline bubbles are worst, whereas cost per token is measured at the saturation knee where bubbles amortize, so the two metrics are evaluated at different points on the same curve and need not agree).",
  "Controlled experiment: for each layout find the maximum concurrency that still satisfies the TPOT p95 and TTFT p95 SLO, compute tokens per second per GPU at exactly that point, divide by the GPU-hour cost including the replica headroom the failure model requires, and compare those figures rather than raw latency.",
  "Rollback gate: publish the cost figure with the SLO, the load forecast and an expiry date; if measured load leaves the forecast envelope or the SLO is renegotiated, the comparison is void and reverts to the previously validated configuration pending re-measurement.",
  ["Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
   "A latency-only framing omits cost per served token, which is the metric the deployment decision is actually made on.",
   "Single-request latency is measured at concurrency one while cost is determined at the saturation knee, so the two can rank layouts differently.",
   "Replica headroom for failure tolerance and recovery time are real costs that the latency comparison ignores entirely.",
   "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."],
  ["Maximum SLO-satisfying concurrency per layout, with the SLO stated as TTFT p95 and TPOT p95 thresholds.",
   "Tokens per second per GPU measured at that concurrency, with dispersion across repeated runs.",
   "Replica count required by the failure and headroom model, plus measured recovery time after a simulated rank loss.",
   "Load forecast, published envelope and expiry date attached to the cost figure, with the counters that assert the envelope still holds."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}, 0.80),
]

assert len(STANCES) == N

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
sel = corpus[START:START + N]
assert len(sel) == N

with open(OUT, "w") as f:
    for rec, st in zip(sel, STANCES):
        num, headline, mech, hyp, exp, rb, risks, evid, qd, conf = st
        m = {x["role"]: x["content"] for x in rec["messages"]}
        ca = (
            f"Analytical stance under test: Stance {num} - {headline}\n"
            + FRAME
            + "\n" + mech
            + "\nFalsifiable hypothesis H" + str(num) + ": " + hyp
            + "\n" + exp
            + "\n" + rb
            + "\n\n" + CRITIQUE
        )
        out = {
            "source_id": rec["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": m["user"],
            "source_assistant": m["assistant"],
            "corrected_answer": ca,
            "quality_dimensions": qd,
            "risks": risks,
            "evidence_required": evid,
            "confidence": conf,
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
print("WROTE", OUT, "ids", sel[0]["id"], "..", sel[-1]["id"])
