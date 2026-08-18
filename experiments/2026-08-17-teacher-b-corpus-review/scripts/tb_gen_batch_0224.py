import json, os, re, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
SRC = f"{EXP}/scripts/tb_gen_batch_0223.py"
OUT = f"{EXP}/results/train-batch-0224.jsonl"
START = 2230
N = 10

# Reuse the frame/critique text verbatim from the 0223 generator by importing it.
sys.path.insert(0, f"{EXP}/scripts")
src = open(SRC).read()
ns = {}
frame_src = src.split('FRAME = """', 1)[1].split('"""', 1)[0]
crit_src = src.split('CRITIQUE = """', 1)[1].split('"""', 1)[0]
FRAME = frame_src
CRITIQUE = crit_src

COMMON_RISK = "Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning."
COMMON_RISK2 = "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them."

STANCES = [
 (240,
  "Crossing the node boundary makes the fabric, not the axis, the deciding variable: RoCE and InfiniBand small-message latency floors set the maximum viable tensor-parallel degree.",
  "Inside a node, tensor-parallel all-reduces ride NVLink/NVSwitch and their small-message latency floor is in the single-digit microseconds. The moment a tensor-parallel group spans hosts, every one of the two collectives per layer per token traverses a NIC, a switch and a host memory path. RDMA over Converged Ethernet removes the CPU from the data path but does not remove the wire and switching latency floor, and it inherits the lossless-Ethernet configuration burden: priority flow control, explicit congestion notification and buffer tuning. A misconfigured lossless domain produces pause storms whose latency signature looks exactly like an unfavourable parallel layout. Pipeline parallelism crossing the same boundary sends one hidden-state tensor per token per stage boundary, which is one small message instead of two collectives per layer, so PP is the axis that tolerates a node crossing and TP is the axis that does not.",
  "H240: for a fixed model, the TP degree that minimises TPOT drops when the group is forced to span two hosts, and the drop is proportional to the ratio of inter-host to intra-host small-message all-reduce latency (ESTIMATE; derivation: decode collectives are latency-bound rather than bandwidth-bound because the payload is one activation row per token per layer, so the per-collective cost tracks the fabric latency floor and the total added cost is layers times two times that floor; if the measured optimum is unchanged, the collective was not on the critical path and the claim is refuted).",
  "Controlled experiment: measure the small-message all-reduce latency floor with a standalone collective microbenchmark at the exact model message size, first with all ranks intra-node and then with the group split across two hosts; then run the identical serving trace under both placements at fixed TP degree, and separately sweep TP degree under the split placement; hold the collective library version, protocol and congestion configuration pinned and logged in every arm.",
  "Rollback gate: if the cross-host arm's TPOT p95 exceeds the SLO, or if pause-frame or congestion-notification counters are non-zero on any port in the path during the run, the cross-host layout is not promoted and the service reverts to the single-node incumbent until the lossless domain is proven clean.",
  [COMMON_RISK,
   "Tensor parallelism extended across a node boundary places two latency-bound collectives per layer per token on the wire, which the source item does not distinguish from the intra-node case.",
   "RoCE lossless configuration errors produce pause storms whose latency signature is indistinguishable from a bad parallel layout without port counters.",
   "Fabric latency floors are properties of the deployment, not the model, so a benchmark on one cluster does not transfer to another.",
   COMMON_RISK2],
  ["Standalone small-message all-reduce latency at the exact model message size, intra-node and cross-node, same rank count.",
   "Per-port pause-frame, ECN-marked and congestion counters for every switch and NIC on the path, sampled across each run.",
   "TPOT p95 and TTFT p95 per placement and per TP degree, three repeats each, with the collective library version and protocol logged.",
   "Topology description showing exactly which ranks landed on which host and which switch ports carried the traffic."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}, 0.80),

 (241,
  "GPUDirect RDMA and GPUDirect Storage change where bytes cross, so a layout comparison that ignores whether the data path bounces through host memory is measuring plumbing rather than parallelism.",
  "Without GPUDirect RDMA, a tensor leaving a GPU for the network is staged into host memory and copied out by the NIC, which adds two PCIe traversals and a host memory round trip per transfer. With GPUDirect RDMA the NIC reads GPU memory directly, provided the NIC and GPU sit under a PCIe topology that permits peer-to-peer and the driver stack has it enabled. Whether this path is active is not visible in latency numbers alone; it must be read out of the collective library's debug output or the driver. The same applies on the storage side: GPUDirect Storage removes a host bounce for weight loading and for KV offload, which changes replica start time and any KV-migration scheme. Two arms that differ silently in whether peer-to-peer was enabled differ by a plumbing constant, not by parallel axis.",
  "H241: enabling the direct GPU-to-NIC path leaves the ranking of tensor versus pipeline parallelism unchanged while shifting both arms' absolute latency by a roughly constant amount, so any ranking change observed when it is toggled indicates a confounded experiment (ESTIMATE; derivation: the host-bounce cost is per transfer and largely independent of which axis issues the transfer, so removing it is close to an additive shift; a ranking inversion would mean the two axes issue transfers of materially different size or count, which must then be measured directly).",
  "Controlled experiment: for each layout, run with the direct path enabled and with it disabled, confirming the actual state from collective library debug output and driver reporting rather than from configuration intent; hold everything else pinned; report absolute latency and the between-arm delta for each toggle state, plus replica cold-start time to capture the storage-side effect.",
  "Rollback gate: if the direct path cannot be confirmed active from driver or library output in the production placement, the benchmark result obtained with it active is not used for the promotion decision; revert to the incumbent and re-benchmark in the placement that production will actually run.",
  [COMMON_RISK,
   "Whether GPU-to-NIC peer-to-peer is active depends on PCIe topology and driver state and is frequently different between benchmark and production hosts.",
   "A host-memory bounce on the data path is an additive latency constant that is easily misattributed to the parallel layout.",
   "Storage-side path differences change replica start and KV-offload behaviour, which affects recovery and autoscaling rather than steady-state latency, and so escape a latency-only comparison.",
   COMMON_RISK2],
  ["Collective library debug output and driver reporting confirming whether the direct GPU-to-NIC path was active in each arm.",
   "PCIe topology map showing GPU-NIC affinity and any switch or root-complex boundary between them.",
   "Absolute and between-arm latency for both toggle states, three repeats each.",
   "Replica cold-start time per arm, to capture the weight-loading path effect."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}, 0.79),

 (242,
  "KV cache capacity, not latency, is usually the binding constraint, and the two axes divide it differently, so the layout question is really a concurrency-per-replica question.",
  "Tensor parallelism shards the KV cache across every rank, so per-GPU KV bytes fall by the TP degree and the concurrency a replica can hold rises accordingly. Pipeline parallelism assigns whole layers to stages, so each stage holds the KV for its own layers only; aggregate capacity is comparable but it is partitioned by layer rather than by head, which constrains how a stage can be rebalanced independently. In a latency-sensitive service the practical failure mode is not a slow token but a request that cannot be admitted, or a preemption that evicts a partially generated sequence and forces recomputation. That recomputation cost lands in the same TPOT percentile the comparison is trying to optimise, which means a layout that looks slower per step can be faster in the tail simply because it preempts less.",
  "H242: at the concurrency where the incumbent begins preempting, the layout with the larger per-replica KV headroom has a lower TPOT p99 despite an equal or worse TPOT p50 (ESTIMATE; derivation: preemption forces recomputation of an evicted sequence's prefill, a cost far larger than one decode step, and it lands in the upper tail; below the preemption threshold the two arms should differ only by per-step cost, which is where p50 is decided).",
  "Controlled experiment: sweep concurrency through and past the preemption onset for each layout; instrument preemption and eviction events, recomputed token counts and KV-block occupancy directly from the engine; report TPOT p50, p95 and p99 separately, plus admitted-versus-rejected request counts, at each concurrency point.",
  "Rollback gate: if the promoted layout preempts at a concurrency below the production peak with the declared safety margin, it is not promoted regardless of its median latency; revert to the incumbent and re-decide with KV headroom as a first-class acceptance criterion.",
  [COMMON_RISK,
   "The binding constraint at production concurrency is usually KV capacity rather than per-step latency, and the source item frames the choice purely as latency.",
   "Preemption and recomputation costs land in the upper tail, so a median-only comparison can promote the layout that fails worse under load.",
   "Long-context traffic consumes KV superlinearly relative to request count, so headroom measured on short prompts does not transfer.",
   COMMON_RISK2],
  ["Per-GPU KV-block occupancy and free-block trajectory per layout across the concurrency sweep.",
   "Preemption and eviction event counts with recomputed token totals, per layout and concurrency point.",
   "TPOT p50, p95 and p99 reported separately, plus admission and rejection counts.",
   "Production prompt-length and generation-length distributions used to size the sweep, with their measurement date."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 4}, 0.80),

 (243,
  "Cost per request, not latency alone, decides the layout, and the two axes have different utilisation profiles at the same latency, so the decision needs an efficiency frontier rather than a winner.",
  "A layout that meets the SLO at half the GPU-hours per request is the better engineering answer even if it is slightly slower, and a layout that meets the SLO only by leaving devices idle is expensive in a way a latency chart cannot show. Pipeline parallelism at low concurrency wastes capacity in bubbles; tensor parallelism at high degree wastes capacity in collectives that grow while useful work per rank shrinks. Both waste modes appear as reduced tokens per GPU-second at fixed latency, which is the correct comparison unit. Presenting a single latency number for each arm discards the axis on which the money is actually spent, and it hides that the right answer is often a frontier: the maximum throughput each layout sustains while still inside the latency SLO.",
  "H243: at the throughput where both layouts exactly meet the TPOT p95 SLO, tokens per GPU-second differ by more than run-to-run dispersion, and the ordering by cost differs from the ordering by latency at low load (ESTIMATE; derivation: the two waste modes have opposite load dependence - the pipeline bubble shrinks as concurrency rises while collective overhead per useful token is roughly load-independent - so the cost-ordering crossover need not coincide with the latency-ordering at any single load point).",
  "Controlled experiment: for each layout, sweep offered load and record the maximum sustained throughput at which TPOT p95 remains inside the SLO; report tokens per GPU-second at that point, along with per-device utilisation and idle fraction; run both arms on identical hardware with identical trace and identical SLO definition, and repeat three times to establish dispersion.",
  "Rollback gate: if the promoted layout's tokens per GPU-second at the SLO boundary is worse than the incumbent's by more than the pre-declared margin, it is not promoted on latency grounds alone; revert to the incumbent and record the cost regression as the reason.",
  [COMMON_RISK,
   "Latency-only comparison omits cost per request, which is usually the decision-relevant quantity once both arms meet the SLO.",
   "Efficiency ordering and latency ordering can cross, so a single load point can promote the wrong layout.",
   "Idle capacity in pipeline bubbles and redundant work in wide collectives are both invisible in a latency chart.",
   COMMON_RISK2],
  ["Maximum sustained throughput inside the TPOT p95 SLO for each layout, three repeats, with dispersion reported.",
   "Tokens per GPU-second at the SLO boundary for each layout, on identical hardware.",
   "Per-device utilisation and idle-fraction traces for both arms at the SLO boundary.",
   "Written SLO definition, identical across arms, including the percentile and the window over which it is computed."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}, 0.79),

 (244,
  "Numerics are not invariant across parallel layouts: reduction order and kernel selection change bitwise outputs, so a layout change is a model change until parity is demonstrated.",
  "Floating-point addition is not associative. Changing the tensor-parallel degree changes how a matrix product is partitioned and therefore the order in which partial sums are reduced, which changes low-order bits. Different shard shapes also select different kernels, some of which accumulate in different precision. Pipeline parallelism does not partition individual GEMMs and so perturbs numerics far less, but it can still change kernel selection through batch shape. Small numerical differences are amplified by sampling: a token that flips near a decision boundary changes the entire continuation. Shipping a layout change without an output-parity check is therefore shipping a behaviour change that will be reported as a quality regression with no obvious cause, days after the latency win was celebrated.",
  "H244: greedy-decoded outputs on a fixed prompt set diverge between tensor-parallel degrees at a non-zero rate, and the divergence rate is materially higher across TP degrees than across pipeline-stage counts (ESTIMATE; derivation: changing TP degree repartitions every GEMM's reduction and changes kernel selection, whereas changing pipeline depth reassigns whole layers without repartitioning the reductions inside them; if divergence rates are equal, kernel selection rather than reduction order dominates and must be logged directly).",
  "Controlled experiment: decode a fixed prompt set greedily with a fixed seed under each layout; report the exact-match rate and first-divergence-position distribution; separately evaluate task quality on a held-out set per layout against a pre-declared acceptance band; log the kernel backend actually selected in each arm so divergence can be attributed to reduction order or to kernel choice.",
  "Rollback gate: if quality on the held-out set falls outside the pre-declared band, the layout is rejected regardless of its latency; if outputs diverge but quality is inside the band, the change ships only with the divergence documented and any output-hash-dependent downstream consumer notified in advance.",
  [COMMON_RISK,
   "Layout changes alter reduction order and kernel selection, so outputs are not bitwise stable across arms.",
   "Sampling amplifies small numerical differences into entirely different continuations, which surfaces as an unexplained quality regression.",
   "Downstream consumers that cache or hash outputs break silently when the layout changes.",
   COMMON_RISK2],
  ["Greedy exact-match rate between layouts on a fixed prompt set with a fixed seed.",
   "Held-out task quality per layout, scored against a pre-declared acceptance band agreed before the runs.",
   "Kernel backend and accumulation precision actually selected per layout, logged from the engine.",
   "List of downstream consumers that depend on output stability, with sign-off recorded before promotion."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 4}, 0.81),

 (245,
  "The layout is chosen once and lived with for months, so the decision must be evaluated against traffic drift and model succession, not against today's trace.",
  "A tensor-parallel degree is baked into how weights are sharded and loaded, into the replica's memory footprint, and often into the autoscaling unit. Changing it later means a rolling redeploy of every replica and a re-tuning of concurrency limits. Meanwhile the traffic changes: prompt lengths grow as product features add context, generation lengths change as prompts change, and the next model in the family has a different layer count and hidden size, which moves both the collective cost and the KV footprint. A layout chosen to be exactly optimal for today's trace is chosen to be fragile. The defensible choice is the one that is near-optimal across the plausible range of next quarter's traffic and the announced next model, and that can be changed without a redeploy if possible.",
  "H245: the layout that is optimal on the current trace remains inside the declared latency band across a projected traffic shift of the magnitude seen in the last two quarters, and across the next model's shape (ESTIMATE; derivation: replay the historical trace shift forward as a sensitivity sweep and re-run the comparison at the projected prompt and generation length distributions; if the optimum moves outside the band, the layout is over-fitted to the current trace and a more robust degree should be chosen).",
  "Controlled experiment: re-run the layout comparison at three traffic points - current, projected next quarter, and the historical extreme observed - and, where the next model's shape is known, at that shape too; report the latency band each layout stays inside across all points, and choose on worst-case-inside-band rather than best-case-at-current-trace.",
  "Rollback gate: if traffic drifts past the range the decision was validated over, the decision record expires automatically and the comparison is re-run before the next change; if the incumbent still meets the SLO, no layout change ships purely on a marginal current-trace win.",
  [COMMON_RISK,
   "Optimising the layout for the current trace produces a choice that expires as prompt and generation length distributions drift.",
   "Changing tensor-parallel degree later requires a rolling redeploy and concurrency re-tuning, so the switching cost is not symmetric with the decision cost.",
   "The next model in the family changes layer count, hidden size and therefore both collective cost and KV footprint.",
   COMMON_RISK2],
  ["Historical prompt and generation length distributions over at least two quarters, with the observed drift magnitude quantified.",
   "Layout comparison re-run at current, projected and historical-extreme traffic points.",
   "Announced shape of the next model in the family, where available, with the comparison re-run at that shape.",
   "Documented cost and procedure for changing tensor-parallel degree in production, including redeploy time and blast radius."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}, 0.79),

 (246,
  "Multi-tenancy and noisy neighbours dominate tail latency in shared clusters, so a layout compared on an idle cluster is compared under conditions production never sees.",
  "In a shared cluster the fabric, the host PCIe topology and sometimes the GPUs themselves are contended. A tensor-parallel group is only as fast as its slowest rank on every single collective, because the all-reduce is a synchronisation point: one straggler rank delays all of them, every layer, every token. Pipeline parallelism is more forgiving of a transient straggler because a slow stage is absorbed partly by micro-batch queueing rather than stalling every peer simultaneously. This means TP's tail latency is far more sensitive to interference than its median suggests, and a benchmark run on a quiet cluster systematically flatters TP. The relevant measurement is under representative background load, not in isolation.",
  "H246: injecting representative background interference degrades TPOT p99 more for the tensor-parallel arm than for the pipeline arm, while degrading p50 comparably (ESTIMATE; derivation: an all-reduce completes at the speed of the slowest participant, so per-collective straggler effects compound across layers and land in the tail; pipeline stages queue rather than synchronise per token, so a transient straggler is partly absorbed; equal tail degradation would indicate the interference did not reach the collective path and the injection must be validated).",
  "Controlled experiment: run each layout in isolation and under a calibrated, reproducible background load applied to the fabric and to the host, with the interference level recorded; report p50, p95 and p99 for each combination; verify the interference actually reached the intended path with per-rank collective timing rather than assuming it did.",
  "Rollback gate: if the promoted layout misses the SLO under representative interference even though it met it in isolation, it is not promoted; revert to the incumbent and either provide isolation guarantees or re-decide under the contended conditions production actually provides.",
  [COMMON_RISK,
   "Benchmarks run on a quiet cluster systematically flatter the layout that is most synchronisation-heavy.",
   "An all-reduce completes at the pace of the slowest rank, so one straggler affects every peer on every layer and every token.",
   "Interference injection that does not actually reach the collective path produces a false negative and a false sense of robustness.",
   COMMON_RISK2],
  ["Per-rank collective timing distributions showing straggler behaviour under each interference level.",
   "p50, p95 and p99 per layout, in isolation and under calibrated background load, three repeats each.",
   "Description and calibration of the interference source, with evidence it reached the intended path.",
   "Cluster tenancy and isolation guarantees actually in force in production, documented rather than assumed."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 4}, 0.80),

 (247,
  "Mixture-of-expert architectures replace the tensor-versus-pipeline framing with an expert-parallel all-to-all whose cost is routing-dependent and load-imbalanced, and the question must be re-posed.",
  "When the model routes tokens to a subset of experts, the dominant communication is an all-to-all that dispatches tokens to the ranks holding their chosen experts and gathers the results back. Unlike an all-reduce, its cost depends on the routing distribution: if a batch's tokens concentrate on a few experts, those ranks become stragglers and every other rank waits. Capacity factors and token dropping are then policy choices that trade quality against tail latency. Tensor parallelism can still be applied inside an expert, and pipeline parallelism across layers, so the axes compose rather than compete. Posing the question as tensor versus pipeline for such a model omits the axis that actually dominates its latency profile.",
  "H247: for a routed model, TPOT p99 correlates with a measured routing-imbalance statistic across the trace more strongly than with the tensor-parallel degree (ESTIMATE; derivation: the all-to-all completes when the most heavily loaded expert rank finishes, so tail latency tracks the maximum-to-mean expert load ratio per step, whereas TP degree affects only the intra-expert GEMM partitioning; a weak correlation would indicate the all-to-all is not on the critical path and the dispatch cost should then be measured directly).",
  "Controlled experiment: instrument per-step expert load distribution and all-to-all duration; run the same trace at several tensor-parallel degrees and at several capacity factors; regress TPOT p99 on the imbalance statistic and on TP degree separately, and report both, alongside token-drop rate and its quality effect.",
  "Rollback gate: if reducing tail latency requires a capacity factor whose token-drop rate moves held-out quality outside the pre-declared band, the change is rejected; revert to the incumbent capacity factor and address imbalance through routing or placement rather than dropping.",
  [COMMON_RISK,
   "For routed architectures the dominant communication is a routing-dependent all-to-all that the tensor-versus-pipeline framing omits entirely.",
   "Expert load imbalance makes tail latency depend on the input distribution, so a benchmark trace with different routing statistics does not transfer.",
   "Capacity factor and token dropping trade quality for latency, and the trade is invisible unless quality is measured at the same setting.",
   COMMON_RISK2],
  ["Per-step expert load distribution with a maximum-to-mean imbalance statistic, over the production trace.",
   "All-to-all duration per step, separated from the rest of the step time.",
   "TPOT p99 versus TP degree and versus capacity factor, with token-drop rate reported at each setting.",
   "Held-out quality at each capacity factor, scored against a pre-declared acceptance band."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}, 0.78),

 (248,
  "Prefix and KV cache reuse across requests changes the arithmetic the comparison rests on, because a large share of prefill may never execute.",
  "Production traffic is rarely a set of independent prompts. Shared system prompts, few-shot preambles and multi-turn conversations mean a substantial fraction of prefill tokens can be served from a cached prefix instead of computed. When cache hit rates are high, prefill cost collapses and the service becomes almost purely decode-bound, which shifts the balance toward the axis with the lowest per-step synchronous cost. The hit rate is a property of traffic and of the cache eviction policy, and it interacts with the layout, because the two axes partition the cache differently and therefore evict differently under pressure. A benchmark driven by synthetic independent prompts has a hit rate near zero and measures a workload the service does not have.",
  "H248: the layout ranking measured on a synthetic zero-reuse trace differs from the ranking measured on a replayed production trace with its natural prefix reuse, and the gap widens with the measured cache hit rate (ESTIMATE; derivation: prefix reuse removes prefill work, raising the decode share of total time, and the axes differ more in per-step decode cost than in prefill cost, so a higher hit rate amplifies the decode-side difference).",
  "Controlled experiment: replay a production trace preserving request ordering, session identity and arrival timestamps so prefix reuse occurs naturally; report the measured cache hit rate per arm and require it to match within dispersion before comparing latency; run the synthetic zero-reuse trace as a separate labelled arm rather than as the primary evidence.",
  "Rollback gate: if the two layouts' cache hit rates differ beyond dispersion on the same trace, the latency comparison is invalid and discarded; investigate the eviction difference, fix it, and re-run before any promotion decision.",
  [COMMON_RISK,
   "Synthetic independent-prompt benchmarks have near-zero prefix reuse and measure a workload production does not have.",
   "Cache hit rate is itself layout-dependent through eviction behaviour, so it is a confounder rather than a constant.",
   "Replaying a trace without preserving session identity and ordering destroys the reuse structure being measured.",
   COMMON_RISK2],
  ["Measured prefix cache hit rate per arm on the replayed production trace, reported before any latency comparison.",
   "Production trace replay preserving session identity, request ordering and arrival timestamps, with evidence of fidelity.",
   "Latency percentiles for the production-trace arm and the synthetic zero-reuse arm, reported separately and labelled.",
   "Cache eviction policy and capacity per layout, pinned and recorded."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}, 0.79),

 (249,
  "The correct deliverable is a scoped decision record whose validity is the intersection of every conditioning variable in this batch, with monitored invariants, an expiry and a rehearsed rollback.",
  "This batch surfaced conditioning variables each individually capable of reversing a layout ranking: whether the group crosses a node boundary and what the fabric's small-message latency floor is, whether the direct GPU-to-NIC path is active, whether KV capacity or per-step latency is the binding constraint, whether the decision is made on latency or on cost at the SLO boundary, whether output parity across layouts has been demonstrated, how far traffic will drift before the next re-evaluation, how much interference the shared cluster imposes, whether the architecture is dense or routed, and what the natural prefix reuse rate of production traffic is. A recommendation is valid only inside the intersection of the ranges over which all of these were held or measured. The honest artifact states that envelope, attaches a monitor to each variable, sets an expiry, and rehearses the revert before it is needed.",
  "H249: every SLO breach during the validity window is preceded by at least one recorded invariant leaving its declared band, and no breach occurs with all invariants in band (ESTIMATE; derivation: the invariant set was constructed from exactly the variables shown above to move the ranking, so a breach with all of them in band demonstrates a missing variable and refutes the completeness of the model rather than the layout choice).",
  "Controlled experiment: publish the decision record with an explicit band for each conditioning variable, wire a monitor to each band, and treat the validity window as a running experiment; at expiry re-run the comparison and score the recorded prediction against the observed outcome instead of starting the analysis from scratch.",
  "Rollback gate: any monitored invariant leaving its band, or expiry reached without re-validation, automatically reverts the service to the incumbent layout; the revert path is rehearsed outside production in advance so that it requires no new decision under pressure.",
  [COMMON_RISK,
   "A recommendation stated without the intersection of conditions under which it was validated expires silently and is later cited as if unconditional.",
   "Conditioning variables identified but never converted into monitored invariants leave violations to surface as incidents rather than alerts.",
   "Without an expiry and a rehearsed rollback, the claim outlives its evidence and reverting becomes an unpractised emergency change.",
   COMMON_RISK2],
  ["Decision record listing the chosen layout, validated envelope, per-variable invariant bands, monitors, expiry date and rollback procedure.",
   "Monitor definitions with alert thresholds for each conditioning variable, plus evidence that each fires in test.",
   "Rehearsal evidence that the rollback path executed successfully outside production, with its measured duration.",
   "Re-validation run performed at expiry and scored against the prediction recorded at promotion time."],
  {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}, 0.81),
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
