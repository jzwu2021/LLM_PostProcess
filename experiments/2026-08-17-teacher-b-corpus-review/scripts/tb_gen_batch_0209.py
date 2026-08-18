import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0209.jsonl"
START, N = 2080, 10

src = open(f"{EXP}/scripts/tb_gen_batch_0205.py").read()
ns = {}
exec(src.split("STANCES = [")[0], ns)
COMMON = ns["COMMON"]

STANCES = [
 ("Stance 90 - Quantisation format decides the layout, because weight and KV precision change both the memory floor and the collective volume.",
  "The parallel degree needed is set by whether the weights plus the KV working set fit in one device's memory, and that floor moves with "
  "precision. Dropping weights to a 4-bit format can remove the memory reason for tensor parallelism entirely, at which point the layout "
  "argument dissolves into a replication argument. Precision also changes the arithmetic intensity of each GEMM and, where the engine "
  "reduces in a wider type, the bytes moved by each collective. Falsifiable hypothesis H90: at a lower weight precision the minimum "
  "memory-feasible tensor degree drops, and the configuration at that lower degree meets the SLO at higher throughput than the higher-degree "
  "configuration required at full precision (ESTIMATE; derivation: minimum tensor degree is the ceiling of total resident bytes over per-device "
  "capacity, and total resident bytes scale roughly linearly with weight precision). Controlled experiment: hold model, traffic and engine "
  "fixed, vary only weight and KV precision, recompute the minimum feasible degree at each precision, and measure SLO-sustaining throughput "
  "plus an output-quality check at every point. Evidence required: resident bytes per device per precision, the dequantisation kernel path "
  "actually taken, collective payload dtype and volume, and a task-level quality comparison against the full-precision baseline. Rollback "
  "gate: revert any precision reduction whose quality delta on the held-out task suite exceeds the pre-declared bound, regardless of the "
  "latency win it buys.",
  ["Quantisation can remove the memory justification for the parallel layout, invalidating the whole comparison.",
   "Dequantisation kernels may be unoptimised at some shapes, so lower precision does not always mean lower latency.",
   "Precision changes output quality, so a latency win must be paid for with a quality measurement."],
  ["Resident bytes per device at each precision, with the minimum feasible tensor degree recomputed.",
   "The dequantisation kernel path taken and the dtype and volume of each collective.",
   "A task-level output quality comparison against the full-precision baseline with a pre-declared bound."]),

 ("Stance 91 - Prefix and KV cache reuse can dominate the layout effect, because a cache hit removes the prefill the argument is about.",
  "Systems with prefix caching or a disaggregated KV store serve repeated system prompts and shared conversation heads without recomputing "
  "prefill. When hit rate is high, most requests skip the compute-heavy phase where the two layouts differ most, and the measured gap "
  "collapses toward the decode-only difference. Cache reuse also interacts with sharding: under tensor parallelism each rank holds a slice of "
  "the cached KV, so a reuse path must reconstruct or address the same sharding as the serving layout. Falsifiable hypothesis H91: the "
  "inter-layout latency gap measured at a realistic cache hit rate is smaller than the gap measured with caching disabled, because prefill "
  "work is where the layouts diverge and hits eliminate it (ESTIMATE; derivation: expected prefill cost scales with one minus hit rate, so "
  "any prefill-borne layout difference scales with the same factor). Controlled experiment: replay the same trace with the cache cold, warm "
  "and disabled against both arms, reporting the measured hit rate alongside every latency figure. Evidence required: measured cache hit rate "
  "and reuse-length distribution for the replayed trace, the sharding scheme of cached KV and how it is addressed on reuse, and paired "
  "cache-on and cache-off results per arm. Rollback gate: reject any layout comparison that does not state the cache hit rate in force, since "
  "the same arms can rank differently at different hit rates.",
  ["A layout comparison run with caching disabled does not predict behaviour of a cached production system.",
   "Cached KV must match the serving layout's sharding, so cache reuse is not layout-neutral.",
   "Hit rate varies with traffic mix, so the ranking can move without any configuration change."],
  ["Measured cache hit rate and reuse-length distribution for the exact replayed trace.",
   "The sharding and addressing scheme of cached KV under each layout.",
   "Paired cache-cold, cache-warm and cache-disabled results for both arms."]),

 ("Stance 92 - Prefill/decode disaggregation reframes the question, because it lets each phase take its own layout.",
  "If prefill and decode run on separate device pools connected by a KV transfer path, the layout no longer has to be a single compromise: "
  "the compute-bound prefill pool can take a layout that maximises arithmetic throughput while the memory-bandwidth-bound decode pool takes "
  "one that maximises KV capacity and minimises per-step overhead. The cost is a KV transfer on every request, whose latency adds to TTFT and "
  "whose bandwidth becomes a new bottleneck. Falsifiable hypothesis H92: a disaggregated deployment meets the TTFT and TPOT SLO at higher "
  "throughput than the best single-layout colocated configuration, provided KV transfer time per request stays below the TTFT slack "
  "(ESTIMATE; derivation: transfer time is KV bytes per request divided by achieved link bandwidth, and disaggregation only pays when that "
  "term is smaller than the scheduling interference it removes). Controlled experiment: build both a colocated and a disaggregated arm on the "
  "same device budget and traffic, and measure TTFT decomposed into queueing, prefill and transfer. Evidence required: per-request KV bytes "
  "transferred and achieved link bandwidth, TTFT decomposed by phase, pool sizing and utilisation on both sides, and behaviour under a "
  "deliberate imbalance between prefill and decode demand. Rollback gate: collapse back to a colocated layout if measured transfer time or "
  "pool imbalance pushes TTFT past the SLO at target load.",
  ["Disaggregation introduces a KV transfer that adds to TTFT and can become the new bottleneck.",
   "Two pools must be sized against a traffic mix that shifts, so imbalance is a standing operational risk.",
   "The added transfer path is a new failure domain with its own timeouts and partial-failure modes."],
  ["Per-request KV bytes transferred and the achieved, not nominal, bandwidth of the transfer path.",
   "TTFT decomposed into queueing, prefill and KV transfer components.",
   "Pool utilisation on both sides under a deliberately imbalanced prefill-to-decode ratio."]),

 ("Stance 93 - Multi-tenancy and noisy neighbours can invalidate a clean single-tenant comparison.",
  "Benchmarks usually run one model alone on the node, but production nodes host sidecars, other replicas, log shippers and sometimes a second "
  "model. Contention appears on host CPU for tokenisation and scheduling, on PCIe for host-device transfers and on the shared interconnect, "
  "and the two layouts have different sensitivity: a collective-heavy layout suffers more from interconnect contention, while a host-bound "
  "pipeline suffers more from CPU contention. Falsifiable hypothesis H93: re-running both arms with a representative co-tenant load changes "
  "the sign or magnitude of the inter-layout gap relative to the isolated runs, showing the isolated benchmark was not decision-grade "
  "(ESTIMATE; derivation: each arm's latency includes a contention term proportional to its demand on the contended resource, and the arms "
  "demand different resources). Controlled experiment: repeat the identical comparison with and without a pinned, reproducible co-tenant "
  "workload, recording per-resource contention metrics in both conditions. Evidence required: co-tenant definition and its resource demand, "
  "host CPU, PCIe and interconnect contention counters per arm per condition, cgroup and affinity settings in force, and paired isolated "
  "versus contended results. Rollback gate: do not promote a layout on isolated-benchmark evidence alone when the production node is shared; "
  "require the contended repeat first.",
  ["Isolated benchmarks omit contention that production nodes always have.",
   "The two layouts load different shared resources, so contention can reorder them.",
   "cgroup, affinity and NUMA settings differ between benchmark and production hosts and silently change results."],
  ["A pinned, reproducible co-tenant workload definition with its resource demand.",
   "Host CPU, PCIe and interconnect contention counters for each arm in both conditions.",
   "cgroup, CPU affinity and NUMA placement settings recorded for every run."]),

 ("Stance 94 - Autoscaling and cold-start behaviour belong in the decision, because a layout's recovery time sets real availability.",
  "Latency at steady state is only half the service; the other half is how fast a replica comes back. Layouts differ in weight-loading "
  "pattern, warmup and graph-capture cost, and in how many devices must be simultaneously available before a replica can start at all. A "
  "layout needing more devices per replica has coarser scaling granularity and a longer window during which capacity is missing after a "
  "failure. Falsifiable hypothesis H94: time from process start to first token served at SLO differs materially between the layouts, and the "
  "layout with the larger device group has a longer effective capacity gap after an instance loss (ESTIMATE; derivation: capacity gap equals "
  "restart time multiplied by the fraction of fleet capacity one replica represents, and both factors move with devices per replica). "
  "Controlled experiment: kill a replica under steady load in each arm and measure time to full capacity restoration plus the error and "
  "latency excursion during the gap. Evidence required: cold-start timeline broken into weight load, warmup and graph capture; devices per "
  "replica and the resulting scaling granularity; and measured capacity-restoration time and error budget consumed per arm. Rollback gate: "
  "reject a layout whose measured restart time leaves the fleet below required capacity for longer than the error budget permits, even if its "
  "steady-state latency is better.",
  ["Steady-state latency ignores restart time, which often dominates availability.",
   "Larger device groups per replica mean coarser autoscaling and bigger capacity holes on failure.",
   "Warmup and graph capture costs are frequently excluded from benchmark timings."],
  ["A cold-start timeline decomposed into weight load, warmup and graph capture per arm.",
   "Devices per replica and the resulting scaling granularity and fleet-capacity fraction.",
   "Measured capacity-restoration time and error budget consumed during a killed-replica drill."]),

 ("Stance 95 - Failure-domain arithmetic favours the layout that loses less capacity per device failure.",
  "In a synchronous group, one failed device takes the whole group out. A layout with a larger group size therefore converts a single-device "
  "fault into a larger capacity loss, and it also raises the probability that any given replica is affected, since the replica fails if any of "
  "its devices fails. This is a reliability argument that is independent of latency and often points the opposite way from the benchmark. "
  "Falsifiable hypothesis H95: expected capacity loss per unit time is higher for the larger-group layout at equal per-device failure rate, "
  "and a fault-injection campaign reproduces the predicted ordering (ESTIMATE; derivation: replica availability is the per-device availability "
  "raised to the group size, so unavailability grows approximately in proportion to group size for small failure rates). Controlled "
  "experiment: inject single-device faults at a controlled rate into both arms under load and measure realised capacity loss and request "
  "failures, comparing against the predicted ordering. Evidence required: per-device failure and recovery statistics from the fleet, group "
  "size per layout, fault-injection campaign results with realised capacity loss, and the health-check semantics that determine how fast a "
  "degraded group is evicted. Rollback gate: require the reliability arithmetic and an injection result alongside any latency-based layout "
  "recommendation before it is approved.",
  ["Larger synchronous groups amplify single-device failures into larger capacity losses.",
   "Reliability arguments often contradict latency benchmarks and are routinely omitted from decision memos.",
   "Health-check semantics decide eviction speed and are usually untested for the failure shape in question."],
  ["Per-device failure and recovery statistics from the actual fleet, not vendor figures.",
   "Group size per layout and the derived replica availability arithmetic.",
   "Fault-injection results with realised capacity loss and measured eviction time."]),

 ("Stance 96 - Batching policy and scheduler configuration are confounders large enough to swap the ranking.",
  "Continuous batching, chunked prefill, max-num-batched-tokens, admission limits and preemption policy each move latency by more than many "
  "layout differences. If the two arms are run with the engine's defaults, those defaults may be tuned for one layout's shape and not the "
  "other's, so the experiment silently compares scheduler tuning rather than parallelism. Falsifiable hypothesis H96: independently tuning "
  "scheduler parameters within each arm changes the inter-layout gap by more than the gap measured at shared defaults, showing the default "
  "comparison measured tuning rather than layout (ESTIMATE; derivation: each arm's latency is a function minimised over its scheduler "
  "parameter space, and comparing unminimised points compares arbitrary interior values). Controlled experiment: run a bounded parameter "
  "sweep per arm, take each arm's best SLO-sustaining configuration, and compare optima rather than defaults. Evidence required: the full "
  "scheduler parameter set per run, the sweep grid and results, per-arm best configurations, and queue-depth and preemption counters "
  "throughout. Rollback gate: refuse to publish a layout verdict derived from default scheduler settings; require per-arm tuning to a "
  "documented stopping rule first.",
  ["Comparing engine defaults compares tuning artefacts, not parallel layouts.",
   "Defaults are often tuned for one layout's shape and disadvantage the other.",
   "Chunked prefill and preemption policy interact with layout, so their effects are not separable by inspection."],
  ["The complete scheduler parameter set recorded for every run.",
   "The per-arm sweep grid, its results and the documented stopping rule.",
   "Queue-depth, admission-refusal and preemption counters captured throughout each run."]),

 ("Stance 97 - Statistical design decides whether the measured gap means anything at all.",
  "A single run per arm, reported as a mean, cannot distinguish a real layout effect from run-to-run variance caused by placement, thermal "
  "state, background daemons and traffic randomness. Tail metrics are noisier than means and need more samples, so a p99 comparison from one "
  "short run is close to uninformative. Falsifiable hypothesis H97: the between-run variance within a single arm is comparable to the "
  "between-arm difference, meaning the reported gap is not distinguishable from noise at the sample size used (ESTIMATE; derivation: the "
  "uncertainty of an estimated percentile shrinks with the square root of sample count, so short runs leave wide intervals around tail "
  "statistics). Controlled experiment: run each arm at least five times in interleaved order with fresh placement each time, and report "
  "per-run values with confidence intervals rather than a single mean. Evidence required: raw per-run results for every repeat, the "
  "randomisation and interleaving schedule, confidence intervals on every reported percentile, and the pre-declared minimum effect size worth "
  "acting on. Rollback gate: do not act on a layout difference whose confidence interval includes zero, and do not declare equivalence from a "
  "wide interval either.",
  ["Single-run comparisons cannot separate layout effects from run-to-run variance.",
   "Tail percentiles need far more samples than means, so short runs give uninformative p99 figures.",
   "Non-interleaved runs confound layout with drift in machine state over the session."],
  ["Raw per-run results for at least five interleaved repeats per arm.",
   "The randomisation and interleaving schedule, with placement refreshed between repeats.",
   "Confidence intervals on every reported percentile and a pre-declared minimum actionable effect size."]),

 ("Stance 98 - Cost per sustained request, not latency, is the decision variable once both arms meet the SLO.",
  "If two configurations both satisfy the latency objective, the remaining question is which delivers the required throughput on fewer "
  "devices, including the devices idled by pipeline bubbles or held by a draft model or a separate prefill pool. A layout that is slightly "
  "faster but needs more devices per unit of sustained throughput is the worse choice, and framing the comparison purely in latency hides "
  "this. Falsifiable hypothesis H98: ranking the memory-feasible configurations by devices per SLO-sustaining request per second produces a "
  "different winner than ranking them by p95 latency at fixed load (ESTIMATE; derivation: latency at fixed load and throughput at fixed SLO "
  "are different functionals of the same latency-versus-load curve, so their argmaxima need not coincide). Controlled experiment: for each "
  "configuration, ramp load until the SLO is first violated, record the sustained rate, and divide device count by that rate to get the cost "
  "metric. Evidence required: latency-versus-load curves to the SLO violation point for every configuration, device counts including idle and "
  "auxiliary devices, and the resulting cost-per-sustained-request table. Rollback gate: do not approve a layout on a fixed-load latency win "
  "alone; require the cost-per-sustained-request comparison over all feasible configurations.",
  ["Fixed-load latency comparisons ignore how much load each configuration can actually sustain.",
   "Idle and auxiliary devices are often excluded from the device count, understating true cost.",
   "The latency winner and the cost winner frequently differ, so the framing choice decides the outcome."],
  ["Latency-versus-load curves extended to the first SLO violation for every configuration.",
   "Full device counts including bubble-idled, draft-model and auxiliary-pool devices.",
   "A cost-per-sustained-request table covering all memory-feasible configurations."]),

 ("Stance 99 - Close by stating what this record is not, so it cannot be mistaken for adjudicated ground truth.",
  "This record is provisional single-lane review output. It was written without executing any benchmark on the asker's hardware, without "
  "access to their topology or traffic, and without sight of any other reviewer's answer on the same item. Everything above is mechanism, "
  "boundary conditions, hypotheses and experimental design; no figure in it is MEASURED, and every quantitative statement carries an ESTIMATE "
  "tag with its derivation so a reader can audit the reasoning instead of adopting the number. The source pair is degenerate - its assistant "
  "turn is a grading rubric, not an answer - so there is no prior answer to agree or disagree with. Falsifiable hypothesis H99: records "
  "carrying an explicit provenance and authority bound are cited as ground truth by downstream steps at a lower rate than otherwise identical "
  "records without one, which would demonstrate the label does work rather than decorate (ESTIMATE; derivation: in the absence of an explicit "
  "bound, confident structured prose is read as authoritative regardless of how it was produced). Controlled experiment: attach this block to "
  "every record and audit every downstream consumer for any step that treats these as gold. Evidence required: the lane-isolation audit trail "
  "showing no teacher-A output was read during generation, a pre-declared inter-lane agreement metric computed only after both lanes are "
  "frozen, and a register of downstream consumers. Rollback gate: no training run, evaluation result or capability claim may cite these "
  "records as ground truth before an inter-lane agreement result exists and is published with them.",
  ["Provisional single-lane review is routinely over-read as adjudicated ground truth downstream.",
   "Confident structured prose signals an authority the provenance does not support.",
   "Without a lane-isolation audit, apparent inter-lane agreement may reflect contamination rather than convergence."],
  ["An audit trail showing the teacher-A lane was not read at any point during generation of this record.",
   "A pre-declared inter-lane agreement metric computed only after both lanes are frozen.",
   "A register of every downstream consumer of these records and how each used them."]),
]

CRITIQUE = (
"Critique of the source item: the prompt is a legitimate infrastructure question and does ask for assumptions, a "
"falsifiable hypothesis, measurements, confounders and rollback criteria, but the corpus pair is degenerate - the "
"assistant turn contains only a rubric describing what an answer should contain, not an answer. There is therefore no "
"substantive content to keep, and the item is rewritten into a complete response that supplies the mechanism, the "
"boundary conditions that flip the recommendation, an explicit falsifiable hypothesis, a single-variable controlled "
"experiment, the evidence artifacts required to adjudicate it, and a rollback gate. Every quantitative claim is labelled "
"ESTIMATE and carries its derivation; no value here is MEASURED, because no benchmark run was performed for this review. "
"This output is provisional teacher-B review material, not expert gold, and it is not evidence about any model's domain "
"capability."
)


def main():
    with open(CORPUS) as f:
        lines = f.readlines()[START:START + N]
    assert len(lines) == N, len(lines)
    assert len(STANCES) == N
    out = []
    for i, line in enumerate(lines):
        d = json.loads(line)
        m = {x["role"]: x["content"] for x in d["messages"]}
        su, sa = m["user"], m["assistant"]
        title, body, risks, ev = STANCES[i]
        ca = f"Analytical stance under test: {title}\n\n{COMMON}\n{body}\n\n{CRITIQUE}"
        out.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": ca,
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 2,
            },
            "risks": [
                "Source pair is degenerate: the assistant turn is a grading rubric rather than an answer.",
                "A bare TP-versus-PP verdict without interconnect, context length and concurrency context is not decidable.",
            ] + risks,
            "evidence_required": [
                "Interconnect topology dump and NCCL transport selection log for every arm of the comparison.",
                "Concurrency-resolved p50/p95/p99 TTFT and TPOT curves rather than mean end-to-end latency.",
            ] + ev,
            "confidence": 0.62,
        })
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out))


main()
