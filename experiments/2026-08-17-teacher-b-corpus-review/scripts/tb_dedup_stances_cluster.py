"""Authored teacher-B exemplars for the deduplicating stage, cluster families 514-521.

Each entry is (head, body, quality_dimensions, risks, evidence_required, confidence).
"""

QD = (2, 2, 2)


def _body(mech, hyp, falsi, metrics, est, exp, conf, roll):
    return f"""Mechanism. {mech}

Falsifiable hypothesis. H1: {hyp} Falsified if {falsi}

Metrics. {metrics} {est}

Controlled experiment. {exp}

Confounders. {conf}

Rollback criteria. {roll}"""


FAM_514 = [
 ("STANCE 200 - Separate the three claims RDMA bundles together, since kernel bypass, zero copy and offloaded transport fail independently and are measured differently.",
  _body(
   "RDMA is usually presented as one property but delivers three: the data path avoids kernel involvement, payloads are not copied into intermediate buffers, and the transport runs on the adapter rather than on host cores. A deployment can obtain one without the others, for example by registering memory correctly but still copying at the application layer, and the resulting performance looks like RDMA failing when it is in fact partially bypassed.",
   "host CPU time per transferred byte falls to near zero and copy counts on the data path are zero, so all three properties are active rather than only the transport offload.",
   "CPU time per byte remains comparable to the socket baseline, which indicates the application is still copying and the bypass is not reaching the payload.",
   "Host CPU cycles per transferred byte, memory copies observed on the data path, achieved bandwidth against link rate, completion latency distribution, and the equivalent figures for a socket baseline on the same hardware.",
   "All of these are MEASURED against the socket baseline; any bandwidth figure quoted from adapter specifications rather than from the deployment is an ESTIMATE and should not be used for planning.",
   "Run identical transfer sizes over the RDMA path and a socket path on the same hosts and links, reporting all three properties separately rather than reporting end-to-end bandwidth alone, since bandwidth can be adequate while CPU cost is unchanged.",
   "Memory registration cost is paid once and is invisible in steady-state measurements while dominating short-lived connections. Some stacks silently fall back to a copy path when registration fails, which looks like working RDMA at lower speed.",
   "If the CPU-per-byte property is absent, fix the application data path rather than tuning transport parameters. Treat any RDMA benefit claim without a same-hardware socket baseline as unsupported and do not carry it into a capacity plan.",
  )),
 ("STANCE 201 - Measure the completion latency distribution rather than the mean, because collective performance is set by the slowest participant.",
  _body(
   "A collective operation completes when its last participant completes, so the operation's latency tracks the tail of the per-link distribution rather than its mean. An interconnect change that improves mean latency while widening the tail can make collectives slower, and a mean-only measurement will report the change as an improvement.",
   "collective completion time correlates with the tail of the per-link completion distribution and not with its mean, so the tail is the quantity that must be optimised.",
   "collective time tracks the mean and is insensitive to the tail, which would indicate the collective is bandwidth-bound rather than synchronisation-bound at this message size.",
   "Full per-operation completion latency distribution including high percentiles, collective completion time, per-rank arrival skew at the collective, and message size.",
   "Distributions are MEASURED per rank; attributing collective slowdown to a specific link is an ESTIMATE unless per-link counters are collected simultaneously.",
   "Sweep message size across the range where the collective transitions from latency-bound to bandwidth-bound, recording the full distribution at each size, so the regime is identified rather than assumed.",
   "Rank arrival skew from unrelated compute imbalance produces the same signature as network tail latency and must be separated by timestamping arrival at the collective. Background traffic on shared links varies with cluster occupancy.",
   "Revert any interconnect or firmware change that widens the tail even if the mean improves, and record the tail percentile the decision was made on so later comparisons use the same statistic.",
  )),
 ("STANCE 202 - Confirm the path in use before attributing results to it, because silent fallback to a slower transport is the common explanation for absent gains.",
  _body(
   "Communication libraries select a transport at initialisation and fall back when a requirement is unmet, often logging at a verbosity that is disabled by default. A deployment can therefore be running over a TCP path while every configuration file names RDMA. Verifying the selected transport from runtime output and adapter counters must precede any performance interpretation.",
   "adapter-level counters account for the transferred bytes, confirming traffic traverses the intended path rather than a fallback.",
   "adapter counters show negligible traffic while the application transfers data, which localises the problem to transport selection rather than to tuning.",
   "Bytes reported by adapter counters against bytes transferred by the application, the transport named in initialisation output, and the count of fallback events logged at increased verbosity.",
   "Counter agreement is MEASURED; a transport named in configuration but not confirmed by counters is an ESTIMATE of what the system is doing and is frequently wrong.",
   "Enable the communication library's transport-selection logging, run a fixed transfer, and reconcile application bytes against adapter counters, treating any material discrepancy as the finding rather than proceeding to tuning.",
   "Multiple adapters may be present and counters read from the wrong one show no traffic. Container network namespaces can hide the adapter from the counter query while the transfer still uses it.",
   "Stop performance work until the path is confirmed, since every measurement before that point describes an unknown configuration. Record the verification output alongside any published result so the path is auditable later.",
  )),
]

FAM_515 = [
 ("STANCE 203 - Establish whether the tail is congestion or configuration before tuning, because lossless-fabric settings and queue buildup produce different signatures.",
  _body(
   "Converged Ethernet relies on priority flow control and explicit congestion notification to approximate losslessness. Tail latency can arise from queue buildup under load, which congestion notification is meant to control, or from a misconfiguration where flow control is not applied consistently on every hop and the fabric silently drops or pauses inappropriately. The two require opposite responses, so distinguishing them precedes any tuning.",
   "pause frames and congestion notification marks are present on the congested hops and scale with offered load, indicating a congestion-control tuning problem rather than a configuration gap.",
   "pause and mark counters remain near zero while the tail grows, which points to inconsistent configuration on some hop or to a non-network cause.",
   "Pause frame counts per port and direction, congestion notification marks, switch queue depth and drop counters, per-hop configuration consistency, and the latency tail correlated with each.",
   "Counters are MEASURED per port; attributing the tail to a specific hop is an ESTIMATE unless counters are collected on every hop simultaneously with the latency measurement.",
   "Sweep offered load while collecting switch and adapter counters and the latency distribution together, so the counters that move with the tail are identified rather than inferred from a single loaded snapshot.",
   "Priority mapping must match end to end, and a single mismatched hop invalidates the lossless assumption without producing an obvious error. Shared tenancy means background load varies independently of the test.",
   "Fix configuration consistency before tuning any congestion parameter, since tuning on an inconsistent fabric produces settings that will not transfer. Revert parameter changes individually and record per-hop configuration state with any published result.",
  )),
 ("STANCE 204 - Check whether incast is the mechanism, because many-to-one collective phases overwhelm a single egress port regardless of aggregate capacity.",
  _body(
   "Collective patterns concentrate traffic from many senders onto one receiver during specific phases. The receiver's egress port becomes the bottleneck even when the fabric has ample aggregate bandwidth, producing queue buildup and a tail that no amount of additional core capacity relieves. The distinguishing evidence is that congestion localises to receiver ports and appears only during those phases.",
   "queue buildup and marks localise to the receiving port during the many-to-one phase, so the tail is an incast effect rather than a fabric capacity shortfall.",
   "congestion appears on core links rather than at receivers, which indicates a capacity or routing problem instead.",
   "Per-port queue depth and mark rate resolved in time, the collective phase timeline, sender fan-in per receiver, and latency tail aligned to the phase boundaries.",
   "Port-level congestion is MEASURED with timestamps aligned to the collective phases; the fan-in implied by the algorithm is an ESTIMATE that must be checked against the library's actual implementation.",
   "Run the collective in isolation with per-port counters sampled finely enough to resolve its phases, and compare against a synthetic many-to-one pattern with known fan-in to confirm the signature matches.",
   "Counter sampling coarser than the phase duration averages the buildup away and shows nothing. Some collective algorithms change fan-in with message size, so a single size does not characterise the pattern.",
   "Change the collective algorithm or its chunking to reduce fan-in before changing fabric parameters, since incast is a traffic-shape problem. If a fabric parameter is changed, revert it if the phase-aligned signature persists, because that indicates it was not the mechanism.",
  )),
 ("STANCE 205 - Pin the routing behaviour, because flow-level hashing can place multiple heavy flows on one path and produce a tail that looks like congestion control failing.",
  _body(
   "Equal-cost multipath selects a path by hashing flow identifiers, which is effective for many small flows and poor for few large ones. Collective traffic consists of a small number of long-lived heavy flows, so collisions are likely and persistent: two flows share a link while parallel links stay idle. The resulting tail is a placement problem, and congestion-control tuning cannot fix it.",
   "utilisation is uneven across equal-cost links during the collective, so the tail is caused by flow placement rather than by insufficient aggregate capacity.",
   "utilisation is even across parallel links while the tail persists, which returns the diagnosis to congestion control or to the endpoints.",
   "Per-link utilisation across the equal-cost set during the collective, the number of distinct flows, the flow-to-link mapping, and the latency tail correlated with the imbalance.",
   "Link utilisation is MEASURED; the flow-to-link mapping is often an ESTIMATE inferred from utilisation unless the switch exposes the hash result directly, and that inference should be stated.",
   "Run the collective repeatedly and record per-link utilisation each time, since hashing depends on port numbers that vary between runs; a placement problem shows as run-to-run variance in both utilisation and the tail.",
   "Run-to-run variance from hashing is easily mistaken for measurement noise if only one run is taken. Adaptive routing, where available, changes the mechanism entirely and must be known to be enabled or not.",
   "Prefer increasing flow count through connection striping or enabling adaptive routing over tuning congestion parameters when imbalance is measured. Record which routing mode was active with every result, since results do not transfer between modes.",
  )),
]

FAM_516 = [
 ("STANCE 206 - Validate that the direct path is actually taken before measuring its benefit, because the staged path is the silent fallback.",
  _body(
   "Direct device-to-device transfer over the network requires driver, topology and memory-registration conditions to all hold. When any fails, the stack stages through host memory and the transfer still succeeds at lower speed. A validation plan that begins with performance measurement therefore cannot distinguish a small benefit from a fallback, and must first establish which path is in use.",
   "host memory bandwidth consumed during the transfer is near zero and adapter counters account for the bytes, confirming the direct path.",
   "host memory traffic scales with the transfer size, which demonstrates staging regardless of the achieved bandwidth.",
   "Host memory bandwidth during transfer, adapter byte counters, achieved transfer bandwidth, host CPU utilisation, and the topology distance between the device and the adapter.",
   "Path confirmation is MEASURED through counters; the topology-implied capability is an ESTIMATE from the device tree and does not establish that the path is being used.",
   "Compare a transfer with the direct path deliberately disabled against one with it enabled, on the same hosts, checking host memory traffic in both, so the counter signature of each path is known before any benefit is claimed.",
   "Devices and adapters under different root complexes may not support the direct path at all, making the comparison a topology question rather than a configuration one. Driver version changes the requirements without changing configuration.",
   "Do not report a benefit without the counter evidence for the path. If the direct path is unavailable due to topology, record that as a placement constraint rather than continuing to tune, since no software change will enable it.",
  )),
 ("STANCE 207 - Scope the benefit to the transfer sizes and frequencies the workload actually produces.",
  _body(
   "Direct-path benefits are largest for large contiguous transfers and can be negative for small ones, where per-transfer setup dominates. A validation using large synthetic buffers therefore measures a regime the workload may never enter. The transfer-size distribution of the real workload must define the measurement range, not the other way round.",
   "the workload's transfer-size distribution falls predominantly in the range where the direct path measured a benefit, so the synthetic result transfers to production.",
   "the workload's transfers concentrate below the crossover size, in which case the measured benefit does not apply and may be a cost.",
   "Transfer-size distribution from the production workload, bandwidth and latency for both paths across that size range, the crossover size, and the share of workload bytes and transfers above it.",
   "Both the size distribution and the per-size performance are MEASURED; the aggregate benefit computed by weighting one by the other is an ESTIMATE and should be reported as such.",
   "Instrument the production workload to obtain the size distribution, then benchmark both paths across exactly that range including the small sizes, and report the benefit weighted by the observed distribution rather than at the best size.",
   "Batching upstream changes the size distribution, so a benefit measured at one batch configuration does not transfer to another. Share of bytes and share of transfers give different answers, and the latency-relevant one is the transfer share.",
   "Roll back the enablement if the weighted benefit is negative or within run-to-run variation, and re-run the weighting whenever batching policy changes, since the distribution rather than the hardware determines the outcome.",
  )),
 ("STANCE 208 - Include a correctness and stability arm, because the direct path changes memory coherence and failure behaviour, not only speed.",
  _body(
   "Bypassing host memory changes when data is visible to which agent and how failures are surfaced. Errors that would have appeared as a host-side copy failure can instead manifest as silently stale data or as a hang in a completion queue. A validation plan measuring only bandwidth can qualify a configuration that is faster and occasionally wrong, which is worse than the staged path.",
   "outputs are bit-identical between the direct and staged paths across a sustained run, and induced faults surface as reported errors rather than as hangs or silent corruption.",
   "any divergence or unreported fault appears, which disqualifies the configuration irrespective of its measured speed.",
   "Bit-level output comparison between paths, count of unreported faults under injection, completion-queue error counts, hang occurrences, and sustained-run duration.",
   "Output identity and fault behaviour are MEASURED over the sustained run; the absence of rare corruption is an ESTIMATE bounded by the run length, which must be stated with the result.",
   "Run both paths on identical inputs with fixed seeds for a duration long enough to cover the deployment's fault rate, comparing outputs bitwise and injecting link and device faults to observe how each path reports them.",
   "Short runs cannot observe rare conditions, so an uneventful validation is weak evidence. Non-deterministic reduction order can produce benign divergence that must be distinguished from corruption before it is reported.",
   "Disqualify on any correctness divergence regardless of performance, and state the observation window with the result so that a later, longer-running failure is understood as outside the validated envelope rather than as a contradiction.",
  )),
]

FAM_517 = [
 ("STANCE 209 - Establish that storage is on the critical path before evaluating a storage-path optimisation.",
  _body(
   "Direct storage-to-device paths only help when storage transfer time is a material share of the operation being optimised. In inference serving, weight loading is a startup cost and cache offload is intermittent, so the storage share can be small enough that even a large relative improvement is invisible end to end. Establishing the share first prevents a well-executed evaluation of an irrelevant component.",
   "storage transfer accounts for a material share of the targeted operation's wall-clock time, so a storage-path improvement can produce a proportionate end-to-end effect.",
   "storage is a small share, in which case the maximum achievable end-to-end gain is bounded below what would justify the change regardless of the path's speed.",
   "Wall-clock time of the targeted operation decomposed into storage transfer, host processing and device work, the storage share, and the end-to-end gain implied by removing storage time entirely.",
   "The decomposition is MEASURED; the implied maximum gain from fully eliminating storage time is an ESTIMATE and represents an upper bound that no implementation will reach.",
   "Instrument the operation to attribute time to each stage, then compute the upper bound on gain before running any storage benchmark, so the evaluation is scoped by what is achievable rather than by what is measurable.",
   "Page cache makes repeat runs read from memory rather than storage, collapsing the apparent storage share; caches must be dropped between runs or the effect stated. Prefetching overlaps storage with compute and changes attribution.",
   "Stop the evaluation if the upper bound is below the threshold that would justify the operational change, and record that bound as the reason so the question is not reopened without new information.",
  )),
 ("STANCE 210 - Measure with caches controlled, because the page cache is the single largest confounder in storage evaluation.",
  _body(
   "Operating systems retain recently read data in host memory, so a second read of the same file does not touch storage at all. Benchmarks that iterate over one file therefore measure memory bandwidth after the first iteration, which makes both the baseline and the optimised path look fast and destroys the comparison. Cache state must be explicitly controlled and reported.",
   "measured storage bandwidth is stable across repeated runs with caches dropped and differs materially from the uncontrolled repeat-run figure, confirming the cache was dominating.",
   "the two figures agree, which would indicate the working set already exceeds cache capacity and the control is unnecessary here.",
   "Achieved bandwidth with caches dropped and with caches warm, working-set size against available host memory, run-to-run variation under each condition, and the device-level read counters.",
   "Bandwidth under each cache condition is MEASURED; device counters must corroborate the drop, since a bandwidth figure alone is an ESTIMATE of whether storage was actually touched.",
   "Run the benchmark with caches dropped before each iteration and with them warm, reporting both, and corroborate the dropped-cache runs against device-level read counters to confirm the reads reached storage.",
   "Dropping caches requires privileges that may be unavailable in containers, in which case working-set size must exceed memory instead. Storage devices have their own caches that cannot be dropped from the host.",
   "Report both cache conditions rather than choosing one, and treat any published storage result without a stated cache condition as uninterpretable, including previously published internal results.",
  )),
 ("STANCE 211 - Compare against a properly tuned conventional path, not against an unoptimised default.",
  _body(
   "A direct storage path is frequently compared against single-threaded buffered reads, which is not what a tuned conventional path looks like. Parallel readers, direct I/O and appropriate request sizes close much of the gap. Without that baseline the measured advantage attributes to the new path improvements that were available without it.",
   "the tuned conventional baseline achieves substantially higher throughput than the default one, so the advantage attributed to the direct path shrinks once the comparison is fair.",
   "tuning the conventional path yields little improvement, which strengthens the case that the remaining gap is attributable to the path itself.",
   "Throughput of default buffered, tuned conventional and direct paths under identical cache conditions, the tuning parameters used for each, and host CPU cost per byte for all three.",
   "All three are MEASURED under the same conditions; a comparison against an untuned baseline is an ESTIMATE of the wrong quantity and should not be reported as a path comparison.",
   "Tune the conventional path first by sweeping reader concurrency and request size to its own optimum, then compare at each path's best configuration rather than at a single shared setting that favours one.",
   "Tuning the conventional path consumes host CPU that may be needed elsewhere, so throughput parity does not mean equivalence and CPU cost must be reported alongside. Filesystem choice affects the achievable conventional throughput.",
   "Withdraw any advantage claim not measured against the tuned baseline, and record the tuning sweep alongside the result so a later reviewer can see which configuration each number came from.",
  )),
]

FAM_518 = [
 ("STANCE 212 - Account for the cache transfer cost, because disaggregating prefill from decode moves the key-value cache across the network on every request.",
  _body(
   "Separating prefill and decode onto different instances requires the cache produced by prefill to reach the decode instance. That volume is proportional to prompt length and to the per-token cache footprint, and it must cross the interconnect within the request's latency budget. The architecture's benefit therefore depends on whether interconnect bandwidth makes that transfer cheap relative to the prefill compute it decouples.",
   "measured cache transfer time is small relative to prefill compute time at the deployment's prompt-length distribution, so disaggregation can profit.",
   "transfer time is comparable to or exceeds prefill time, which makes the architecture a latency regression irrespective of its scheduling benefits.",
   "Cache bytes transferred per request, achieved transfer bandwidth, transfer time against prefill compute time across the prompt-length distribution, and end-to-end time to first token for both architectures.",
   "Transfer time and prefill time are MEASURED on the target interconnect; a transfer time computed from nominal link bandwidth is an ESTIMATE that typically overstates achievable rates substantially.",
   "Measure both architectures on the same hardware across the production prompt-length distribution, reporting time to first token and inter-token latency separately, since disaggregation affects the two differently.",
   "Prefix caching reduces both prefill compute and transferred bytes, shifting the ratio without changing either component's cost model. Transfer can overlap with the start of decode, so serial accounting overstates the penalty.",
   "Roll back to colocated execution if time to first token regresses at the production prompt distribution, and re-evaluate only when interconnect bandwidth or the prompt distribution changes materially, since those are the terms that decide it.",
  )),
 ("STANCE 213 - Name the resource asymmetry the design is meant to exploit, because without one the added complexity has no source of benefit.",
  _body(
   "Disaggregation profits when prefill and decode have different resource profiles that colocation cannot satisfy simultaneously: prefill is compute-intensive and bursty, decode is memory-bandwidth-bound and long-lived. If a deployment's workload does not exhibit that asymmetry, or if the hardware pool is homogeneous, the separation adds a network hop and a failure mode without addressing any constraint.",
   "measured per-phase resource profiles differ enough that a colocated instance is idle in one dimension while saturated in another, establishing the asymmetry the design targets.",
   "both phases saturate the same resource, in which case separation cannot improve utilisation and the added hop is pure cost.",
   "Per-phase compute utilisation and memory bandwidth utilisation, the fraction of time each resource is saturated, phase duration distribution, and utilisation of both resources under colocation.",
   "Per-phase utilisation is MEASURED under production load; the utilisation predicted after separation is an ESTIMATE and must be validated after any migration rather than assumed from the plan.",
   "Profile both phases separately under production load on colocated hardware, then compare the achievable utilisation of a separated configuration on the same total resources, holding the request trace fixed.",
   "Continuous batching already interleaves phases and recovers some of the same utilisation, so the colocated baseline must use it or the comparison is unfair. Short prompts make prefill negligible and remove the asymmetry entirely.",
   "Do not adopt the architecture on general reasoning about phase differences; require the measured asymmetry on the actual workload. If the asymmetry disappears after a workload change, revisit the architecture rather than tuning around it.",
  )),
 ("STANCE 214 - Enumerate the new failure modes, because splitting a request across two instances introduces partial-failure states that colocation does not have.",
  _body(
   "Once a request's prefill and decode run on different instances, the request can be in a state where prefill completed but the cache did not arrive, or where the decode instance fails holding a cache that cannot be reproduced without repeating prefill. These states need explicit handling, and their absence is a correctness and availability risk that no throughput measurement will reveal.",
   "every partial-failure state has a defined handling path with a bounded recovery cost, so no request can be lost or silently stalled.",
   "some state has no defined handling and manifests as a hang or a lost request under injection, which is a design gap rather than a tuning issue.",
   "Enumerated failure states with their handling paths, request loss and stall counts under injected instance and link failures, recovery latency per state, and cache memory retained after a failed transfer.",
   "Behaviour under injection is MEASURED; the enumeration's completeness is an ESTIMATE and should be revisited after every incident, since incidents are the main source of states the design did not anticipate.",
   "Inject decode-instance loss, transfer failure and prefill-instance loss at controlled points in the request lifecycle, recording the outcome for each, and confirm cache memory is reclaimed after each failure rather than leaking.",
   "Retries can mask lost requests in aggregate success metrics while doubling prefill cost invisibly. Timeouts tuned for colocated operation are usually too short for the transfer path and cause spurious failures.",
   "Do not promote the architecture until each enumerated state has a tested handling path. If an unhandled state is found in production, revert to colocated operation for the affected traffic rather than adding a retry, since a retry repeats the full prefill cost.",
  )),
]

FAM_519 = [
 ("STANCE 215 - Ask what the orchestration layer does that the serving engine does not, because overlapping responsibilities produce two schedulers fighting each other.",
  _body(
   "A serving engine already performs admission, batching and cache management within an instance. An orchestration layer adds routing, autoscaling and cross-instance placement. Where the two overlap, both make decisions on stale views of the same state, producing oscillation: the router sends load to an instance the engine is already about to reject. The boundary must be explicit before the layer is adopted.",
   "the responsibilities assigned to each layer are disjoint, so no decision is made concurrently by both on the same state.",
   "both layers act on queue depth or cache occupancy independently, which predicts oscillation under load and is observable as routing instability.",
   "Routing decision rate and its variance under steady load, rejection rate at instances the router selected, queue depth oscillation amplitude, and the staleness of the state each layer acts on.",
   "Oscillation and staleness are MEASURED under a steady load test; the claim that the layers are disjoint is an ESTIMATE derived from documentation until the oscillation test is run.",
   "Apply a constant offered load and observe whether routing decisions and queue depths settle or oscillate, since a disjoint boundary produces convergence and an overlapping one produces a persistent cycle whose period reveals the feedback delay.",
   "Autoscaling introduces its own slower feedback loop that can be mistaken for routing oscillation and must be disabled during the test. Load generators with their own retry logic add a third loop.",
   "Assign each contested decision to exactly one layer and disable it in the other rather than tuning both. If oscillation persists after the boundary is made disjoint, revert the orchestration layer rather than adding damping, since damping hides the coupling without removing it.",
  )),
 ("STANCE 216 - Measure cache-aware routing against the simplest baseline, because its benefit depends entirely on prefix reuse in the actual traffic.",
  _body(
   "Routing requests to the instance holding a matching prefix saves prefill work, but only when requests share prefixes and the matching instance has capacity. If traffic has little reuse, the routing constraint reduces load-balancing freedom and increases queueing without saving any compute. The reuse structure of the real traffic decides which effect dominates.",
   "measured prefix reuse in production traffic is high enough that cache-aware routing saves more prefill compute than the queueing it adds costs.",
   "reuse is low or is concentrated within a window shorter than the cache retention time, in which case the routing constraint is a net cost.",
   "Prefix reuse rate and reuse-gap distribution in production traffic, prefill compute saved by cache hits, queue wait attributable to routing constraints, cache hit rate per instance, and end-to-end latency against a least-loaded baseline.",
   "Reuse and hit rates are MEASURED from a replayed production trace; savings projected from a hit rate without measuring the added queueing are an ESTIMATE of only one side of the trade-off.",
   "Replay one production trace through cache-aware routing and through least-loaded routing on identical hardware, reporting both the compute saved and the queue wait added rather than the hit rate alone.",
   "Reuse is highly workload-dependent and a trace from one period may not represent another, particularly around deployment or prompt-template changes. Cache eviction under load reduces the hit rate exactly when routing pressure is highest.",
   "Fall back to least-loaded routing if end-to-end latency regresses, and re-measure reuse whenever prompt templates change, since a template change can eliminate reuse without any change to traffic volume.",
  )),
 ("STANCE 217 - Require that any autoscaling signal be measured against its actuation delay, because scaling a model server is slow relative to load changes.",
  _body(
   "Bringing up an additional model-serving replica requires scheduling, weight loading and warm-up, which takes long enough that a load spike is often over before the replica is ready. An autoscaler tuned on request-rate signals without accounting for that delay will add capacity after the incident and remove it before the next one, amplifying rather than damping the variation.",
   "the measured time from scaling decision to serving readiness is long relative to the load's autocorrelation time, so reactive scaling on request rate cannot track it.",
   "readiness time is short relative to load variation, in which case reactive scaling is viable and the remaining question is signal choice.",
   "Time from scaling decision to first served request decomposed into scheduling, weight load and warm-up, the load autocorrelation time, replica count against offered load over time, and the phase lag between them.",
   "Readiness time is MEASURED directly; the autocorrelation time is MEASURED from the traffic history, and any predicted benefit from a predictive scaler is an ESTIMATE until it is run in shadow.",
   "Measure readiness end to end under realistic image and weight sizes, then overlay replica count and offered load on one timeline to make the phase lag visible rather than inferring adequacy from a scaling-event count.",
   "Warm caches and pre-pulled images make test readiness much faster than production readiness on a cold node. Scaling limits and quota can delay actuation in ways that do not appear in a small test.",
   "Prefer provisioning for the measured peak over reactive scaling when readiness exceeds the autocorrelation time, and evaluate any predictive scaler in shadow before it is allowed to act, since a mistimed scale-down is more damaging than static over-provisioning.",
  )),
]

FAM_520 = [
 ("STANCE 218 - Separate configuration failures from topology failures, because both present as a hung collective with no useful error.",
  _body(
   "A multi-node collective that hangs gives almost no diagnostic information by default. The cause is usually either a configuration mismatch, where ranks disagree about the interface or the rendezvous endpoint, or a topology limitation, where a required path does not exist between two devices. These need different responses, and the first diagnostic step is to obtain the library's own topology and transport output rather than to change settings.",
   "the library's initialisation output shows a transport selection or ring construction that differs from the intended topology, localising the cause to configuration.",
   "the constructed topology matches intent and the hang persists, which moves the cause to a specific link or rank rather than to configuration.",
   "Library initialisation and topology output per rank, selected transport per rank pair, the rank at which the collective stalls, and interface enumeration on each host.",
   "The selection output is MEASURED from the run; the intended topology is a design ESTIMATE that must be compared against it rather than assumed to be what the library derived.",
   "Reproduce with the library's diagnostic verbosity enabled and collect output from every rank rather than from rank zero only, since the divergent rank is usually not the one that reports the failure.",
   "Container network namespaces and multiple interfaces make ranks enumerate different interfaces on identical hosts. Firewall rules can block the rendezvous port while the data path is fine, producing a hang at initialisation that resembles a data-path fault.",
   "Change one setting at a time and record the initialisation output with each, since a working configuration reached through several simultaneous changes cannot be attributed and will not survive the next environment change.",
  )),
 ("STANCE 219 - Establish the small-scale baseline before debugging at full scale, because bisection is faster than inspection.",
  _body(
   "A collective that fails across many nodes is hard to instrument and slow to iterate on. Reducing to two ranks on one host, then two hosts, then the full topology localises the failure to the step where it first appears, which usually identifies the responsible dimension directly. Starting at full scale spends most of the effort on reproduction rather than on diagnosis.",
   "the failure appears at a specific step in the scale-up sequence, isolating the responsible dimension to the one that changed at that step.",
   "the failure appears only at full scale with every smaller configuration passing, which points to a scale-dependent resource limit rather than to a configuration or link fault.",
   "Pass or fail outcome at each configuration in the scale-up sequence, the dimension changed at each step, time to reproduce at each scale, and resource limits at the failing scale.",
   "Outcomes are MEASURED at each step; the inference that the dimension changed at the failing step is the cause is an ESTIMATE that must be confirmed by reverting only that dimension.",
   "Run the sequence from single-host multi-device through two hosts to full scale, changing one dimension per step, and record the outcome and the library's topology output at each so the failing step has evidence attached.",
   "Intermittent failures can pass a small configuration by chance, so each step needs enough repetitions to distinguish a pass from a lucky run. Resource limits such as file descriptors and pinned memory scale with rank count and produce genuinely scale-only failures.",
   "Do not apply a fix found at small scale to production without re-running the full sequence, since a change that resolves a two-host failure can introduce a different failure at scale. Record the sequence outcome with the fix.",
  )),
 ("STANCE 220 - Distinguish a hang from extreme slowness, because they have different causes and the default symptom is identical.",
  _body(
   "A collective that has not completed may be deadlocked or may be progressing at a small fraction of expected speed, for example over a fallback transport. Both appear as a job that does not finish. Timeout configuration alone cannot separate them; progress must be observable, either through per-rank instrumentation or through network counters showing whether bytes are still moving.",
   "adapter counters show bytes continuing to move during the apparent hang, establishing slow progress rather than deadlock.",
   "counters are static while the job is stalled, which confirms a deadlock and redirects diagnosis to rank participation and ordering.",
   "Adapter byte counters sampled over time during the stall, per-rank progress markers, achieved bandwidth against expectation, and the set of ranks that have entered the collective.",
   "Counter movement is MEASURED during the stall; an inference of deadlock from the absence of completion alone is an ESTIMATE and is wrong often enough to mislead the entire investigation.",
   "Sample counters and per-rank progress markers during the stall rather than after the timeout, since post-mortem output cannot distinguish the two, and record which ranks had entered the collective at the time of sampling.",
   "Very slow progress will eventually trigger a timeout and be reported as a hang, so the report's wording is not evidence. Some ranks may exit early on error, leaving the rest genuinely deadlocked as a secondary effect of a different primary fault.",
   "Diagnose slow progress as a transport or routing problem and deadlock as a participation or ordering problem; do not raise timeouts to make a slow-progress case pass, since that converts a diagnosable fault into a latent one.",
  )),
]

FAM_521 = [
 ("STANCE 221 - Schedule gang-wise, because a distributed inference job with partially allocated ranks holds resources without serving anything.",
  _body(
   "A tensor-parallel deployment cannot serve until every rank is present. A scheduler that allocates ranks incrementally can leave a job holding most of its devices while waiting for the last one, blocking other jobs and producing zero throughput. All-or-nothing allocation with a bounded wait converts that indefinite hold into a clean rejection the operator can act on.",
   "under mixed job sizes, gang scheduling yields higher aggregate served throughput than incremental allocation because fewer devices are held by non-serving jobs.",
   "throughput is equal or lower, which would indicate fragmentation losses from all-or-nothing allocation exceed the partial-allocation waste at this job-size mix.",
   "Devices held by non-serving jobs over time, aggregate served throughput, job wait time distribution, allocation failure rate, and fragmentation of the free device pool.",
   "Held devices and throughput are MEASURED in a scheduling simulation or a controlled cluster; the benefit at a different job-size mix is an ESTIMATE and must be re-derived for that mix.",
   "Replay one job trace through both policies with identical arrival times and job sizes, reporting held-but-idle device time alongside throughput, since throughput alone hides the mechanism.",
   "Job-size mix dominates the result, so a trace with uniform sizes will not show the effect. Preemption policies interact with both schemes and must be held fixed or the comparison mixes two changes.",
   "Bound the gang wait explicitly and reject rather than queue indefinitely, so a starved job is visible. Revert to the prior policy if allocation failure rate rises without a throughput gain, since that indicates fragmentation is dominating.",
  )),
 ("STANCE 222 - Make recovery cost explicit, because restarting a model-serving replica reloads weights and rebuilds caches before it serves anything.",
  _body(
   "Recovery for a stateless service is fast, so a restart-on-failure policy is nearly free. A model-serving replica must load large weights, warm up kernels and rebuild any prefix cache, during which it consumes a device and serves nothing. A policy that restarts aggressively can therefore keep a significant share of the fleet permanently in recovery under a moderate failure rate.",
   "the product of failure rate and measured recovery duration accounts for a material share of fleet device time, so recovery cost constrains the restart policy.",
   "that product is negligible, in which case an aggressive restart policy is inexpensive and the design question is elsewhere.",
   "Measured recovery duration decomposed into scheduling, weight load, warm-up and cache rebuild, observed failure rate per replica-hour, fleet device time in recovery, and served throughput during recovery.",
   "Recovery duration and failure rate are MEASURED; the fleet-time share is an ESTIMATE computed from them and is sensitive to a failure rate observed over too short a window.",
   "Measure recovery end to end on a cold node with production weight sizes, then combine with the observed failure rate to compute the fleet share, rather than assuming recovery resembles a warm restart.",
   "Warm page cache and pre-pulled images make measured recovery far faster than a genuine cold recovery, which is the case that matters. Correlated failures cause many simultaneous recoveries, so the average share understates the worst case.",
   "Prefer draining and repairing over immediate restart when recovery is long relative to the failure interval, and revert any policy change that increases fleet time in recovery even if it reduces the count of failed requests.",
  )),
 ("STANCE 223 - Decide whether the failure domain is the replica or the job, because tensor-parallel ranks fail together and data-parallel replicas do not.",
  _body(
   "Within a tensor-parallel group, the loss of one rank makes the group unable to serve, so the failure domain is the whole group. Across data-parallel replicas the domain is a single replica. A recovery policy that treats every device as an independent unit will restart one rank into a group that has already failed, and a policy that treats the job as atomic will needlessly restart healthy replicas. The parallelism layout determines which applies.",
   "the observed correlation of failures within tensor-parallel groups is near total while correlation across data-parallel replicas is low, matching the layout-derived failure domains.",
   "failures correlate across data-parallel replicas as well, which indicates a shared cause such as a host, rack or dependency rather than the parallelism layout.",
   "Failure correlation within and across parallel groups, the parallelism layout mapped onto physical hosts and racks, restart counts by unit, and time to restored service per policy.",
   "Correlations are MEASURED from incident history; the mapping of logical ranks onto physical failure domains is an ESTIMATE unless it is read from the actual placement rather than from the intended one.",
   "Analyse incident history grouped by the actual physical placement, and compare restart-by-rank against restart-by-group policies on that history for time to restored service and wasted restarts.",
   "Placement can drift from the intended layout after rescheduling, so the logical mapping may not reflect physical reality. A shared host or power domain creates correlation that has nothing to do with parallelism and will be misattributed to it.",
   "Set the recovery unit from the measured failure domain rather than from convenience, and re-derive it whenever placement or parallelism configuration changes, since a policy correct for one layout is wrong for another.",
  )),
]

RISKS = ["Generated review content is provisional and is not expert-verified gold.",
         "Cluster-level conclusions depend on topology and workload that must be measured rather than assumed."]
EVIDENCE = ["Counters and timings collected on the target cluster, with the configuration recorded alongside.",
            "Explicit labelling of estimated versus measured quantities."]
CONFIDENCE = 0.71

STANCES = {
    fam: [(head, body, QD, list(RISKS), list(EVIDENCE), CONFIDENCE) for head, body in entries]
    for fam, entries in (
        (514, FAM_514), (515, FAM_515), (516, FAM_516), (517, FAM_517),
        (518, FAM_518), (519, FAM_519), (520, FAM_520), (521, FAM_521),
    )
}
