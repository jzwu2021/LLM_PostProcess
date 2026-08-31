"""Cluster interconnect and fabric mechanisms (topic: networking)."""
from __future__ import annotations

from core import Mechanism, Quant, Setting, fmt_int, gib, register


def q_rdma_cpu(s: Setting) -> Quant:
    return Quant(
        label="the three separate properties an RDMA path is expected to deliver",
        steps=[
            "Kernel bypass: the data path does not enter the operating system on each transfer",
            "Zero copy: the payload is not staged through an intermediate buffer",
            "Transport offload: protocol processing runs on the adapter rather than on host cores",
            f"Each can be absent independently, so a deployment on {s.interconnect} may hold one "
            f"without the others",
        ],
        value="three properties that fail independently and are measured separately",
        interpretation=(
            "A single end-to-end bandwidth number cannot distinguish which of the three is active, "
            "which is why an RDMA benefit claim needs a same-hardware socket baseline and host CPU "
            "accounting rather than a throughput figure alone."),
    )


def q_incast(s: Setting) -> Quant:
    senders = max(s.gpu_count - 1, 1)
    payload = 2 * s.hidden * s.dtype_bytes
    return Quant(
        label="the fan-in one receiver absorbs during a many-to-one collective phase",
        steps=[
            f"Fleet size {s.gpu_count}, so up to {senders} senders target one receiver in the "
            f"gather phase",
            f"Payload per sender per token = 2 * {fmt_int(s.hidden)} * {s.dtype_bytes} = "
            f"{fmt_int(payload)} B",
            f"Arriving simultaneously: {senders} * {fmt_int(payload)} = {gib(payload * senders)} "
            f"at one port",
            f"The receiver's egress capacity is one port's worth regardless of fabric aggregate "
            f"bandwidth",
        ],
        value=f"{senders}-way fan-in delivering {gib(payload * senders)} into a single port per token",
        interpretation=(
            "Aggregate fabric capacity does not relieve this. The bottleneck is one port during one "
            "phase, so the fix is traffic shape rather than more core bandwidth."),
    )


def q_ecmp_collision(s: Setting) -> Quant:
    flows = max(s.gpu_count // 2, 1)
    return Quant(
        label="why a handful of heavy flows defeats equal-cost path hashing",
        steps=[
            f"Collective traffic consists of roughly {flows} long-lived heavy flows, not many small ones",
            "Each flow is hashed to one path and stays there for its lifetime",
            f"With few flows and several equal-cost paths, collisions are likely and persistent",
            "Two colliding flows share one link's capacity while parallel links stay idle",
        ],
        value=f"about {flows} long-lived flows hashed independently onto a small path set",
        interpretation=(
            "Hashing is designed for many short flows. With few long ones the placement is effectively "
            "a lottery re-drawn on every run, which is why the tail varies run to run."),
    )


def q_pfc_scope(s: Setting) -> Quant:
    return Quant(
        label="the number of hops that must agree for a lossless fabric to behave losslessly",
        steps=[
            "Priority flow control is hop-by-hop: each link pauses independently",
            "Every hop on the path must map the traffic to the same priority class",
            f"On a fleet of {s.gpu_count} devices over {s.interconnect}, a path crosses several "
            f"switch hops in each direction",
            "A single mismatched hop breaks the lossless property for the whole path",
        ],
        value="every hop on the path, in both directions, must agree on the priority mapping",
        interpretation=(
            "This is a configuration-consistency property rather than a tuning parameter. Tuning "
            "congestion control on an inconsistent fabric produces settings that will not transfer."),
    )


def q_gdr_stage(s: Setting) -> Quant:
    payload = s.kv_bytes_per_token * s.ctx
    return Quant(
        label="the host memory traffic a staged transfer adds that a direct path avoids",
        steps=[
            f"A full-context cache transfer moves {gib(payload)}",
            "On the direct path that traffic crosses device to adapter without touching host memory",
            f"Staged, it is written to host memory and read back: {gib(2 * payload)} of host traffic",
            "Host memory bandwidth is shared with every other process on the machine",
        ],
        value=f"{gib(2 * payload)} of host memory traffic per transfer on the staged path, zero on the direct one",
        interpretation=(
            "Host memory traffic is the counter that distinguishes the two paths. Achieved bandwidth "
            "alone cannot, because a staged path can still reach a respectable figure."),
    )


def q_collective_tail(s: Setting) -> Quant:
    return Quant(
        label="why collective completion tracks the tail rather than the mean of per-link latency",
        steps=[
            f"A collective over {s.gpu_count} participants completes when the last one completes",
            "The expected maximum of many samples sits far above their mean",
            f"With {s.gpu_count} participants, the completion time approximates a high quantile of "
            f"the per-link distribution",
            "Improving the mean while widening the tail therefore makes collectives slower",
        ],
        value=f"completion tracks roughly the {100 - 100 // max(s.gpu_count, 1)}th percentile of per-link latency",
        interpretation=(
            "Reporting mean link latency for a collective workload measures the wrong statistic. Any "
            "fabric change must be judged on the tail it produces at the participant count in use."),
    )


def q_transfer_budget(s: Setting) -> Quant:
    payload = s.kv_bytes_per_token * (s.ctx // 2)
    budget_s = s.slo_ms / 1000.0
    return Quant(
        label="the bandwidth a cross-instance cache transfer needs to fit the latency budget",
        steps=[
            f"Half-context cache payload = {fmt_int(s.kv_bytes_per_token)} * {fmt_int(s.ctx // 2)} = "
            f"{gib(payload)}",
            f"Latency budget for the whole request is {s.slo_ms} ms = {budget_s:.3f} s",
            f"If the transfer may consume a tenth of it: {budget_s / 10:.4f} s",
            f"Required rate = {gib(payload)} / {budget_s / 10:.4f} s = "
            f"{payload / (1024 ** 3) / (budget_s / 10):.1f} GiB/s",
        ],
        value=f"{payload / (1024 ** 3) / (budget_s / 10):.1f} GiB/s to keep the transfer within a tenth of the budget",
        interpretation=(
            f"Compare that against achieved rather than nominal link rate on {s.interconnect}. "
            f"Achievable rates are commonly well below the nominal figure, and the gap decides "
            f"whether cache movement is viable at all."),
    )


def q_topology_distance(s: Setting) -> Quant:
    return Quant(
        label="why device-to-adapter placement decides whether a direct path exists at all",
        steps=[
            "A direct device-to-adapter path requires both to sit under a compatible topology branch",
            "Devices and adapters under different root complexes may have no direct route",
            f"On {s.gpu_count} devices with {s.interconnect}, placement is decided at provisioning",
            "No driver or runtime setting creates a path the hardware topology does not provide",
        ],
        value="a placement constraint fixed at provisioning, not a runtime setting",
        interpretation=(
            "When the topology does not permit the path, further tuning cannot help. That conclusion "
            "should be recorded as a placement finding so the question is not reopened as a software "
            "problem."),
    )


def q_rendezvous(s: Setting) -> Quant:
    return Quant(
        label="the separation between the rendezvous path and the data path",
        steps=[
            "Initialisation contacts a coordination endpoint to exchange ranks and addresses",
            "The data path afterwards uses different ports and often a different interface entirely",
            f"Across {s.gpu_count} participants, every rank must reach the coordination endpoint first",
            "A blocked coordination port produces a hang before any data transfer is attempted",
        ],
        value="two independent paths: coordination first, then bulk data on different ports",
        interpretation=(
            "A hang at initialisation and a hang during transfer have different causes and different "
            "evidence. Confirming which path failed is the first step, and it is cheap."),
    )


def q_flow_striping(s: Setting) -> Quant:
    base = max(s.gpu_count // 2, 1)
    return Quant(
        label="how many flows are needed before path hashing spreads load evenly",
        steps=[
            f"Current flow count is roughly {base}, one per communicating pair",
            "Hash-based placement approaches even utilisation only when flows greatly outnumber paths",
            f"Striping each logical connection over several channels multiplies the flow count",
            f"At four channels per pair the count rises from {base} to {base * 4}",
        ],
        value=f"{base} flows now, {base * 4} after four-way striping",
        interpretation=(
            "More flows make the hash behave statistically rather than as a lottery. This changes the "
            "variance of link utilisation, which is the quantity that produces the tail."),
    )


register(
    Mechanism(
        key="rdma_property_bundle", topic="networking",
        title="RDMA bundles three properties that fail independently and are measured separately",
        concepts=("rdma", "kernel_bypass", "benchmarking"),
        symptom="An RDMA-enabled path delivers acceptable bandwidth while host CPU utilisation remains as high as it was on the socket path.",
        chain="Kernel bypass, zero copy and transport offload are separate mechanisms, and a deployment can obtain the transport offload while the application still copies the payload, leaving CPU cost unchanged.",
        metric="Host CPU cycles per transferred byte, compared against a socket baseline on the same hardware.",
        signature="Adapter counters account for the bytes while host CPU per byte stays close to the socket baseline, which isolates the copy rather than the transport.",
        confounders=(
            "Memory registration cost, which is paid once and is invisible in steady-state measurement while dominating short connections.",
            "A silent fallback to a copy path when registration fails, which presents as working RDMA at lower speed.",
            "Other processes on the host contributing to the CPU measurement.",
        ),
        fixes=(
            "Run the same transfer over a socket path on the same hosts and compare all three properties rather than bandwidth alone.",
            "Fix the application data path so the payload is registered and not copied before hand-off.",
            "Confirm from adapter counters that the intended path carries the bytes before any tuning begins.",
        ),
        rollback="Stop performance work and re-establish the baseline if adapter counters do not account for the transferred bytes, since every measurement before that point describes an unknown configuration.",
        options=("comparing against a same-hardware socket baseline", "fixing the application data path to avoid the copy"),
        tradeoff="whether the missing property is in the transport or in the application above it",
        flip="the copy turns out to be inside a library the team does not control, at which point the baseline comparison is all that remains",
        falsifier="host CPU per byte falls to near zero and copy counts on the data path are zero",
        wrong_claim="The fabric is RDMA-capable and the configuration enables it, so transfers are zero copy.",
        wrong_why="Configuration expresses intent while counters express behaviour, and a stack that cannot satisfy a requirement falls back silently to a copy path that still reports as RDMA.",
        threshold="Require adapter counters to account for the transferred bytes before any RDMA benefit is credited.",
        cost="Purchasing RDMA-capable hardware for a path that still copies buys adapter capability the application never reaches.",
        scaling="The CPU saving matters more as transfer volume grows, so a partially bypassed path degrades relatively worse at scale.",
        quant=q_rdma_cpu,
    ),
    Mechanism(
        key="collective_incast", topic="networking",
        title="many-to-one collective phases overwhelm one receiver port regardless of fabric capacity",
        concepts=("incast", "collectives", "congestion"),
        symptom="Tail latency spikes during specific phases of a collective while overall fabric utilisation looks comfortable.",
        chain="Gather-style phases direct traffic from many senders onto one receiver, so that receiver's port queues while parallel links stay idle, and aggregate capacity does not relieve a single-port bottleneck.",
        metric="Per-port queue depth and congestion marks, sampled finely enough to resolve individual collective phases.",
        signature="Queue buildup localises at receiver ports and appears only during the many-to-one phase, aligned to the phase boundaries in time.",
        confounders=(
            "Core link congestion, which indicates a capacity or routing problem rather than incast.",
            "Rank arrival skew from compute imbalance, which delays the phase without any queueing.",
            "Counter sampling coarser than the phase duration, which averages the buildup away entirely.",
        ),
        fixes=(
            "Change the collective algorithm or its chunking so fewer senders target one receiver at once.",
            "Stagger the phase across sub-groups so fan-in is bounded rather than fleet-wide.",
            "Enable and tune congestion notification, which shapes the arrival rate but does not reduce the fan-in.",
        ),
        rollback="Revert an algorithm change if the phase-aligned queueing signature persists, since that means fan-in was not reduced by it.",
        options=("reducing fan-in through algorithm or chunking choice", "staggering the phase across sub-groups"),
        tradeoff="whether the traffic shape can be changed without lengthening the collective's critical path",
        flip="staggering lengthens the collective beyond the latency it saves, at which point the fabric rather than the shape has to absorb the burst",
        falsifier="congestion appears on core links rather than at receiver ports",
        wrong_claim="Fabric utilisation is well below capacity, so the network cannot be the source of these spikes.",
        wrong_why="Aggregate utilisation averages across links and time, while incast is one port during one phase, so the condition is invisible in exactly the metric being quoted.",
        threshold="Investigate incast when queue buildup concentrates at receiver ports during a known many-to-one phase.",
        cost="Adding fabric capacity to relieve incast pays for links that stay idle during the phase that hurts.",
        scaling="Fan-in grows with participant count, so the effect worsens as the deployment scales even at constant per-node traffic.",
        quant=q_incast,
    ),
    Mechanism(
        key="ecmp_flow_collision", topic="networking",
        title="path hashing places few heavy flows badly and produces a tail that resembles congestion",
        concepts=("ecmp", "routing", "tail_latency"),
        symptom="The same job shows materially different tail latency from run to run with no configuration change between runs.",
        chain="Equal-cost paths are selected by hashing flow identifiers, and collective traffic consists of a small number of long-lived heavy flows, so collisions are likely, persistent for the flow's lifetime, and re-drawn on each run as ports change.",
        metric="Per-link utilisation across the equal-cost set during the collective, collected on several runs.",
        signature="Utilisation is uneven across parallel links, and both the imbalance and the tail vary run to run while the configuration is fixed.",
        confounders=(
            "Genuine capacity shortfall, which shows as even utilisation at a high level rather than as imbalance.",
            "Adaptive routing being enabled, which changes the mechanism entirely and must be known.",
            "Single-run measurement, which cannot distinguish placement variance from measurement noise.",
        ),
        fixes=(
            "Stripe each logical connection over several channels so the flow count rises and hashing behaves statistically.",
            "Enable adaptive routing where the fabric supports it, so placement responds to load.",
            "Pin or engineer paths for the heaviest flows where the fabric permits explicit routing.",
        ),
        rollback="Revert striping if per-flow ordering requirements are violated or if the added connection count exhausts adapter resources.",
        options=("striping connections to raise the flow count", "enabling adaptive routing where supported"),
        tradeoff="whether the fabric offers load-responsive placement or only static hashing",
        flip="adaptive routing becomes available and handles placement directly, at which point striping adds connection overhead for no further benefit",
        falsifier="utilisation is even across parallel links while the tail persists",
        wrong_claim="Run-to-run variance in the tail is measurement noise, since nothing in the configuration changed.",
        wrong_why="Path assignment depends on ephemeral port numbers that differ per run, so the placement itself changes between runs even though the configuration does not.",
        threshold="Treat uneven utilisation across an equal-cost set during a collective as a placement finding rather than a capacity one.",
        cost="Adding links to a fabric whose problem is placement buys capacity that hashing will continue to leave idle.",
        scaling="Collision probability falls as flow count rises, so the problem is worst in the small-flow-count regime that collectives naturally produce.",
        quant=q_ecmp_collision,
    ),
    Mechanism(
        key="pfc_configuration_consistency", topic="networking",
        title="a lossless fabric requires every hop to agree, and one mismatch silently breaks it",
        concepts=("roce", "priority_flow_control", "configuration"),
        symptom="A converged Ethernet fabric configured as lossless still shows drops and erratic tail latency under load.",
        chain="Flow control is applied hop by hop and depends on consistent priority mapping along the whole path, so a single hop that maps the traffic differently will drop or fail to pause, and nothing reports the inconsistency.",
        metric="Per-hop priority mapping and pause frame counters, collected on every hop of the path in both directions.",
        signature="Pause counters are present on some hops and absent on others along the same path, identifying where the mapping diverges.",
        confounders=(
            "Genuine congestion within a correctly configured fabric, which shows consistent pause behaviour on all hops.",
            "Shared tenancy varying background load independently of the test.",
            "Counters read from the wrong direction, since mapping can differ per direction.",
        ),
        fixes=(
            "Audit the priority mapping on every hop in both directions and reconcile it before tuning anything.",
            "Restrict the traffic to a path set whose configuration has been verified, if a full audit is not immediately possible.",
            "Remove the lossless assumption and design for loss, if consistency cannot be maintained operationally.",
        ),
        rollback="Undo congestion-control parameter changes made before the audit, since settings tuned on an inconsistent fabric are not valid once it is consistent.",
        options=("auditing and reconciling the per-hop mapping", "restricting traffic to a verified path set"),
        tradeoff="whether the fabric's configuration can be made and kept consistent across every hop",
        flip="the fabric spans administrative boundaries where consistency cannot be enforced, at which point designing for loss is the honest choice",
        falsifier="pause and mark counters are consistent across every hop while the tail persists",
        wrong_claim="The fabric is configured lossless end to end, so drops must be a hardware fault.",
        wrong_why="Lossless behaviour is a property of every hop agreeing rather than of an intended configuration, and a single divergent hop produces exactly this symptom without any hardware fault.",
        threshold="Require verified per-hop priority mapping on the whole path before any congestion parameter is tuned.",
        cost="Tuning on an inconsistent fabric produces settings that must be discarded, so the effort is spent twice.",
        scaling="The probability that some hop diverges grows with path length and fabric size, so larger fabrics need the audit automated rather than manual.",
        quant=q_pfc_scope,
    ),
    Mechanism(
        key="gdr_silent_staging", topic="networking",
        title="a direct device-to-network path falls back to host staging without saying so",
        concepts=("gpudirect", "rdma", "validation"),
        symptom="A direct transfer path was enabled and the measured improvement is far smaller than expected, with no errors anywhere.",
        chain="The direct path requires driver, topology and registration conditions to hold together, and when any fails the stack stages the payload through host memory, which still succeeds and still reports as configured.",
        metric="Host memory bandwidth consumed during the transfer, alongside adapter byte counters.",
        signature="Host memory traffic scales with transfer size on the staged path and stays near zero on the direct one, which distinguishes them regardless of achieved bandwidth.",
        confounders=(
            "Topology genuinely not supporting the path, which is a provisioning constraint rather than a configuration fault.",
            "Driver version changing the requirements without any configuration change.",
            "Small transfer sizes where setup cost dominates and both paths look similar.",
        ),
        fixes=(
            "Measure host memory traffic with the direct path deliberately disabled and enabled, so each path's counter signature is known.",
            "Verify the device-to-adapter topology distance before attributing the shortfall to configuration.",
            "Record the path as unavailable and stop tuning, if the topology does not permit it.",
        ),
        rollback="Withdraw any published benefit figure that was not accompanied by the host-traffic evidence, since it cannot be attributed to the path.",
        options=("measuring host memory traffic on both paths", "verifying device-to-adapter topology distance"),
        tradeoff="whether the shortfall is a configuration fault or a topology limit",
        flip="the topology turns out to forbid the path, at which point configuration work is wasted and placement is the only lever",
        falsifier="host memory bandwidth stays near zero during the transfer while adapter counters account for the bytes",
        wrong_claim="The transfer completed at good bandwidth, so the direct path is working.",
        wrong_why="A staged path also completes at respectable bandwidth, so throughput cannot distinguish the two; only host memory traffic can.",
        threshold="Require near-zero host memory traffic during the transfer before the direct path is treated as active.",
        cost="Reporting a benefit that came from a staged path misdirects the next round of hardware purchasing.",
        scaling="The gap between paths widens with transfer size, so a validation done on small transfers understates both the benefit and the fallback penalty.",
        quant=q_gdr_stage,
    ),
    Mechanism(
        key="collective_tail_statistic", topic="networking",
        title="collective performance is set by the tail of per-link latency, never by its mean",
        concepts=("collectives", "tail_latency", "measurement"),
        symptom="A fabric change improved average link latency in benchmarks and made the distributed job slower.",
        chain="A collective completes only when its last participant completes, so its duration tracks a high quantile of the per-link distribution, and a change that improves the mean while widening the tail moves the quantity that matters in the wrong direction.",
        metric="Full per-link completion latency distribution including high percentiles, at the participant count actually in use.",
        signature="Collective completion time correlates with the distribution's tail and is insensitive to its mean across the sweep.",
        confounders=(
            "Rank arrival skew from compute imbalance, which produces the same signature without any network effect.",
            "Message size placing the collective in a bandwidth-bound rather than latency-bound regime.",
            "Averaging percentiles across shards, which is not a percentile of the combined distribution.",
        ),
        fixes=(
            "Report the full distribution rather than the mean for any fabric comparison.",
            "Timestamp rank arrival at the collective so compute skew can be separated from network tail.",
            "Evaluate fabric changes at the participant count in production, since the relevant quantile moves with it.",
        ),
        rollback="Revert a fabric or firmware change that widens the tail even if the mean improved, and record the percentile the decision was made on.",
        options=("reporting the full latency distribution", "timestamping rank arrival to separate compute skew"),
        tradeoff="whether the delay originates before the collective or inside it",
        flip="message sizes grow enough that the collective becomes bandwidth-bound, at which point the mean regains relevance and the tail argument weakens",
        falsifier="collective time tracks the mean and is insensitive to the tail across the message-size sweep",
        wrong_claim="Average link latency improved by a third, so distributed training and serving will both get faster.",
        wrong_why="Collectives are governed by their slowest participant, so an improvement in the mean paired with a wider tail makes them slower rather than faster.",
        threshold="Judge fabric changes on the quantile corresponding to the participant count, not on the mean.",
        cost="A fabric procured on mean-latency evidence can degrade the workload it was bought for.",
        scaling="The governing quantile rises with participant count, so a change that is neutral at small scale can be harmful at full scale.",
        quant=q_collective_tail,
    ),
    Mechanism(
        key="cache_transfer_budget", topic="networking",
        title="moving cache between instances must fit inside the request's latency budget",
        concepts=("disaggregation", "bandwidth", "slo"),
        symptom="A disaggregated design improved scheduling flexibility and pushed time to first token past its objective.",
        chain="Separating phases across instances requires the produced cache to cross the interconnect within the request's remaining budget, and that volume is proportional to prompt length, so the architecture's viability is decided by achieved bandwidth rather than by the scheduling benefit.",
        metric="Cache bytes transferred per request and achieved transfer rate, compared against the latency budget.",
        signature="Transfer time is a material share of time to first token and scales with prompt length rather than with load.",
        confounders=(
            "Prefix caching reducing both prefill work and transferred bytes, which shifts the ratio without changing either cost model.",
            "Transfer overlapping the start of decode, so serial accounting overstates the penalty.",
            "Nominal link rate being used in place of achieved rate, which understates the transfer time substantially.",
        ),
        fixes=(
            "Measure achieved transfer rate on the actual interconnect rather than quoting the nominal figure.",
            "Overlap the transfer with the start of decode where the runtime permits it.",
            "Return to colocated execution for the prompt-length range where the transfer cannot fit the budget.",
        ),
        rollback="Revert to colocated execution if time to first token regresses at the production prompt-length distribution, and re-evaluate only when bandwidth or that distribution changes.",
        options=("measuring achieved transfer rate on the real interconnect", "overlapping transfer with the start of decode"),
        tradeoff="whether achieved bandwidth carries the cache within the share of the budget allotted to it",
        flip="prompts grow long enough that even an overlapped transfer exceeds its share of the budget, at which point colocation is the only option that meets the objective",
        falsifier="transfer time is a negligible share of time to first token across the production prompt-length distribution",
        wrong_claim="The interconnect is rated far above what the cache transfer needs, so movement cost is not a concern.",
        wrong_why="Nominal rates are not achieved rates, and the achievable fraction on a shared fabric under concurrent collective traffic is commonly far lower than the rating.",
        threshold="Require the transfer to fit within a stated fraction of the first-token budget at the production prompt-length distribution.",
        cost="A disaggregated deployment that misses its latency objective spends extra devices to serve worse than a colocated one.",
        scaling="Transferred volume grows linearly with prompt length, so the design fails first on exactly the long-context traffic that motivates disaggregation.",
        quant=q_transfer_budget,
    ),
    Mechanism(
        key="topology_placement_limit", topic="networking",
        title="some fabric paths are decided at provisioning and cannot be created by configuration",
        concepts=("topology", "placement", "provisioning"),
        symptom="A direct path is configured, the driver reports support, and the path is never selected on some hosts and always selected on others.",
        chain="Whether a direct route exists between a device and an adapter is a property of how they are attached, so hosts provisioned with different attachment layouts differ in capability, and no runtime setting changes the attachment.",
        metric="Device-to-adapter topology distance per host, enumerated and compared against which hosts select the direct path.",
        signature="Path selection correlates exactly with topology distance and not with any software version or configuration difference.",
        confounders=(
            "Driver version differences between hosts, which also correlate with path selection for a different reason.",
            "Container namespaces hiding the adapter from enumeration while the transfer still uses it.",
            "Firmware settings that disable the capability independently of topology.",
        ),
        fixes=(
            "Enumerate topology distance on every host and record it as a scheduling attribute.",
            "Constrain placement so latency-sensitive workloads land only on hosts that support the path.",
            "Re-provision or replace the hosts whose layout forbids the path, if the workload requires it fleet-wide.",
        ),
        rollback="Remove the placement constraint if it fragments the pool badly enough to hurt scheduling, and record the capability difference instead of enforcing it.",
        options=("recording topology distance as a scheduling attribute", "constraining placement to supporting hosts"),
        tradeoff="whether enough supporting hosts exist to serve the workload without fragmenting the pool",
        flip="the supporting host pool becomes too small to schedule against, at which point re-provisioning rather than constraint is the answer",
        falsifier="path selection varies across hosts with identical topology distance",
        wrong_claim="The driver reports the capability as supported, so the path will be used.",
        wrong_why="Driver support is necessary and not sufficient; the route must also exist in the attachment layout, and where it does not the stack silently uses another path.",
        threshold="Treat topology distance as a hard placement constraint rather than a preference for workloads that depend on the direct path.",
        cost="Chasing a topology limit as a software problem consumes engineering time on a question no configuration can answer.",
        scaling="Heterogeneous fleets accumulate layout variants over successive procurements, so the problem grows with fleet age rather than with fleet size.",
        quant=q_topology_distance,
    ),
    Mechanism(
        key="rendezvous_vs_datapath", topic="networking",
        title="initialisation and bulk transfer use different paths and fail for different reasons",
        concepts=("nccl", "bootstrap", "diagnosis"),
        symptom="A distributed job hangs at startup with no error, and the fabric passes every bandwidth test.",
        chain="Ranks first exchange addresses through a coordination endpoint on one port set, then move data on another, so a blocked or misconfigured coordination path produces a hang before any bulk transfer is attempted and leaves data-path tests entirely uninformative.",
        metric="Which ranks reached the coordination endpoint, and whether any bytes appear on the data path at all.",
        signature="No traffic appears on the data path while some ranks are blocked before rendezvous completes, so the failure precedes any transfer.",
        confounders=(
            "A genuine data-path fault, which shows partial traffic followed by a stall rather than none at all.",
            "Container network namespaces making ranks enumerate different interfaces on identical hosts.",
            "A slow rather than blocked rendezvous, which eventually completes and looks like an intermittent fault.",
        ),
        fixes=(
            "Collect initialisation diagnostics from every rank rather than from the coordinating rank alone.",
            "Verify reachability of the coordination endpoint and its port from every participant before testing bandwidth.",
            "Pin the interface selection explicitly where hosts enumerate multiple interfaces.",
        ),
        rollback="Undo interface pinning if it excludes a rank that was previously reachable, since an over-narrow selection converts a partial failure into a total one.",
        options=("collecting initialisation diagnostics from every rank", "verifying coordination endpoint reachability first"),
        tradeoff="whether the failure occurs before or after rendezvous completes",
        flip="rendezvous is confirmed complete on all ranks, at which point the investigation moves entirely to the data path",
        falsifier="data-path traffic is present while the job is stalled",
        wrong_claim="Bandwidth tests pass between all node pairs, so the network is fine and the problem is in the framework.",
        wrong_why="Bandwidth tests exercise the data path, while the hang occurs during coordination on a different port set, so a passing test carries no information about the failing path.",
        threshold="Confirm rendezvous completion on every rank before spending any effort on data-path diagnosis.",
        cost="Bandwidth testing a healthy data path while the coordination path is blocked spends the diagnostic window on the wrong layer.",
        scaling="The chance that at least one rank cannot reach the coordination endpoint grows with participant count, so large jobs hang more often for this reason than small ones.",
        quant=q_rendezvous,
    ),
    Mechanism(
        key="flow_count_striping", topic="networking",
        title="raising the flow count makes path hashing behave statistically instead of as a lottery",
        concepts=("ecmp", "striping", "utilisation"),
        symptom="Link utilisation across parallel paths is persistently uneven during collectives, and the imbalance pattern changes on every restart.",
        chain="Hash-based placement approaches even distribution only when flows greatly outnumber paths, and collective traffic produces roughly one flow per communicating pair, so the sample is too small for the hash to average out.",
        metric="Number of distinct flows during the collective and the variance of per-link utilisation across the equal-cost set.",
        signature="Utilisation variance falls as flow count rises, and the tail improves with it in the same sweep.",
        confounders=(
            "Adaptive routing masking the effect where it is enabled, which changes the mechanism.",
            "Adapter resource limits capping the achievable connection count.",
            "Ordering requirements that forbid spreading one logical stream over several channels.",
        ),
        fixes=(
            "Stripe each logical connection over several channels and sweep the channel count against utilisation variance.",
            "Confirm the library supports multiple channels per pair before assuming the flow count can be raised.",
            "Move to adaptive routing instead, where the fabric offers it, since it addresses placement directly.",
        ),
        rollback="Reduce the channel count if adapter connection resources are exhausted or if ordering guarantees are violated, both of which cost more than the tail improvement.",
        options=("striping connections over more channels", "moving to adaptive routing where available"),
        tradeoff="whether flow count can be raised without breaking ordering or exhausting adapter resources",
        flip="adapter connection limits are reached before utilisation evens out, at which point only load-responsive routing can help",
        falsifier="utilisation variance is unchanged as flow count rises",
        wrong_claim="The fabric has ample aggregate bandwidth, so uneven link utilisation is harmless.",
        wrong_why="The collective is limited by its slowest path rather than by aggregate capacity, so an idle parallel link does not compensate for a congested one.",
        threshold="Raise the flow count until per-link utilisation variance falls below the band at which the tail stops improving.",
        cost="Idle parallel links represent purchased capacity that placement prevents the workload from using.",
        scaling="Required flow count grows with the number of equal-cost paths, so wider fabrics need proportionally more striping to stay balanced.",
        quant=q_flow_striping,
    ),
)
