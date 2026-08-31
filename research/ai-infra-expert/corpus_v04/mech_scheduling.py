"""Cluster scheduling, orchestration and capacity mechanisms (topic: scheduling)."""
from __future__ import annotations

from core import Mechanism, Quant, Setting, fmt_int, gib, register


def q_gang(s: Setting) -> Quant:
    held = s.tp - 1
    return Quant(
        label="the devices a partially allocated job holds while serving nothing",
        steps=[
            f"A tensor-parallel job needs all {s.tp} ranks before it can serve",
            f"Incremental allocation can leave it holding {held} devices while waiting for the last",
            f"Those {held} devices are unavailable to other jobs and produce zero output",
            f"On a fleet of {s.gpu_count}, that is "
            f"{held / max(s.gpu_count, 1) * 100:.1f}% of capacity held idle by one waiting job",
        ],
        value=f"{held} devices held idle, {held / max(s.gpu_count, 1) * 100:.1f}% of the fleet",
        interpretation=(
            "All-or-nothing allocation converts an indefinite hold into a clean rejection the operator "
            "can act on. The cost of gang scheduling is fragmentation; the cost of not having it is "
            "capacity held by jobs that cannot start."),
    )


def q_recovery_share(s: Setting) -> Quant:
    return Quant(
        label="the fleet time consumed by recovery at a given failure rate",
        steps=[
            f"Recovery holds {s.tp} devices while loading {gib(s.weight_bytes)} and warming up",
            "Fleet time in recovery = failure rate per replica-hour times recovery duration",
            f"At one failure per replica per 100 hours and a 10-minute recovery: "
            f"{10 / 60 / 100 * 100:.2f}% of replica time",
            f"Across {s.gpu_count} devices at TP{s.tp}, that is "
            f"{s.gpu_count // max(s.tp, 1)} replicas each paying it",
        ],
        value=f"recovery holds {s.tp} devices per event and scales with the failure rate",
        interpretation=(
            "The product of failure rate and recovery duration is the quantity that constrains restart "
            "policy. Either term alone is uninformative, which is why aggressive restart policies "
            "survive review."),
    )


def q_fragmentation(s: Setting) -> Quant:
    jobs = max(s.gpu_count // max(s.tp, 1), 1)
    leftover = s.gpu_count % max(s.tp, 1)
    return Quant(
        label="the devices stranded by an indivisible job size",
        steps=[
            f"Fleet {s.gpu_count} devices, job size TP{s.tp}",
            f"Whole jobs that fit: floor({s.gpu_count} / {s.tp}) = {jobs}",
            f"Stranded remainder: {s.gpu_count} mod {s.tp} = {leftover}",
            f"Those {leftover} devices cannot host a job of this shape at all",
        ],
        value=f"{jobs} whole jobs with {leftover} device(s) stranded",
        interpretation=(
            "Stranded capacity is a property of the job shape against the fleet shape, not of the "
            "scheduler's quality. Mixing job shapes reduces it; a single mandated shape guarantees it."),
    )


def q_queue_priority(s: Setting) -> Quant:
    return Quant(
        label="the wait a low-priority job faces under strict priority at this utilisation",
        steps=[
            "Under strict priority, a low-priority job runs only when no high-priority job is waiting",
            f"At high utilisation on {s.gpu_count} devices, the idle gaps are short and unpredictable",
            "Expected wait grows without bound as high-priority arrival rate approaches capacity",
            "No amount of queueing time converts into a guarantee, because priority has no ageing",
        ],
        value="unbounded expected wait for the lowest priority class as utilisation approaches capacity",
        interpretation=(
            "Strict priority is a starvation policy by construction. If low-priority work must "
            "eventually run, the policy needs ageing or a reserved share, and that has to be designed "
            "rather than assumed."),
    )


def q_autoscale_lag(s: Setting) -> Quant:
    return Quant(
        label="the lag between a scaling decision and served capacity",
        steps=[
            "Scheduling the replica, pulling the image and claiming devices happen first",
            f"Then {gib(s.weight_bytes)} of weights load across {s.tp} devices",
            "Then graph capture and warm-up run before the first token is served",
            f"Only after all of that does the replica contribute to serving {s.concurrency} requests",
        ],
        value=f"{gib(s.weight_bytes)} of weight transfer plus capture and warm-up before any capacity arrives",
        interpretation=(
            "Reactive scaling is effective only when this lag is short relative to how long load stays "
            "elevated. When it is not, capacity arrives after the incident and is removed before the "
            "next one."),
    )


def q_failure_domain(s: Setting) -> Quant:
    replicas = max(s.gpu_count // max(s.tp, 1), 1)
    return Quant(
        label="the size of the failure domain implied by this parallel layout",
        steps=[
            f"Within a tensor-parallel group of {s.tp}, losing one rank disables the whole group",
            f"So the failure domain is {s.tp} devices, not one",
            f"Across {replicas} replicas, losing one replica costs "
            f"{1 / replicas * 100:.1f}% of serving capacity",
            f"A host outage hosting several ranks can disable more than one group at once",
        ],
        value=f"a {s.tp}-device failure domain, with {replicas} independent replica(s) on the fleet",
        interpretation=(
            "Recovery policy must act on the group rather than the device. Restarting a single rank "
            "into a group that has already failed wastes the restart and delays the real recovery."),
    )


def q_bin_packing(s: Setting) -> Quant:
    return Quant(
        label="why free device count is not the same as schedulable capacity",
        steps=[
            f"A job needs {s.tp} devices on a topology that permits its collectives",
            "Free devices scattered across hosts may satisfy the count and not the topology",
            f"On {s.interconnect}, a group split across hosts pays a different collective cost "
            f"than one contained within a host",
            "So a scheduler reporting sufficient free capacity can still fail to place the job",
        ],
        value=f"{s.tp} devices are needed together, not merely {s.tp} devices somewhere",
        interpretation=(
            "Capacity dashboards that count free devices overstate schedulable capacity. The gap is "
            "the fragmentation the topology constraint imposes, and it grows as the fleet fills."),
    )


def q_preemption_cost(s: Setting) -> Quant:
    return Quant(
        label="what preempting a serving replica costs compared with preempting a batch job",
        steps=[
            "A batch job checkpoints and resumes from its last checkpoint",
            f"A serving replica must reload {gib(s.weight_bytes)} and rewarm before serving again",
            "In-flight requests are lost or must be retried elsewhere, adding load during recovery",
            f"Prefix cache state built up over the replica's lifetime is discarded entirely",
        ],
        value=f"{gib(s.weight_bytes)} reload plus lost cache and in-flight requests per preemption",
        interpretation=(
            "Preemption policies designed for batch workloads misprice serving replicas. The cost is "
            "not proportional to the work interrupted; it is a fixed, large restart."),
    )


def q_utilisation_target(s: Setting) -> Quant:
    return Quant(
        label="why a high average utilisation target guarantees queueing",
        steps=[
            "Arrivals are variable, so instantaneous demand exceeds the average regularly",
            f"At {s.concurrency} concurrent requests, short bursts routinely exceed the mean by a wide margin",
            "Running the average close to capacity leaves no room to absorb those bursts",
            f"Queue wait then enters the {s.slo_ms} ms budget before any per-request work begins",
        ],
        value=f"headroom must cover the burst above the mean, not just the mean, within {s.slo_ms} ms",
        interpretation=(
            "A utilisation target and a latency objective are the same decision expressed twice. "
            "Setting them independently produces a target that the objective cannot meet."),
    )


def q_multitenancy(s: Setting) -> Quant:
    return Quant(
        label="the interference surface between two jobs sharing a device",
        steps=[
            f"Device memory {s.mem_gb} GB is partitioned, so each tenant's cache budget shrinks",
            "Memory bandwidth is shared and not partitioned, so each tenant's decode slows",
            f"On {s.interconnect}, collective traffic from one tenant delays the other's",
            "Only the memory partition is visible in per-tenant accounting; the rest is not",
        ],
        value=f"one partitioned resource ({s.mem_gb} GB) and at least two shared ones",
        interpretation=(
            "Per-tenant memory limits create an appearance of isolation that bandwidth and fabric "
            "sharing do not honour. Latency objectives cannot be guaranteed per tenant on shared "
            "devices without measuring the interference."),
    )


register(
    Mechanism(
        key="gang_scheduling", topic="scheduling",
        title="incremental allocation lets a distributed job hold devices it cannot yet use",
        concepts=("gang_scheduling", "allocation", "utilisation"),
        symptom="Fleet utilisation reads high while served throughput is low and several jobs are stuck in a starting state.",
        chain="A distributed serving job cannot begin until every rank is allocated, so a scheduler that grants devices incrementally leaves partially allocated jobs holding capacity that produces nothing while blocking jobs that could run.",
        metric="Devices held by jobs that are allocated but not yet serving, sampled over time.",
        signature="Held-but-idle device time is a material share of fleet time, and it concentrates in jobs waiting on their final ranks.",
        confounders=(
            "Slow startup on fully allocated jobs, which also shows devices busy without serving.",
            "Image pull or weight load time, which is unavoidable rather than an allocation defect.",
            "Jobs deliberately held warm as standby, which are idle by design.",
        ),
        fixes=(
            "Bound the time a job may hold a partial allocation and release it when the bound expires.",
            "Switch to all-or-nothing allocation so a job holds devices only when it can start.",
            "Reserve a placement plan before granting any device, so the shape is known to be satisfiable.",
        ),
        rollback="Return to incremental allocation if allocation failure rates rise without a throughput gain, which indicates fragmentation now dominates the waste it removed.",
        options=("bounding how long a partial allocation may be held", "switching to all-or-nothing allocation"),
        tradeoff="whether the waste from partial holds exceeds the fragmentation that all-or-nothing introduces",
        flip="the job-size mix becomes uniform enough that fragmentation dominates, at which point strict gang allocation costs more than it saves",
        falsifier="held-but-idle device time is negligible while utilisation and throughput still diverge",
        wrong_claim="Utilisation is above 90%, so the fleet is well used and more capacity is required.",
        wrong_why="Allocation-based utilisation counts devices held by jobs that are producing nothing, so it reports the symptom as if it were evidence of demand.",
        threshold="Treat held-but-idle device time above a small share of fleet time as an allocation defect rather than as startup cost.",
        cost="Capacity purchased on an allocation-based utilisation signal buys devices to relieve a scheduling behaviour rather than real demand.",
        scaling="The waste per stuck job grows with the job's device count, so larger parallel degrees make incremental allocation progressively worse.",
        quant=q_gang,
    ),
    Mechanism(
        key="serving_recovery_cost", topic="scheduling",
        title="restart policies designed for stateless services misprice model-serving replicas",
        concepts=("recovery", "restart_policy", "availability"),
        symptom="A moderate failure rate keeps a visible fraction of the fleet permanently in a starting state.",
        chain="Recovering a serving replica requires loading the full weight set, capturing graphs and rebuilding caches, so restart duration is large and fixed, and multiplying it by even a modest failure rate consumes a real share of fleet capacity.",
        metric="Fleet device time spent in recovery, computed as failure rate times measured recovery duration.",
        signature="Time in recovery scales with the product of failure rate and recovery duration rather than with load, and does not fall when traffic falls.",
        confounders=(
            "Correlated failures producing many simultaneous recoveries, which makes an average understate the worst case.",
            "Warm page cache in testing, which makes measured recovery far faster than genuine cold recovery.",
            "Deliberate rolling restarts during deployment, which are planned rather than failure-driven.",
        ),
        fixes=(
            "Measure recovery end to end on a cold node with production weight sizes before setting any restart policy.",
            "Drain and repair rather than restart immediately, where recovery is long relative to the failure interval.",
            "Pre-stage weights and images so transfer is off the recovery critical path.",
        ),
        rollback="Return to the previous restart policy if fleet time in recovery rises, even if the count of failed requests falls, since recovery time is the constrained resource.",
        options=("draining and repairing instead of restarting immediately", "pre-staging weights to shorten recovery"),
        tradeoff="whether recovery duration is short relative to the interval between failures",
        flip="pre-staging stops helping because capture and warm-up rather than transfer dominate recovery, at which point only holding warm replicas reduces the exposure",
        falsifier="the product of failure rate and recovery duration is a negligible share of fleet time",
        wrong_claim="Automatic restart on failure is the standard policy and costs nothing when the service is healthy.",
        wrong_why="A serving replica in recovery holds its full device allocation and serves nothing, so the policy's cost is the failure rate multiplied by a large fixed duration rather than zero.",
        threshold="Treat fleet time in recovery above a small stated share as a constraint on restart aggressiveness.",
        cost="Devices held by recovering replicas are billed at full price for the entire recovery window.",
        scaling="Recovery duration grows with model size while failure rate grows with fleet size, so the product worsens on both axes simultaneously.",
        quant=q_recovery_share,
    ),
    Mechanism(
        key="shape_fragmentation", topic="scheduling",
        title="an indivisible job shape strands devices that no job of that shape can use",
        concepts=("fragmentation", "placement", "capacity_planning"),
        symptom="The fleet reports free devices while new jobs of the standard shape cannot be placed.",
        chain="Jobs require a fixed number of devices with a compatible topology, so a fleet whose free capacity does not divide into that shape holds a remainder that is free, countable and unusable.",
        metric="Free devices grouped by whether they form a placeable set of the required shape.",
        signature="Free device count exceeds the job requirement while no placeable set exists, and the gap equals the remainder of the fleet size against the job shape.",
        confounders=(
            "Devices reserved or cordoned for maintenance, which are unavailable for a different reason.",
            "Topology constraints excluding otherwise free devices, which is a separate limit on the same capacity.",
            "Accounting lag after a job exits, which shows devices as busy briefly after release.",
        ),
        fixes=(
            "Report schedulable sets rather than free device count on the capacity dashboard.",
            "Permit more than one job shape so the remainder can host something.",
            "Choose parallel degrees that divide the host and fleet sizes evenly.",
        ),
        rollback="Return to a single job shape if mixing shapes degrades collective performance for the primary workload, since placement flexibility is not worth a slower main path.",
        options=("reporting schedulable sets instead of free devices", "permitting more than one job shape"),
        tradeoff="whether a second job shape can be served well enough to justify the placement complexity",
        flip="the secondary shape performs badly enough on the leftover topology that running it is worse than leaving the devices idle",
        falsifier="a placeable set of the required shape exists whenever the free count is sufficient",
        wrong_claim="There are free GPUs in the cluster, so the scheduler should be able to place this job.",
        wrong_why="The job needs a specific number of devices together on a compatible topology, and a count of free devices scattered across the fleet does not establish that such a set exists.",
        threshold="Track schedulable sets of the deployed shape rather than free device count as the capacity signal.",
        cost="Stranded devices are purchased, powered and unusable, and they do not appear as waste in a free-capacity metric.",
        scaling="The stranded remainder is bounded by the job shape, so it is a fixed loss per host or per fleet that matters most on small fleets.",
        quant=q_fragmentation,
    ),
    Mechanism(
        key="strict_priority_starvation", topic="scheduling",
        title="strict priority without ageing is a starvation policy by construction",
        concepts=("priority", "starvation", "fairness"),
        symptom="Low-priority work that used to complete overnight now never completes, while high-priority work is served normally.",
        chain="Under strict priority a lower class runs only in the gaps left by higher classes, so as high-priority demand approaches capacity the gaps vanish and the expected wait for the lower class grows without bound.",
        metric="Wait time distribution per priority class, reported with its tail rather than its mean.",
        signature="Low-class wait grows without settling while high-class wait stays flat, and the divergence tracks high-class utilisation rather than total load.",
        confounders=(
            "A genuine increase in low-priority job size, which lengthens runtime rather than wait.",
            "Preemption of running low-priority jobs, which produces the same symptom through a different mechanism.",
            "Quota exhaustion for the low-priority tenant, which blocks admission independently of priority.",
        ),
        fixes=(
            "Add ageing so a job's effective priority rises with its wait, bounding starvation.",
            "Reserve a minimum share of capacity for the lower class rather than relying on gaps.",
            "Separate the classes onto distinct pools if the objectives are genuinely incompatible.",
        ),
        rollback="Remove the reservation if high-priority objectives are missed, and record which objective is being traded rather than tuning both at once.",
        options=("adding ageing so waiting raises effective priority", "reserving a minimum share for the lower class"),
        tradeoff="whether high-priority objectives can absorb the capacity a reservation removes",
        flip="high-priority demand grows to consume the whole fleet, at which point any reservation directly breaches its objective and the classes must be separated",
        falsifier="low-class wait settles at a stable value as high-class utilisation rises",
        wrong_claim="Low-priority jobs are simply waiting their turn; they will run when the cluster is quieter.",
        wrong_why="Under strict priority with no ageing there is no mechanism that makes a waiting job more likely to run, so an indefinite wait is the designed behaviour rather than a temporary backlog.",
        threshold="Bound low-class wait explicitly through ageing or reservation rather than leaving it to residual capacity.",
        cost="Work that never runs consumes queue state and planning attention while returning nothing.",
        scaling="Starvation onset moves earlier as high-priority utilisation rises, so a policy that worked at moderate load fails abruptly rather than gradually.",
        quant=q_queue_priority,
    ),
    Mechanism(
        key="autoscaling_actuation_lag", topic="scheduling",
        title="autoscaling a model server is too slow to track load unless the lag is measured first",
        concepts=("autoscaling", "readiness", "capacity_planning"),
        symptom="Autoscaling is configured and healthy, and capacity consistently arrives after the load spike has passed.",
        chain="A scaling decision must be followed by scheduling, weight transfer, graph capture and warm-up before any request is served, and if that lag exceeds the duration of the load excursion the added capacity arrives too late and is removed before the next one.",
        metric="Time from scaling decision to first served token, decomposed into its stages, compared against the load's autocorrelation time.",
        signature="Replica count and offered load are visibly out of phase by approximately the measured readiness time.",
        confounders=(
            "Quota or admission delays in the cluster scheduler, which extend actuation before any loading starts.",
            "Pre-pulled images and warm page cache in testing, which understate production readiness.",
            "Load generator retries inflating the apparent demand signal.",
        ),
        fixes=(
            "Measure readiness end to end on a genuinely cold node before tuning any scaling parameter.",
            "Pre-provision for the measured peak where readiness exceeds the excursion duration.",
            "Evaluate any predictive scaler in shadow mode before allowing it to act.",
        ),
        rollback="Disable reactive scaling and hold a fixed replica count if replica churn rises without a throughput improvement, since churn consumes the capacity it is meant to add.",
        options=("measuring readiness before tuning scaling parameters", "pre-provisioning for the measured peak"),
        tradeoff="whether readiness time is short relative to how long load stays elevated",
        flip="readiness falls far enough through staging and smaller artifacts that reactive scaling can track load, at which point static provisioning is the more expensive choice",
        falsifier="replica count and offered load are in phase, which means actuation is fast enough",
        wrong_claim="The autoscaler is configured with the right thresholds, so capacity will follow demand.",
        wrong_why="Threshold correctness governs when the decision is made, not how long it takes to take effect, and the delay rather than the trigger is what decides whether the capacity is useful.",
        threshold="Treat reactive scaling as ineffective when measured readiness exceeds the duration over which load stays elevated.",
        cost="Replicas that become ready after the spike and are torn down before the next one are paid for and never used.",
        scaling="Readiness grows with model size while excursions do not, so scaling becomes less viable as models grow.",
        quant=q_autoscale_lag,
    ),
    Mechanism(
        key="failure_domain_mismatch", topic="scheduling",
        title="the recovery unit must match the parallel layout, not the device",
        concepts=("failure_domain", "recovery", "parallelism"),
        symptom="A single device failure triggers one restart, the group still does not serve, and the restart repeats.",
        chain="Within a tensor-parallel group every rank is required for the group to serve, so the failure domain is the group rather than the device, and restarting one rank into an already-failed group neither restores service nor releases the remaining ranks.",
        metric="Failure correlation within parallel groups compared against correlation across independent replicas.",
        signature="Failures correlate almost perfectly within a group and weakly across replicas, matching the domain the layout implies.",
        confounders=(
            "A shared host or power domain creating correlation unrelated to the parallel layout.",
            "Placement drifting from the intended layout after rescheduling, so the logical map is stale.",
            "A cascading failure caused by the restart itself rather than by the original fault.",
        ),
        fixes=(
            "Set the recovery unit to the parallel group rather than the device.",
            "Read the physical placement rather than the intended layout when computing the domain.",
            "Re-derive the domain whenever the parallel configuration or placement policy changes.",
        ),
        rollback="Return to per-device restart if group restart is found to recycle healthy ranks without shortening time to restored service.",
        options=("setting the recovery unit to the parallel group", "reading physical placement rather than intended layout"),
        tradeoff="whether the observed failure correlation matches the layout-derived domain",
        flip="failures begin correlating across replicas as well, which indicates a shared cause that the layout does not describe and that group restart will not address",
        falsifier="failures correlate no more strongly within groups than across replicas",
        wrong_claim="One GPU failed, so restarting that one process is the minimal correct response.",
        wrong_why="The remaining ranks in the group are already unable to serve, so a single-rank restart addresses part of a unit that fails and recovers as a whole.",
        threshold="Set the recovery unit from the measured failure correlation rather than from device boundaries.",
        cost="Repeated single-rank restarts consume recovery cycles while the group remains out of service.",
        scaling="The domain grows with parallel degree, so larger degrees lose more capacity per fault and recover more slowly.",
        quant=q_failure_domain,
    ),
    Mechanism(
        key="free_devices_vs_placeable", topic="scheduling",
        title="free device count overstates capacity because placement has topology constraints",
        concepts=("placement", "topology", "capacity_planning"),
        symptom="Capacity dashboards show sufficient free devices while placement requests are rejected.",
        chain="A distributed job requires its devices to sit on a topology that supports its collectives, so free devices scattered across hosts satisfy the count without satisfying the constraint, and the scheduler correctly refuses.",
        metric="Free devices that form a topologically valid set of the required shape, counted separately from total free devices.",
        signature="The two counts diverge as the fleet fills, and the divergence tracks how scattered the free devices are rather than how many there are.",
        confounders=(
            "Cordoned or draining nodes, which reduce free capacity for maintenance reasons.",
            "Quota limits blocking admission independently of placement feasibility.",
            "Anti-affinity rules imposed for availability, which restrict placement deliberately.",
        ),
        fixes=(
            "Publish placeable-set count alongside free device count on the capacity signal.",
            "Defragment by draining and repacking during low-demand windows.",
            "Relax the topology constraint for jobs that can tolerate a cross-host group, and measure what it costs them.",
        ),
        rollback="Restore the topology constraint if the relaxed placement degrades collective performance beyond the capacity it recovered.",
        options=("publishing placeable-set count as the capacity signal", "defragmenting during low-demand windows"),
        tradeoff="whether the workload can tolerate a group spread across hosts",
        flip="the workload becomes latency-sensitive enough that a cross-host group misses its objective, at which point defragmentation is the only route to capacity",
        falsifier="placeable-set count matches free device count as the fleet fills",
        wrong_claim="We have capacity, the scheduler just is not placing the job, so the scheduler is at fault.",
        wrong_why="The scheduler is enforcing a topology requirement the job declared, and a count of free devices does not establish that a valid set exists.",
        threshold="Use placeable-set count rather than free device count whenever a capacity decision is being made.",
        cost="Capacity purchased on a free-device signal arrives into the same fragmented state and does not become placeable.",
        scaling="Divergence between the two counts grows as fleet utilisation rises, so the signal is least accurate exactly when it matters most.",
        quant=q_bin_packing,
    ),
    Mechanism(
        key="serving_preemption_mispricing", topic="scheduling",
        title="preempting a serving replica costs a full restart, not the work interrupted",
        concepts=("preemption", "recovery", "multitenancy"),
        symptom="Enabling preemption for batch work improved batch throughput and degraded serving availability far more than expected.",
        chain="Batch jobs resume from a checkpoint proportional to the work lost, while a preempted serving replica must reload the entire weight set, rebuild caches and shed in-flight requests, so its preemption cost is fixed and large regardless of how long it had been running.",
        metric="Time from preemption to restored serving capacity, compared against the batch work the preemption enabled.",
        signature="Recovery time after preemption is independent of how long the replica had been serving, which distinguishes it from checkpoint-based resumption.",
        confounders=(
            "In-flight request retries adding load elsewhere, which degrades other replicas at the same time.",
            "Prefix cache loss lowering measured performance after recovery for reasons unrelated to the restart itself.",
            "Concurrent deployment activity producing restarts that are not preemptions.",
        ),
        fixes=(
            "Exclude serving replicas from preemption and reserve their capacity explicitly.",
            "Price preemption by measured recovery cost rather than by elapsed runtime.",
            "Drain the replica gracefully before reclaiming it, so in-flight requests complete elsewhere.",
        ),
        rollback="Restore the preemption exclusion if serving availability falls, and treat the batch throughput gain as forfeited rather than tunable.",
        options=("draining the replica before reclaiming it", "excluding serving replicas from preemption entirely"),
        tradeoff="whether graceful draining is fast enough to make reclamation useful to the batch workload",
        flip="drain time grows long enough that reclamation no longer helps the batch queue, at which point exclusion is the honest policy",
        falsifier="recovery time after preemption scales with how long the replica had been running",
        wrong_claim="Preemption reclaims idle capacity, so it is free when the serving replica is lightly loaded.",
        wrong_why="Reclamation cost is set by restart duration rather than by current load, so a lightly loaded replica is exactly as expensive to preempt as a busy one.",
        threshold="Require the measured restart cost to be included in any preemption decision for a serving replica.",
        cost="Batch throughput gained by preempting serving capacity is paid for in serving availability and in duplicated startup work.",
        scaling="Restart cost grows with model size, so preemption becomes progressively worse value as served models grow.",
        quant=q_preemption_cost,
    ),
    Mechanism(
        key="utilisation_target_conflict", topic="scheduling",
        title="a utilisation target and a latency objective are one decision stated twice",
        concepts=("utilisation", "queueing", "slo"),
        symptom="A fleet meeting its utilisation target consistently misses its latency objective, and neither team believes it owns the conflict.",
        chain="Arrivals vary, so instantaneous demand exceeds the mean regularly, and running the average close to capacity removes the headroom those bursts need, which converts directly into queue wait inside the latency budget.",
        metric="Queue wait as a share of end-to-end latency, plotted against average utilisation.",
        signature="Queue wait rises sharply above a utilisation knee while service time per request stays flat, locating the loss in waiting rather than in work.",
        confounders=(
            "A genuine per-request slowdown, which raises service time rather than wait time.",
            "Client retries inflating arrivals and thus apparent demand.",
            "Batch composition changing with load, which alters service time independently.",
        ),
        fixes=(
            "Derive the utilisation target from the latency objective and the observed arrival variability rather than setting it independently.",
            "Publish the headroom requirement explicitly so capacity planning and latency ownership share one number.",
            "Shed or defer the traffic class that does not need the objective, freeing headroom for the class that does.",
        ),
        rollback="Lower the utilisation target back if latency does not recover, since that indicates service time rather than queueing is the constraint.",
        options=("deriving the utilisation target from the latency objective", "shedding the traffic class that does not need the objective"),
        tradeoff="whether the loss is in waiting or in per-request service time",
        flip="arrival variability falls enough that the mean predicts the peak, at which point a higher utilisation target becomes compatible with the objective",
        falsifier="service time per request rises alongside wait time as utilisation grows",
        wrong_claim="We are only at 80% utilisation, so there is headroom and latency problems must be elsewhere.",
        wrong_why="Average utilisation says nothing about instantaneous demand, and at realistic arrival variability an 80% average already leaves insufficient room to absorb bursts within the budget.",
        threshold="Set the utilisation target so that burst demand above the mean still fits within the latency objective.",
        cost="Meeting a utilisation target that the latency objective cannot support produces breaches that are attributed to capacity rather than to the target.",
        scaling="Relative burst size falls as request volume grows, so large fleets can safely run at higher utilisation than small ones and a single fleet-wide target misfits both.",
        quant=q_utilisation_target,
    ),
    Mechanism(
        key="multitenant_interference", topic="scheduling",
        title="per-tenant memory limits create isolation that shared bandwidth does not honour",
        concepts=("multitenancy", "interference", "isolation"),
        symptom="Two tenants stay within their allocated memory and both report worse latency than when running alone.",
        chain="Device memory can be partitioned per tenant, but memory bandwidth, interconnect capacity and scheduler attention are shared, so a tenant that respects its memory limit can still consume the bandwidth another tenant's decode phase depends on.",
        metric="Per-tenant achieved memory bandwidth and step time, measured alone and while co-located.",
        signature="Memory stays within limits for both tenants while achieved bandwidth per tenant falls in proportion to the other's activity.",
        confounders=(
            "Memory partitioning itself reducing each tenant's cache and thus its batch efficiency.",
            "One tenant's traffic pattern changing at the same time for unrelated reasons.",
            "Host-level contention from processes outside either tenant.",
        ),
        fixes=(
            "Measure co-located performance rather than inferring isolation from the memory partition.",
            "Co-schedule tenants whose phases differ so bandwidth demand does not coincide.",
            "Give each tenant whole devices where a latency objective must actually be guaranteed.",
        ),
        rollback="Separate the tenants onto whole devices if measured interference exceeds the objective's margin, rather than adjusting the memory partition further.",
        options=("co-scheduling tenants whose phases differ", "giving each tenant whole devices"),
        tradeoff="whether the tenants' bandwidth demands can be kept from coinciding",
        flip="both tenants become bandwidth-heavy at the same times, at which point no co-schedule avoids the interference and separation is required",
        falsifier="each tenant's achieved bandwidth is unchanged by the other's activity",
        wrong_claim="Each tenant has its own memory partition, so they are isolated from one another.",
        wrong_why="Memory is the one partitioned resource; bandwidth and fabric are shared, and decode is bandwidth-bound, so the resource that governs performance is exactly the one not isolated.",
        threshold="Do not offer a per-tenant latency objective on shared devices unless the measured interference fits inside its margin.",
        cost="Selling an isolation guarantee that the hardware does not provide creates obligations that can only be met by later separating the tenants.",
        scaling="Interference grows with the number of co-located tenants, so density gains are offset by per-tenant degradation that is not visible in memory accounting.",
        quant=q_multitenancy,
    ),
)
