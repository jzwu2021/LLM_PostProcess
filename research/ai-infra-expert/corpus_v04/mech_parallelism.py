"""Parallelism and distributed-execution mechanisms (topic: parallelism)."""
from __future__ import annotations

from core import Mechanism, Quant, Setting, fmt_int, gib, register


def q_tp_allreduce(s: Setting) -> Quant:
    bytes_per_tok = 2 * s.hidden * s.dtype_bytes
    per_layer = bytes_per_tok * 2 * (s.tp - 1) / max(s.tp, 1)
    total = per_layer * s.layers
    return Quant(
        label="the all-reduce traffic a single decoded token forces across the tensor-parallel group",
        steps=[
            f"Each transformer layer reduces a hidden-size activation twice per token: "
            f"2 * {fmt_int(s.hidden)} * {s.dtype_bytes} = {fmt_int(bytes_per_tok)} B",
            f"Ring all-reduce moves 2(N-1)/N of that per rank at TP{s.tp}: "
            f"{fmt_int(per_layer)} B per layer",
            f"Across {s.layers} layers: {fmt_int(total)} B = {gib(total)} per token per rank",
            f"At {s.concurrency} concurrent requests each producing one token per step, the group "
            f"moves {gib(total * s.concurrency)} per step",
        ],
        value=f"{gib(total)} per token per rank, {gib(total * s.concurrency)} per step at full batch",
        interpretation=(
            f"This traffic is latency-critical and synchronous: the step cannot retire until it "
            f"completes. On {s.interconnect} it is the term that decides whether TP{s.tp} is faster "
            f"than a smaller degree with a larger batch."),
    )


def q_pp_bubble(s: Setting) -> Quant:
    stages = max(s.tp, 2)
    micro = max(s.concurrency // stages, 1)
    bubble = (stages - 1) / (micro + stages - 1)
    return Quant(
        label="the pipeline bubble fraction at this stage count and microbatch count",
        steps=[
            f"Assume {stages} pipeline stages and {micro} microbatches in flight",
            f"Bubble fraction = (stages - 1) / (microbatches + stages - 1)",
            f"= ({stages} - 1) / ({micro} + {stages} - 1) = {bubble:.3f}",
            f"So {bubble * 100:.1f}% of device time is spent waiting rather than computing",
        ],
        value=f"a bubble fraction of {bubble:.3f}, i.e. {bubble * 100:.1f}% idle stage time",
        interpretation=(
            "The bubble shrinks only by raising microbatch count, which raises memory and latency. "
            "It cannot be removed by faster interconnect, because it is a scheduling gap rather than "
            "a transfer cost."),
    )


def q_tp_shard(s: Setting) -> Quant:
    total = s.weight_bytes
    per = s.weight_bytes_per_gpu
    return Quant(
        label="the per-device weight footprint under this tensor-parallel degree",
        steps=[
            f"Total weights = {s.params_b}e9 params * {s.dtype_bytes} B = {gib(total)}",
            f"Sharded across TP{s.tp}: {gib(total)} / {s.tp} = {gib(per)} per device",
            f"Device capacity is {s.mem_gb} GiB, so weights occupy "
            f"{per / (s.mem_gb * (1024 ** 3)) * 100:.1f}% of it",
            f"Remaining for cache, activations and runtime reserve: "
            f"{gib(s.mem_gb * (1024 ** 3) - per)}",
        ],
        value=f"{gib(per)} of weights per device, leaving {gib(s.mem_gb * (1024 ** 3) - per)}",
        interpretation=(
            "Raising the parallel degree buys cache capacity by shrinking the weight share, but it "
            "also multiplies the synchronous collective traffic per token. The two move in opposite "
            "directions, which is why the degree cannot be chosen from memory alone."),
    )


def q_dp_replica(s: Setting) -> Quant:
    replicas = max(s.gpu_count // max(s.tp, 1), 1)
    return Quant(
        label="how many independent replicas this fleet supports at the current parallel degree",
        steps=[
            f"Fleet size {s.gpu_count} devices, tensor-parallel degree {s.tp}",
            f"Replicas = floor({s.gpu_count} / {s.tp}) = {replicas}",
            f"Each replica serves up to {s.concurrency} concurrent requests independently",
            f"Fleet-level concurrency = {replicas} * {s.concurrency} = {replicas * s.concurrency}",
        ],
        value=f"{replicas} replica(s), giving fleet concurrency of {replicas * s.concurrency}",
        interpretation=(
            "Replica count and parallel degree consume the same devices. Any argument for raising the "
            "degree is simultaneously an argument for fewer independent failure domains and less "
            "aggregate concurrency."),
    )


def q_collective_sync(s: Setting) -> Quant:
    return Quant(
        label="the cost of one straggler rank in a synchronous collective",
        steps=[
            f"A collective over TP{s.tp} completes only when the last rank arrives",
            f"If one rank is late by d, all {s.tp} ranks pay d on that step",
            f"Wasted device time per step = ({s.tp} - 1) * d",
            f"Over a decode sequence the delay repeats every step, so d multiplies by the output length",
        ],
        value=f"one late rank costs {s.tp - 1} times its own delay, once per step",
        interpretation=(
            "Straggler cost is multiplied by group size and by output length, which is why a small, "
            "persistent per-rank imbalance is far more damaging than an occasional large one."),
    )


def q_seq_parallel(s: Setting) -> Quant:
    act = s.ctx * s.hidden * s.dtype_bytes
    return Quant(
        label="the activation memory a full-length prefill holds before sequence parallelism",
        steps=[
            f"One layer's activation for a full prompt = {fmt_int(s.ctx)} * {fmt_int(s.hidden)} * "
            f"{s.dtype_bytes} = {gib(act)}",
            f"Peak retains several such buffers concurrently for the layer being computed",
            f"Sharding the sequence across TP{s.tp} reduces each rank's share to {gib(act / s.tp)}",
            f"The saving is per layer and applies only during prefill, not during decode",
        ],
        value=f"{gib(act)} per layer buffer, reduced to {gib(act / s.tp)} per rank when sharded",
        interpretation=(
            "This is a prefill-phase saving. It relieves peak memory during long-prompt admission and "
            "does nothing for the steady-state decode footprint, so it changes admission limits rather "
            "than sustained concurrency."),
    )


def q_expert_parallel(s: Setting) -> Quant:
    tokens = s.concurrency
    bytes_tok = s.hidden * s.dtype_bytes
    return Quant(
        label="the all-to-all volume a mixture-of-experts layer moves per step",
        steps=[
            f"Each of {tokens} tokens is routed to remote experts and back",
            f"Payload per token per direction = {fmt_int(s.hidden)} * {s.dtype_bytes} = "
            f"{fmt_int(bytes_tok)} B",
            f"Two directions: {fmt_int(2 * bytes_tok)} B per token per layer",
            f"At {tokens} tokens and {s.layers} layers: {gib(2 * bytes_tok * tokens * s.layers)} per step",
        ],
        value=f"{gib(2 * bytes_tok * tokens * s.layers)} of all-to-all traffic per step at full batch",
        interpretation=(
            f"All-to-all is bounded by the slowest pairwise path rather than by aggregate bandwidth, so "
            f"on {s.interconnect} the placement of experts matters more than the total link count."),
    )


def q_tp_latency_floor(s: Setting) -> Quant:
    return Quant(
        label="the per-step latency floor imposed by collective synchronisation",
        steps=[
            f"Every decode step contains at least two synchronising collectives per layer",
            f"Across {s.layers} layers that is at least {2 * s.layers} synchronisation points per token",
            f"Each carries a fixed launch and handshake cost independent of payload size",
            f"At a p99 objective of {s.slo_ms} ms, that fixed cost is charged on every one of the "
            f"output tokens the objective covers",
        ],
        value=f"at least {2 * s.layers} synchronisation points per generated token",
        interpretation=(
            "The floor is set by count of synchronisations, not by bytes, so it does not improve with "
            "a faster link. Reducing the parallel degree removes synchronisation points and is the "
            "only lever that moves it."),
    )


def q_uneven_shard(s: Setting) -> Quant:
    heads = s.kv_heads
    return Quant(
        label="whether the attention head count divides evenly across this parallel degree",
        steps=[
            f"Key-value heads = {heads}, tensor-parallel degree = {s.tp}",
            f"{heads} / {s.tp} = {heads / s.tp:.3f}, remainder {heads % s.tp}",
            f"An uneven split forces padding or an unbalanced assignment",
            f"Padding cost is paid on every layer of every step, not once at load",
        ],
        value=(f"{'divides evenly' if heads % s.tp == 0 else 'does NOT divide evenly'}: "
               f"{heads} heads over TP{s.tp}"),
        interpretation=(
            "Divisibility is a hard constraint on the parallel plan, not a tuning preference. Where it "
            "fails, the runtime either refuses to launch or pads, and padding consumes capacity on "
            "every step."),
    )


def q_pp_memory_skew(s: Setting) -> Quant:
    stages = max(s.tp, 2)
    per_stage = s.weight_bytes / stages
    return Quant(
        label="the memory imbalance across pipeline stages",
        steps=[
            f"Weights divide roughly evenly: {gib(s.weight_bytes)} / {stages} = {gib(per_stage)} "
            f"per stage",
            f"In-flight activations do not: the first stage holds state for every microbatch still "
            f"in the pipeline",
            f"With {stages} stages, stage 1 retains up to {stages} times the activation state of the "
            f"last stage",
            f"Peak memory is therefore set by stage 1, while capacity is provisioned per stage equally",
        ],
        value=f"{gib(per_stage)} of weights per stage, with stage 1 holding up to {stages}x the activations",
        interpretation=(
            "Provisioning identical devices per stage wastes capacity on the last stage and runs the "
            "first stage close to its limit. The imbalance is structural and is not visible in an "
            "average memory metric."),
    )


register(
    Mechanism(
        key="tp_allreduce_cost", topic="parallelism",
        title="tensor parallelism trades memory relief for synchronous all-reduce traffic on every token",
        concepts=("tensor_parallelism", "collectives", "interconnect"),
        symptom="Raising the tensor-parallel degree freed device memory but made per-token latency worse rather than better.",
        chain="Each layer reduces activations across the whole tensor-parallel group twice per token, and that traffic is synchronous, so the step cannot retire until the slowest link completes it.",
        metric="All-reduce time as a share of step time, measured per layer rather than end to end.",
        signature="Step time rises with parallel degree while per-device compute time falls, and the difference is accounted for by collective time.",
        confounders=(
            "A smaller per-device batch after resharding, which reduces arithmetic efficiency independently of communication.",
            "Kernel selection changing with the new shard shape, which alters compute time without any communication effect.",
            "Background traffic from a co-located job sharing the same links, which inflates collective time on its own schedule.",
        ),
        fixes=(
            "Reduce the tensor-parallel degree to the smallest that fits weights and cache, and re-measure per-token latency.",
            "Confine the tensor-parallel group within one high-bandwidth domain so no collective crosses the slow link.",
            "Move to a pipeline or replica split for the dimension that does not fit, accepting its own scheduling cost.",
        ),
        rollback="Restore the previous parallel degree if per-token latency does not improve within one measurement window, and revert any batch-size change made alongside it.",
        options=("reducing the tensor-parallel degree", "confining the group to one high-bandwidth domain"),
        tradeoff="whether the interconnect carries the per-token reduction faster than the memory relief is worth",
        flip="weights or cache no longer fit at the lower degree, at which point memory rather than latency becomes the binding constraint",
        falsifier="collective time is a small share of step time while step time still rises with parallel degree",
        wrong_claim="Tensor parallelism splits the work across more GPUs, so more GPUs will always make each token faster.",
        wrong_why="It splits arithmetic but adds a synchronous reduction per layer, and past the point where communication dominates, adding devices increases per-token latency.",
        threshold="Treat collective time above roughly a third of step time as evidence the parallel degree is too high for this interconnect.",
        cost="Devices added to a group that is already communication-bound are paid for at full price and return less throughput than the same devices used as an independent replica.",
        scaling="Traffic per rank approaches a constant as the degree grows while synchronisation points grow linearly, so the penalty accelerates with degree.",
        quant=q_tp_allreduce,
    ),
    Mechanism(
        key="pipeline_bubble", topic="parallelism",
        title="pipeline parallelism leaves a scheduling bubble that only microbatch count can shrink",
        concepts=("pipeline_parallelism", "utilisation", "microbatching"),
        symptom="Device utilisation is uniformly moderate across all stages and throughput is well below the sum of the stages' individual capability.",
        chain="Stages must fill and drain around each batch, so at any moment some stages have no work, and the idle fraction is fixed by the ratio of stage count to microbatches in flight.",
        metric="Idle time per stage per batch, expressed as a fraction of wall-clock stage time.",
        signature="Idle fraction matches the value predicted from stage and microbatch counts, and falls as microbatch count rises while stage count is held fixed.",
        confounders=(
            "Stage imbalance, where one stage is genuinely slower and starves the rest for a different reason.",
            "Communication delay between adjacent stages, which adds idle time that microbatching does not remove.",
            "Host-side input starvation, which idles the first stage and propagates downstream.",
        ),
        fixes=(
            "Raise the number of microbatches in flight until the predicted bubble is acceptable.",
            "Rebalance layers across stages so no single stage sets the step period.",
            "Reduce the stage count and take the memory cost elsewhere, since the bubble scales with stage count.",
        ),
        rollback="Reduce microbatch count again if memory pressure or first-token latency regresses, because both grow with the number of microbatches in flight.",
        options=("raising the number of in-flight microbatches", "reducing the pipeline stage count"),
        tradeoff="whether memory and first-token latency can absorb more in-flight microbatches",
        flip="added microbatches no longer fit in memory or push first-token latency past its objective, at which point the stage count itself has to fall",
        falsifier="idle fraction stays flat as microbatch count rises, which points to stage imbalance or starvation rather than to the bubble",
        wrong_claim="Utilisation is around 70% on every stage, which is close enough to full and leaves nothing to recover.",
        wrong_why="Uniform partial utilisation across stages is the signature of the bubble rather than of a well-packed pipeline, and the missing fraction is recoverable by microbatch count without any hardware change.",
        threshold="Size microbatches so the predicted bubble fraction stays below roughly a tenth of stage time.",
        cost="The bubble is paid as accelerator hours during which the stage holds its weights and computes nothing.",
        scaling="Bubble fraction grows with stage count and shrinks with microbatch count, so deeper pipelines need proportionally more in-flight work to stay efficient.",
        quant=q_pp_bubble,
    ),
    Mechanism(
        key="tp_degree_memory_tradeoff", topic="parallelism",
        title="the tensor-parallel degree is chosen by cache headroom and communication together, never by memory alone",
        concepts=("tensor_parallelism", "capacity_planning", "kv_cache"),
        symptom="A parallel degree chosen to make the weights fit leaves the service with adequate memory and disappointing throughput.",
        chain="Raising the degree shrinks each device's weight share and frees memory for cache, but it also adds a synchronous reduction per layer, so the degree that maximises cache is rarely the degree that maximises tokens per second.",
        metric="Tokens per second and free cache bytes recorded together at each candidate parallel degree.",
        signature="Free cache rises monotonically with degree while throughput peaks at an intermediate degree and then declines.",
        confounders=(
            "Batch size changing with the degree, which moves throughput for reasons unrelated to sharding.",
            "Kernel autotuning selecting different implementations per shard shape, which shifts compute time.",
            "Weight quantisation applied at the same time, which changes the memory term without changing communication.",
        ),
        fixes=(
            "Sweep the parallel degree and record throughput and free cache jointly rather than choosing on memory.",
            "Quantise weights so a lower degree fits, then re-run the sweep from the new memory baseline.",
            "Split the deployment into more replicas at the lowest workable degree, trading per-request latency for aggregate throughput.",
        ),
        rollback="Return to the previously deployed degree if throughput at the new degree is not better on the same trace, and re-run the comparison before adopting any batch-size change made with it.",
        options=("sweeping the degree and choosing on measured throughput", "quantising weights to permit a lower degree"),
        tradeoff="whether memory headroom or collective time is the binding term at the current degree",
        flip="quantisation costs measurable output quality, at which point the memory relief is no longer free and the higher degree returns",
        falsifier="throughput rises monotonically with parallel degree across the whole sweep",
        wrong_claim="The model needs TP8 because that is the smallest degree at which the weights fit, so the parallelism question is settled.",
        wrong_why="Fitting is a lower bound on the degree, not a choice of it, and the degree above the fitting point is decided by measured throughput rather than by whether the weights load.",
        threshold="Accept the lowest degree at which weights and the target cache budget both fit and measured throughput is within a stated margin of the sweep maximum.",
        cost="Choosing the degree on memory alone typically buys communication overhead that shows up as reduced tokens per accelerator hour.",
        scaling="The gap between the memory-optimal and throughput-optimal degree widens as model size grows relative to device memory.",
        quant=q_tp_shard,
    ),
    Mechanism(
        key="replica_vs_degree", topic="parallelism",
        title="parallel degree and replica count compete for the same devices",
        concepts=("data_parallelism", "capacity_planning", "availability"),
        symptom="Aggregate fleet throughput fell after a change that improved single-request latency.",
        chain="Devices spent on raising the parallel degree are devices removed from the replica count, so per-request latency improves while the number of independent serving units, and with it aggregate concurrency, falls.",
        metric="Fleet-level tokens per second and admitted concurrency, measured alongside single-request latency rather than instead of it.",
        signature="Single-request latency improves while fleet throughput and admitted concurrency both fall, in proportion to the lost replica count.",
        confounders=(
            "A routing change deployed at the same time, which redistributes load independently of replica count.",
            "Warm-up on the newly created replicas, which depresses throughput temporarily after any reshard.",
            "A shift in the request mix toward longer generations, which lowers throughput without any topology change.",
        ),
        fixes=(
            "Report single-request latency and fleet throughput together for every candidate topology.",
            "Choose the topology from the objective that is actually binding, and record which one that is.",
            "Split the fleet into pools with different degrees so latency-sensitive and throughput-sensitive traffic do not share one compromise.",
        ),
        rollback="Restore the previous topology if fleet throughput falls without a corresponding latency requirement being met, and re-measure after replicas have warmed.",
        options=("reporting both objectives before changing topology", "splitting the fleet into separate pools per objective"),
        tradeoff="whether the workload is bound by per-request latency or by aggregate concurrency",
        flip="traffic becomes uniform enough that two pools cannot be kept usefully busy, at which point splitting wastes more than it recovers",
        falsifier="fleet throughput is unchanged after replica count falls, which means the fleet was not concurrency-bound",
        wrong_claim="Latency improved after the reshard, so the new topology is strictly better.",
        wrong_why="The reshard consumed replicas, so the comparison is incomplete until aggregate throughput and admitted concurrency are reported alongside the latency gain.",
        threshold="Require both a latency number and a fleet-throughput number before any topology change is approved.",
        cost="A topology optimised for a latency objective the workload does not have pays for idle aggregate capacity.",
        scaling="The lost replica count grows with the parallel degree, so the aggregate penalty compounds as the degree rises.",
        quant=q_dp_replica,
    ),
    Mechanism(
        key="straggler_amplification", topic="parallelism",
        title="a synchronous collective multiplies one slow rank across the whole group",
        concepts=("collectives", "stragglers", "tail_latency"),
        symptom="Step time is set by a value well above the median rank's compute time, and the excess does not appear in any single rank's profile.",
        chain="A collective completes only when its last participant arrives, so a persistent small delay on one rank is paid by every other rank on every step, and the group's throughput tracks the slowest member rather than the average.",
        metric="Per-rank arrival time at the collective, timestamped and compared across ranks on the same step.",
        signature="One rank consistently arrives last by a stable margin, and step time exceeds median rank compute time by approximately that margin.",
        confounders=(
            "Network tail latency, which delays completion after arrival rather than delaying arrival itself.",
            "Clock skew between hosts, which can manufacture an apparent arrival difference that is not real.",
            "Rotating slowness from thermal or power capping, which moves between ranks over time rather than sitting on one.",
        ),
        fixes=(
            "Identify the consistently late rank and check its device clocks, power cap and co-tenancy before changing anything shared.",
            "Rebalance any per-rank asymmetry in the shard assignment so no rank carries more work than its peers.",
            "Replace or drain the host if the delay is hardware-bound, since no software tuning removes a capped device.",
        ),
        rollback="If draining a suspected host does not reduce step time, return it to service rather than continuing to remove capacity on an unconfirmed hypothesis.",
        options=("identifying and repairing the single late rank", "rebalancing the shard assignment across ranks"),
        tradeoff="whether the delay is bound to one host or is distributed across the assignment",
        flip="the lateness moves between ranks over time, at which point per-rank repair chases a symptom and the assignment or the environment is the real target",
        falsifier="arrival times are evenly distributed across ranks while step time still exceeds median compute time",
        wrong_claim="Every rank shows normal utilisation, so the slowdown must be in the network rather than in the ranks.",
        wrong_why="A rank waiting inside a collective reports as busy, so uniform utilisation is consistent with one rank being late and the rest blocking on it.",
        threshold="Investigate when the slowest rank's arrival exceeds the median rank's by more than the run-to-run variation band.",
        cost="Every rank in the group is billed for the waiting time, so one capped device wastes the whole group's accelerator hours.",
        scaling="The waste is proportional to group size and repeats every step, so it grows with both parallel degree and output length.",
        quant=q_collective_sync,
    ),
    Mechanism(
        key="sequence_parallel_prefill", topic="parallelism",
        title="sequence parallelism relieves prefill activation peaks and not decode footprint",
        concepts=("sequence_parallelism", "activation_memory", "prefill"),
        symptom="Long prompts trigger out-of-memory during admission while steady-state serving of the same model is comfortable.",
        chain="Activation buffers during prefill scale with prompt length while decode activations do not, so peak memory is set by the longest admitted prompt rather than by sustained load.",
        metric="Peak device memory during prefill against steady-state decode memory, sampled at the same instant as the admitted prompt length.",
        signature="Peak memory tracks the longest prompt in the step and returns to a much lower plateau once prefill completes.",
        confounders=(
            "Cache reservation for the admitted request, which also rises with prompt length but persists after prefill.",
            "Allocator caching, which retains freed blocks and makes the peak appear not to recede.",
            "Concurrent admission of several long prompts, which multiplies the peak without any change in the per-prompt cost.",
        ),
        fixes=(
            "Cap the prompt tokens admitted per step so peak activation memory has an explicit bound.",
            "Enable sequence parallelism so the activation buffer is sharded across the group during prefill.",
            "Route long prompts to a pool with more memory headroom rather than sizing the whole fleet for the tail.",
        ),
        rollback="Restore the previous admission cap if time to first token for long prompts regresses beyond its objective, since the cap trades that latency for peak safety.",
        options=("capping prompt tokens admitted per step", "enabling sequence parallelism for the prefill phase"),
        tradeoff="whether peak activation memory or first-token latency is the binding objective",
        flip="prompts become long enough that any workable per-step cap makes first-token latency unacceptable, at which point sharding the activation is the only remaining lever",
        falsifier="peak memory is flat with respect to admitted prompt length",
        wrong_claim="The service has plenty of free memory in steady state, so a long prompt cannot cause an out-of-memory failure.",
        wrong_why="Steady-state free memory describes the decode phase, and prefill peaks transiently well above it, so the relevant headroom is the peak rather than the plateau.",
        threshold="Bound admitted prompt tokens per step so predicted peak activation stays within the memory reserve at the deployed parallel degree.",
        cost="Sizing the whole fleet for the longest prompt buys memory that is idle for the great majority of requests.",
        scaling="Peak activation grows linearly with prompt length and with the number of long prompts admitted together, so the tail of the length distribution sets the requirement.",
        quant=q_seq_parallel,
    ),
    Mechanism(
        key="expert_parallel_alltoall", topic="parallelism",
        title="expert parallelism converts routing decisions into all-to-all traffic",
        concepts=("mixture_of_experts", "all_to_all", "placement"),
        symptom="A sparse model with fewer active parameters serves more slowly than a dense model of the same active size.",
        chain="Routing sends each token to experts that live on other devices and back again, so every layer adds a dispatch and combine exchange whose cost is set by the slowest pairwise path rather than by aggregate bandwidth.",
        metric="All-to-all time per layer, separated into wait time and transfer time.",
        signature="Exchange time is dominated by wait rather than transfer, and it tracks the most loaded destination rather than the total volume.",
        confounders=(
            "Expert load imbalance, which lengthens the exchange through queueing at one destination rather than through link capacity.",
            "Capacity-factor padding, which inflates transferred volume with tokens that will be discarded.",
            "Co-located traffic from other layers overlapping the exchange, which makes attribution by wall clock unreliable.",
        ),
        fixes=(
            "Place frequently co-activated experts within one high-bandwidth domain so the common exchange stays local.",
            "Reduce the expert-parallel degree so fewer exchanges cross the slow path.",
            "Change the routing or capacity policy so destination load is even, since the exchange tracks the worst destination.",
        ),
        rollback="Revert a placement change if all-to-all time does not fall within one measurement window, because placement changes interact with the scheduler and are easy to misattribute.",
        options=("placing co-activated experts within one high-bandwidth domain", "reducing the expert-parallel degree"),
        tradeoff="whether the exchange is bound by the slowest link or by destination-side imbalance",
        flip="expert co-activation becomes uniform so no placement is better than any other, at which point only the degree matters",
        falsifier="exchange time is dominated by transfer rather than wait and scales with total volume",
        wrong_claim="The sparse model activates a fraction of the parameters, so it should be proportionally faster than the dense equivalent.",
        wrong_why="Active parameter count governs arithmetic, while the exchange cost it introduces is governed by placement and imbalance, and that cost has no counterpart in the dense model.",
        threshold="Investigate placement when all-to-all wait time exceeds transfer time on the same exchange.",
        cost="Devices spent on expert parallelism deliver less throughput than the same devices in a dense replica whenever the exchange dominates.",
        scaling="Exchange cost grows with expert-parallel degree and with batch size together, so it worsens exactly where sparse models are supposed to win.",
        quant=q_expert_parallel,
    ),
    Mechanism(
        key="collective_latency_floor", topic="parallelism",
        title="synchronisation count, not payload size, sets the per-token latency floor",
        concepts=("collectives", "latency", "tensor_parallelism"),
        symptom="Upgrading to a faster interconnect improved bandwidth benchmarks but left per-token latency essentially unchanged.",
        chain="Each layer contributes a fixed number of synchronisation points per token, and each carries a launch and handshake cost independent of payload, so a floor exists that more bandwidth cannot lower.",
        metric="Fixed per-collective overhead measured at the smallest payload, multiplied by the synchronisation count per token.",
        signature="Collective time per token is nearly independent of payload size across the small-payload range and scales with the number of collectives instead.",
        confounders=(
            "Kernel launch overhead on the compute side, which also produces payload-independent cost.",
            "Graph capture being disabled, which adds per-launch cost that resembles synchronisation overhead.",
            "CPU-side scheduling delay between layers, which appears inside the same interval.",
        ),
        fixes=(
            "Enable graph capture so launch and handshake costs are amortised across the step.",
            "Fuse or reduce the number of collectives per layer where the framework supports it.",
            "Reduce the parallel degree, which removes synchronisation points entirely rather than making them faster.",
        ),
        rollback="Disable graph capture again if memory grows beyond the reserve or if shape variability causes recapture on every step, since both cancel the benefit.",
        options=("enabling graph capture to amortise launch cost", "reducing the parallel degree to remove synchronisations"),
        tradeoff="whether the floor comes from per-launch overhead or from the count of synchronisation points",
        flip="shape variability forces frequent recapture, at which point capture stops amortising and the degree becomes the only lever",
        falsifier="collective time scales with payload size across the operating range, which means bandwidth rather than a fixed floor is the limit",
        wrong_claim="The new fabric doubles bandwidth, so per-token latency should improve substantially.",
        wrong_why="Decode collectives carry small payloads, so their cost is dominated by fixed per-operation overhead rather than by transfer time, and bandwidth barely enters the result.",
        threshold="Treat collective time that is flat across a payload sweep as evidence that count rather than bandwidth is the constraint.",
        cost="Purchasing interconnect bandwidth to fix a synchronisation-count problem buys capability the decode phase cannot use.",
        scaling="Synchronisation count grows with layer count and with parallel degree, so the floor rises with model depth independently of link speed.",
        quant=q_tp_latency_floor,
    ),
    Mechanism(
        key="shard_divisibility", topic="parallelism",
        title="head counts that do not divide the parallel degree force padding or refusal",
        concepts=("tensor_parallelism", "configuration", "validation"),
        symptom="A parallel configuration that looks reasonable either refuses to launch or launches and runs measurably slower than a neighbouring configuration.",
        chain="Attention heads are partitioned across the group, so a degree that does not divide the head count either cannot form a valid assignment or must pad it, and padding is recomputed on every layer of every step.",
        metric="Head count divided by parallel degree, with the remainder reported explicitly alongside per-step time.",
        signature="A configuration with a non-zero remainder shows per-step time above its neighbours by roughly the padded fraction.",
        confounders=(
            "Kernel selection differing between shard shapes, which changes speed without any padding involved.",
            "Memory pressure at the smaller degree, which slows the comparison configuration for an unrelated reason.",
            "Grouped-query attention making key-value heads differ from attention heads, so the wrong count is checked.",
        ),
        fixes=(
            "Validate divisibility of both attention and key-value head counts before launch and refuse rather than pad.",
            "Choose a parallel degree from the divisors of the head count rather than from device availability alone.",
            "Change the served model configuration if no workable divisor exists at the required memory footprint.",
        ),
        rollback="Return to the previous validated degree if a newly permitted configuration shows per-step time above its neighbours, since that is the padding signature.",
        options=("validating divisibility and refusing invalid degrees", "choosing the degree from the divisors of the head count"),
        tradeoff="whether a valid divisor exists that also satisfies the memory requirement",
        flip="no divisor satisfies the memory requirement, at which point the model configuration rather than the parallel plan has to change",
        falsifier="a configuration with a non-zero remainder performs the same as its evenly dividing neighbours",
        wrong_claim="The framework accepted the configuration and it is serving traffic, so the parallel degree is valid.",
        wrong_why="Acceptance means the runtime found a workable assignment, possibly by padding, and padding is a per-step cost rather than a launch-time error.",
        threshold="Require an exact zero remainder for both attention and key-value head counts before a parallel degree is deployable.",
        cost="Padded shards consume arithmetic on positions that carry no information, on every layer of every step.",
        scaling="The padded fraction grows as the degree approaches the head count, so the penalty is worst exactly at the high degrees chosen for large models.",
        quant=q_uneven_shard,
    ),
    Mechanism(
        key="pipeline_memory_skew", topic="parallelism",
        title="pipeline stages hold unequal activation state even when weights are balanced",
        concepts=("pipeline_parallelism", "activation_memory", "capacity_planning"),
        symptom="One pipeline stage runs close to its memory limit while the others have comfortable headroom, despite an even weight split.",
        chain="Earlier stages retain activation state for every microbatch still in flight downstream, so their peak memory scales with the number of stages while later stages hold almost none.",
        metric="Peak memory per stage, reported per stage rather than averaged across the pipeline.",
        signature="Peak memory decreases monotonically from the first stage to the last, with the ratio tracking the stage count.",
        confounders=(
            "Uneven layer assignment, which skews weights rather than activations and responds to rebalancing instead.",
            "Allocator caching differing per stage, which distorts reported peaks without changing live usage.",
            "The first stage also holding input batching buffers, which adds a fixed rather than a stage-count-dependent term.",
        ),
        fixes=(
            "Assign fewer layers to the early stages so weight footprint offsets their larger activation retention.",
            "Reduce the number of in-flight microbatches, accepting a larger pipeline bubble in exchange.",
            "Provision the early stages on higher-memory devices rather than sizing every stage identically.",
        ),
        rollback="Restore the even layer assignment if throughput falls after rebalancing, since moving layers changes stage compute time as well as memory.",
        options=("shifting layers away from the early stages", "reducing the number of in-flight microbatches"),
        tradeoff="whether the memory pressure can be relieved without enlarging the pipeline bubble",
        flip="the bubble grows past its objective once microbatches are reduced, at which point layer assignment rather than microbatch count must absorb the imbalance",
        falsifier="peak memory is flat across stages, which means the retention model does not apply to this schedule",
        wrong_claim="Weights are split evenly across the stages, so memory should be balanced and the outlier must be a leak.",
        wrong_why="Weight balance says nothing about activation retention, and the first stage legitimately holds state for every microbatch still in the pipeline.",
        threshold="Investigate when the first stage's peak exceeds the last stage's by more than the ratio predicted from the stage and microbatch counts.",
        cost="Provisioning every stage for the first stage's peak buys memory the later stages never use.",
        scaling="The imbalance grows with stage count, so deeper pipelines skew harder even as the weight split becomes finer.",
        quant=q_pp_memory_skew,
    ),
)
