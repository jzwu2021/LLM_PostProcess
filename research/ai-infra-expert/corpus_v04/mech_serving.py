"""Serving-runtime mechanisms (topic: serving)."""
from __future__ import annotations

from core import GIB, Mechanism, Quant, Setting, fmt_int, gib, register


def q_kv_ceiling(s: Setting) -> Quant:
    per_tok = s.kv_bytes_per_token
    per_req = per_tok * s.ctx
    budget = s.kv_budget_bytes
    seats = max(budget // per_req, 0)
    return Quant(
        label="the KV cache footprint of one full-context request and the concurrency it permits",
        steps=[
            f"KV bytes per token = 2 * layers * kv_heads * head_dim * dtype_bytes "
            f"= 2 * {s.layers} * {s.kv_heads} * {s.head_dim} * {s.dtype_bytes} = {fmt_int(per_tok)} B",
            f"At full context {fmt_int(s.ctx)}: {fmt_int(per_tok)} * {fmt_int(s.ctx)} = {gib(per_req)} per request",
            f"Weights per GPU = {s.params_b}e9 * {s.dtype_bytes} / TP{s.tp} = {gib(s.weight_bytes_per_gpu)}",
            f"KV budget = {s.mem_gb} GiB * 0.90 - {gib(s.weight_bytes_per_gpu)} = {gib(budget)}",
            f"Full-context seats = floor({gib(budget)} / {gib(per_req)}) = {seats}",
        ],
        value=f"{gib(per_req)} per full-context request, giving {seats} full-context seats",
        interpretation=(
            f"The configured concurrency of {s.concurrency} is only reachable because real requests "
            f"do not all hold full context; at {seats} full-context seats the deployment is oversubscribed "
            f"by design and depends on the length distribution staying short."),
    )


def q_recompute(s: Setting) -> Quant:
    per_tok = s.kv_bytes_per_token
    avg_ctx = s.ctx // 2
    lost = per_tok * avg_ctx
    return Quant(
        label="the work destroyed by preempting one mid-flight request",
        steps=[
            f"Assume a preempted request holds half of context: {fmt_int(avg_ctx)} tokens",
            f"KV discarded = {fmt_int(per_tok)} B/token * {fmt_int(avg_ctx)} = {gib(lost)}",
            f"Recompute cost is one prefill over {fmt_int(avg_ctx)} tokens, which is compute-bound "
            f"and competes with the decode work of every other request in the batch",
            f"At concurrency {s.concurrency}, preempting 10% of the batch replays "
            f"{fmt_int(avg_ctx * s.concurrency // 10)} prompt tokens per preemption round",
        ],
        value=f"{gib(lost)} of cache and one full prefill of {fmt_int(avg_ctx)} tokens per preempted request",
        interpretation=(
            "Preemption is not a soft degradation. It converts memory pressure into additional compute "
            "at exactly the moment the device is already saturated, which is why it produces a latency "
            "cliff rather than a slope."),
    )


def q_prefill_block(s: Setting) -> Quant:
    tokens = s.ctx
    return Quant(
        label="the decode delay imposed by admitting one full-length prefill",
        steps=[
            f"A full-length prefill processes {fmt_int(tokens)} tokens in one or a few forward passes",
            f"Every decode request in the batch waits for that step to retire, so all "
            f"{s.concurrency} in-flight requests take the delay simultaneously",
            f"The p99 target is {s.slo_ms} ms; a single prefill step measured at even 300 ms consumes "
            f"{300 * 100 // s.slo_ms}% of the budget for every concurrent decode",
        ],
        value=f"one admitted {fmt_int(tokens)}-token prefill delays all {s.concurrency} decodes by a full step",
        interpretation=(
            f"The cost is shared across the batch, not paid by the prefilling request, which is why "
            f"per-request latency attribution does not reveal it and why the {s.slo_ms} ms target is "
            f"missed by requests that did nothing expensive."),
    )


def q_fragmentation(s: Setting) -> Quant:
    block = 16
    per_tok = s.kv_bytes_per_token
    waste_tok = block // 2
    waste = per_tok * waste_tok
    budget = s.kv_budget_bytes
    lost_seats = (waste * s.concurrency) / max(per_tok * s.ctx, 1)
    return Quant(
        label="the capacity lost to internal fragmentation at a block size of 16 tokens",
        steps=[
            f"Blocks are allocated whole, so the last block of each request is on average half unused: "
            f"{block} / 2 = {waste_tok} tokens",
            f"Wasted bytes per request = {waste_tok} * {fmt_int(per_tok)} = {gib(waste)}",
            f"At concurrency {s.concurrency}: {gib(waste * s.concurrency)} of the {gib(budget)} KV budget",
            f"Expressed as full-context seats: {lost_seats:.2f} seats",
        ],
        value=f"{gib(waste * s.concurrency)} wasted, roughly {lost_seats:.2f} full-context seats",
        interpretation=(
            "Fragmentation scales with block size and with concurrency, not with context length, so it "
            "is worst exactly in the high-concurrency short-request regime that block allocation was "
            "adopted to serve."),
    )


def q_bandwidth(s: Setting) -> Quant:
    wpg = s.weight_bytes_per_gpu
    per_tok = s.kv_bytes_per_token
    b1 = wpg + per_tok * (s.ctx // 2)
    b32 = wpg + per_tok * (s.ctx // 2) * 32
    return Quant(
        label="bytes read per decode step at batch 1 versus batch 32",
        steps=[
            f"Weights re-read every step, independent of batch: {gib(wpg)} per GPU",
            f"KV read per request at half context = {fmt_int(per_tok)} * {fmt_int(s.ctx // 2)} = {gib(per_tok * (s.ctx // 2))}",
            f"Batch 1: {gib(wpg)} + {gib(per_tok * (s.ctx // 2))} = {gib(b1)} for 1 token",
            f"Batch 32: {gib(wpg)} + 32 * {gib(per_tok * (s.ctx // 2))} = {gib(b32)} for 32 tokens",
            f"Bytes per generated token fall from {gib(b1)} to {gib(b32 / 32)}",
        ],
        value=f"{gib(b1)} per token at batch 1 against {gib(b32 / 32)} per token at batch 32",
        interpretation=(
            "The weight term is amortised by batching and the KV term is not, so the benefit of larger "
            "batches saturates once KV traffic dominates. That saturation point, not the device's "
            "compute, is what bounds useful batch size here."),
    )


def q_overreserve(s: Setting) -> Quant:
    per_tok = s.kv_bytes_per_token
    real = s.ctx // 4
    reserved = per_tok * s.ctx
    used = per_tok * real
    budget = s.kv_budget_bytes
    return Quant(
        label="concurrency lost when capacity is reserved at max context instead of realised length",
        steps=[
            f"Reserving at max context {fmt_int(s.ctx)}: {gib(reserved)} per request",
            f"Realised length is often nearer {fmt_int(real)} tokens: {gib(used)} per request",
            f"Seats at reserved sizing = floor({gib(budget)} / {gib(reserved)}) = {budget // max(reserved, 1)}",
            f"Seats at realised sizing = floor({gib(budget)} / {gib(used)}) = {budget // max(used, 1)}",
        ],
        value=f"{budget // max(reserved, 1)} seats reserved against {budget // max(used, 1)} seats achievable",
        interpretation=(
            "Static reservation at the maximum turns a length distribution into a worst case for every "
            "request. The gap is the entire argument for paged, incrementally grown allocation."),
    )


def q_truncation(s: Setting) -> Quant:
    budget = 768
    return Quant(
        label="the share of an evaluation invalidated by a generation cap",
        steps=[
            f"A cap of {budget} output tokens is below what a full reasoning answer needs",
            f"Any answer reaching the cap is cut mid-derivation and scores as incomplete",
            f"On this deployment the context is {fmt_int(s.ctx)}, so the cap is a harness setting, "
            f"not a model or memory limit",
            f"If 97 of 100 answers hit the cap, 97% of the score measures the cap and not the model",
        ],
        value=f"a {budget}-token cap can invalidate essentially the entire evaluation",
        interpretation=(
            "Truncation is a harness defect that masquerades as a capability result. Any comparison "
            "across two runs with different caps is uninterpretable regardless of the sample size."),
    )


def q_coldstart(s: Setting) -> Quant:
    buckets = 8
    return Quant(
        label="the one-time work a process pays before its measurements mean anything",
        steps=[
            f"Weight load moves {gib(s.weight_bytes)} into {s.gpu_count} devices, "
            f"{gib(s.weight_bytes_per_gpu)} per GPU at TP{s.tp}",
            "Graph capture and kernel autotuning run once per distinct shape bucket encountered",
            f"With roughly {buckets} length buckets across a {fmt_int(s.ctx)} context, that is "
            f"{buckets} separate capture events spread over early traffic",
            "The prefix cache is empty, so early requests pay full prefill that later ones do not",
        ],
        value=f"{gib(s.weight_bytes)} of load plus roughly {buckets} capture events before steady state",
        interpretation=(
            "None of this recurs in steady state, so a measurement taken across it describes the "
            "start-up transient rather than the service. The interval is a property of the shape "
            "set encountered, not of elapsed time, which is why a fixed wait is not a reliable "
            "substitute for checking that the metric has flattened."),
    )


def q_queue(s: Setting) -> Quant:
    return Quant(
        label="the latency an unbounded queue adds once arrivals exceed service rate",
        steps=[
            f"With admitted concurrency {s.concurrency}, arrivals above the service rate accumulate "
            f"rather than being rejected",
            f"Queue wait grows linearly with the excess and with time, without bound",
            f"The p99 target of {s.slo_ms} ms is breached by queue wait alone, before any per-request "
            f"work is measured",
            f"Every queued request still consumes a client connection and a retry budget upstream",
        ],
        value=f"unbounded growth: the {s.slo_ms} ms target fails on wait, not on service time",
        interpretation=(
            "An unbounded queue converts an overload into a latency failure that looks like a "
            "performance regression. Rejecting early is worse for one request and better for the "
            "service, and that trade must be made explicitly."),
    )


def q_prefix(s: Setting) -> Quant:
    shared = s.ctx // 2
    per_tok = s.kv_bytes_per_token
    return Quant(
        label="the prefill saved by a prefix cache hit and the memory it holds to do so",
        steps=[
            f"Assume a shared prefix of {fmt_int(shared)} tokens across requests",
            f"Prefill avoided per hit = {fmt_int(shared)} tokens of compute",
            f"Cache held to enable it = {fmt_int(per_tok)} * {fmt_int(shared)} = {gib(per_tok * shared)}",
            f"That memory is unavailable for KV seats while it is retained",
        ],
        value=f"{fmt_int(shared)} tokens of prefill saved per hit, at {gib(per_tok * shared)} held",
        interpretation=(
            "The benefit is entirely conditional on reuse. With no reuse the retained prefix is pure "
            "capacity loss, so the hit rate must be measured on real traffic before the feature is "
            "credited with anything."),
    )


register(
    Mechanism(
        key="kv_capacity_ceiling", topic="serving",
        title="KV cache capacity, not compute, sets the concurrency ceiling",
        concepts=("kv_cache", "capacity_planning", "concurrency"),
        symptom="Throughput stops improving as concurrency is raised, and added requests only add latency.",
        chain="Each admitted request reserves KV proportional to its context, so device memory runs out before arithmetic units do, and the scheduler stops admitting rather than running slower.",
        metric="KV block occupancy at the moment admission stalls, sampled per step rather than averaged.",
        signature="Occupancy sits at its ceiling while compute utilisation stays well below saturation.",
        confounders=(
            "A CPU-side tokenisation or scheduling bottleneck, which also caps throughput without saturating the device.",
            "An upstream connection or concurrency limit that never lets the server see the offered load.",
            "Background eviction from an unrelated tenant, which frees and refills memory on its own schedule.",
        ),
        fixes=(
            "Lower the per-request reservation by capping max output length to the realised distribution rather than the model maximum.",
            "Enable or increase KV quantisation so each token costs fewer bytes, accepting a measured quality check.",
            "Add devices or raise tensor parallelism so weights take a smaller share per GPU and more memory remains for KV.",
        ),
        rollback="Revert if p99 latency worsens or if quality checks regress after a KV precision change; restore the previous reservation and re-measure occupancy before trying again.",
        options=("capping the per-request reservation", "adding accelerators to enlarge the KV budget"),
        tradeoff="whether the length distribution, not the model maximum, is what fills memory",
        flip="the workload shifts toward long contexts, at which point capping stops recovering seats and only rejects work",
        falsifier="occupancy is well below its ceiling when admission stalls, which moves the cause to the scheduler or to an upstream limit",
        wrong_claim="Throughput is flat because the GPUs are compute-saturated, so we need faster accelerators.",
        wrong_why="Decode is memory-bound and admission here stops on memory capacity, so compute saturation is neither observed nor implied; faster arithmetic would leave the ceiling exactly where it is.",
        threshold="Treat KV occupancy above 90% at the admission decision as the binding constraint, and below 70% as evidence the constraint is elsewhere.",
        cost="Buying accelerators to relieve a reservation policy problem pays for memory that a configuration change would have freed at no capital cost.",
        scaling="The ceiling tightens as context grows, because reservation is linear in context while the memory budget is fixed.",
        quant=q_kv_ceiling,
    ),
    Mechanism(
        key="preemption_recompute", topic="serving",
        title="preemption under memory pressure causes recompute storms",
        concepts=("preemption", "kv_cache", "tail_latency"),
        symptom="Tail latency jumps in steps rather than degrading smoothly, and total token throughput falls while the device stays busy.",
        chain="When memory is exhausted the scheduler evicts an in-flight request, discarding its cache, and the request must re-run prefill later, adding compute at the moment the device is already contended.",
        metric="Preemption and restore events per minute, joined to the requests they affected.",
        signature="Affected requests show a second prefill after their first token, and the throughput dip aligns with the preemption events rather than with arrival rate.",
        confounders=(
            "Arrival bursts, which raise tail latency without any eviction occurring.",
            "Prefix cache eviction, which also causes recomputation but on a different trigger and without a preemption event.",
            "A slow downstream consumer applying backpressure, which stalls requests in a way that resembles eviction.",
        ),
        fixes=(
            "Reduce admitted concurrency so the scheduler stops reaching the eviction threshold.",
            "Enable swapping of evicted cache to host memory where interconnect bandwidth makes restore cheaper than recompute.",
            "Reduce per-request reservation so more requests fit and eviction is not reached at this load.",
        ),
        rollback="Revert if preemption events do not fall within one traffic cycle, since the mitigation is then addressing a symptom whose cause lies elsewhere.",
        options=("lowering admitted concurrency", "swapping evicted cache to host memory"),
        tradeoff="whether restoring from host memory over the available interconnect is faster than recomputing prefill",
        flip="the throughput given up by holding concurrency down exceeds the tail latency it recovers, at which point paying for restore is the cheaper trade",
        falsifier="preemption counters stay at zero across the whole window in which tail latency stepped up",
        wrong_claim="The latency steps are garbage collection or a driver hiccup, since nothing in the request path changed.",
        wrong_why="Preemption is in the request path and is invisible in per-request timings because the recompute is attributed to the victim's second prefill rather than to the event that caused it.",
        threshold="Any sustained non-zero preemption rate under normal load is a capacity finding; treat more than one preemption per hundred requests as the dominant tail contributor.",
        cost="Recompute is paid in device time that produced no output, so a preempting deployment buys accelerator hours it then discards.",
        scaling="The effect worsens superlinearly with load, because each recompute consumes the capacity that would have prevented the next eviction.",
        quant=q_recompute,
    ),
    Mechanism(
        key="prefill_decode_interference", topic="serving",
        title="a long prefill blocks every concurrent decode in the same step",
        concepts=("continuous_batching", "prefill", "tpot"),
        symptom="Inter-token latency spikes for requests that are producing short outputs and doing nothing unusual.",
        chain="Continuous batching runs prefill and decode in the same scheduler step, so a long prompt's forward pass occupies the step and every decoding request waits for it to retire.",
        metric="Per-step duration paired with the prompt-token count admitted in that step.",
        signature="Step duration correlates with admitted prompt tokens, and the inter-token spikes of unaffected requests align exactly with those steps.",
        confounders=(
            "Device contention from a co-located process, which lengthens steps without any large prompt.",
            "A change in batch composition from the admission policy, which alters step time independently of prompt length.",
            "Clock or sampling skew in the metric pipeline, which can manufacture apparent alignment.",
        ),
        fixes=(
            "Enable chunked prefill so a long prompt is split across steps and no single step is dominated by it.",
            "Cap the prompt tokens admitted per step so step duration has an explicit bound.",
            "Separate prefill and decode onto different instances, accepting the cache transfer cost that creates.",
        ),
        rollback="Revert chunking or the admission cap if time to first token regresses beyond its objective, since the mitigation trades first-token latency for inter-token stability.",
        options=("chunked prefill within one instance", "disaggregating prefill and decode onto separate instances"),
        tradeoff="whether the cache transfer between instances is cheap relative to the prefill work it decouples",
        flip="prompts grow long enough that chunking no longer bounds step time without starving prefill progress",
        falsifier="step duration is uncorrelated with admitted prompt tokens across the whole load range",
        wrong_claim="The spikes belong to the long-prompt requests, so they are expected and not a problem.",
        wrong_why="The cost falls on the concurrent decodes rather than on the long-prompt request, so per-request attribution shows nothing while the objective is missed by requests that did no expensive work.",
        threshold="Bound admitted prompt tokens per step so that worst-case step duration stays under a stated fraction of the inter-token objective.",
        cost="Buying capacity to absorb the spikes pays for hardware to hide a scheduling policy that a token budget would fix.",
        scaling="Severity grows with batch size, because a single long prefill delays proportionally more concurrent decodes.",
        quant=q_prefill_block,
    ),
    Mechanism(
        key="block_fragmentation", topic="serving",
        title="paged KV block size trades allocator efficiency against internal fragmentation",
        concepts=("paged_attention", "memory_fragmentation", "block_size"),
        symptom="Admitted concurrency is lower than the memory arithmetic predicts, with no eviction and no error.",
        chain="Cache is allocated in whole blocks, so each request's final block is partly unused, and that waste is paid per request rather than per token.",
        metric="Allocated blocks against live tokens, expressed as wasted positions per request.",
        signature="Waste per request is close to half the block size and is insensitive to context length while scaling with concurrency.",
        confounders=(
            "External fragmentation in the allocator, which wastes memory through a different mechanism and responds to different fixes.",
            "Retained prefix cache blocks, which look like waste but are held deliberately.",
            "Reserved workspace for activations, which is fixed overhead rather than fragmentation.",
        ),
        fixes=(
            "Reduce block size and re-measure both waste and allocator overhead, since only the pair is informative.",
            "Align request admission so short requests share blocks where the runtime supports it.",
            "Change the allocation strategy to grow blocks incrementally, which is a structural change to the memory manager.",
        ),
        rollback="Restore the previous block size if allocator overhead or step time regresses by more than the capacity recovered.",
        options=("reducing the block size", "keeping the block size and admitting fewer, longer requests"),
        tradeoff="whether the recovered capacity exceeds the allocator and locality cost of smaller blocks",
        flip="request lengths become aligned to the block size by upstream padding, at which point waste disappears and only the overhead remains",
        falsifier="measured waste is far below half a block per request, which means lengths are aligned and the model does not apply",
        wrong_claim="Fragmentation is negligible because the block size is small relative to the context length.",
        wrong_why="Waste scales with request count rather than with context, so a small block size relative to context says nothing; at high concurrency the aggregate is a significant share of the budget.",
        threshold="Investigate when wasted positions per request exceed half the block size, since that indicates padding or alignment beyond ordinary rounding.",
        cost="Capacity lost to fragmentation is capacity paid for and not served, and it appears in spend as under-utilised accelerators rather than as an error.",
        scaling="Total waste is linear in concurrency and independent of context, so it is worst in the short-request high-concurrency regime.",
        quant=q_fragmentation,
    ),
    Mechanism(
        key="decode_bandwidth_bound", topic="serving",
        title="decode is memory-bandwidth bound, so batching amortises weights but not KV",
        concepts=("roofline", "memory_bandwidth", "batching"),
        symptom="Compute utilisation reads low while the service is clearly at its limit, and adding batch size helps at first and then stops helping.",
        chain="Each decode step re-reads the full weight set regardless of batch size and re-reads each request's cache, so the step is limited by bytes moved; batching divides the weight term across more tokens but leaves the cache term untouched.",
        metric="Bytes read per generated token, split into the weight component and the KV component.",
        signature="Bytes per token fall steeply with early batch increases and then flatten as the KV term dominates, with achieved bandwidth near the device's measured peak throughout.",
        confounders=(
            "Kernel launch overhead at very small batch sizes, which also produces poor efficiency but disappears differently.",
            "Prefix cache hits, which reduce KV traffic without any change in batch size.",
            "A quantisation change, which alters bytes per weight and shifts the whole curve.",
        ),
        fixes=(
            "Raise batch size to the point where bytes per token flattens, and no further.",
            "Reduce the KV term through grouped-query attention or cache quantisation, which is what moves the flattening point.",
            "Change to hardware with higher memory bandwidth, which raises the ceiling rather than the efficiency.",
        ),
        rollback="Reduce batch size back to the measured flattening point if inter-token latency regresses, since batch beyond that point costs latency for no throughput.",
        options=("increasing batch size", "reducing bytes per cached token"),
        tradeoff="whether the weight term or the KV term dominates bytes read per token at the operating batch size",
        flip="context lengths grow so that the KV term dominates from the smallest batch, making further batching ineffective",
        falsifier="achieved bandwidth stays far below the device's measured peak while compute is also low, indicating neither resource is the limit",
        wrong_claim="GPU utilisation is only 30%, so we are wasting most of the hardware and should raise the batch size until it reaches 90%.",
        wrong_why="The reported utilisation counter reflects arithmetic occupancy, and decode is bound by bytes moved rather than arithmetic; raising batch past the bandwidth limit adds latency without adding tokens.",
        threshold="Stop increasing batch size once bytes per generated token improve by less than a stated margin per doubling.",
        cost="Provisioning for compute in a bandwidth-bound phase buys arithmetic capacity that the workload cannot reach.",
        scaling="The flattening point moves toward smaller batches as context grows, because the per-request cache term grows while the weight term does not.",
        quant=q_bandwidth,
    ),
    Mechanism(
        key="max_len_overreservation", topic="serving",
        title="reserving capacity at maximum context turns a length distribution into a worst case",
        concepts=("admission_control", "capacity_planning", "kv_cache"),
        symptom="The service admits far fewer concurrent requests than the memory arithmetic suggests, while most requests are short.",
        chain="If capacity is reserved at the configured maximum context rather than grown with realised length, every short request occupies a long request's footprint and the memory budget is consumed by space that is never written.",
        metric="Reserved bytes against written bytes per request, aggregated over the realised length distribution.",
        signature="The ratio of reserved to written capacity tracks the ratio of maximum context to median realised length, and is insensitive to load.",
        confounders=(
            "Genuine long-context traffic, which consumes the reservation legitimately.",
            "A client setting an explicit maximum output length, which changes reservation without changing realised length.",
            "Prefix cache retention, which occupies memory that a naive audit attributes to reservation.",
        ),
        fixes=(
            "Lower the configured maximum context to the realised distribution's upper percentile rather than the model's limit.",
            "Enable incremental block growth so allocation follows realised length.",
            "Route long-context traffic to a separate pool so one distribution does not size the other.",
        ),
        rollback="Restore the previous maximum immediately if long requests begin failing admission, since the mitigation converts a capacity problem into a rejection problem.",
        options=("lowering the configured maximum context", "routing long contexts to a dedicated pool"),
        tradeoff="whether long requests are a small enough tail that excluding them recovers more than it rejects",
        flip="long-context traffic grows into a material share, at which point a lowered maximum rejects real demand rather than recovering waste",
        falsifier="reserved and written bytes are close, which means allocation already follows realised length and the mechanism is absent",
        wrong_claim="We must raise max_model_len to the model's limit so we never reject a long request.",
        wrong_why="Raising the maximum raises the reservation for every request under static allocation, so it reduces admitted concurrency for the common case in order to admit a rare one.",
        threshold="Investigate when reserved capacity exceeds written capacity by more than a factor of two across the realised distribution.",
        cost="Reserved-but-unwritten memory is paid for at the accelerator's full price and serves nothing.",
        scaling="The waste ratio grows with the gap between configured maximum and median realised length, not with load.",
        quant=q_overreserve,
    ),
    Mechanism(
        key="generation_cap_truncation", topic="serving",
        title="an output-token cap silently truncates answers and invalidates evaluation",
        concepts=("evaluation", "truncation", "harness"),
        symptom="Scores are uniformly low across a benchmark, with answers that read as competent until they stop mid-sentence.",
        chain="The harness caps generated tokens below what a complete answer requires, so the model is scored on a fragment, and the score measures the cap rather than the model.",
        metric="Finish-reason distribution and the share of responses terminating at exactly the cap.",
        signature="A spike of responses ending at precisely the cap value, with finish reason length rather than stop.",
        confounders=(
            "A stop sequence firing early, which also truncates but at varying lengths rather than at the cap.",
            "A genuinely terse model, which produces short answers with a stop finish reason.",
            "Context exhaustion, which truncates for a different reason and at a different boundary.",
        ),
        fixes=(
            "Raise the cap above the observed answer-length distribution and re-run.",
            "Report the truncation rate alongside every score so a capped run is never compared with an uncapped one.",
            "Redesign the harness so the cap is derived from the task rather than hard-coded.",
        ),
        rollback="Discard and re-run any comparison whose arms used different caps; there is no correction that rescues the existing numbers.",
        options=("raising the cap and re-running", "keeping the cap and scoring only untruncated answers"),
        tradeoff="whether the truncated subset is a random sample or is systematically the harder questions",
        flip="answers grow to fill whatever cap is set, at which point raising it defers the problem instead of removing it and the answer-length distribution itself has to be bounded",
        falsifier="the truncation rate is near zero, which removes the cap as an explanation for the scores",
        wrong_claim="The base model scores near zero on this benchmark, so the fine-tune is clearly necessary.",
        wrong_why="A near-zero score produced under truncation is a measurement of the harness, and comparing it against a fine-tune run with a different cap compares two harnesses rather than two models.",
        threshold="Treat any evaluation with a truncation rate above a few percent as uninterpretable and re-run rather than adjusting the score.",
        cost="Every accelerator hour spent on a truncated evaluation is spent producing a number that cannot be used.",
        scaling="The problem worsens as tasks require longer reasoning, so a cap that was adequate for one task set silently invalidates the next.",
        quant=q_truncation,
    ),
    Mechanism(
        key="cold_start_measurement_bias", topic="serving",
        title="measurements taken in the first minutes after a deploy describe warm-up rather than the service",
        concepts=("benchmarking", "warmup", "graph_capture"),
        symptom="A freshly deployed build benchmarks poorly and the same build benchmarks well an hour later with no change made in between.",
        chain="Immediately after start the runtime is still capturing graphs, autotuning kernel choices for each new shape and serving every request against an empty prefix cache, so early measurements include one-time costs that steady-state serving never pays again.",
        metric="Time per output token as a function of minutes since process start, plotted rather than averaged.",
        signature="The curve falls steeply for the first shape buckets encountered and then flattens, and the flattening point coincides with capture and autotuning completing.",
        confounders=(
            "Page cache warming on the host, which speeds weight access on a second run for a reason unrelated to the runtime.",
            "Prefix cache filling with real traffic, which improves latency without any runtime state changing.",
            "Load ramping during the same window, which changes batch composition while the curve is still falling.",
        ),
        fixes=(
            "Discard a stated warm-up interval before recording any benchmark, and publish that interval with the result.",
            "Drive the expected shape buckets deliberately at start so capture and autotuning complete before traffic arrives.",
            "Persist autotuning results across restarts where the runtime supports it, so the cost is paid once per build rather than per process.",
        ),
        rollback="Discard and re-run any comparison whose arms had different warm-up treatment; there is no correction that rescues those numbers after the fact.",
        options=("discarding a stated warm-up interval before recording", "driving the expected shapes deliberately at start"),
        tradeoff="whether the shape set the service will see is known well enough to be driven in advance",
        flip="request shapes turn out to be too varied to enumerate, at which point only a warm-up interval and honest reporting remain",
        falsifier="time per output token is flat from the first request onward",
        wrong_claim="The new build is slower in our benchmark, so the change regressed performance.",
        wrong_why="The benchmark captured graph capture, autotuning and an empty prefix cache, which the previous build had already paid for in a long-running process, so the comparison is between a cold process and a warm one rather than between two builds.",
        threshold="Require every published benchmark to state its warm-up interval and to show the metric had flattened before recording began.",
        cost="Rejecting a good build or accepting a bad one on warm-up artefacts costs a release cycle and the engineering that follows it.",
        scaling="Warm-up cost grows with the number of distinct shape buckets, so services with variable prompt lengths take longer to reach steady state.",
        quant=q_coldstart,
    ),
    Mechanism(
        key="unbounded_queue_overload", topic="serving",
        title="an unbounded queue converts overload into unbounded latency instead of rejection",
        concepts=("admission_control", "queueing", "slo"),
        symptom="Latency grows without limit under sustained load while throughput stays flat and no errors are returned.",
        chain="When arrivals exceed service rate and the queue has no bound, excess work accumulates rather than being refused, so every admitted request pays a wait that grows with the backlog.",
        metric="Queue wait as a share of end-to-end latency, sampled per request rather than averaged.",
        signature="Wait time grows monotonically with backlog while service time per request stays constant.",
        confounders=(
            "A genuine per-request slowdown, which raises service time rather than wait time.",
            "Client-side retries, which inflate arrivals and can be mistaken for organic demand.",
            "A downstream dependency stalling, which fills the queue for a reason unrelated to capacity.",
        ),
        fixes=(
            "Bound the queue and reject beyond it, returning a retryable status rather than a slow success.",
            "Add a deadline so a request whose remaining budget cannot be met is refused rather than served late.",
            "Shed load by priority so the rejection falls on the traffic the service chooses rather than at random.",
        ),
        rollback="Restore the previous bound if rejection rate rises without a latency improvement, which indicates the bound is below the service's real capacity.",
        options=("bounding the queue with rejection", "adding a per-request deadline"),
        tradeoff="whether clients handle an explicit rejection better than a late response",
        flip="clients retry aggressively on rejection, at which point rejecting increases offered load and a deadline is the safer control",
        falsifier="service time per request grows alongside wait time, which shows a slowdown rather than a queueing effect",
        wrong_claim="No errors are being returned, so the service is healthy and the latency is a client-side problem.",
        wrong_why="An unbounded queue produces exactly this signature: success responses delivered arbitrarily late, which is a more damaging failure than an explicit rejection and is invisible in an error-rate objective.",
        threshold="Bound the queue so that worst-case wait at full backlog stays within the latency objective.",
        cost="Work performed for a request whose client has already timed out is accelerator time bought and discarded.",
        scaling="Backlog grows linearly with the excess arrival rate and with time, so a small persistent overload eventually produces an arbitrarily large latency.",
        quant=q_queue,
    ),
    Mechanism(
        key="prefix_cache_reuse_dependency", topic="serving",
        title="prefix cache benefit is entirely conditional on measured reuse",
        concepts=("prefix_caching", "workload_characterisation", "routing"),
        symptom="A prefix cache was enabled and neither latency nor throughput moved, while memory available for new requests fell.",
        chain="Retaining a shared prefix saves prefill only when later requests actually share it; without reuse the retained blocks are pure capacity loss and reduce the seats available for admission.",
        metric="Prefix cache hit rate on production traffic, together with the reuse-gap distribution between matching requests.",
        signature="Hit rate is low while retained blocks occupy a material share of the KV budget, and admitted concurrency falls by roughly that share.",
        confounders=(
            "A prompt template change, which can eliminate reuse without any change in traffic volume.",
            "Cache eviction under load, which lowers hit rate exactly when pressure is highest and looks like absent reuse.",
            "Routing that spreads matching requests across replicas, so reuse exists globally but not on any one instance.",
        ),
        fixes=(
            "Measure the hit rate and the reuse gap before crediting the feature with any benefit.",
            "Route requests by prefix so matching requests reach the instance holding the cache.",
            "Bound retention by a window derived from the measured reuse gap rather than by available memory.",
        ),
        rollback="Disable retention if hit rate stays below the level at which saved prefill exceeds the lost admission capacity.",
        options=("prefix-aware routing", "least-loaded routing with a shorter retention window"),
        tradeoff="whether saved prefill compute exceeds the queueing added by constraining placement",
        flip="reuse falls or becomes spread across replicas, at which point prefix-aware routing only adds queueing",
        falsifier="hit rate is high and retained capacity is small, which makes the feature beneficial and removes the hypothesis",
        wrong_claim="Prefix caching is a strict improvement, so it should be enabled everywhere by default.",
        wrong_why="It trades admission capacity for saved prefill, and that trade is negative whenever reuse is low, so it is workload-conditional rather than strictly beneficial.",
        threshold="Require a measured hit rate high enough that saved prefill exceeds the admission capacity given up before retention is enabled.",
        cost="Retained blocks with no reuse are memory paid for at accelerator price and used to store nothing that will be read.",
        scaling="Reuse falls as the replica count rises under load-balanced routing, so the feature weakens exactly as the fleet grows.",
        quant=q_prefix,
    ),
)
