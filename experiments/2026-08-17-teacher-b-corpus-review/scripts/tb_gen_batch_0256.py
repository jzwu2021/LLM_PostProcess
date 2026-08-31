import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0256.jsonl"
START, END = 2550, 2560

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
 ("STANCE 131 - Check whether memory rather than compute binds the capacity factor, because raising capacity to stop drops enlarges activation buffers that may not fit.",
  """Mechanism. Capacity factor sets the per-expert buffer size, and those buffers are allocated for every expert on the device regardless of how many tokens actually arrive. Raising the factor to eliminate drops therefore raises peak activation memory proportionally, competing with weights and KV cache for the same HBM. On a memory-bound deployment the reachable capacity factor is capped by allocation rather than by throughput, and a recommendation to raise it is infeasible rather than merely expensive.

Falsifiable hypothesis. H1: at the current deployment the capacity factor that would eliminate drops requires peak activation memory exceeding available headroom, so the drop rate cannot be driven to zero without reducing batch size or resharding. Falsified if the required factor fits within headroom, which makes this a straightforward throughput trade rather than a feasibility constraint.

Metrics. Peak activation memory and total HBM occupancy as functions of capacity factor, headroom after weights and KV cache, the capacity factor required to reach zero drops on the observed load distribution, achieved batch size at each factor, token-drop rate, all-to-all time and p99 step time. Memory figures are MEASURED from allocator statistics rather than computed from shapes, because allocator fragmentation and workspace buffers are not visible in a shape calculation.

Controlled experiment. Sweep the capacity factor upward on a fixed replayed stream, recording measured peak memory and drop rate at each point until either drops reach zero or allocation fails, and record which occurs first. Repeat at two batch sizes, since the memory-versus-drops trade is mediated by batch size and a single-batch-size sweep cannot show the frontier.

Confounders. Allocator caching makes peak memory depend on allocation history, so each point needs a clean process rather than a live reconfiguration. KV cache sizing is often dynamic and will silently absorb freed memory, hiding headroom. Fragmentation grows over a long-running process, so a short sweep understates steady-state pressure.

Rollback criteria. Revert to the recorded prior capacity factor if allocation failures or out-of-memory events occur at any load level, since an OOM is a hard availability failure while a drop is a quality degradation. Capacity factor and maximum batch size must be recorded and reverted as a single configuration unit, because reverting one without the other leaves an untested combination."""),
 ("STANCE 132 - Report the useful-token fraction, because padded capacity is real compute that throughput counters happily count as work.",
  """Mechanism. Expert kernels are executed over the full capacity buffer, with unused slots filled by padding. The device performs those FLOPs and the throughput counter records them, so a deployment with severe imbalance and generous capacity can show high utilisation while doing little useful work. Utilisation and achieved FLOPs are therefore not evidence of efficiency in a sparse MoE, and the honest denominator is the fraction of processed slots that carried a real token.

Falsifiable hypothesis. H1: the useful-token fraction at the deployed capacity factor is materially below one, so a substantial share of measured expert FLOPs is padding. Falsified if the fraction is close to one, which would mean capacity is tightly matched to load and utilisation figures can be read at face value.

Metrics. Useful-token fraction per expert and aggregated, padded slots processed, expert FLOPs split into useful and padding, device utilisation and achieved FLOPs for contrast, GPU-seconds per useful token, capacity factor, token-drop rate and p99 step time. The useful fraction is MEASURED by comparing dispatched slot counts against buffer sizes; any currency figure derived from wasted GPU-seconds is an ESTIMATE with its rate stated.

Controlled experiment. Instrument dispatch to record both the real slot count and the buffer size per expert per microbatch, then compute the fraction directly rather than inferring it from aggregate utilisation. Sweep the capacity factor and plot useful fraction against drop rate to expose the frontier, since the two move in opposite directions and the operating point is a choice on that curve.

Confounders. Some kernels skip masked rows and do not actually execute padding FLOPs, so the waste is implementation-dependent and must be confirmed against the kernel rather than assumed. Utilisation counters aggregate across layers including dense ones, diluting the effect. Very small microbatches make buffer granularity dominate, exaggerating the padded fraction.

Rollback criteria. Do not present utilisation or achieved FLOPs as efficiency evidence for this service once the useful fraction is known to be below the pre-declared floor; restate prior capacity claims that used them. Any capacity reduction taken to improve the fraction must revert if drop rate or eval-set quality crosses its threshold."""),
 ("STANCE 133 - Examine the expert kernel's shape sensitivity, because uneven per-expert token counts produce inefficient GEMMs even when the totals look balanced.",
  """Mechanism. Expert computation is a batch of matrix multiplications whose row counts are the per-expert token counts. Kernel efficiency depends on those shapes: very small row counts underutilise tensor cores and pay fixed launch and tiling overheads, while a grouped implementation is limited by the largest group. Two configurations with identical total tokens and identical dispersion can therefore differ substantially in achieved throughput purely through shape effects, which no token-count metric will reveal.

Falsifiable hypothesis. H1: achieved FLOPs efficiency of the expert GEMMs varies materially across the observed range of per-expert token counts, with a threshold below which efficiency degrades sharply. Falsified if efficiency is flat across the observed range, which would mean shape effects are immaterial here and token counts are a sufficient summary.

Metrics. Achieved FLOPs efficiency per expert GEMM as a function of row count, distribution of per-expert row counts, kernel launch count per layer, time in expert GEMMs versus in the collective, grouped-kernel padding to alignment boundaries, p99 step time and useful-token fraction. Efficiency curves are MEASURED by microbenchmarking the deployed kernel at the observed shapes, independently of the full model.

Controlled experiment. Microbenchmark the expert GEMM in isolation across the empirical distribution of row counts to obtain the efficiency curve, then weight that curve by the observed distribution to predict aggregate efficiency and compare the prediction against the profiled full-model figure. A mismatch indicates an effect the shape model omits, such as memory-bound behaviour at small sizes.

Confounders. Kernel selection is autotuned and can differ between the microbenchmark and the full model, so the autotuner cache state must be matched. Alignment padding quantises row counts, making the curve stepped rather than smooth. Concurrent kernels on other streams change achieved efficiency relative to an isolated benchmark.

Rollback criteria. Do not adopt a routing or capacity change justified solely by improved token-count balance if the shape-weighted efficiency prediction worsens; require the profiled full-model figure to confirm. Any kernel or backend change made in response must revert on a fixed eval-set quality regression, since numerics differ across kernel implementations."""),
 ("STANCE 134 - Treat dropless variable-size routing as a latency trade rather than a free fix, because removing drops makes step time depend on the worst batch.",
  """Mechanism. Dropless implementations size expert buffers to the actual token counts instead of a fixed capacity, so no token is discarded. The cost is that kernel shapes and memory footprint now vary per microbatch, and step time becomes a function of the realised maximum expert load. A configuration that was predictable becomes load-dependent, and the tail of the step-time distribution is set by the most skewed microbatch rather than by a constant capacity.

Falsifiable hypothesis. H1: switching to dropless routing eliminates token drops while increasing p99 step time relative to the fixed-capacity configuration, because the tail follows the maximum per-microbatch expert load. Falsified if p99 step time is unchanged or improves, which would make dropless strictly better here and remove the trade.

Metrics. Token-drop rate, distribution of maximum per-expert load per microbatch, p50 and p99 step time under each scheme, peak and variance of activation memory, allocator fragmentation over a long run, fixed eval-set quality, and useful-token fraction. Step-time distributions and memory variance are MEASURED over a long window, because the tail behaviour of a variable-shape scheme cannot be characterised on a short one.

Controlled experiment. Run both schemes on the same replayed stream at matched parallelism and batch composition, over a window long enough to sample the tail of the microbatch skew distribution. Report the full step-time distribution rather than a summary, and record allocation failures separately from latency, since variable footprints fail differently from constant ones.

Confounders. Variable shapes defeat kernel autotuning caches and CUDA graph capture, which can dominate the measured difference and is an implementation property rather than an algorithmic one. Memory fragmentation accumulates, so early and late portions of a long run differ. Quality comparison is confounded because the dropless scheme computes a different function than the dropping one.

Rollback criteria. Revert to fixed capacity if p99 step time or allocation stability regresses, treating drop elimination as insufficient justification on its own. Because the two schemes compute different functions, the eval-set quality comparison is mandatory before either is adopted, and the prior scheme must remain selectable by a single configuration flag."""),
 ("STANCE 135 - Measure prefill and decode separately, because their routing statistics differ so sharply that a pooled histogram describes neither.",
  """Mechanism. During prefill many tokens are processed at once, so each expert receives a large sample and per-expert counts approach the underlying routing distribution. During decode each sequence contributes one token per step, so the number of tokens per microbatch is small and per-expert counts are dominated by sampling noise and by which sequences happen to be co-batched. The same model exhibits smooth balance in prefill and severe transient skew in decode, and pooling the two produces a figure that matches neither regime.

Falsifiable hypothesis. H1: per-expert load dispersion in decode microbatches is substantially higher than in prefill microbatches at the same checkpoint, and the decode regime dominates the observed tail latency. Falsified if dispersion is comparable across phases, which would justify a single pooled analysis and a single capacity setting.

Metrics. Per-expert load dispersion by phase, tokens per microbatch by phase, token-drop rate by phase, all-to-all bytes and time by phase, step-time distribution by phase, share of end-to-end latency contributed by each phase, and useful-token fraction by phase. Phase-tagged metrics are MEASURED by stamping the phase on each microbatch record rather than inferred from batch size.

Controlled experiment. Tag every microbatch with its phase and recompute all balance metrics per phase on a fixed replayed window. Then sweep capacity factor independently per phase, since the optimum differs, and confirm whether the serving stack even permits phase-specific capacity before recommending it.

Confounders. Chunked prefill blurs the phase boundary, producing microbatches containing both, so a two-way tag is insufficient and mixed microbatches need their own class. Continuous batching mixes sequences at different stages, so decode microbatch composition depends on the scheduler. Very short prompts make prefill resemble decode statistically.

Rollback criteria. Do not apply a capacity factor derived from pooled statistics to a phase-disaggregated deployment; revert to the prior setting and re-derive per phase. If the serving stack cannot express per-phase capacity, record that as a constraint rather than reporting the pooled optimum as achievable."""),
 ("STANCE 136 - Treat expert load as an output of the scheduler, because under continuous batching the composition of every microbatch is a scheduling decision.",
  """Mechanism. The set of sequences co-resident in a microbatch is chosen by the admission and scheduling policy, and that set determines which tokens are routed together. Two deployments serving identical traffic with different scheduling policies will therefore see different per-microbatch expert distributions. Attributing the resulting skew to the router or to the traffic is a misattribution: the scheduler is an active participant in the load pattern and is far more easily changed than either.

Falsifiable hypothesis. H1: holding the checkpoint and the arriving request stream fixed, changing the scheduling policy materially changes per-microbatch expert load dispersion and p99 step time. Falsified if dispersion is invariant to scheduling policy, which would establish the router and traffic as the only drivers and remove the scheduler from the intervention set.

Metrics. Per-microbatch expert load dispersion under each scheduling policy, microbatch composition statistics covering sequence count and stage mix, queue wait time, admitted concurrency, p99 step time, end-to-end p99 latency, token-drop rate and useful-token fraction. Dispersion under each policy is MEASURED on the same replayed arrival stream so the traffic is genuinely held constant.

Controlled experiment. Replay one recorded arrival stream, including timing, through each scheduling policy at fixed capacity, placement and parallelism, and compare dispersion and tail step time. A policy that groups sequences with similar routing profiles would be expected to reduce dispersion, so include such a grouping policy explicitly rather than only comparing existing defaults.

Confounders. Scheduling changes admitted concurrency, which changes both batch size and queueing delay, so end-to-end latency moves for reasons unrelated to expert balance and the step-time term must be isolated. Routing-aware grouping requires knowing routing before execution, which is only possible approximately and adds its own cost. Arrival replay without timing fidelity destroys the queueing behaviour being studied.

Rollback criteria. Revert the scheduling policy if end-to-end p99 latency or fairness across tenants regresses, even when step-time dispersion improves, since scheduling changes have effects far beyond this defect. Any routing-aware grouping must be disabled by a single flag and must not be allowed to starve any tenant, which requires a per-tenant wait-time guardrail."""),
 ("STANCE 137 - Convert step-time distributions into request-level tails explicitly, because a request spans many steps and its latency is a sum over them rather than a single draw.",
  """Mechanism. Imbalance manifests per step, but users experience per request. A request generating many tokens accumulates many step times, so the request-level tail depends on both the step-time distribution and the correlation between consecutive steps. If skew is independent across steps, averaging suppresses it and the request tail is tighter than the step tail suggests; if skew is persistent, it accumulates and the request tail is worse. The mapping is therefore an empirical question about autocorrelation, not an arithmetic one.

Falsifiable hypothesis. H1: per-step expert-load skew is positively autocorrelated across consecutive decode steps, so request-level p99 latency is worse than an independence assumption predicts. Falsified if skew is uncorrelated across steps, in which case the independence model is adequate and step-level p99 overstates the user-visible tail.

Metrics. Step-time distribution, autocorrelation of per-step skew and of step time at short lags, output-token count distribution per request, request-level p99 latency, predicted request-level p99 under an independence model for contrast, and token-drop rate. Autocorrelation and the realised request tail are MEASURED; the independence prediction is an ESTIMATE included specifically so it can be falsified.

Controlled experiment. Record per-step times tagged by request over a long window, compute the autocorrelation of skew, and compare the realised request-level tail against the independence prediction built from the same step-time distribution. Stratify by output length, since long generations accumulate more steps and the two regimes should diverge if autocorrelation is present.

Confounders. Requests of different lengths mix different numbers of draws, so pooling across lengths obscures the accumulation effect. A request's steps share batch composition with co-resident requests, inducing correlation that is a scheduling artifact rather than a routing property. Preemption and requeueing insert gaps that are not step time but appear in end-to-end latency.

Rollback criteria. Do not set capacity or placement targets against step-level tails alone once the mapping is measured; restate targets at the request level. If an intervention improves step p99 but leaves request p99 unchanged beyond the noise band, treat it as unproven at the user-visible level and do not claim a latency benefit for it."""),
 ("STANCE 138 - Account for speculative decoding, because draft and verification steps change both the token volume entering the routers and the number of steps a request takes.",
  """Mechanism. With speculative decoding, a draft model proposes several tokens and the target model verifies them in a single forward pass. That pass routes multiple tokens at once rather than one, which changes per-microbatch expert statistics toward the prefill regime. Acceptance rate then determines how many steps the request actually needs. Both effects alter the relationship between expert imbalance and user-visible latency, in opposite directions, so speculative configuration must be recorded with any MoE tail measurement.

Falsifiable hypothesis. H1: enabling speculative decoding reduces per-microbatch expert load dispersion, because each verification pass routes several tokens, while leaving the aggregate routing distribution unchanged. Falsified if dispersion is unchanged, which would mean verification batching is too small to affect the statistics at this configuration.

Metrics. Per-microbatch expert dispersion with speculation enabled and disabled, tokens routed per verification pass, acceptance rate, steps per request, all-to-all bytes per accepted token, p99 step time, request-level p99 latency, token-drop rate and fixed eval-set quality. Acceptance rate and dispersion are MEASURED per configuration; any projected latency benefit combining the two effects is an ESTIMATE until measured end to end.

Controlled experiment. Run the same replayed request stream with speculation enabled and disabled at otherwise identical configuration, reporting dispersion, acceptance and request-level latency together. Because acceptance rate depends on the draft model and on input type, stratify by input type rather than reporting a single acceptance figure.

Confounders. Rejected draft tokens still consumed routing and expert compute, so cost per accepted token is the correct denominator and raw token counts overstate efficiency. Speculation changes batch composition and can disable other fast paths such as graph capture. Draft-model quality varies by domain, so acceptance and therefore the entire effect is mix-dependent.

Rollback criteria. Disable speculation by a single configuration flag if request-level p99 or eval-set quality regresses, and re-derive any capacity factor tuned under speculation before reverting, since the two settings were validated together. Do not carry a dispersion measurement taken with speculation enabled into a deployment that runs without it."""),
 ("STANCE 139 - Record the chunked-prefill configuration, because chunk size directly sets how many tokens each microbatch routes and therefore the granularity of imbalance.",
  """Mechanism. Chunked prefill splits a long prompt into fixed-size pieces that are scheduled alongside decode work. The chunk size determines how many prefill tokens appear in a microbatch, which in turn sets the sample size from which per-expert counts are drawn. Small chunks make each microbatch statistically noisy and increase transient skew; large chunks smooth the distribution but lengthen the step and delay co-resident decode requests. Chunk size is therefore a direct control on the imbalance the system experiences.

Falsifiable hypothesis. H1: per-microbatch expert load dispersion decreases monotonically as chunk size increases, while decode inter-token latency for co-resident requests increases, establishing a direct trade controlled by a single parameter. Falsified if dispersion is insensitive to chunk size, which would remove chunking from the intervention set.

Metrics. Per-microbatch expert dispersion, token-drop rate and useful-token fraction at each chunk size, decode inter-token latency for co-resident requests, time to first token for the chunked request, p99 step time, all-to-all time, and mixed-microbatch share. All are MEASURED per chunk size on the same replayed stream; the recommended operating point is an ESTIMATE until validated at production load.

Controlled experiment. Sweep chunk size on a fixed arrival stream containing both long prompts and active decodes, holding capacity, placement and parallelism fixed, and record both the balance metrics and the decode-side latency impact. Report the pair rather than the dispersion alone, since improving balance by enlarging chunks is paid for by co-resident decode requests.

Confounders. Chunk size interacts with the maximum batched token budget, so raising it can reduce the number of co-scheduled sequences and change composition twice over. Mixed microbatches containing prefill and decode have hybrid statistics and must be classified separately. Prompt length distribution determines how often chunking engages at all.

Rollback criteria. Revert chunk size to the recorded prior value if decode inter-token latency or time to first token breaches its objective, since balance is not a user-visible metric and latency is. Chunk size and the batched token budget must be reverted together as one configuration unit, because the pair was validated jointly."""),
 ("STANCE 140 - Do not attribute KV cache pressure to the mixture of experts, because the cache scales with attention state and context length rather than with expert count.",
  """Mechanism. In a sparse MoE only the feed-forward blocks are replicated into experts; attention and its key-value state are unchanged. The KV cache therefore has the same size as the corresponding dense model at the same layer count, head configuration and context length. Memory pressure on an MoE deployment is dominated by expert weights and activation buffers, not by a larger cache, and conflating these leads to interventions aimed at the wrong allocation.

Falsifiable hypothesis. H1: measured KV cache bytes per token on this deployment match the value computed from the attention configuration and are independent of expert count and capacity factor. Falsified if KV bytes per token vary with expert configuration, which would indicate an implementation that stores per-expert state and would change the memory analysis entirely.

Metrics. Measured KV bytes per token, computed KV bytes per token from layer count, key-value head count, head dimension and dtype, expert weight bytes resident per device, activation buffer bytes as a function of capacity factor, total HBM occupancy broken into these components, and achieved maximum concurrency. The component split is MEASURED from allocator statistics; the computed KV figure is an ESTIMATE derived from configuration and must be reconciled against the measurement.

Controlled experiment. Vary expert count and capacity factor while holding attention configuration and context length fixed, and confirm the KV component is invariant while activation and weight components move. This isolates the attribution directly rather than arguing it from architecture documentation.

Confounders. Paged allocators round KV to block granularity, so measured bytes exceed the computed figure by a quantisation term that must be accounted before declaring a mismatch. Some implementations reserve a fixed KV pool independent of load, which hides variation. Quantised KV changes the dtype term and must be recorded.

Rollback criteria. Reject and correct any capacity plan that attributes cache growth to the expert count, and re-derive it from the measured component split before provisioning. If a KV pool was resized on that mistaken attribution, revert it to the recorded prior size and re-measure achieved concurrency, since an oversized pool silently reduces the memory available for activation buffers."""),
]

DECISIONS = ["rewrite"] * 10

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

CONF = [0.73,0.72,0.7,0.71,0.74,0.72,0.71,0.69,0.72,0.75]

RISKS = [
 ["Source answer treats capacity overflow as a throughput trade without checking whether memory permits raising capacity at all.",
  "Allocator caching makes peak memory depend on allocation history, so live reconfiguration understates the true peak.",
  "Dynamic KV sizing silently absorbs freed memory and hides the headroom being measured."],
 ["Source answer counts token dropping and padding without requiring a useful-work denominator.",
  "Utilisation and achieved FLOPs count padding as work, so an imbalanced deployment can appear efficient.",
  "Whether padding FLOPs are actually executed is kernel-dependent and must be confirmed rather than assumed."],
 ["Source answer treats per-expert token counts as sufficient without considering kernel shape efficiency.",
  "Autotuned kernel selection can differ between an isolated microbenchmark and the full model.",
  "Alignment padding quantises row counts, making the efficiency curve stepped rather than smooth."],
 ["Source answer compares routing and capacity policies without noting that dropless routing converts drops into a variable-latency tail.",
  "Variable shapes defeat autotuning caches and graph capture, which can dominate the measured difference.",
  "Dropless and dropping schemes compute different functions, so quality comparison is mandatory rather than optional."],
 ["Source answer requests routing distribution and batch composition without separating prefill from decode.",
  "Chunked prefill produces mixed microbatches that belong to neither phase and need their own class.",
  "Continuous batching makes decode microbatch composition a scheduler output rather than a traffic property."],
 ["Source answer lists batch composition as a measurement without identifying the scheduler as its cause.",
  "Scheduling changes admitted concurrency and queueing delay, moving end-to-end latency for reasons unrelated to expert balance.",
  "Arrival replay without timing fidelity destroys the queueing behaviour under study."],
 ["Source answer targets tail latency without relating per-step imbalance to per-request accumulation.",
  "Steps within a request share batch composition with co-resident requests, inducing correlation that is a scheduling artifact.",
  "Pooling requests of different output lengths obscures the accumulation effect."],
 ["Source answer does not record whether speculative decoding is active, although it changes tokens routed per pass and steps per request.",
  "Rejected draft tokens still consume routing and expert compute, so raw token counts overstate efficiency.",
  "Draft-model acceptance is domain-dependent, making the entire effect mix-dependent."],
 ["Source answer treats batch composition as given rather than as set by the chunked-prefill configuration.",
  "Chunk size interacts with the batched token budget and can change composition through two paths at once.",
  "Improving balance by enlarging chunks is paid for by co-resident decode requests."],
 ["Source answer does not guard against attributing memory pressure to the expert count rather than to attention state.",
  "Paged allocators round KV to block granularity, producing a quantisation term that can be mistaken for a mismatch.",
  "A fixed reserved KV pool hides variation and can silently reduce memory available for activation buffers."],
]

EVID = [
 ["Measured peak activation memory and HBM occupancy from allocator statistics at each capacity factor, each on a clean process.",
  "Capacity sweep to either zero drops or allocation failure, repeated at two batch sizes to expose the memory-versus-drops frontier."],
 ["Dispatched slot counts compared against buffer sizes per expert per microbatch to compute the useful-token fraction directly.",
  "Useful fraction plotted against drop rate across a capacity sweep, with kernel confirmation of whether padding FLOPs execute."],
 ["Microbenchmark of the deployed expert GEMM across the empirical row-count distribution with matched autotuner cache state.",
  "Shape-weighted efficiency prediction compared against the profiled full-model figure, with kernel launch counts recorded."],
 ["Full step-time distribution and maximum per-expert load per microbatch under both schemes over a window long enough to sample the tail.",
  "Allocation failures and fragmentation recorded separately from latency, with fixed eval-set quality measured under both schemes."],
 ["Phase stamped on every microbatch record, with all balance metrics recomputed per phase and mixed microbatches classified separately.",
  "Independent per-phase capacity sweep, plus confirmation that the serving stack can express per-phase capacity before it is recommended."],
 ["Per-microbatch dispersion measured under each scheduling policy on one replayed arrival stream including original timing.",
  "Step-time term isolated from queueing delay, with per-tenant wait-time guardrails recorded for any routing-aware grouping policy."],
 ["Per-step times tagged by request over a long window with autocorrelation of skew computed at short lags.",
  "Realised request-level p99 compared against an independence-model prediction built from the same step-time distribution, stratified by output length."],
 ["Per-microbatch dispersion, tokens per verification pass and acceptance rate measured with speculation enabled and disabled.",
  "All-to-all bytes and cost reported per accepted token, with acceptance stratified by input type and eval-set quality held fixed."],
 ["Per-microbatch dispersion, drop rate and useful-token fraction recorded across a chunk-size sweep on a fixed arrival stream.",
  "Decode inter-token latency and time to first token reported alongside dispersion, with mixed-microbatch share and batched token budget recorded."],
 ["Measured KV bytes per token from allocator statistics reconciled against the value computed from the attention configuration.",
  "Component split of HBM into KV, expert weights and activation buffers while expert count and capacity factor are varied at fixed attention configuration."],
]

rows = []
for i, s in enumerate(src):
    m = {x["role"]: x["content"] for x in s["messages"]}
    head, body = STANCES[i]
    tc, ic, os_ = QD[i]
    rows.append({
        "source_id": s["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": DECISIONS[i],
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": head + "\n\n" + body,
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": RISKS[i],
        "evidence_required": EVID[i],
        "confidence": CONF[i],
    })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(rows))
