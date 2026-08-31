"""Authored teacher-B exemplars for the deduplicating stage, design families 501-503.

Each entry is (head, body, quality_dimensions, risks, evidence_required, confidence).
"""

QD = (2, 2, 2)

FAM_501 = [
 ("STANCE 161 - Decompose the step before blaming the draft model: a speculative step costs draft generation plus one target forward, and only the second is amortised over accepted tokens.",
  """Mechanism. A speculative step runs the draft model k times and the target model once over the k+1 positions. The target forward is the expensive term but it is shared across every accepted token, while draft time is paid in full regardless of acceptance. If draft latency is a large fraction of target latency, the arithmetic stops working: the break-even acceptance rate rises above what the draft can deliver and the system is slower than plain decoding.

Falsifiable hypothesis. H1: measured draft time per step divided by target forward time exceeds the ratio at which the observed acceptance rate can break even, so speculation cannot profit at this configuration. Falsified if the ratio sits below break-even and the slowdown persists, which moves the cause to synchronisation or batching rather than to the draft-target cost balance.

Metrics. Draft latency per proposed token, target forward latency at speculative width and at width one, accepted tokens per step, acceptance rate by position within the proposal, wall-clock time per accepted token with and without speculation, and the break-even acceptance rate implied by the measured latency ratio. Latencies and acceptance are MEASURED per step; the break-even rate is an ESTIMATE derived from them and must show its derivation.

Controlled experiment. Run the identical request set through the target alone and through the speculative path on the same hardware, seeds and batch policy, recording per-step draft and target times separately rather than only end-to-end latency. Sweep the proposal width, since break-even moves with width and a single width cannot show whether the configuration is merely mistuned.

Confounders. Acceptance rate depends on the input distribution, so a draft tuned on one domain fails on another and a pooled rate hides it. The target forward at width k is not k times the width-one cost, because it is one batched pass, so a naive cost model overstates the saving. Draft and target may share the device, in which case their times are not additive under concurrency.

Rollback criteria. Disable speculation by a single serving flag if time per accepted token exceeds the non-speculative baseline at production width, and re-measure rather than tuning width first. Because speculation changes batch composition, any capacity or concurrency setting tuned with it enabled must be reverted together with it."""),
 ("STANCE 162 - Measure acceptance by position, because a proposal whose first token is usually rejected wastes the entire remaining proposal and averages conceal that shape.",
  """Mechanism. Verification is sequential: once a proposed token is rejected, every later token in that proposal is discarded regardless of whether it would have matched. The useful yield of a proposal is therefore governed by the position of the first rejection, not by the mean per-token acceptance. Two configurations with identical average acceptance can differ greatly in accepted tokens per step if one rejects early and the other rejects late.

Falsifiable hypothesis. H1: the distribution of first-rejection position is concentrated at the earliest positions, so accepted tokens per step is far below the value implied by the mean acceptance rate. Falsified if first rejections are uniformly distributed across positions, in which case the mean is an adequate summary and width tuning is the remaining lever.

Metrics. Acceptance rate at each proposal position, distribution of first-rejection position, accepted tokens per step with its full distribution rather than its mean, discarded proposed tokens per step, draft compute spent on discarded positions, and time per accepted token. Position-wise acceptance is MEASURED from verification records; any predicted yield from a width change is an ESTIMATE until that width is run.

Controlled experiment. Log per-position accept and reject outcomes for a fixed replayed request set, then compute the first-rejection distribution and the realised yield. Sweep proposal width and check whether yield saturates, since saturation indicates the useful width is bounded by early rejection and further width only adds discarded draft compute.

Confounders. Sampling temperature changes acceptance because verification compares distributions, so temperature must be pinned and reported. Prompt-dependent difficulty varies by request type, so stratification is required. Some implementations accept a corrected token after the first rejection, which changes the yield definition and must be stated.

Rollback criteria. Reduce width to the measured saturation point rather than disabling speculation outright when yield saturates early, and revert to the recorded prior width if time per accepted token regresses. Any width change must be recorded with the temperature and draft checkpoint it was validated against, since both move the acceptance curve."""),
 ("STANCE 163 - Check whether speculation is competing with batching: at high concurrency the target model is already saturated and the extra positions displace other requests rather than filling idle capacity.",
  """Mechanism. Speculative decoding profits when the target model is latency-bound and underutilised, because verifying k positions costs little more than verifying one. Under continuous batching at high load the device is already throughput-bound, and the extra verification positions consume capacity that would otherwise serve concurrent requests. The single-stream benefit therefore inverts into a fleet-level throughput loss, which end-to-end latency for one request cannot reveal.

Falsifiable hypothesis. H1: speculation improves single-stream latency while reducing aggregate throughput at production concurrency, so the sign of its benefit depends on load. Falsified if throughput is unchanged or improved at production concurrency, which would make speculation beneficial across the operating range.

Metrics. Single-stream time per token and aggregate tokens per second at several concurrency levels with speculation enabled and disabled, admitted concurrency, queue wait, batch occupancy, accepted tokens per step, and end-to-end p99 latency. Throughput at each concurrency level is MEASURED under a load sweep; a benefit claimed from a single-stream measurement is an ESTIMATE of unknown scope.

Controlled experiment. Sweep offered load from idle to SLO breach with speculation enabled and disabled, holding model, hardware and batching policy fixed, and report the crossover concurrency at which the sign changes. A single operating point cannot locate that crossover and is the most common way this decision is made incorrectly.

Confounders. Speculation changes memory footprint through additional activation and draft state, which can reduce the maximum batch size and confound the throughput comparison. Draft and target sharing the device interact through the scheduler. Warm-up and graph capture differ between the two paths.

Rollback criteria. Gate speculation on measured load rather than enabling it globally, and disable it above the measured crossover by a single policy parameter. If the crossover cannot be measured because load cannot be varied safely, treat the fleet-level benefit as unproven and keep speculation off in the saturated regime."""),
]

FAM_502 = [
 ("STANCE 164 - Read the symptom literally before diagnosing: low compute utilisation with saturated bandwidth is the expected signature of memory-bound decode, not a defect to be removed.",
  """Mechanism. Incremental decode multiplies a weight matrix by a single-token activation per request. Arithmetic intensity is therefore very low: each weight byte is used for a handful of operations, so the kernel is limited by how fast weights and KV entries can be read from memory rather than by arithmetic throughput. Low compute utilisation alongside near-peak bandwidth is what a correctly functioning decode phase looks like, and treating it as an anomaly leads to optimisations that cannot help.

Falsifiable hypothesis. H1: measured achieved bandwidth is close to the device's attainable peak and measured arithmetic intensity sits on the memory-bound side of the roofline knee, so the phase is bandwidth-limited by construction. Falsified if achieved bandwidth is well below attainable peak while compute utilisation is also low, which indicates neither resource is saturated and points to launch overhead or stalls.

Metrics. Achieved memory bandwidth against attainable peak from a microbenchmark on the same device, arithmetic intensity per decode kernel, compute utilisation, bytes read per generated token split into weights and KV, batch size, and time per output token. Attainable peak and achieved bandwidth are MEASURED; the roofline knee is an ESTIMATE computed from device specifications and must be reconciled with the microbenchmark rather than taken from a datasheet.

Controlled experiment. Measure attainable bandwidth with a streaming microbenchmark on the same hardware and build, then compare the decode kernel's achieved bandwidth against it. Sweep batch size and observe whether bytes per token fall as weights are amortised across more requests, which is the direct test of the memory-bound explanation.

Confounders. Quantisation changes bytes per weight and moves the intensity, so precision must be pinned. Prefix caching changes KV read volume independently of batch size. Profilers report bandwidth over different windows, so a counter compared against a datasheet peak rather than a measured one will mislead.

Rollback criteria. Do not pursue compute-side optimisations while the phase is demonstrated to be memory-bound; record the roofline position as the justification for whichever direction is chosen. If a change raises compute utilisation without improving time per output token, revert it, since utilisation is not the objective."""),
 ("STANCE 165 - Separate weight traffic from KV traffic, because they scale differently with batch and context and the correct fix differs for each.",
  """Mechanism. Decode reads two distinct things from memory: model weights, which are shared across every request in the batch, and KV cache entries, which are private to each sequence and grow with context length. Increasing batch size amortises weight traffic but increases KV traffic proportionally. A single aggregate bandwidth number cannot say which term dominates, yet the interventions are opposite: larger batches and quantised weights address one, while shorter context, KV quantisation and eviction address the other.

Falsifiable hypothesis. H1: at the deployed batch size and context length, KV traffic exceeds weight traffic per decode step, so batching further will not reduce bytes per token materially. Falsified if weight traffic dominates, in which case increasing batch size is the direct remedy and context management is secondary.

Metrics. Bytes read per decode step split into weights and KV, both measured rather than computed; batch size; per-sequence context length distribution; bytes per generated token; achieved bandwidth; and time per output token. The split is MEASURED by instrumenting the kernels or by differencing controlled configurations; a split derived only from shape arithmetic is an ESTIMATE and must be labelled.

Controlled experiment. Hold context fixed and sweep batch size, then hold batch fixed and sweep context length, recording bytes per token in each sweep. Weight-dominated regimes show bytes per token falling with batch and flat with context; KV-dominated regimes show the opposite, so the two sweeps identify the regime without needing kernel-level instrumentation.

Confounders. Prefix caching removes KV reads for shared prefixes and varies with traffic. Grouped-query attention reduces KV heads and shifts the balance substantially, so the head configuration must be reported. Paged allocators read at block granularity, so short sequences read more than their token count implies.

Rollback criteria. Revert a batch-size increase if time per output token or p99 latency regresses, since larger batches trade latency for throughput and the trade is only worthwhile in the weight-dominated regime. Any KV quantisation must revert on a fixed eval-set quality regression, because it changes the served function rather than only the memory traffic."""),
 ("STANCE 166 - Rule out launch overhead and stalls before accepting the memory-bound story, because a decode step that is idle between kernels also shows low compute utilisation.",
  """Mechanism. At small batch sizes each decode step issues many short kernels. If kernel launch, synchronisation or Python-side scheduling occupies a significant share of the step, the device is idle for part of it. That idleness depresses compute utilisation and, because no memory traffic occurs while idle, it also depresses average bandwidth below attainable peak. The symptom then resembles a memory-bound phase while the actual constraint is host-side overhead, which graph capture or kernel fusion removes.

Falsifiable hypothesis. H1: the sum of kernel execution times is materially less than the wall-clock step time, indicating device idle gaps attributable to launch and synchronisation rather than to memory pressure. Falsified if kernel time accounts for nearly all of step time, which confirms the device is busy and the phase is genuinely bandwidth-limited.

Metrics. Wall-clock step time, summed kernel execution time, device idle gap per step, kernel launch count per step, time in synchronisation, achieved bandwidth measured only over active kernel windows rather than over the whole step, and time per output token. The gap analysis is MEASURED from a kernel timeline; attributing the gap to a specific cause before the timeline is captured is an ESTIMATE.

Controlled experiment. Capture a kernel timeline for a fixed replayed workload at the production batch size and compute the idle fraction, then enable graph capture or fusion and re-measure. If the idle fraction falls and time per output token improves while achieved bandwidth over active windows is unchanged, the original constraint was overhead rather than bandwidth.

Confounders. Tracing itself adds overhead and can inflate the measured gap, so tracer inertness must be established first. Bandwidth averaged over the whole step is depressed by idle time and will falsely suggest headroom. Graph capture is incompatible with dynamic shapes in some stacks, so the remedy may be unavailable rather than merely unused.

Rollback criteria. Revert graph capture or fusion if it breaks dynamic-shape handling or changes numerics enough to move fixed eval-set quality, and re-measure the idle fraction rather than assuming the change helped. Report achieved bandwidth over active kernel windows in all future analyses, since the whole-step average conflates two different constraints."""),
]

FAM_503 = [
 ("STANCE 167 - Fix the workload before comparing the systems, because tokens per second is a property of the request distribution as much as of the server.",
  """Mechanism. Throughput depends on prompt length, generation length, arrival pattern and concurrency. Two systems benchmarked on different distributions are not being compared at all, and the difference in their published numbers can be entirely explained by workload. A reproducible comparison therefore begins by pinning the request trace itself, including per-request prompt and generation token counts and arrival timestamps, and replaying that identical trace against both systems.

Falsifiable hypothesis. H1: replaying one pinned request trace against both systems reduces the throughput gap to within the measured run-to-run band, showing the published difference was workload rather than implementation. Falsified if a material gap persists under the identical trace, which localises the difference in the systems and justifies deeper comparison.

Metrics. Per-request prompt and generation token counts, arrival timestamps, achieved tokens per second split into prompt and generated tokens, requests per second, concurrency, p50 and p99 end-to-end latency, and run-to-run variation across repeats. The trace and all counts are MEASURED and published as raw artifacts; any normalised or extrapolated throughput figure is an ESTIMATE.

Controlled experiment. Replay the identical trace against both systems on identical hardware with identical model revision, tokenizer, precision and parallelism, repeating each run enough times to establish a variation band before any difference is interpreted. Report generated tokens per second separately from total, since counting prompt tokens rewards whichever system prefills more aggressively.

Confounders. Tokenizer differences change the token count for the same text, so tokens per second is not comparable across tokenizers without reporting characters or requests as well. Stopping criteria differ, so one system may generate fewer tokens per request by design. Warm-up and cache state differ between runs and must be controlled explicitly.

Rollback criteria. Withdraw any published comparison not accompanied by its raw trace and configuration, and do not select a system on a figure that cannot be regenerated. If a procurement or migration decision has already been made on such a figure, re-run the pinned comparison before committing further capacity to it."""),
 ("STANCE 168 - Report throughput and latency as a curve rather than a point, because any system can be tuned to win on either at the expense of the other.",
  """Mechanism. Raising batch size and queue depth increases tokens per second while lengthening per-request latency. A single throughput figure therefore encodes an unstated operating point, and a system tuned for maximum batching will always appear faster than one tuned for responsiveness. The comparison is only meaningful as a frontier: throughput achieved at each latency constraint, measured by sweeping offered load until the constraint is breached.

Falsifiable hypothesis. H1: the ranking of the two systems reverses between a strict and a relaxed latency constraint, so no single throughput number can order them. Falsified if one system dominates across the entire frontier, which would make a single figure defensible provided the constraint is stated.

Metrics. Achieved tokens per second at each offered load level, p50, p95 and p99 end-to-end latency at each level, time to first token, time per output token, admitted concurrency, queue wait, and the load at which each latency objective is breached. All points are MEASURED under a load sweep; the maximum sustainable throughput at a given objective is read from the curve rather than extrapolated.

Controlled experiment. Sweep offered load against both systems on the pinned trace, recording the full latency distribution at each point, and plot throughput against the latency constraint. Stop each sweep at the objective breach rather than at saturation, since throughput beyond the breach describes a configuration nobody would run.

Confounders. Client-side load generators saturate before the server does and cap the measured curve, so the generator must be verified as not the bottleneck. Autoscaling changes capacity mid-sweep and must be disabled. Queueing discipline differs between systems and changes the latency distribution shape at equal throughput.

Rollback criteria. Do not adopt a system on a throughput figure whose latency constraint is unstated; require the frontier. If the frontiers cross, record the operating point that matters for this service and choose against that point explicitly, so the choice can be revisited when the requirement changes."""),
 ("STANCE 169 - Verify that both systems produced comparable output, because a throughput advantage obtained by generating different text is not a throughput advantage.",
  """Mechanism. Serving systems differ in default sampling parameters, stopping criteria, maximum token limits and prompt templating. Any of these can shorten generations and inflate requests per second while lowering tokens per second, or the reverse. Without checking what was produced, a comparison measures two different tasks. At temperature zero with identical model and tokenizer, the two systems should emit closely matching text, and a divergence is itself a finding.

Falsifiable hypothesis. H1: at temperature zero with identical model revision, tokenizer, prompt rendering and stopping rules, the two systems produce token sequences that agree on the large majority of requests. Falsified if outputs diverge materially, which means the throughput comparison is invalid until the cause of divergence is identified and eliminated.

Metrics. Exact-match rate of generated token sequences between systems at temperature zero, generated tokens per request in each system, finish-reason distribution, truncation rate, prompt token counts after each system's templating, and fixed eval-set quality for both. Output agreement and token counts are MEASURED; any explanation for divergence is an ESTIMATE until isolated by a controlled change.

Controlled experiment. Run the pinned trace at temperature zero through both systems and compare outputs token by token before comparing any performance metric. Where divergence appears, bisect on the candidate causes by aligning one element at a time, starting with prompt rendering and stopping rules, since those change the task rather than the numerics.

Confounders. Numeric differences across kernels produce small divergence that is not a configuration defect and must be distinguished from templating differences by inspecting where sequences first differ. Default maximum token limits differ and silently truncate. Special-token handling in templating changes prompt length and therefore prefill cost.

Rollback criteria. Do not report a performance comparison while output agreement is below the pre-declared threshold; fix the alignment first and re-run. If a decision was already taken on a comparison with unverified outputs, re-run the agreement check before acting further, since the entire result may be an artifact of one system doing less work."""),
]

RISKS_501 = [
 ["Source answer lists draft and target latency as measurements without deriving the break-even acceptance rate they imply.",
  "The target forward at speculative width is one batched pass, so a naive per-token cost model overstates the achievable saving.",
  "Draft and target sharing a device makes their latencies non-additive under concurrency."],
 ["Source answer asks for acceptance rate by position but not for the first-rejection distribution that determines yield.",
  "Sampling temperature changes acceptance because verification compares distributions, so an unpinned temperature invalidates the curve.",
  "Implementations that accept a corrected token after the first rejection use a different yield definition."],
 ["Source answer treats speculation as workload-independent, although its benefit inverts between idle and saturated regimes.",
  "Speculation changes memory footprint and can reduce maximum batch size, confounding the throughput comparison.",
  "A single-stream measurement cannot locate the concurrency at which the benefit changes sign."],
]
EVID_501 = [
 ["Per-step draft and target latencies recorded separately on identical hardware, seeds and batch policy.",
  "Break-even acceptance rate derived from the measured latency ratio with its derivation shown, across a proposal-width sweep."],
 ["Per-position accept and reject outcomes yielding the first-rejection distribution and realised accepted tokens per step.",
  "Width sweep showing whether yield saturates, with temperature and draft checkpoint pinned and recorded."],
 ["Tokens per second and latency measured across a load sweep from idle to objective breach with speculation enabled and disabled.",
  "Crossover concurrency at which the sign of the benefit changes, with maximum batch size recorded under both configurations."],
]

RISKS_502 = [
 ["Source answer frames the observation as a defect without testing whether it is the expected memory-bound signature.",
  "Comparing achieved bandwidth against a datasheet peak rather than a measured attainable peak misleads the diagnosis.",
  "Quantisation and prefix caching change bytes per token independently of the phase being studied."],
 ["Source answer asks to separate weight and KV reads but does not state that they scale oppositely with batch and context.",
  "Grouped-query attention changes the KV head count and shifts the balance substantially.",
  "Paged allocators read at block granularity, so short sequences read more than their token count implies."],
 ["Source answer treats low compute utilisation as evidence of a bandwidth limit without excluding device idle time.",
  "Bandwidth averaged over the whole step is depressed by idle gaps and falsely suggests headroom.",
  "Tracing adds overhead and can inflate the measured idle fraction unless tracer inertness is established."],
]
EVID_502 = [
 ["Attainable peak bandwidth from a streaming microbenchmark on the same device and build, compared against decode achieved bandwidth.",
  "Batch-size sweep showing bytes per token falling as weights amortise, with precision and prefix-cache state pinned."],
 ["Bytes per decode step split into weight and KV traffic by instrumentation or by differencing controlled configurations.",
  "Two sweeps holding context fixed while varying batch and holding batch fixed while varying context, with head configuration reported."],
 ["Kernel timeline giving summed kernel time against wall-clock step time and the resulting device idle fraction.",
  "Achieved bandwidth recomputed over active kernel windows only, before and after enabling graph capture or fusion."],
]

RISKS_503 = [
 ["Source answer freezes many configuration elements but does not pin the request trace that determines throughput.",
  "Tokenizer differences change token counts for identical text, making tokens per second non-comparable.",
  "Differing stopping criteria mean one system generates fewer tokens per request by design."],
 ["Source answer asks for a reproducible comparison without requiring a throughput-versus-latency frontier.",
  "A client-side load generator can saturate before the server and silently cap the measured curve.",
  "Queueing discipline differs between systems and changes the latency distribution at equal throughput."],
 ["Source answer compares performance without requiring that both systems produced comparable output.",
  "Default maximum token limits differ between systems and silently truncate generations.",
  "Kernel-level numeric differences produce divergence that must be distinguished from templating differences."],
]
EVID_503 = [
 ["Pinned request trace with per-request prompt and generation token counts and arrival timestamps, published as a raw artifact.",
  "Repeated runs establishing a run-to-run variation band before any difference is interpreted, with generated tokens reported separately from prompt tokens."],
 ["Throughput and full latency distribution recorded at each offered load level up to the objective breach for both systems.",
  "Verification that the load generator is not the bottleneck and that autoscaling was disabled for the duration of the sweep."],
 ["Token-by-token output comparison at temperature zero with identical model revision, tokenizer, prompt rendering and stopping rules.",
  "Finish-reason and truncation-rate distributions for both systems, with the first divergence position recorded where outputs differ."],
]

CONF = [0.73, 0.72, 0.74]

STANCES = {}
for fam, bodies, risks, evid in (
    (501, FAM_501, RISKS_501, EVID_501),
    (502, FAM_502, RISKS_502, EVID_502),
    (503, FAM_503, RISKS_503, EVID_503),
):
    STANCES[fam] = [
        (head, body, QD, risks[i], evid[i], CONF[i])
        for i, (head, body) in enumerate(bodies)
    ]
