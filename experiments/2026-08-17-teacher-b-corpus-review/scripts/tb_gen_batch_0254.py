import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0254.jsonl"
START, END = 2530, 2540

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

# Rows 0-7 are the redundant-calculator template; rows 8-9 are the sparse-MoE template.
STANCES = [
 ("STANCE 111 - Replay the tool from its recorded responses rather than calling it live, so tool nondeterminism and drift are not counted as policy variance.",
  """Mechanism. A live tool introduces its own variability into every arm: latency fluctuates, transient errors occur, and results can change if the tool consults any mutable state. That variance enters the measured behaviour because the policy conditions on the observation it receives. Serving recorded responses keyed on the canonicalised argument tuple removes the tool as a source of variance, leaving the policy and the harness as the only moving parts.

Falsifiable hypothesis. H1: replaying with recorded tool responses reduces the control-versus-control variance of the unnecessary-call rate by more than a quarter relative to live-tool replay. Falsified if the two variances are indistinguishable, which would mean tool variance is negligible here and live replay is acceptable.

Metrics. Control-versus-control variance of the unnecessary-call rate under recorded and live tool modes, tool response cache miss rate during replay, share of calls whose arguments were unseen in the recording, tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. Both variances are MEASURED from repeated runs; any claimed sensitivity gain is an ESTIMATE until the intervention arm is run under both modes.

Controlled experiment. Record tool responses over a logged window, then run the same item set twice in each mode with recorded seeds. Arguments absent from the recording must be surfaced as explicit misses rather than silently forwarded to the live tool, because a partial recording reintroduces the variance it was meant to remove and does so non-uniformly across arms.

Confounders. Recorded latency is replayed as a constant, which removes a real source of behavioural pressure and can therefore understate redundancy driven by timeouts. An intervention that changes which arguments are requested increases the miss rate in one arm only, biasing the comparison. Recordings age as the tool changes, so the recording's tool version must be pinned and reported.

Rollback criteria. Discard any comparison whose miss rate differs materially between arms and re-record to cover the intervention's argument distribution before re-running. Do not promote a result obtained under constant replayed latency to a claim about timeout-driven behaviour; that claim requires the degraded-tool sweep run separately."""),
 ("STANCE 112 - Attribute the cost to the team whose traffic creates it, because an unowned platform cost produces no fix regardless of how well it is measured.",
  """Mechanism. Redundant calls consume shared prefill capacity and tool quota that appear on the platform's bill rather than on the originating team's. The team that could change the prompt, the scaffold or the tool schema therefore sees no signal, while the team that sees the signal cannot change the behaviour. This is a measurement-to-incentive gap, and it is testable by checking whether cost is reported at the granularity at which the behaviour is controlled.

Falsifiable hypothesis. H1: redundant-call cost is currently reported only at platform granularity, and attributing it to originating teams changes the ranking of cost drivers enough to move at least one team's prioritisation. Falsified if per-team attribution reproduces the platform-level ranking and no owner's priorities change, which would make chargeback a reporting change with no behavioural consequence.

Metrics. Redundant-call GPU-seconds and tool invocations attributed per originating team, the share of total cost that is attributable versus unattributable, the ranking of teams before and after attribution, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate. Attributed GPU-seconds are an ESTIMATE unless read from per-caller accounting counters, and the derivation must be shown.

Controlled experiment. Join call records to an owner via a caller identifier already present in the request path, compute attributed cost for a full billing period, and compare the resulting ranking against the platform-level view. Report the unattributable residual explicitly rather than distributing it proportionally, since proportional allocation manufactures precision that the join does not support.

Confounders. Shared scaffolds and shared prompts mean several teams contribute to one behaviour, so single-owner attribution overstates responsibility. Caller identifiers are frequently absent on internal traffic, producing a large residual. Cost rates are internal allocations rather than marginal costs, so an attributed saving may free no capacity.

Rollback criteria. Do not enforce quotas or budgets on attributed figures whose unattributable residual exceeds the pre-declared ceiling; publish the attribution as advisory until the join coverage improves. Any quota introduced later must be revertible by a single per-team parameter and must not be allowed to reduce tool-required-stratum accuracy for the throttled team."""),
 ("STANCE 113 - Use pre-period covariates to reduce variance before asking for more traffic: the same window can detect a smaller effect if the estimator conditions on prior behaviour.",
  """Mechanism. Trajectories differ enormously in their propensity to call tools, and that propensity is largely stable and observable before the experiment starts. An estimator that ignores it spends most of its variance budget on between-unit heterogeneity that has nothing to do with the treatment. Conditioning on a pre-period covariate, such as each cohort's historical call rate, removes that component and shrinks the interval without collecting a single additional request.

Falsifiable hypothesis. H1: the pre-period call rate explains a material share of the outcome variance, so a covariate-adjusted estimator narrows the confidence interval by more than a tenth relative to the unadjusted difference. Falsified if the covariate is uncorrelated with the outcome and the interval is unchanged, which would mean the adjustment adds complexity for nothing.

Metrics. Correlation between pre-period and in-period call rates, unadjusted and adjusted effect estimates with their intervals, realised variance reduction, minimum detectable effect under each estimator, and the standard behavioural set of tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate. The variance reduction is MEASURED on held-out data; the projected traffic saving it implies is an ESTIMATE.

Controlled experiment. Fix the covariate definition and the adjustment procedure before looking at in-period outcomes, then compute both estimators on the same data and report both. Validate on a control-versus-control replay first, where the true effect is zero, to confirm the adjusted estimator is unbiased rather than merely narrower.

Confounders. A covariate computed from a period that overlaps the treatment absorbs part of the effect and biases the estimate toward zero. Units without pre-period history are dropped, which selects for tenured cohorts. Choosing the covariate after seeing outcomes is a specification search that invalidates the interval.

Rollback criteria. Report the unadjusted estimate alongside the adjusted one always, and treat a disagreement between them beyond the noise band as a signal that the adjustment is misspecified rather than as a better answer. If the control-versus-control check shows a nonzero adjusted effect, abandon the adjustment for this design."""),
 ("STANCE 114 - Declare an alpha-spending rule before monitoring the experiment continuously, because repeatedly checking a running comparison manufactures significance.",
  """Mechanism. Rollouts are watched on dashboards, and the decision to stop is taken when the numbers look convincing. Each look is an opportunity to cross a threshold by chance, so the realised false-positive rate under continuous monitoring is far above the nominal one. This is a property of the stopping rule rather than of the data, and it is the most common way a null intervention gets shipped in a system that is otherwise well instrumented.

Falsifiable hypothesis. H1: under the current informal stopping practice, a control-versus-control comparison monitored continuously crosses the nominal significance threshold at least once with high probability. Falsified if repeated null replays rarely cross the threshold, which would indicate the effective number of looks is small enough that the naive test is adequate.

Metrics. Number of looks taken, realised crossing rate of control-versus-control comparisons under the same monitoring cadence, nominal versus realised false-positive rate, the pre-declared alpha-spending schedule, the boundary at each look, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length and recovery rate. The realised crossing rate is MEASURED by replaying null comparisons, not assumed.

Controlled experiment. Run a set of control-versus-control comparisons through the identical monitoring and stopping process used for real rollouts, counting how often a stop would have been triggered. Then re-run the same process under a sequential boundary with a pre-declared spending schedule and confirm the crossing rate matches the intended level.

Confounders. Looks taken informally, for example glancing at a dashboard without recording a decision, are invisible to the accounting yet still influence stopping. Correlated metrics multiply the effective number of tests when any of several dashboards could trigger a stop. Early looks on small samples have wide intervals that invite over-interpretation.

Rollback criteria. Any rollout stopped early without a pre-declared boundary is treated as inconclusive rather than positive, and the change reverts to control pending a properly bounded re-run. The spending schedule and the look count must be recorded before the first look, and an unrecorded look invalidates the analysis rather than being retrofitted into it."""),
 ("STANCE 115 - Pre-declare the segments before reporting heterogeneous effects, because post-hoc subgroup discovery will always find a segment where the intervention worked.",
  """Mechanism. The population is naturally divisible along many axes: task class, tenant, session length, model version, tool type. Searching those axes after seeing the aggregate result guarantees a favourable subgroup, and the resulting claim is unfalsifiable because the hypothesis was formed from the same data used to test it. Pre-declaring a small segment set converts subgroup reporting from advocacy into measurement.

Falsifiable hypothesis. H1: the intervention effect differs across the pre-declared segments by more than the within-segment noise bands, establishing genuine heterogeneity rather than a single average effect. Falsified if all segment effects lie within one band of the pooled effect, which supports reporting the pooled number and abandoning segment-level claims.

Metrics. Effect and interval per pre-declared segment with per-segment denominators, the pooled effect, a formal interaction test, the number of segments declared, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate. Segment effects are MEASURED but must be reported with multiplicity-adjusted intervals; any unadjusted segment claim is an ESTIMATE of unknown reliability.

Controlled experiment. Register the segment list and the interaction test before the arms are run, sized so each segment has a pre-declared minimum denominator. Report every declared segment including those with null or negative effects, since selective reporting of the declared set reintroduces exactly the bias the pre-declaration was meant to remove.

Confounders. Segments correlate with each other, so an apparent effect in one may be a shadow of another and requires joint modelling to separate. Small segments have wide intervals that will occasionally look large. Segment membership can itself be affected by the treatment when the segmenting variable is measured in-period.

Rollback criteria. Do not deploy to a segment on the strength of a segment-level result that fails the multiplicity-adjusted threshold; treat it as a hypothesis for a future pre-registered study. If deployment was already scoped by a post-hoc segment, revert to the pooled scope and re-declare the segments before any further targeting."""),
 ("STANCE 116 - Check whether the arms interfere through shared caches and pools, because two arms on one cluster are not independent units.",
  """Mechanism. Standard experimental analysis assumes one unit's assignment does not affect another's outcome. On a shared serving cluster that assumption fails structurally: the arms share a prefix cache, a KV pool, an admission queue and a tool rate limit. An arm that emits fewer calls frees capacity that the other arm consumes, so the control is improved by the treatment and the measured difference understates the true effect, or is distorted in a direction that depends on which resource binds.

Falsifiable hypothesis. H1: running the arms on isolated cache and pool resources changes the measured effect by more than the noise band relative to running them co-resident. Falsified if isolated and co-resident measurements agree, which would license the cheaper co-resident design for this system.

Metrics. Effect measured under co-resident and isolated configurations, cross-arm prefix cache sharing rate, KV pool occupancy by arm, tool rate-limit headroom consumed by arm, admission queue depth by arm, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. Sharing rates are MEASURED from cache and pool instrumentation rather than inferred from latency.

Controlled experiment. Run the comparison twice, once with arms co-resident and once with arm-dedicated cache and pool resources at matched capacity per arm, and compare the two effect estimates. Matching capacity per arm is essential, since dedicating resources also changes the per-arm capacity and would otherwise introduce a second difference.

Confounders. Isolation changes batch composition, and under continuous batching that alone shifts latency and cost. Dedicated pools are typically smaller, which raises queueing independently of the intervention. Tool-side rate limits may be global regardless of serving isolation, leaving one interference channel open even in the isolated configuration.

Rollback criteria. Prefer the isolated estimate when the two disagree, and state the co-resident estimate as the lower bound rather than discarding it. Return dedicated resources to the shared pool immediately after the experiment by the recorded configuration change, since standing dedicated capacity is a durable cost incurred for a temporary purpose."""),
 ("STANCE 117 - Prove the instrumentation is inert before trusting anything it reports, because adding logging to the decision path changes timing and can change behaviour.",
  """Mechanism. The telemetry required to adjudicate redundancy sits on the hot path: it serialises arguments, hashes them and writes records. That work adds latency between turns, can alter scheduling and batch formation, and in some frameworks introduces synchronisation. If the instrumented and uninstrumented systems behave differently, then every rate the instrumentation reports describes a system that only exists while it is being measured.

Falsifiable hypothesis. H1: enabling the instrumentation leaves final outputs byte-identical and p95 latency within the noise band on a fixed replay. Falsified if outputs differ or latency shifts beyond the band, which means the instrument perturbs the system and its readings must be corrected or the instrument redesigned.

Metrics. Byte-level output identity rate between instrumented and uninstrumented replays at temperature zero, p95 and p99 latency under each, per-call instrumentation overhead in microseconds, dropped-record rate, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length and recovery rate. Output identity and overhead are MEASURED; any correction applied to reported rates because of a detected perturbation is an ESTIMATE and must be labelled.

Controlled experiment. Replay a fixed item set at temperature zero with instrumentation off and on, comparing outputs token by token before comparing any rate. Then repeat under representative load, since an overhead invisible at idle can matter when it lands inside a batching window. Record the dropped-record rate, because sampling under load makes the instrument's own coverage load-dependent.

Confounders. Asynchronous logging hides overhead from the request path but can still contend for CPU and cause scheduling jitter. Buffered writers drop records precisely under the load conditions of most interest. Comparing at temperature zero removes sampling noise but also removes the regime where a small timing change could flip a sampled decision.

Rollback criteria. Disable the instrumentation by a single flag if output identity fails, and treat all rates collected under the perturbing version as provisional pending re-collection with a corrected instrument. Instrumentation must be sampled rather than removed only after the inertness check passes, so that coverage reduction is a deliberate choice rather than a silent failure under load."""),
 ("STANCE 118 - Build a minimal deterministic reproduction before touching production, because an intervention that cannot be demonstrated on a small failing case is not yet understood.",
  """Mechanism. The defect is currently described statistically, as a rate over heterogeneous traffic. A rate is a poor debugging instrument: it cannot distinguish several mechanisms with the same aggregate signature, and it forces every hypothesis test to be an expensive fleet-level experiment. A minimal case that reproduces the redundant call deterministically at fixed seed converts the problem into one that can be examined directly and iterated on in seconds.

Falsifiable hypothesis. H1: there exists a prompt, tool schema and seed under which the policy emits a redundant call deterministically across repeated runs at temperature zero. Falsified if no such case can be constructed after a bounded search, which would indicate the behaviour is genuinely stochastic or context-dependent and must be studied statistically.

Metrics. Reproduction rate of the minimal case across repeated runs and across seeds, number of context elements that can be removed while preserving the behaviour, sensitivity of the behaviour to each removed element, and, once an intervention exists, whether it removes the behaviour in the minimal case. Reproduction rate is MEASURED; transfer from the minimal case to fleet traffic is an ESTIMATE until measured on the fleet.

Controlled experiment. Start from a logged trajectory exhibiting the defect, then reduce it by removing turns, context elements and tool schema fields one at a time, keeping the reduction only when the behaviour persists across repeated runs. The final case is the minimal context that triggers the behaviour, and the removal log is itself a sensitivity analysis identifying which elements matter.

Confounders. A minimal case is by construction unrepresentative and may exercise a rare mechanism rather than the dominant one, so it must never be used to estimate a rate. Temperature-zero determinism can be broken by nondeterministic kernels or batch-composition effects, so determinism must be verified rather than assumed. Reduction can accidentally cross into a different mechanism that produces a superficially similar output.

Rollback criteria. Do not generalise from the minimal case: any intervention validated on it must still pass the fleet-level comparison before rollout, and a fleet-level failure overrides the minimal-case success. Keep the minimal case as a regression test pinned to the checkpoint and serving build on which it was verified, and re-verify it after any of those change rather than assuming it still reproduces."""),
 ("STANCE 119 - Report expert load as a distribution over experts and microbatches rather than an average, and count dropped tokens as a correctness change rather than a capacity statistic.",
  """Mechanism. In a sparse MoE layer the router assigns tokens to experts, and each expert has a capacity per microbatch set by the capacity factor. Tokens exceeding capacity are dropped or padded. An average load figure can look healthy while individual experts overflow on particular microbatches, and the dropped tokens never reach their selected expert, so the model computes a different function than the one that was evaluated offline. That is a silent correctness change presented as a utilisation number.

Falsifiable hypothesis. H1: at the deployed capacity factor, the tail of the per-expert per-microbatch load distribution produces a non-zero token-drop rate, and raising the capacity factor removes those drops at a measurable all-to-all and memory cost. Falsified if the drop rate is already zero across the observed batch composition, which would mean capacity is not binding and the tail latency has another cause.

Metrics. Per-expert token counts per microbatch with their full distribution rather than the mean, capacity factor, token-drop and padding rates, all-to-all time per layer, per-device compute and memory, end-to-end p95 and p99 latency, and output-quality checks on a fixed eval set to detect the correctness effect of drops. Drop rates and all-to-all times are MEASURED from instrumentation; any projected quality effect of a capacity change is an ESTIMATE until the eval set is re-run.

Controlled experiment. Sweep the capacity factor on a fixed replayed batch stream, holding placement, batch composition and parallelism fixed, and record the drop rate, all-to-all time and eval-set quality at each point. Because drops depend on batch composition, repeat the sweep on at least two contrasting compositions rather than a single representative stream.

Confounders. Batch composition drives routing skew, so a change in traffic mix looks like a routing change. Padding hides drops from throughput counters while still consuming compute. Capacity changes shift memory footprint and can trigger a different attention or communication path, altering latency for reasons unrelated to routing.

Rollback criteria. Revert any capacity-factor change if eval-set quality declines or if p99 latency breaches the objective, and treat a non-zero drop rate on the deployed configuration as a correctness defect requiring disclosure rather than as an acceptable efficiency trade. The prior capacity factor must be a single revertible serving parameter recorded with the model version."""),
 ("STANCE 120 - Measure imbalance at the device and link level, not the expert level, because the tail is set by the slowest rank and placement decides which rank that is.",
  """Mechanism. Experts are placed onto devices, and several experts commonly share one device. A perfectly balanced expert-level histogram can still produce severe device-level imbalance if the busiest experts are co-located. Because the all-to-all is a collective, every rank waits for the slowest, so the step time is determined by the worst device rather than by the average expert. Expert-level metrics therefore cannot predict the tail that operators actually experience.

Falsifiable hypothesis. H1: device-level load imbalance exceeds expert-level imbalance under the current placement, and re-placing the co-located hot experts reduces p99 step time by more than the run-to-run noise band without changing the router. Falsified if device-level and expert-level imbalance coincide, which would mean placement is neutral and the imbalance originates in routing.

Metrics. Per-device token counts and busy time, per-link all-to-all bytes and time, the ratio of maximum to mean device load, rank wait time at the collective, p99 step time, achieved interconnect bandwidth, and end-to-end p95 and p99 service latency. Device and link figures are MEASURED from profiler and fabric counters; any predicted gain from a proposed placement is an ESTIMATE until the placement is run.

Controlled experiment. Profile one fixed replayed batch stream, recording per-device busy time and per-link transfer alongside the per-expert histogram, then compute both imbalance ratios from the same trace. Change only the expert-to-device mapping, holding the router, capacity factor, batch composition and parallelism fixed, and re-measure. Repeat on a second batch composition to check that the placement is not overfitted to one traffic mix.

Confounders. Topology is heterogeneous, so two devices are not equidistant and a balanced token count can still produce unbalanced link time. Other tenants sharing the fabric contend for bandwidth, which appears as imbalance. Profiler overhead perturbs the very step times being measured, so profiled and unprofiled step times must be compared before the trace is trusted.

Rollback criteria. Revert the placement to the recorded prior mapping if p99 step time or service latency regresses, or if the improvement does not reproduce on the second batch composition, since a placement tuned to one traffic mix will regress when the mix shifts. Placement must be expressed as a versioned configuration artifact so the revert is exact rather than reconstructed."""),
]

DECISIONS = ["rewrite"] * 10

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

CONF = [0.71,0.68,0.72,0.73,0.72,0.7,0.73,0.69,0.74,0.73]

RISKS = [
 ["Source answer does not remove the tool as an independent source of variance in the comparison.",
  "Recorded latency replayed as a constant understates redundancy driven by timeouts.",
  "An intervention that changes the requested argument distribution raises the recording miss rate in one arm only."],
 ["Source answer measures a platform cost without attributing it to the team able to change the behaviour.",
  "Shared scaffolds mean several teams contribute to one behaviour, so single-owner attribution overstates responsibility.",
  "Missing caller identifiers on internal traffic produce a large unattributable residual."],
 ["Source answer plans a comparison without using available pre-period information to reduce variance.",
  "A covariate window overlapping the treatment absorbs part of the effect and biases the estimate toward zero.",
  "Choosing the covariate after seeing outcomes is a specification search that invalidates the interval."],
 ["Source answer does not constrain how often the running comparison may be inspected before stopping.",
  "Informal dashboard looks influence stopping while remaining invisible to any alpha accounting.",
  "Correlated dashboards multiply the effective number of tests that could trigger a stop."],
 ["Source answer permits subgroup claims without pre-declaring the segments to be reported.",
  "Small segments have wide intervals that will occasionally appear large by chance.",
  "Segment membership measured in-period can itself be affected by the treatment."],
 ["Source answer treats the two arms as independent units although they share cache, pool and rate-limit resources.",
  "Freed capacity from the treatment arm improves the control arm, distorting the measured difference.",
  "Dedicated pools are smaller and raise queueing independently of the intervention."],
 ["Source answer relies on hot-path telemetry without establishing that the telemetry does not change behaviour.",
  "Asynchronous logging hides overhead from the request path while still causing scheduling jitter.",
  "Buffered writers drop records precisely under the load conditions of greatest interest."],
 ["Source answer studies the defect only in aggregate, where several mechanisms share one statistical signature.",
  "A minimal case is unrepresentative by construction and may exercise a rare mechanism rather than the dominant one.",
  "Temperature-zero determinism can be broken by nondeterministic kernels or batch-composition effects."],
 ["Source answer lists routing distribution and capacity overflow as metrics without requiring the full per-microbatch distribution.",
  "Averaged expert load can look healthy while individual experts overflow and drop tokens on particular microbatches.",
  "Padding conceals drops from throughput counters while still consuming compute."],
 ["Source answer treats placement as one item in a list rather than as the factor that determines which rank sets the tail.",
  "A balanced expert histogram can still produce severe device-level imbalance when hot experts are co-located.",
  "Profiler overhead perturbs the step times being measured, and fabric contention from other tenants appears as imbalance."],
]

EVID = [
 ["Control-versus-control variance of the unnecessary-call rate measured under both recorded-tool and live-tool replay modes.",
  "Explicit recording miss rate per arm with the recording's tool version pinned, and unseen arguments surfaced rather than forwarded live."],
 ["Per-team attributed GPU-seconds and tool invocations joined on the caller identifier already present in the request path.",
  "Unattributable residual reported explicitly rather than allocated proportionally, with the cost-rate derivation recorded."],
 ["Correlation between pre-period and in-period call rates computed on held-out data with the covariate definition fixed in advance.",
  "Unadjusted and adjusted estimates reported together, validated on a control-versus-control replay where the true effect is zero."],
 ["Realised crossing rate of control-versus-control comparisons run through the identical monitoring and stopping process.",
  "Pre-declared alpha-spending schedule with the boundary at each look and a recorded look count taken before the first look."],
 ["Pre-registered segment list with per-segment minimum denominators and a formal interaction test.",
  "Effects and multiplicity-adjusted intervals reported for every declared segment including null and negative results."],
 ["Effect estimated under co-resident and arm-isolated cache and pool configurations at matched per-arm capacity.",
  "Cross-arm cache sharing rate, KV pool occupancy and tool rate-limit headroom read from cache and pool instrumentation."],
 ["Byte-level output identity between instrumented and uninstrumented replays at temperature zero on a fixed item set.",
  "Per-call overhead and dropped-record rate measured under representative load as well as at idle."],
 ["Reduction log recording which context elements could be removed while the behaviour persisted across repeated runs.",
  "Reproduction rate across seeds with the checkpoint and serving build pinned, plus fleet-level confirmation before any rollout."],
 ["Per-expert per-microbatch token-count distribution with token-drop and padding rates at the deployed capacity factor.",
  "Capacity-factor sweep on at least two contrasting batch compositions with all-to-all time and fixed eval-set quality recorded at each point."],
 ["Per-device busy time and per-link all-to-all bytes and time from profiler and fabric counters on one fixed batch stream.",
  "Maximum-to-mean device load ratio and rank wait time before and after an expert-to-device remapping, repeated on a second batch composition."],
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
