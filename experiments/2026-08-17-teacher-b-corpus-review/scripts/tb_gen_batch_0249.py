import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0249.jsonl"
START, END = 2480, 2490

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
 ("STANCE 61 - Redundant calculator calls are a KV-cache economics problem before they are a policy problem: price the defect in prefill GPU-seconds or the intervention will never be prioritized correctly.",
  """Mechanism. Each redundant tool call forces one extra observation append and one extra decode turn. On a disaggregated vLLM or NVIDIA Dynamo deployment the append invalidates nothing but does extend the prefix, so the next turn re-prefills only the delta when prefix caching is healthy, and re-prefills the whole prompt when the block was evicted between turns. The cost of a redundant call is therefore bimodal: cheap on a cache hit, roughly linear in full prompt length on a miss. Any single average latency number hides this bimodality and will misprice the intervention.

Falsifiable hypothesis. H1: the prefill GPU-seconds attributable to redundant calls is dominated by cache-miss turns, with at least 70% of the wasted GPU-seconds coming from under 30% of the redundant calls. Falsified if wasted GPU-seconds are near-uniform across redundant calls, which would mean cache residency is not the lever and only reducing call count helps.

Metrics. Per redundant call: prefix cache hit or miss, prefilled token count, prefill GPU-seconds, KV block reuse depth. Aggregate: unnecessary-call rate (UCR), wasted prefill GPU-seconds per completed task, p95 end-to-end task latency, tool success rate, final answer correctness on the tool-required stratum, mean trajectory length, and recovery rate after a failed tool call. Cost figures are ESTIMATE unless read from serving counters; when derived, state the derivation as prefilled_tokens x measured tokens-per-GPU-second on the same GPU class, not a vendor spec sheet number.

Controlled experiment. Two arms on one frozen checkpoint, identical decode config and identical hardware pool, replayed against the same logged request window. Arm A is production policy. Arm B adds a stop/no-tool option surfaced by a system-prompt clause plus a decode-time constraint that suppresses the tool-call token when the answer branch already dominates. Pin autoscaling, pin the cache eviction policy, and record the KV block size, because changing block size between arms silently changes the hit rate and invalidates every cost delta.

Confounders. Traffic mix shifts change the natural cache hit rate independent of policy. Warm-up effects inflate arm A if it runs first, so interleave the arms. Shorter trajectories from arm B raise the hit rate on their own, which means part of any observed GPU-second saving is a mechanical consequence of fewer turns and must be decomposed rather than credited to better tool judgment.

Rollback criteria. Revert if tool-required-stratum accuracy drops by more than 1 absolute point, if recovery rate after a failed tool call falls at all, or if p95 latency regresses. Keep the previous system prompt and constraint config as a single flag flip and rehearse the revert before the experiment, not after."""),
 ("STANCE 62 - Treat the calculator as an untrusted dependency: the correct intervention is a cheap verifier in front of the tool, not a smarter policy behind it.",
  """Mechanism. The policy cannot know that the answer is already known, because that fact lives in the trajectory context, not in the tool interface. A verifier placed between the policy and the tool sees both the proposed call arguments and the already-observed values, and can short-circuit an exact repeat deterministically. This converts a probabilistic policy defect into a deterministic dedupe, at the cost of a false-suppression risk when arguments are similar but not identical.

Falsifiable hypothesis. H1: at least 60% of redundant calculator calls are exact-argument repeats of a call already present in the trajectory, so a pure dedupe verifier with no model in the loop removes the majority of them. Falsified if under 30% are exact repeats, which would mean the redundancy is semantic rather than syntactic and dedupe is the wrong mechanism.

Metrics. Verifier precision and recall against adjudicated redundancy labels, false-suppression rate, suppressed-call count, unnecessary-call rate, tool success rate, final correctness on the tool-required stratum, trajectory length, tool p95 latency, and recovery rate. Also track verifier added latency, which is the boundary condition: a verifier costing more than the tool call it suppresses is a net loss.

Controlled experiment. Run the verifier in shadow mode first, logging what it would suppress without suppressing it, over a fixed logged window. Measure precision, recall and the counterfactual correctness impact by adjudicating a stratified sample of would-be suppressions. Only after shadow precision clears the pre-declared bar do you enable enforcement, and then only on a canary slice with the same request mix.

Confounders. Argument normalization choices, such as whitespace and float formatting, change the exact-repeat rate substantially and must be fixed and version-pinned before the shadow run. Adjudicator drift across sessions inflates apparent precision, so use blind duplicate adjudication on a subsample and report inter-rater agreement.

Rollback criteria. Revert enforcement if false-suppression exceeds the pre-declared ceiling, if tool-required-stratum accuracy drops beyond the noise band established from control-versus-control replay, or if verifier p95 added latency exceeds the measured tool call p95. The verifier must be a single config flag with the classifier version recorded in every log line so a revert is unambiguous."""),
 ("STANCE 63 - Redundancy is a reward-shaping artifact: if the training signal never priced a call, no inference-time patch will hold.",
  """Mechanism. If the outcome reward credits only final correctness, calling a tool is free in expectation and weakly positive under uncertainty, so the policy converges to calling whenever it is cheap. Redundant calls are then the rational equilibrium of the objective, not a bug in the decoder. Inference-time suppression fights that equilibrium and will regress the moment traffic shifts, because the underlying preference ordering is unchanged.

Falsifiable hypothesis. H1: introducing an explicit per-call cost term into the preference or reward signal reduces the unnecessary-call rate durably, and the reduction persists on a strictly later held-out traffic window, whereas a prompt-only intervention decays by more than half over the same window. Falsified if both interventions decay at the same rate, which would mean the drift is driven by traffic composition rather than by objective misspecification.

Metrics. Unnecessary-call rate over time on a rolling later window, tool success rate, final correctness on the tool-required stratum, trajectory length, tool latency, recovery rate after tool failure, plus a stop/no-tool evaluation set where the correct behavior is to answer without calling. Report the cost-term coefficient and the accuracy-versus-UCR frontier rather than a single operating point, because the coefficient trades exactly along that frontier.

Controlled experiment. Three arms trained from the same base checkpoint with identical data order and seeds: control, prompt-only suppression, and preference optimization with a per-call cost term. Evaluate all three on the same frozen eval harness, including a strictly later held-out window collected after training data cutoff. Pre-register the coefficient sweep and the analysis plan commit hash so the frontier is not selected post hoc.

Confounders. Reward hacking, where the policy answers without calling on tool-required items to avoid the cost, is the dominant failure mode and is invisible unless the tool-required stratum is reported separately. Evaluation harness drift between training rounds silently moves the frontier. Sample size on the tool-required stratum is usually the binding constraint, so state realized power, not just planned power.

Rollback criteria. Revert to control if tool-required-stratum accuracy falls below the pre-declared floor, if recovery rate after a failed tool call degrades, or if the stop/no-tool eval improves while overall task success drops, which is the signature of cost-term overshoot. Keep the control checkpoint served behind a flag so the revert is a routing change, not a retrain."""),
 ("STANCE 64 - Measure redundancy per decision opportunity, not per trajectory: rate denominators chosen for convenience will manufacture false wins.",
  """Mechanism. Unnecessary calls per trajectory falls automatically when trajectories get shorter, regardless of whether judgment improved. The invariant quantity is the fraction of decision points at which the policy chose to call while the answer was already derivable from context. Normalizing by decision opportunities decouples the metric from trajectory-length side effects and makes any intervention that merely truncates trajectories visibly neutral.

Falsifiable hypothesis. H1: at least a third of the apparent reduction reported under a per-trajectory denominator disappears under a per-decision-opportunity denominator for interventions that shorten trajectories. Falsified if the two denominators track each other within the control-versus-control noise band, which would mean denominator choice is not load-bearing for this defect.

Metrics. Primary: redundant calls divided by adjudicated decision opportunities. Secondary: per-trajectory unnecessary-call rate reported alongside it for comparability with historical dashboards, mean and p95 trajectory length, tool success rate, tool latency, final correctness on the tool-required stratum, and recovery rate. Every rate must carry its denominator inline in the dashboard, since a rate without its denominator is not auditable.

Controlled experiment. Recompute both denominators over the same logged window for the same arms, then run a control-versus-control replay with identical seeds to establish the noise band for each metric before comparing any intervention against it. The manipulated variable in the first pass is only the denominator, so any divergence is arithmetic, not behavioral.

Confounders. Adjudicating what counts as a decision opportunity is itself a judgment call and drifts across annotators, so freeze the rubric and version it. Trajectories truncated by turn limits create censored denominators that bias the rate downward. Retries after transient tool failures inflate the denominator without representing genuine judgment, so classify them separately.

Rollback criteria. This intervention is a measurement change, so the rollback is a dashboard revert with historical series preserved under both denominators. Block any downstream policy rollout that was justified solely by a per-trajectory improvement until it is reproduced under the per-decision-opportunity denominator with a confidence interval that excludes the noise band."""),
 ("STANCE 65 - Separate can-answer from should-answer: the policy needs a calibrated self-estimate of answer availability, and calibration is what you measure.",
  """Mechanism. A redundant call happens when the policy's internal estimate of whether the answer is already available is miscalibrated upward on uncertainty. The tractable target is not the call decision but the calibration curve of that estimate. If the estimate is well calibrated, any threshold gives a controllable point on the precision-recall frontier; if it is miscalibrated, no threshold is stable and every deployment retunes it by hand.

Falsifiable hypothesis. H1: the answer-availability estimate is systematically underconfident, with expected calibration error above 0.1 and the bias concentrated in the high-availability bins where redundant calls cluster. Falsified if expected calibration error is under 0.05 and residual redundancy persists, which would locate the defect in the decision rule rather than in the estimate.

Metrics. Expected calibration error and reliability diagram on held-out adjudicated items, redundant-call rate at each threshold, false-suppression rate at each threshold, tool success, final correctness on the tool-required stratum, trajectory length, tool latency, and recovery rate. Any threshold recommendation must be reported with the calibration set size, because a threshold fitted on a few hundred items is an ESTIMATE with a wide interval, not a MEASURED operating point.

Controlled experiment. Fit calibration on one time-disjoint window and evaluate on a strictly later one, never on a random split of the same window, since trajectory-level correlation across a shared window leaks. Compare a temperature-scaled estimate against the raw estimate at matched redundancy-reduction levels, so the comparison is at equal effect size rather than at equal threshold.

Confounders. Trajectory-level correlation makes item-level intervals far too narrow unless clustered by trajectory. Distribution shift between calibration and evaluation windows is the main reason field thresholds decay. Adjudication noise puts a ceiling on measurable calibration error that must be estimated from duplicate labels and reported.

Rollback criteria. Revert the threshold to the previous value if false suppression exceeds the pre-declared ceiling on the tool-required stratum, or if the reliability diagram measured in production diverges from the offline one beyond the pre-declared band, which indicates the calibration window no longer represents live traffic. Record the classifier and threshold version in every log line so the revert is verifiable after the fact."""),
 ("STANCE 66 - Run the intervention as a staged rollout with a re-measured control at every rung, because harness drift, not the policy, explains most multi-week gains.",
  """Mechanism. Redundancy metrics are computed by an adjudication and logging pipeline that changes underneath the experiment: prompt templates, tool schemas, retry policy and log sampling all move. A single pre-period baseline compared against a late post-period therefore mixes policy effect with pipeline drift. Carrying a re-measured control arm at each rung makes drift observable as a moving control, and the intervention effect becomes the difference against the contemporaneous control rather than against a stale number.

Falsifiable hypothesis. H1: the contemporaneous control's unnecessary-call rate itself moves by more than half the claimed intervention effect over the rollout window. Falsified if the control stays inside its replay noise band across all rungs, which would license the simpler pre-post comparison for this system.

Metrics. Per rung: marginal and cumulative unnecessary-call rate for both arms, tool success, tool p95 latency, final correctness on the tool-required stratum, trajectory length, recovery rate after tool failure, and stop/no-tool eval score. Also log rung entry and exit timestamps, traffic share, and the config hash of prompt, tool schema and decode settings, since an unrecorded config change makes the whole series uninterpretable.

Controlled experiment. Ramp traffic share in pre-declared rungs, holding the control arm at a fixed nonzero share throughout. At each rung, first verify control-versus-control stability against the replay noise band, then evaluate the treatment difference. Do not advance a rung on a point estimate alone; require the interval to exclude the noise band, and record the interim-look count so the significance claim accounts for repeated testing.

Confounders. Traffic composition drifts by hour and day, so rungs of unequal duration are not comparable. Shared caches between arms leak effects across them, particularly prefix caches keyed on prompt content. Novelty effects in early rungs inflate the initial delta.

Rollback criteria. Halt and revert the current rung if tool-required-stratum accuracy drops beyond the pre-declared floor, if recovery rate degrades, or if the control arm itself moves outside its noise band, since that means the measurement is untrustworthy and no rollout decision can be made. Each rung must have an exercised revert record, meaning the revert path was actually executed at least once in staging, not merely documented."""),
 ("STANCE 67 - Fix the tool schema before touching the model: an idempotent, result-caching calculator interface makes most redundancy harmless and the rest visible.",
  """Mechanism. If the calculator endpoint is content-addressed, so that the same expression returns a cached result with an explicit cache-hit marker, then a repeated call costs a lookup instead of a compute and, more importantly, self-labels as redundant in the trace. This moves detection out of adjudication and into the tool layer, where it is deterministic. The residual redundancy after this change is exactly the semantically-equivalent-but-syntactically-different subset, which is the part that genuinely requires a model-side fix.

Falsifiable hypothesis. H1: after making the tool idempotent with an explicit cache-hit marker, the tool-side cache-hit rate agrees with adjudicated exact-repeat redundancy within 5 absolute points, making the tool layer a usable ground-truth source. Falsified if agreement is worse than 15 points, which would indicate argument normalization is capturing something different from what adjudicators call redundant.

Metrics. Tool cache-hit rate, adjudicated redundancy rate on a stratified sample, agreement between the two, tool p95 latency split by hit and miss, unnecessary-call rate, tool success rate, final correctness on the tool-required stratum, trajectory length, and recovery rate. Report cache memory footprint and eviction rate, because a cache that thrashes reintroduces the compute cost it was meant to remove.

Controlled experiment. Deploy the cache in a passive mode that records hits but still recomputes, so correctness is provably unchanged while the hit-rate series is collected. Compare recomputed and cached results byte-for-byte to prove idempotence on live traffic before enabling the fast path. Then enable the fast path on a canary slice with the same request mix and confirm identical final answers on a replayed window.

Confounders. Nondeterministic tools, such as anything depending on time, randomness or external state, break idempotence and will silently return stale results. Argument normalization aggressiveness trades hit rate against correctness. Cache key collisions across tenants are a correctness and isolation hazard and must be keyed with a tenant identifier.

Rollback criteria. Disable the fast path immediately if any byte mismatch appears between cached and recomputed results during the passive comparison, if eviction rate exceeds the pre-declared ceiling, or if p95 latency fails to improve, which would mean the cache is pure overhead. The passive mode must remain available as the fallback state."""),
 ("STANCE 68 - Report an accuracy-versus-redundancy frontier, not a single operating point, or you will trade correctness away without noticing.",
  """Mechanism. Every mechanism that reduces redundant calls also removes some calls that were load-bearing. The intervention therefore does not have a scalar effect; it has a curve. Reporting one operating point conceals where on that curve the system sits and makes two interventions with different curves look interchangeable. The decision-relevant object is the frontier of tool-required-stratum accuracy against unnecessary-call rate, swept over the intervention's own strength parameter.

Falsifiable hypothesis. H1: the frontiers of the prompt-only and verifier-based interventions cross, so neither dominates and the correct choice depends on the accuracy floor the product accepts. Falsified if one frontier dominates the other at every swept level beyond the noise band, in which case the dominated intervention should be dropped outright.

Metrics. Swept over the strength parameter: unnecessary-call rate, tool-required-stratum accuracy, overall task success, false-suppression rate, trajectory length, tool p95 latency, and recovery rate after a failed tool call. Each point carries a confidence interval clustered by trajectory. Any GPU-cost figure attached to a frontier point is an ESTIMATE derived from measured prefill token counts times measured tokens-per-GPU-second, and must be labeled as such rather than presented as a billing number.

Controlled experiment. Sweep the strength parameter over a pre-declared grid on a frozen checkpoint and a frozen eval harness, evaluating every grid point on the identical logged window with identical seeds. Include the unmodified control as a grid point so the frontier is anchored. Pre-register the grid and the analysis plan commit hash; a grid extended after seeing results turns the frontier into a selection artifact.

Confounders. Selection of the operating point after seeing test results is the dominant bias and is why the grid must be pre-declared. Small tool-required strata give intervals so wide that crossing frontiers are indistinguishable from noise, so report realized sample size per grid point. Harness drift between grid points, if they are run over days rather than in one pass, moves the whole curve.

Rollback criteria. Do not ship any operating point whose tool-required-stratum accuracy interval overlaps the pre-declared floor. If the shipped point later drifts across the floor in production, revert to the control grid point, which must remain deployable as a single parameter change with no retrain required."""),
 ("STANCE 69 - Distinguish redundancy from retry: transient tool failures and NCCL-style timeouts generate call patterns that look identical in logs and require the opposite fix.",
  """Mechanism. A call that repeats after a timeout, a truncated response or a transport-level error is a retry, and suppressing it degrades recovery. A call that repeats after a successful, already-observed result is redundancy, and suppressing it is the goal. In logs both appear as consecutive calls with similar arguments. Without a status-classified observation record, any dedupe or suppression rule will silently attack the recovery path, which is the more expensive failure.

Falsifiable hypothesis. H1: at least 15% of apparent redundant calls are retries following a non-success observation, so a naive dedupe rule that ignores observation status measurably degrades recovery rate. Falsified if retries are under 3% of apparent redundancy, which would make status-blind dedupe acceptable for this workload.

Metrics. Observation status distribution, retry rate, recovery rate after a failed tool call, adjudicated redundancy rate excluding retries, unnecessary-call rate, tool success rate, tool p95 latency split by success and failure, final correctness on the tool-required stratum, and trajectory length. Recovery rate is the guardrail metric and must be reported at every stage of any suppression rollout.

Controlled experiment. Instrument the tool boundary to emit an explicit status for every observation, then partition the historical apparent-redundancy population by status. Run a shadow-mode dedupe rule in two variants, status-blind and status-aware, over the same logged window and compare their would-be suppressions against adjudicated labels, measuring how many retries the status-blind variant would have killed.

Confounders. Client-side retries hidden inside the tool SDK never reach the agent log and make retries invisible at the policy layer. Timeouts that eventually succeed are logged as success and miscount. Load-dependent failure rates mean the retry fraction measured under light load underestimates production, so measure under representative load.

Rollback criteria. Revert any suppression rule to status-aware or to fully off if recovery rate after a failed tool call declines at all relative to the contemporaneous control, since recovery is a strict guardrail rather than a tradeable metric. Keep the rule behind a flag with the rule version stamped in every suppressed-call log line."""),
 ("STANCE 70 - Pre-register the analysis plan and publish the null results, because redundancy metrics are noisy enough that unregistered analysis will always find an effect.",
  """Mechanism. Unnecessary-call rate is a low-base-rate metric computed over correlated trajectories with adjudication noise, and it admits many defensible analysis choices: denominator, stratum, retry handling, outlier trimming, and interim-look timing. Each choice moves the point estimate by an amount comparable to the effects being claimed. Fixing every choice before data collection is the only mechanism that makes the resulting interval mean what it says.

Falsifiable hypothesis. H1: the spread of point estimates across defensible-but-unregistered analysis variants on the same dataset exceeds the claimed intervention effect. Falsified if the multiverse spread is under a third of the effect, which would indicate the finding is robust to analyst degrees of freedom and pre-registration is a formality here.

Metrics. Primary metric with its denominator, stratum definitions, retry handling rule, planned sample size and stopping rule, all fixed in a committed analysis plan. Reported outputs: primary unnecessary-call difference with a trajectory-clustered confidence interval, every pre-declared secondary metric including tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate, and the multiverse spread across the pre-declared robustness variants.

Controlled experiment. Commit the analysis plan and record its commit hash before the first data point is collected. Run the two arms to the planned sample size, honoring the stopping rule and logging every interim look. Then execute the registered analysis first and only afterwards any exploratory analysis, labeled as exploratory in the write-up.

Confounders. Optional stopping inflates false positives in proportion to the number of unlogged looks. Adjudicator drift over a long collection window shifts the base rate. Post hoc stratum definitions are the single most effective way to manufacture a result and must be frozen in the plan. Realized sample size below plan is common and must be reported alongside planned, since it silently widens every interval.

Rollback criteria. Do not ship on an exploratory finding. If the registered primary analysis fails to exclude the noise band established by control-versus-control replay, the result is null and must be published as null rather than re-cut. Any shipped intervention must retain a single-flag revert, and the revert must be exercised in staging before rollout so the rollback path is demonstrated rather than assumed."""),
]

DECISIONS = ["rewrite"] * 10

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

CONF = [0.72,0.7,0.68,0.74,0.69,0.71,0.67,0.73,0.7,0.72]

RISKS = [
 ["Source answer is a grading rubric, not an answer, and gives no mechanism for why redundant calls arise.",
  "Costing redundancy by average latency hides the bimodal cache-hit versus cache-miss distribution and misprices the intervention.",
  "Shorter trajectories mechanically raise prefix cache hit rate, so GPU-second savings can be credited to tool judgment that did not improve."],
 ["Source answer names metrics without specifying where the intervention sits in the call path.",
  "Exact-argument dedupe suppresses semantically necessary recomputation when normalization is too aggressive.",
  "A verifier costing more latency than the tool call it suppresses is a net regression that aggregate UCR will not reveal."],
 ["Source answer treats reward or preference signal as an add-on rather than as the cause of the equilibrium.",
  "Adding a per-call cost term invites reward hacking where the policy skips genuinely required calls.",
  "Prompt-only interventions decay under traffic shift and the decay is invisible without a strictly later held-out window."],
 ["Source answer specifies rates without specifying denominators, which is where this defect's measurement bias lives.",
  "Per-trajectory denominators reward trajectory truncation and manufacture apparent improvement.",
  "Turn-limit truncation censors denominators and biases the rate downward."],
 ["Source answer does not separate the policy's answer-availability estimate from the call decision rule.",
  "Thresholds fitted on small calibration sets are unstable and decay under distribution shift.",
  "Item-level intervals computed without trajectory clustering are far too narrow and overstate calibration quality."],
 ["Source answer implies a single before-and-after comparison, which confounds policy effect with harness drift.",
  "Shared prefix caches between arms leak effects across the treatment boundary.",
  "Unlogged interim looks invalidate the significance of any rung-advance decision."],
 ["Source answer places all responsibility on the policy and ignores the tool interface as an intervention point.",
  "Tools with time, randomness or external state dependence are not idempotent and caching them returns stale results.",
  "Cache keys without tenant scoping create cross-tenant correctness and isolation hazards."],
 ["Source answer asks for a single set of measurements rather than a swept tradeoff curve.",
  "Choosing the operating point after seeing test results is a selection artifact that inflates the reported gain.",
  "Small tool-required strata make crossing frontiers indistinguishable from noise."],
 ["Source answer lists recovery as one metric among many without recognizing that retries and redundancy are confusable in logs.",
  "Status-blind dedupe attacks the recovery path, which is a more expensive failure than the redundancy it removes.",
  "SDK-internal retries never reach the agent log, so retry rate measured at the policy layer underestimates the true rate."],
 ["Source answer requires a falsifiable hypothesis but does not constrain the analyst degrees of freedom that determine the result.",
  "Optional stopping with unlogged interim looks inflates false positives in proportion to the number of looks.",
  "Post hoc stratum definitions can manufacture an effect of the same magnitude as the one being claimed."],
]

EVID = [
 ["Per-redundant-call prefix cache hit or miss flag, prefilled token count, and prefill GPU-seconds read from serving counters rather than estimated from spec sheets.",
  "Interleaved two-arm replay on one frozen checkpoint with pinned autoscaling, pinned KV block size, plus p95 task latency and tool-required-stratum accuracy per arm."],
 ["Shadow-mode verifier precision, recall and false-suppression rate against blind duplicate-adjudicated labels with inter-rater agreement reported.",
  "Verifier p95 added latency compared against measured tool call p95, with the argument normalization rule version pinned and recorded."],
 ["Three-arm training run from one base checkpoint with identical data order and seeds, evaluated on a frozen harness including a strictly later held-out window.",
  "Accuracy-versus-UCR frontier over a pre-registered cost-coefficient sweep, with tool-required-stratum accuracy and stop/no-tool eval reported separately."],
 ["Both denominators recomputed over the same logged window, with each dashboard rate displaying its denominator inline.",
  "Control-versus-control replay with identical seeds establishing the per-metric noise band before any intervention comparison."],
 ["Reliability diagram and expected calibration error fitted on one time-disjoint window and evaluated on a strictly later one, with trajectory-clustered intervals.",
  "Threshold sweep reporting false-suppression rate and tool-required-stratum accuracy at matched redundancy-reduction levels, with calibration set size stated."],
 ["Per-rung marginal and cumulative UCR for treatment and a contemporaneous control held at fixed nonzero traffic share, with config hashes recorded per rung.",
  "Logged interim-look count, rung entry and exit timestamps, and an exercised revert record demonstrating the rollback path was actually run."],
 ["Passive-mode byte-for-byte comparison of cached versus recomputed tool results on live traffic, proving idempotence before the fast path is enabled.",
  "Agreement between tool-side cache-hit rate and adjudicated exact-repeat redundancy on a stratified sample, plus cache eviction rate and memory footprint."],
 ["Pre-registered strength-parameter grid with the analysis plan commit hash, evaluated on one frozen checkpoint and one frozen logged window with identical seeds.",
  "Per-grid-point tool-required-stratum accuracy, false-suppression rate and realized sample size with trajectory-clustered confidence intervals."],
 ["Explicit per-observation status emitted at the tool boundary, with apparent-redundancy population partitioned by status.",
  "Shadow-mode comparison of status-blind versus status-aware dedupe over the same window, quantifying retries the status-blind variant would have suppressed."],
 ["Committed analysis plan with its commit hash, planned sample size, stopping rule and stratum definitions fixed before data collection begins.",
  "Registered primary UCR difference with trajectory-clustered interval, all pre-declared secondary metrics reported in full, and the multiverse spread across robustness variants."],
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
