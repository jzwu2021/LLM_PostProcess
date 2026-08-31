import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0252.jsonl"
START, END = 2510, 2520

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
 ("STANCE 91 - Report the rate as a function of the decode configuration: temperature and nucleus mass move the tool-call token probability, so a fix validated at greedy decoding may not survive production sampling.",
  """Mechanism. The choice to emit a tool call is a token-level sampling event. At greedy decoding the decision is determined by the argmax, while under nucleus sampling the tail mass on the call token is drawn with some probability at every opportunity, and that probability compounds over a long trajectory. An intervention that shifts the logit margin will therefore look decisive at temperature zero and partial at production temperature, and the difference is a property of the decode configuration rather than of the policy.

Falsifiable hypothesis. H1: unnecessary-call rate varies monotonically with sampling temperature across the production range, and the intervention effect measured at greedy decoding overstates the effect at production temperature by more than the replay noise band. Falsified if the rate is flat across the temperature sweep, which would mean the decision is far from the sampling boundary and the greedy result transfers.

Metrics. Unnecessary-call rate at each point of a pre-declared temperature and top-p grid, the tool-call token probability at adjudicated decision points, and the intervention effect measured separately at each grid point. Alongside these, tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. The per-grid-point rates are MEASURED; any single headline rate quoted without its decode configuration is an ESTIMATE of unknown scope.

Controlled experiment. Replay one frozen checkpoint over one logged window at each grid point with recorded seeds, running the control and intervention arms interleaved at every point rather than sweeping one arm before the other. Establish a control-versus-control band at each grid point separately, because sampling variance itself grows with temperature and a single pooled band would be wrong at both ends.

Confounders. Repetition and presence penalties interact with the call token specifically, since a repeated call is textually similar to the prior one, so those penalties must be pinned and reported. Seed reuse across grid points creates correlated draws that understate variance. Serving builds sometimes clamp temperature server-side, so the configured value and the effective value can differ and must both be logged.

Rollback criteria. Do not ship an intervention whose effect at the production decode configuration does not exclude the noise band at that configuration, regardless of its greedy-decoding result. If production decode settings later change, treat the intervention as unvalidated at the new setting and re-measure before relying on it; the decode configuration must therefore be pinned and version-recorded alongside the checkpoint."""),
 ("STANCE 92 - Attribute the defect to the earliest causal turn, not the turn where the redundant call appears: the failure is usually a retention decision made several turns earlier.",
  """Mechanism. A call is redundant because a value obtained earlier is no longer usable at the decision point. That unusability was created upstream, by a summarisation step that dropped the value, a rendering step that reformatted it, or a scaffold that never wrote it into state. Instrumenting only the turn that emits the call records the symptom. The causal turn is the last one at which the value was present and usable, and it is recoverable from the trajectory.

Falsifiable hypothesis. H1: for the majority of adjudicated redundant calls, the value was present in an earlier turn and absent or altered at the decision turn, locating the cause upstream of the call site. Falsified if the value is present and unaltered at the decision turn in most cases, which would place the cause in the policy's decision rather than in retention.

Metrics. Distance in turns between the last turn where the value was present and usable and the turn of the redundant call, the share of redundant calls with an identifiable upstream loss event, the loss event type broken out into summarisation, reformatting, eviction and scaffold-state omission, and the standard behavioural set of tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate.

Controlled experiment. Walk each adjudicated redundant call backwards through the rendered prompts and the scaffold state, recording at which turn the value ceased to be present in the form the policy could use. This is a deterministic trace analysis requiring no inference and produces a MEASURED attribution table. Then intervene on the single most common loss event and re-measure, holding every other component fixed.

Confounders. The value may be present in a semantically equivalent but textually different form, so the presence test must be defined and versioned rather than left to string matching. Scaffold state and rendered prompt can disagree, and only the rendered prompt is what the policy saw. Trajectories truncated by retention are missing their own upstream turns, so the attribution population is biased toward short trajectories.

Rollback criteria. Revert any retention or rendering change if final correctness or recovery rate degrades, since keeping more content has its own cost in context budget. Because the attribution table is the basis for the fix, invalidate the table and re-derive it whenever the summariser, renderer or scaffold changes version, rather than carrying it forward."""),
 ("STANCE 93 - Check the safety and content-filtering layer before the policy: a tool result that was blocked or rewritten in flight is invisible to the model, and calling again is the correct response to an empty observation.",
  """Mechanism. Production stacks interpose filters between the tool and the policy. A filter may redact a numeric value that pattern-matches a sensitive format, truncate an oversized response, or replace the payload with a refusal string. The policy then observes an absent or unusable result and calls again. This is a correct response to the observation it received, and no amount of training-side work will remove it, because the information never reached the model.

Falsifiable hypothesis. H1: the filter intervention rate on tool responses is non-negligible, and adjudicated redundant calls are concentrated in trajectories where a filter modified the preceding response. Falsified if redundant calls occur at the same rate in filtered and unfiltered trajectories, which would eliminate the filter path as an explanation.

Metrics. Filter intervention rate by filter rule and by action, covering redaction, truncation and replacement; unnecessary-call rate conditioned on whether the preceding response was filtered; the share of filtered responses that retained the answer in usable form; and the standard behavioural set of tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. Filter counts are MEASURED from the filter's own logs, not inferred from the policy's behaviour.

Controlled experiment. Join tool-call records to filter decision records on the call identifier and compare unnecessary-call rate across filtered and unfiltered strata on the same window, matching on task type and prompt length. Where a rule is implicated, run a two-arm replay in which that single rule is switched from replacement to a structured non-disclosing notice that states a value was withheld, holding everything else fixed.

Confounders. Filters fire more often on particular task types, so the filtered stratum differs in composition and must be matched. A filter that fails open under load produces a time-varying intervention rate. Filter logs and tool logs are written by different services with independent clocks, so joining on timestamp rather than call identifier will mis-associate records.

Rollback criteria. Any change to a filter rule is a safety-surface change and must be reverted immediately if the rate of disclosed sensitive values rises above zero, irrespective of its effect on redundancy. The structured notice must be validated to carry no residual payload, and the prior rule version must remain deployable behind a flag so the revert is a configuration change."""),
 ("STANCE 94 - Locate the intervention relative to the streaming commit boundary: once a tool-call token has been streamed to the client, suppression is no longer available and only cancellation is.",
  """Mechanism. In a streaming deployment tokens leave the server as they are produced. By the time an orchestrator-side suppressor can inspect a complete tool call, part of it has already been committed to the client and, in agent frameworks, possibly to the trace and to downstream consumers. Interventions that assume a clean pre-execution veto are therefore unimplementable in the streaming path, and the feasible set is limited to decode-time masking before commit or cancellation with compensating output after it.

Falsifiable hypothesis. H1: in the current serving configuration the median tool call is fully committed to the client before the orchestrator's suppression hook can execute, making post-hoc suppression infeasible without a visible retraction. Falsified if the hook consistently executes before commit within the measured margin, which would make orchestrator-side suppression a viable intervention point.

Metrics. Time from first tool-call token emission to commit, time from emission to suppressor decision, the fraction of calls where the decision precedes commit, retraction visibility rate at the client, unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length and recovery rate. The timing distributions are MEASURED from server traces; any claimed feasibility margin derived from them is an ESTIMATE until the hook is exercised end to end.

Controlled experiment. Instrument both timestamps on the same request path and measure the margin distribution under representative load rather than at idle, since queueing changes both terms. Then evaluate the two feasible intervention points, decode-time masking and post-commit cancellation, on the same window and compare their behavioural and client-visible effects directly.

Confounders. Client-side buffering can hide a retraction in testing while exposing it for other clients, so retraction visibility must be measured per client type. Non-streaming evaluation harnesses show a margin that does not exist in production. Load changes the margin, so a feasibility conclusion drawn at idle will not hold at peak.

Rollback criteria. Do not deploy post-commit cancellation if retraction is visible to any client type, because a visibly retracted action is a worse user-facing defect than a redundant call. Decode-time masking must be revertible by disabling a single logits processor, and that revert path must be exercised in staging under load before rollout."""),
 ("STANCE 95 - Price both the defect and the rollout against the service error budget, so this work competes on the same scale as every other change to the serving path.",
  """Mechanism. A service has a finite error budget, and every rollout consumes some of it in expectation. Redundant calls consume budget indirectly through latency and capacity pressure, while the experiments and rollouts proposed to remove them consume it directly through the risk of regression. Expressed in raw rates the two are incomparable, and the work gets prioritised by narrative. Expressed in error budget they are comparable to every other candidate change.

Falsifiable hypothesis. H1: the error budget consumed by the redundancy defect over a release period exceeds the expected budget consumed by the proposed rollout, making the change net positive on that scale. Falsified if the rollout's expected consumption equals or exceeds the defect's, which argues for accepting the defect until a cheaper intervention exists.

Metrics. Error budget consumed by latency-objective breaches attributable to redundancy, expected budget consumption of the rollout derived from the canary plan and historical rollout regression rates, remaining budget in the current period, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. Historical rollout regression rates are MEASURED; the expected consumption of this rollout is an ESTIMATE and must show its derivation.

Controlled experiment. Attribute historical objective breaches to causes using existing incident records and per-request telemetry, isolating the share associated with elevated redundancy. Separately, compute the rollout's expected consumption from the canary traffic fraction, the historical probability that a change of this class regresses, and the detection latency. Both figures are produced before the rollout, not after.

Confounders. Attributing breaches to a single cause is unreliable when several factors co-occur, so the attributed share must be reported as an interval. Error budget policy differs across service tiers, so a fleet-level figure hides tier-level exhaustion. Detection latency directly scales expected consumption and is often assumed rather than measured.

Rollback criteria. Halt the rollout when consumption reaches the pre-declared fraction of remaining budget, independent of the behavioural metrics, and revert to the prior configuration. If the period's budget is already exhausted, the change does not ship regardless of its expected value, and that decision is recorded with the figures that produced it."""),
 ("STANCE 96 - Measure under a degraded tool, because redundancy amplifies during tool slowdown and converts a partial dependency failure into a self-inflicted load spike.",
  """Mechanism. When the tool slows or returns errors, an agent disposed to call again does so more often, and each retry adds prefill work and occupies a concurrency slot for longer. The offered load therefore rises exactly when the dependency has least capacity, which is the classic retry-storm shape. A redundancy measurement taken while the tool is healthy cannot observe this regime, so the most operationally significant behaviour is missing from the evidence.

Falsifiable hypothesis. H1: under injected tool latency elevation, calls per task rise superlinearly with tool latency rather than remaining flat, demonstrating amplification. Falsified if calls per task is flat across the injected latency sweep, which would mean the policy does not amplify and the healthy-state measurement is sufficient.

Metrics. Calls per task and unnecessary-call rate as functions of injected tool latency and error rate, tool queue depth, concurrency slot occupancy, end-to-end p95 latency, deadline-exhaustion rate, and the standard behavioural set of tool success excluding injections, tool-required-stratum accuracy, trajectory length and recovery rate. The amplification slope is MEASURED from the sweep; any projected outage impact derived from it is an ESTIMATE.

Controlled experiment. Inject latency and error responses at the tool boundary at pre-declared levels in a replay environment, sweeping the level rather than testing a single degraded point, and confirm the injector is inert at zero. Run the candidate intervention across the same sweep, since an intervention that reduces redundancy in the healthy regime may or may not damp amplification in the degraded one, and those are separate claims.

Confounders. Client and SDK retry logic amplifies independently of the policy and must be disabled or accounted for separately. Circuit breakers change the regime discontinuously once tripped, so the sweep must record breaker state. Timeouts interact with the deadline budget, so a latency injection also shortens the effective budget for later turns.

Rollback criteria. Abort the injection immediately on any breach of the service objective or on breaker trip, and never run this sweep against production traffic. Ship an amplification-damping change only if it holds the healthy-state guardrails, and retain the prior configuration behind a flag so the revert is a single change exercised in staging first."""),
 ("STANCE 97 - Stratify by position within the session: the first call in a session and the twentieth are different decisions, and pooling them hides where the defect actually lives.",
  """Mechanism. Early in a session the context is short, the prefix cache is cold and there is no prior result to reuse, so a call is almost always legitimate. Later the context is long, prior results are present but distant, and cache warmth changes the cost profile. Redundancy is by construction impossible at the first opportunity and increasingly possible later, so a pooled rate is a weighted average over a mechanism that varies systematically with position.

Falsifiable hypothesis. H1: unnecessary-call rate rises with session position and the rise is not explained by trajectory length alone, indicating a position-specific mechanism such as distance to the prior result. Falsified if the rate is flat across position strata once length is controlled, which would justify pooling and redirect attention away from position.

Metrics. Unnecessary-call rate by session position decile, prior-result token distance at each decision point, prefix cache hit rate by position, tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. Position-stratified rates are MEASURED; the denominator for each stratum must be reported because late strata contain far fewer sessions.

Controlled experiment. Compute the stratified rates on a logged window, then control for trajectory length by comparing positions within length-matched cohorts so that long sessions do not dominate late strata. Where the rise persists, test the distance mechanism directly by re-injecting the prior result at a fixed near distance for a randomly assigned subset and comparing against a length-matched padding control.

Confounders. Session length is itself an outcome, since sessions that go wrong run longer, so conditioning on it can induce collider bias and the length-matched comparison must be pre-declared. Sessions are truncated by turn caps, censoring the latest positions. Users who reach late positions are a self-selected population with different task mixes.

Rollback criteria. Revert any re-injection change if context-budget pressure rises to the point that necessary earlier content is evicted, or if final correctness on long sessions declines. Because the change alters prompt assembly for late turns only, the prior assembler must remain available behind a flag and the revert exercised in staging under long-session traffic."""),
 ("STANCE 98 - Bound the telemetry before shipping it: per-call structured logging has a cardinality and retention cost that can exceed the savings it is meant to unlock.",
  """Mechanism. Every proposal here depends on richer per-call telemetry, and telemetry is not free. Labels such as tool name, argument hash, tenant, checkpoint and schema version multiply into a cardinality that drives index size, query cost and retention spend. Adding an unbounded label, most commonly a raw argument or a request identifier, converts a metrics system into an unindexed log store and degrades every other consumer of that system.

Falsifiable hypothesis. H1: the proposed label set produces a time-series cardinality below the pre-declared ceiling at production call volume, and the added storage and query cost is a small fraction of the oracle-strip savings bound. Falsified if projected cardinality exceeds the ceiling or the cost approaches the savings, in which case the telemetry must be sampled or aggregated before it ships.

Metrics. Projected and then measured active time-series count, ingestion volume per day, retention cost at the recorded rate, query latency for the analysis queries on the enlarged index, sampling rate applied, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate. The cardinality projection is an ESTIMATE and its derivation must be shown; the post-deployment cardinality is MEASURED.

Controlled experiment. Enable the full label set on a small pre-declared traffic sample first, measure realised cardinality and ingestion volume, and extrapolate to full volume rather than assuming linearity in the number of labels, since cardinality is multiplicative in the number of distinct label values. Then verify that the analysis queries return within their time budget at the projected index size.

Confounders. Cardinality growth is driven by the highest-cardinality label, so a single unbounded field dominates and averages across labels are misleading. Sampling reduces cost but also reduces the effective sample size of every downstream rate, which must be reconciled with the study's power requirement. Retention policy changes alter cost independently of the label set.

Rollback criteria. Disable the new labels by a single configuration change if cardinality exceeds the ceiling or if ingestion degrades unrelated dashboards, and treat any analysis performed during the excursion as provisional. Sampling rate must be recorded per record so that rates computed during a sampled period are correctly weighted rather than silently biased."""),
 ("STANCE 99 - Track deadline exhaustion rather than mean latency: the operational failure from extra calls is requests that run out of budget, and a mean cannot express that.",
  """Mechanism. Agent requests carry an end-to-end deadline that is consumed turn by turn. Each redundant call spends part of it on a round trip and on the prefill of a longer prompt. Requests near the boundary cross it and fail outright, while requests far from it are unaffected. The harm is therefore concentrated in a tail that a mean latency figure averages away, and the correct outcome measure is the rate at which requests exhaust their budget.

Falsifiable hypothesis. H1: removing redundant calls reduces the deadline-exhaustion rate by proportionally more than it reduces mean latency, showing the harm is concentrated at the boundary. Falsified if both move proportionally, which would mean mean latency is an adequate summary for this traffic and the tail framing adds nothing.

Metrics. Deadline-exhaustion rate, remaining budget at each turn boundary, the distribution of budget consumed per tool call, p95 and p99 end-to-end latency, mean latency reported alongside them for contrast, unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length and recovery rate. Budget consumption per call is MEASURED per request; any projected exhaustion reduction is an ESTIMATE until the arm is run.

Controlled experiment. Compute the counterfactual budget consumption by stripping provably redundant steps from logged trajectories and re-accounting the timeline, which bounds the achievable exhaustion reduction from above without running the model. Then run the intervention arm and compare the realised exhaustion rate against that bound and against a control-versus-control band.

Confounders. Deadlines differ by client and by endpoint, so a pooled exhaustion rate mixes populations with different boundaries. Requests that exhaust their budget are truncated and therefore under-report their own call counts, biasing the redundancy denominator downward exactly where the harm is greatest. Retries at the client level create a new request with a fresh deadline, hiding the failure from the server-side rate.

Rollback criteria. Revert the intervention if the exhaustion rate rises in any client segment, even if the pooled rate improves, because deadline behaviour is a contractual property per client. Deadline values themselves must not be changed as part of this work; loosening a deadline to reduce exhaustion would mask the defect rather than remove it."""),
 ("STANCE 100 - Match monitoring cadence to release cadence: a metric reviewed less often than the system changes cannot defend an improvement, whatever its statistical quality.",
  """Mechanism. Detection latency is the interval between a regression entering production and someone observing it. If releases occur weekly and the redundancy dashboard is reviewed monthly, then in expectation several releases carry an undetected regression, and by the time it is seen the causal release is ambiguous. The defensibility of an improvement is therefore bounded by detection latency, not by the rigour of the study that produced it.

Falsifiable hypothesis. H1: with the current review cadence, the expected number of releases between a regression's introduction and its detection exceeds one, making causal attribution to a single release impossible. Falsified if detection latency is shorter than the release interval, which would make each regression attributable and the cadence adequate.

Metrics. Release interval, dashboard review interval, measured detection latency for seeded regressions, number of releases within one detection latency, alert precision and recall on seeded regressions, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate. Detection latency is MEASURED by seeding, not estimated from intuition.

Controlled experiment. Seed a known regression of the smallest size worth defending into a pre-production or shadow path, and measure how long the existing monitoring takes to surface it and whether the alert fires at all. Repeat across a small set of magnitudes to obtain a detection curve rather than a single point, and record the false-alert rate over the same period so precision is measured alongside recall.

Confounders. Seeded regressions in shadow traffic may not reproduce the production distribution, so the detection curve is optimistic. Alert thresholds tuned during a quiet period fire constantly during a noisy one, and the resulting fatigue suppresses real detections. Dashboard review is a human process whose actual cadence differs from its nominal one.

Rollback criteria. If the detection curve shows that the smallest defensible regression is not caught within one release interval, do not claim the improvement is maintained; either raise the cadence to an automated per-release check or restate the improvement as unmonitored. Any alerting change must be revertible by configuration, and every suppressed or overridden alert must be logged so the override rate is auditable."""),
]

DECISIONS = ["rewrite"] * 10

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

CONF = [0.72,0.7,0.71,0.69,0.68,0.73,0.7,0.72,0.71,0.7]

RISKS = [
 ["Source answer prescribes a controlled experiment without pinning the decode configuration that determines the decision boundary.",
  "An effect validated at greedy decoding can overstate the effect at production sampling temperature.",
  "Server-side temperature clamping makes the configured value differ from the effective value."],
 ["Source answer instruments the turn that emits the call rather than the turn where the value was lost.",
  "A value present in a semantically equivalent but textually different form defeats naive presence testing.",
  "Scaffold state and rendered prompt can disagree, and only the rendered prompt is what the policy observed."],
 ["Source answer assumes the tool result reaching the policy is the result the tool produced.",
  "Filters fire more often on particular task types, so the filtered stratum differs in composition.",
  "Joining filter and tool records on timestamp rather than call identifier mis-associates records across services."],
 ["Source answer proposes suppression without checking whether the streaming path permits a pre-execution veto.",
  "Post-commit cancellation can surface as a client-visible retraction, which is a worse defect than the redundant call.",
  "Non-streaming evaluation harnesses show a feasibility margin that does not exist in production."],
 ["Source answer omits any accounting that makes this work comparable to other changes competing for the same risk budget.",
  "Attributing objective breaches to a single cause is unreliable when several factors co-occur.",
  "Detection latency scales the rollout's expected budget consumption and is usually assumed rather than measured."],
 ["Source answer measures only in the healthy regime, where amplification during tool degradation cannot appear.",
  "Client and SDK retry logic amplifies load independently of the policy and can be mistaken for policy behaviour.",
  "Circuit breakers change the regime discontinuously once tripped, invalidating a smooth sweep."],
 ["Source answer pools decisions across session positions where the redundancy mechanism cannot be uniform.",
  "Session length is itself an outcome, so conditioning on it can induce collider bias.",
  "Turn caps censor the latest positions, which is where the mechanism is strongest."],
 ["Source answer requires per-call telemetry without bounding its cardinality or cost.",
  "A single unbounded label converts a metrics system into an unindexed log store and degrades unrelated consumers.",
  "Sampling to control cost reduces the effective sample size of every downstream rate."],
 ["Source answer measures latency without measuring the deadline budget that actually determines request failure.",
  "Requests that exhaust their deadline are truncated and under-report their own call counts, biasing the denominator.",
  "Client-side retries create a fresh deadline and hide the failure from server-side rates."],
 ["Source answer treats the improvement as durable without stating how quickly a regression would be detected.",
  "Seeded regressions in shadow traffic may not reproduce the production distribution, making the detection curve optimistic.",
  "Alert thresholds tuned in a quiet period cause fatigue in a noisy one, suppressing real detections."],
]

EVID = [
 ["Unnecessary-call rate and tool-call token probability at every point of a pre-declared temperature and top-p grid, with penalties pinned.",
  "Interleaved control and intervention arms at each grid point with per-point control-versus-control bands and both configured and effective decode values logged."],
 ["Backward trace attribution table locating the last turn at which the value was present and usable, with a versioned presence test.",
  "Loss-event breakdown across summarisation, reformatting, eviction and scaffold omission, derived from rendered prompts rather than scaffold state alone."],
 ["Filter decision records joined to tool-call records on the call identifier, with intervention rate by rule and action.",
  "Unnecessary-call rate compared across filtered and unfiltered strata matched on task type and prompt length, plus a single-rule two-arm replay."],
 ["Measured distributions of emission-to-commit and emission-to-suppressor-decision times under representative load.",
  "Retraction visibility measured per client type, with decode-time masking and post-commit cancellation compared on the same window."],
 ["Historical objective-breach attribution with the redundancy-associated share reported as an interval, plus remaining budget for the period.",
  "Expected rollout consumption derived from canary fraction, historical regression rate for this change class and measured detection latency."],
 ["Calls per task and unnecessary-call rate across a pre-declared injected latency and error-rate sweep, with a zero-injection inertness check.",
  "Breaker state and SDK retry configuration recorded per sweep point, with deadline-exhaustion rate measured alongside queue depth."],
 ["Unnecessary-call rate by session-position decile with per-stratum denominators and prior-result token distance.",
  "Length-matched cohort comparison pre-declared before analysis, plus a randomised re-injection arm with a length-matched padding control."],
 ["Realised time-series cardinality and ingestion volume measured on a pre-declared traffic sample before full rollout.",
  "Analysis-query latency at the projected index size, retention cost at the recorded rate, and the applied sampling rate recorded per record."],
 ["Per-request remaining-budget timeline and deadline-exhaustion rate broken out by client and endpoint.",
  "Oracle-strip counterfactual re-accounting of the timeline bounding the achievable exhaustion reduction, compared against the realised arm."],
 ["Measured detection latency and alert recall for seeded regressions across a range of magnitudes, producing a detection curve.",
  "Release interval, actual dashboard review cadence, false-alert rate over the same period, and a log of every suppressed or overridden alert."],
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
