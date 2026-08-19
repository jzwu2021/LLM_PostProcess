import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0250.jsonl"
START, END = 2490, 2500

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
 ("STANCE 71 - Attribute the cost to the serving tier that actually pays it: in a disaggregated prefill/decode deployment redundant calls tax prefill capacity, not decode, and the two scale independently.",
  """Mechanism. In a disaggregated deployment such as NVIDIA Dynamo or a Mooncake-style prefill/decode split, an extra tool observation lengthens the prompt and forces additional prefill work, while the decode tier only pays for the handful of tokens in the call itself. The two tiers have separate pools, separate autoscalers and separate saturation points. Reporting a single aggregate GPU-hour figure therefore hides which pool the defect saturates, and capacity added to the wrong tier will not relieve the queue.

Falsifiable hypothesis. H1: redundant calls raise prefill-tier queue depth at a rate at least three times their effect on decode-tier queue depth, at fixed request rate. Falsified if both tiers move proportionally, which would mean the split does not change the capacity story and a single aggregate figure is adequate.

Metrics. Per tier: queue depth, GPU-seconds per completed task, utilization, and p95 tier-local latency. Cross-tier: KV transfer bytes and p95 transfer latency, prefix cache hit rate at the prefill tier. Behavior: unnecessary-call rate, tool success, final correctness on the tool-required stratum, trajectory length, tool latency and recovery rate. GPU-second attributions are ESTIMATE unless read from per-tier accounting counters; when derived, state the derivation as measured prefill tokens divided by measured tier throughput on the same GPU class.

Controlled experiment. Hold request rate, pool sizes and autoscaling fixed, then inject a controlled level of synthetic redundancy into a replayed logged window and measure the per-tier response. Sweep the induced redundancy level rather than comparing only zero and production levels, so the slope, not just a difference, is measured. Confirm the injector is behaviorally inert on final answers by comparing outputs at zero injection against the untouched control.

Confounders. Autoscaling reacting mid-experiment turns a capacity measurement into a control-loop measurement. Prefix cache warmth differs between arms and shifts prefill cost independently of redundancy. Cross-tier KV transfer over RDMA contends with other traffic on the same fabric, so transfer latency must be read with fabric counters, not inferred from end-to-end latency.

Rollback criteria. Stop the injection immediately if p95 end-to-end latency breaches the service objective or if prefill queue depth exceeds the pre-declared ceiling. Any capacity change made on the basis of this measurement must be revertible by returning pool sizes to the recorded prior values, and that revert must be exercised in staging first."""),
 ("STANCE 72 - The intervention must be evaluated against an adversarial stratum where the answer looks known but is not, otherwise you are measuring on the easy half of the problem.",
  """Mechanism. Suppression rules and calibrated thresholds are fitted on naturally occurring traffic, where genuinely ambiguous cases are rare. The failure mode that matters in production is the case that superficially resembles an already-answered question but requires recomputation, for example a value that was observed earlier but has since been invalidated by a later step. Without a deliberately constructed stratum of such cases, the measured false-suppression rate is an underestimate of unknown size.

Falsifiable hypothesis. H1: false-suppression rate on a constructed adversarial stratum is at least three times the rate measured on natural traffic at the same threshold. Falsified if the two rates agree within their intervals, which would mean natural traffic already covers the hard cases and the adversarial stratum adds no information.

Metrics. False-suppression rate on natural and adversarial strata reported separately and never pooled, tool-required-stratum accuracy, unnecessary-call rate, tool success, trajectory length, tool latency, recovery rate, and the stop/no-tool eval score. Report the adversarial stratum construction rule and its version, since the measured rate is meaningful only relative to a fixed construction procedure.

Controlled experiment. Construct the adversarial stratum by programmatic mutation of logged trajectories, invalidating a previously observed value in a later step while keeping surface form similar, then have adjudicators blind-label a sample to confirm the mutations are genuinely tool-required. Evaluate every candidate threshold on both strata in the same pass on a frozen checkpoint.

Confounders. Mutation artifacts can make adversarial items detectably synthetic, so the policy solves them by pattern rather than by reasoning; blind adjudication plus a human distinguishability check is needed. Over-representing the adversarial stratum in any pooled metric distorts headline numbers, which is why the strata must be reported separately. Adjudicator fatigue on repetitive mutated items degrades label quality.

Rollback criteria. Do not ship a threshold whose adversarial-stratum false-suppression rate exceeds the pre-declared ceiling, regardless of how good the natural-traffic number is. If production recovery rate or tool-required accuracy later drifts toward the adversarial prediction, revert the threshold to the prior value via the recorded single-flag change."""),
 ("STANCE 73 - Bound the claim by base rate: at a low unnecessary-call rate, the sample size needed to detect a meaningful reduction usually exceeds what the team plans to collect.",
  """Mechanism. Unnecessary-call rate is a proportion over decision opportunities, and its variance is driven by the base rate and by trajectory-level clustering. When the base rate is a few percent and outcomes correlate within a trajectory, the effective sample size is far smaller than the raw count of calls. A study powered on the raw count will report a wide interval that is then narrated as a trend, which is how unreplicable results enter the dashboard.

Falsifiable hypothesis. H1: the design effect from trajectory-level clustering exceeds 2, so the effective sample size is less than half the raw call count and the planned study is underpowered for the target effect. Falsified if the intra-trajectory correlation is near zero and the design effect is under 1.2, in which case the naive power calculation stands.

Metrics. Base rate with its denominator, intra-trajectory correlation, design effect, planned versus realized effective sample size, minimum detectable effect at the planned size, and the primary unnecessary-call difference with a trajectory-clustered confidence interval. Alongside these, the standard behavioral set: tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate.

Controlled experiment. Before running the comparison, estimate the intra-trajectory correlation from a historical window and compute the minimum detectable effect at the planned sample size. If the target effect is below that threshold, the correct action is to extend collection or abandon the study, not to run it and interpret the point estimate. Then run control-versus-control replay to confirm the empirical noise band matches the analytic prediction.

Confounders. Pooling across heterogeneous task types inflates apparent variance and can mask a real within-type effect. Truncated trajectories reduce the denominator unevenly across arms. Adjudication noise adds variance that does not shrink with more traffic, only with more adjudication, so it sets a floor on the detectable effect.

Rollback criteria. This is a design gate rather than a deployment, so the rollback is a decision rule: block any rollout justified by an interval that does not exclude the empirically measured control-versus-control noise band. If a rollout already happened under an underpowered result, revert it to control and re-run at adequate power before re-shipping."""),
 ("STANCE 74 - Log the tool-call decision with a stable schema version, because most retrospective analyses of this defect fail on schema drift rather than on statistics.",
  """Mechanism. The record of a tool call is produced by code that changes: argument serialization, observation truncation, status encoding and sampling rate all evolve. A retrospective query spanning that change silently mixes incompatible records, and the resulting time series shows a step that is attributed to policy behavior. Versioning the log schema and refusing to pool across versions makes the drift explicit rather than invisible.

Falsifiable hypothesis. H1: at least one schema change within the analysis window produces an apparent unnecessary-call rate step larger than the intervention effect being claimed. Falsified if the rate series is continuous across every schema boundary within the replay noise band, which would license pooling across versions for this window.

Metrics. Log schema version stamped on every record, records per version, unnecessary-call rate computed per version and never pooled without an explicit bridging analysis, log sampling rate, observation truncation rate, and the standard behavioral set of tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate.

Controlled experiment. Replay a fixed set of trajectories through each log schema version's parser and compare the derived metrics; any difference is parser-induced and quantifies the bridge. Then recompute the historical series with the bridge applied and check whether the previously observed steps persist. The manipulated variable is the parser, so behavioral differences are by construction artifacts.

Confounders. Sampling-rate changes alter which trajectories appear, biasing the population rather than the parsing. Truncated observations make an already-known answer unrecoverable from the log, causing adjudicators to mislabel a legitimate call as redundant. Backfills that rewrite historical records with the new schema destroy the evidence needed for the bridge, so backfills must be written to a sibling path.

Rollback criteria. If the bridging replay shows parser-induced differences exceeding the noise band, invalidate every conclusion drawn across that boundary and re-derive from the versioned series. Retain the prior parser as a runnable artifact so any recomputation is reproducible, and never delete the original raw records."""),
 ("STANCE 75 - Ask whether the calculator should exist in the loop at all: for a policy that can already do the arithmetic, the tool is a latency tax with a correctness ceiling.",
  """Mechanism. A tool is worth its round trip only when it is more accurate or more capable than the policy on the relevant input distribution. If the policy's own arithmetic accuracy on the observed input distribution matches the tool's, every call is pure latency and prefill cost. The decision-relevant comparison is therefore not how to reduce redundant calls but where the accuracy crossover between policy and tool lies as a function of input complexity.

Falsifiable hypothesis. H1: below a measurable input-complexity threshold, the policy's arithmetic accuracy is statistically indistinguishable from the tool's, so gating the tool above that threshold removes most calls at no accuracy cost. Falsified if policy accuracy is below tool accuracy across the whole observed complexity range, which would mean the tool is always justified and only redundancy, not usage, should be reduced.

Metrics. Policy-only and tool-assisted accuracy as functions of an explicit input-complexity measure such as operand count and magnitude, the crossover point with its confidence interval, unnecessary-call rate, tool success, final correctness on the tool-required stratum, trajectory length, tool p95 latency, and recovery rate. Any latency saving quoted is MEASURED from the replay, not inferred from the removed call count.

Controlled experiment. On a frozen checkpoint, evaluate the same items twice, once with the tool available and once with it disabled, holding decode config and seeds identical. Stratify results by the complexity measure and locate the crossover. Then run a gated arm that permits the tool only above the crossover and compare against both baselines on the same window.

Confounders. The complexity measure itself is a modeling choice and a poor one will smear the crossover. Silent arithmetic errors by the policy are not self-reported, so accuracy must be adjudicated against ground truth rather than against the policy's confidence. Tool errors and timeouts are attributed to the tool arm and must be counted, not dropped.

Rollback criteria. Revert to unconditional tool availability if gated-arm accuracy on the tool-required stratum falls below the pre-declared floor, or if the crossover estimate proves unstable across time windows, since an unstable crossover means the gate will silently mis-route as traffic shifts. The gate must be a single threshold parameter with its value logged per request."""),
 ("STANCE 76 - Instrument recovery as a first-class outcome with its own experiment, because every redundancy intervention trades against it and the trade is rarely measured.",
  """Mechanism. Redundant calling and robust recovery are produced by the same disposition: a willingness to call the tool again when the situation is uncertain. Suppressing that disposition necessarily reduces both. Recovery is only exercised when a tool call fails, which is rare in normal traffic, so a redundancy experiment run on natural traffic collects almost no recovery evidence and will report a guardrail that was never actually tested.

Falsifiable hypothesis. H1: under injected tool failures at a controlled rate, an intervention that reduces unnecessary-call rate by a given amount reduces recovery rate by a measurable amount, establishing a nonzero trade slope. Falsified if recovery rate is flat across the intervention strength sweep under injected failures, which would mean the two dispositions are separable in this policy.

Metrics. Recovery rate after an injected tool failure, time to recovery in turns, unnecessary-call rate, tool success excluding injected failures, final correctness on the tool-required stratum, trajectory length, and tool p95 latency. Injected failure rate and failure mode taxonomy, covering timeout, malformed response, and error status, must be reported since recovery differs sharply by mode.

Controlled experiment. Use fault injection at the tool boundary at a pre-declared rate and mode mix, applied identically across arms, on a frozen checkpoint replaying a fixed window. Sweep intervention strength and measure the recovery-versus-redundancy slope. Verify the injector is inert at zero injection rate by comparing outputs against the untouched control.

Confounders. Injected failures that are detectably synthetic, for example with a distinctive error string, let the policy special-case them and inflate recovery. Retries inside the tool SDK mask injected failures before the policy sees them. Recovery adjudication is subjective when the policy produces a partially correct answer, so the rubric must be frozen and duplicate-labeled.

Rollback criteria. Treat recovery as a strict guardrail, not a tradeable metric: revert any intervention whose recovery rate under injection falls below the pre-declared floor even if unnecessary-call rate improved substantially. Fault injection itself must be revertible by a single flag and must never be enabled on production traffic without a separate approved change."""),
 ("STANCE 77 - Compare against a trivial baseline first: a one-line system-prompt clause is the control that most interventions fail to beat, and skipping it inflates every reported gain.",
  """Mechanism. Reported effects for agent behavior changes are usually measured against the untouched production policy, which includes no instruction about redundant calling at all. A single system-prompt clause stating that the tool should not be called when the value is already present in context is nearly free and often captures a large share of the achievable reduction. Any complex intervention's incremental value is the difference against that clause, not against the empty control.

Falsifiable hypothesis. H1: the one-line prompt clause captures at least half the unnecessary-call reduction achieved by the full verifier or preference-optimization intervention. Falsified if the clause achieves under 15% of the reduction, which would justify the engineering cost of the complex path on effect size alone.

Metrics. Unnecessary-call rate for empty control, prompt-clause arm and complex arm, with the incremental difference between clause and complex arm reported as the headline rather than the difference from empty control. Alongside: tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency, recovery rate, and stop/no-tool eval score for all three arms.

Controlled experiment. Three arms on one frozen checkpoint, identical decode config, seeds and logged window, run interleaved rather than sequentially. Pin the prompt clause text and record its hash, since paraphrases of the same instruction produce materially different rates and an unrecorded paraphrase makes the arm irreproducible. Include a control-versus-control replay to establish the noise band.

Confounders. Prompt-clause effects interact with prompt length and position, so adding the clause also shifts prefix cache behavior and token budget. Instruction-following ability varies across model versions, so the clause's effect measured on one checkpoint does not transfer. Novelty in early traffic can inflate the clause arm.

Rollback criteria. If the complex arm's incremental gain over the prompt clause does not exclude the noise band, ship the clause and abandon the complex intervention rather than shipping both. Reverting the clause is a prompt config change with the prior text hash recorded, and it must be exercised in staging before rollout."""),
 ("STANCE 78 - State the boundary of transfer explicitly: results measured on one checkpoint, one tool schema and one traffic mix do not license a claim about the agent in general.",
  """Mechanism. The unnecessary-call rate is a joint property of the policy weights, the prompt, the tool schema, the decode configuration and the input distribution. Changing any one of these moves the rate. A result therefore has a scope defined by the tuple that produced it, and a claim stated without that tuple is unfalsifiable because it cannot be checked against a re-run. Recording the tuple converts a narrative into a reproducible measurement.

Falsifiable hypothesis. H1: holding the intervention fixed and changing only the tool schema's argument format moves the unnecessary-call rate by more than the intervention effect, demonstrating that the result does not transfer across schema variants. Falsified if the rate is stable within the noise band across schema variants, which would support a broader scope claim.

Metrics. Unnecessary-call rate reported jointly with checkpoint hash, prompt hash, tool schema version, decode configuration and traffic window identifier. Behavioral set as usual: tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency, recovery rate, and stop/no-tool eval. Every number in the write-up carries the tuple, and any number quoted without it is treated as an ESTIMATE of unknown scope.

Controlled experiment. Replicate the intervention across a small pre-declared grid of scope variants, changing exactly one element of the tuple at a time, on the same logged window with identical seeds. The output is a transfer table showing where the effect holds, not a single headline number. Include the original configuration as the anchor cell.

Confounders. Partial confounding when a schema change also changes prompt length, which alters cache behavior and token budget simultaneously. Checkpoint changes bundle many differences at once and cannot isolate a cause. Traffic windows from different periods differ in composition, so window identity must be held fixed while other elements vary.

Rollback criteria. Restrict deployment to the cells of the transfer table where the effect was measured and excluded the noise band; treat unmeasured cells as unsupported and gate them behind a separate rollout. If a production configuration drifts out of a measured cell, revert to the prior configuration until the cell is measured, using the recorded tuple as the revert target."""),
 ("STANCE 79 - Make the abstention path explicit in the action space: a policy with no first-class stop action expresses uncertainty by calling the tool again.",
  """Mechanism. If the only available actions are call-the-tool and emit-a-final-answer, then hesitation has nowhere to go except into another call. Adding an explicit low-cost action that states the answer is already available, or that asks a clarifying question, gives uncertainty a cheaper outlet. This is an action-space change rather than a threshold change, and it is testable by whether the new action absorbs probability mass that previously went to redundant calls.

Falsifiable hypothesis. H1: after adding an explicit stop action, at least 40% of previously redundant call decisions are replaced by the stop action rather than by a direct final answer, showing the mass was absorbed rather than merely suppressed. Falsified if the stop action is chosen in under 10% of those decisions, which would mean the action space was not the binding constraint.

Metrics. Action distribution at adjudicated decision points, stop-action usage rate, unnecessary-call rate, tool success, final correctness on the tool-required stratum, trajectory length, tool p95 latency, recovery rate after tool failure, and a stop/no-tool evaluation set where abstention is the correct behavior. Report stop-action precision, meaning how often stopping was actually correct, since an over-used stop action is a new defect rather than a fix.

Controlled experiment. Two arms on the same base checkpoint, identical seeds and window, differing only in the action space and the minimal prompt text needed to describe the new action. Adjudicate a stratified sample of decision points in both arms to attribute where the probability mass moved. Confirm the added action does not change tokenization of existing actions, which would silently perturb the control.

Confounders. Describing the new action lengthens the prompt, which alone shifts behavior; a length-matched placebo clause in the control arm isolates the action-space effect from the prompt-length effect. Adjudicating whether stopping was correct requires ground truth that is unavailable for open-ended tasks, so scope the analysis to items with verifiable answers.

Rollback criteria. Revert the action-space change if stop-action precision falls below the pre-declared floor, if tool-required-stratum accuracy declines, or if recovery rate degrades. Because this is a served-model configuration change rather than a weight change, retain the prior action space behind a routing flag and exercise the revert in staging before rollout."""),
 ("STANCE 80 - Close the loop with a standing regression gate: a one-time reduction that is not defended by a blocking evaluation will be undone by the next unrelated change.",
  """Mechanism. Unnecessary-call rate is not owned by any single component, so it degrades as a side effect of prompt edits, tool schema changes, decode tuning and checkpoint upgrades made for unrelated reasons. Without a blocking gate in the release path, the metric ratchets back toward its old value and no one is accountable, because each individual change moves it only slightly. A gate converts a one-time win into a maintained invariant.

Falsifiable hypothesis. H1: absent a blocking gate, unnecessary-call rate regresses to within a quarter of its pre-intervention level within a fixed number of unrelated releases. Falsified if the rate remains within the noise band of its post-intervention value across that release count, which would mean the improvement is structurally stable and a gate is unnecessary overhead.

Metrics. Gate pass and fail counts by release, unnecessary-call rate per release on a frozen eval set, tool-required-stratum accuracy, stop/no-tool eval score, recovery rate under injected failures, trajectory length, tool p95 latency, and gate runtime, since a gate slower than the release cadence will be bypassed and is therefore not a gate.

Controlled experiment. Run the gate in advisory mode across a pre-declared number of releases, recording what it would have blocked without blocking, and measure the realized regression on production traffic over the same period. Compare the advisory verdicts against the observed production drift to establish that the gate's frozen eval set predicts field behavior before it is given blocking authority.

Confounders. A frozen eval set decays in representativeness as traffic shifts, so its agreement with production must be re-measured periodically rather than assumed. Overfitting to the gate, where changes are tuned to pass it rather than to improve behavior, appears as gate scores improving while production drifts. Flaky gate infrastructure produces false blocks that erode trust and lead to routine overrides.

Rollback criteria. Demote the gate from blocking to advisory if its false-block rate exceeds the pre-declared ceiling or if its agreement with production drift falls below the pre-declared floor, and re-derive the eval set before re-promoting. The demotion must be a single config change, recorded with the gate version, and every override of a blocking gate must be logged with a justification so the override rate is itself auditable."""),
]

DECISIONS = ["rewrite"] * 10

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

CONF = [0.7,0.71,0.73,0.69,0.68,0.72,0.74,0.7,0.69,0.73]

RISKS = [
 ["Source answer aggregates cost without distinguishing the prefill and decode tiers that scale and saturate independently.",
  "Autoscaling reacting mid-experiment converts a capacity measurement into a control-loop measurement.",
  "Cross-tier KV transfer contends with other RDMA fabric traffic, so end-to-end latency misattributes fabric contention to the policy."],
 ["Source answer evaluates only on naturally occurring traffic, where genuinely ambiguous cases are rare.",
  "False-suppression rate measured on natural traffic underestimates the production rate by an unknown factor.",
  "Programmatic mutations can be detectably synthetic, letting the policy pattern-match rather than reason."],
 ["Source answer requires measurements without requiring that the study be powered to detect the target effect.",
  "Trajectory-level clustering makes the effective sample size far smaller than the raw call count.",
  "Adjudication noise sets a variance floor that more traffic cannot reduce, only more adjudication can."],
 ["Source answer assumes the logged record of a tool call is a stable measurement instrument.",
  "Schema and sampling changes produce apparent rate steps that get attributed to policy behavior.",
  "Backfills that rewrite historical records in place destroy the evidence needed to bridge across versions."],
 ["Source answer presumes the tool belongs in the loop and asks only how to reduce redundant use of it.",
  "A poorly chosen input-complexity measure smears the policy-versus-tool accuracy crossover.",
  "Silent policy arithmetic errors are not self-reported and require ground-truth adjudication."],
 ["Source answer lists recovery as a metric without noting that natural traffic rarely exercises it.",
  "Suppressing the call-again disposition reduces redundancy and recovery together, and the trade is usually unmeasured.",
  "SDK-internal retries mask injected failures before the policy observes them, inflating apparent recovery."],
 ["Source answer does not require a trivial baseline, so gains get measured against an empty control.",
  "Prompt-clause paraphrases produce materially different rates, making an unrecorded clause irreproducible.",
  "Adding a clause changes prompt length and position, shifting cache behavior alongside the intended effect."],
 ["Source answer states requirements without scoping the result to the configuration that produced it.",
  "A rate quoted without checkpoint, prompt, schema, decode config and window identity cannot be re-checked.",
  "Checkpoint changes bundle many differences simultaneously and cannot isolate a cause."],
 ["Source answer asks for a stop or no-tool evaluation without questioning whether the action space contains a stop action.",
  "Describing a new action lengthens the prompt, confounding the action-space effect with a prompt-length effect.",
  "An over-used stop action is a new defect and is invisible unless stop-action precision is reported."],
 ["Source answer treats the intervention as a one-time change with no mechanism to defend the improvement.",
  "Unrelated releases each move the rate slightly, ratcheting it back with no single accountable change.",
  "Overfitting to a frozen gate shows as improving gate scores while production behavior drifts."],
]

EVID = [
 ["Per-tier queue depth, utilization and GPU-seconds read from per-tier accounting counters with pool sizes and autoscaling pinned.",
  "Induced-redundancy sweep on a replayed window with KV transfer bytes and p95 transfer latency read from fabric counters, plus a zero-injection inertness check."],
 ["Adversarial stratum construction rule with its version, plus blind adjudication confirming mutated items are genuinely tool-required.",
  "False-suppression rate reported separately for natural and adversarial strata at every candidate threshold, never pooled."],
 ["Intra-trajectory correlation and design effect estimated from a historical window, with minimum detectable effect at the planned sample size.",
  "Control-versus-control replay noise band compared against the analytic prediction, plus planned versus realized effective sample size."],
 ["Log schema version stamped per record with per-version record counts and rates computed per version.",
  "Parser-replay bridge comparing derived metrics across schema versions on a fixed trajectory set, with original raw records retained unmodified."],
 ["Paired tool-enabled and tool-disabled evaluation on one frozen checkpoint with identical seeds, stratified by an explicit input-complexity measure.",
  "Crossover estimate with confidence interval plus a gated arm evaluated against both baselines on the same window, with latency savings measured from replay."],
 ["Fault injection at the tool boundary with a pre-declared rate and failure-mode mix applied identically across arms, plus a zero-rate inertness check.",
  "Recovery rate and time to recovery by failure mode across an intervention-strength sweep, with a frozen duplicate-labeled recovery rubric."],
 ["Three interleaved arms on one frozen checkpoint with the prompt clause text hash recorded and identical decode config and seeds.",
  "Incremental difference between the prompt-clause arm and the complex arm reported as headline, with a control-versus-control noise band."],
 ["Every reported rate stamped with checkpoint hash, prompt hash, tool schema version, decode configuration and traffic window identifier.",
  "Transfer table from a pre-declared one-factor-at-a-time scope grid run on the same window with identical seeds, anchored on the original configuration."],
 ["Adjudicated action distribution at decision points in both arms, attributing where probability mass moved.",
  "Stop-action precision on items with verifiable ground truth, plus a length-matched placebo clause in the control arm to isolate the action-space effect."],
 ["Advisory-mode gate verdicts across a pre-declared number of releases compared against realized production drift over the same period.",
  "Gate false-block rate, gate runtime, agreement between the frozen eval set and production, and a logged justification for every gate override."],
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
