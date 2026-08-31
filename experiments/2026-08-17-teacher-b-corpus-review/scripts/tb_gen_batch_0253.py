import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0253.jsonl"
START, END = 2520, 2530

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
 ("STANCE 101 - Exclude or model the post-rollout warm-up window: the first hours after a checkpoint or config change are a transient, and measuring across them attributes the transient to the change.",
  """Mechanism. Immediately after a rollout the prefix cache is cold, compiled graphs and kernels are being warmed, autoscalers are still converging and traffic is being drained from the previous version. Every one of these shifts latency and, through admission and queueing, the observed behaviour. A measurement window that begins at rollout therefore contains a systematic transient whose sign and magnitude have nothing to do with the intervention.

Falsifiable hypothesis. H1: metrics measured in the first warm-up interval after rollout differ from their steady-state values by more than the control-versus-control noise band, so including that interval biases the comparison. Falsified if the warm-up interval is statistically indistinguishable from steady state, which would license measuring from the moment of rollout.

Metrics. Prefix cache hit rate, p95 latency, queue depth and replica count as functions of time since rollout, together with unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate over the same time axis. The warm-up duration is MEASURED as the time until each series enters its steady-state band, not assumed from a fixed rule of thumb.

Controlled experiment. Perform a no-op rollout, redeploying the identical artifact, and measure how long each series takes to return to its prior band. That duration is the warm-up envelope attributable to deployment mechanics alone. Then apply the same exclusion window to the real intervention rollout, and pre-declare the window before the intervention data is examined.

Confounders. Traffic composition varies by hour, so a warm-up window that coincides with a diurnal shift confuses the two; the no-op rollout must be repeated at comparable times of day. Autoscaler settings differ between environments, so a staging-measured envelope understates production. Cache warmth depends on tenant mix, which itself changes during a drain.

Rollback criteria. This is an analysis gate: discard any comparison whose window overlaps the measured warm-up envelope and re-measure at steady state rather than adjusting the numbers after the fact. If a rollout decision was already taken on a window that included the transient, treat it as unsupported and re-evaluate before proceeding to the next traffic step."""),
 ("STANCE 102 - If the framework issues tool calls concurrently, redundancy appears as duplicate in-flight calls, and any deduplication must be request-scoped and race-safe or it will return the wrong answer.",
  """Mechanism. Sequential reasoning about redundancy assumes the prior result exists before the next decision. With parallel tool calls, two identical calls can be in flight simultaneously, so neither can observe the other's result. A naive cache keyed on the argument tuple will have both requests miss, both execute, and then both write; worse, a cache shared across requests can serve one request's result to another, which is a correctness and isolation defect rather than an optimisation.

Falsifiable hypothesis. H1: a measurable share of duplicate calls are concurrent rather than sequential, so sequential deduplication cannot suppress them. Falsified if effectively all duplicates are separated by at least one completed round trip, which would make a simple sequential cache sufficient.

Metrics. Share of duplicate calls that overlap in time, in-flight duplicate count per request, single-flight coalescing hit rate, cross-request cache serve rate which must be zero, tool success, tool-required-stratum accuracy, unnecessary-call rate, trajectory length, tool p95 latency and recovery rate. Overlap is MEASURED from call start and end timestamps recorded on the same clock.

Controlled experiment. Classify duplicates as concurrent or sequential from the timestamps on a logged window. Then implement single-flight coalescing scoped strictly to one request, deploy it in passive mode where the second caller still executes but its result is compared byte-for-byte against the coalesced one, and only enable the fast path after the passive comparison shows exact agreement on every observation.

Confounders. Clock skew between the orchestrator and the tool service corrupts the overlap classification, so both timestamps must come from one clock. Retries inside the SDK create duplicates that are invisible above the SDK boundary. A cache keyed without a tenant or session scope will appear to have a high hit rate precisely because it is leaking across requests.

Rollback criteria. Disable coalescing immediately by a single flag if any cross-request serve is observed or if the passive comparison shows a single byte difference, since an isolation defect is categorically worse than the redundancy it removes. Coalescing must never be enabled for tools declared impure or time-varying, and that allowlist must be explicit rather than default-open."""),
 ("STANCE 103 - Check how numbers tokenise: the same value rendered differently becomes a different token sequence, and the policy may not recognise its own prior result.",
  """Mechanism. Numeric text is not tokenised uniformly. Digit grouping, thousands separators, trailing zeros, scientific notation and currency symbols all change the token sequence for an identical value. If the tool returns one form and the policy's internal representation of the question expects another, the prior result is present in the context but is not the sequence the policy learned to match against. The observable consequence is a re-call that looks like a reasoning failure but is a serialisation mismatch.

Falsifiable hypothesis. H1: canonicalising numeric rendering in tool responses to a single declared form reduces the unnecessary-call rate by more than the replay noise band. Falsified if the rate is unchanged after canonicalisation, which would show the policy is not sensitive to numeric surface form and would redirect effort to the decision itself.

Metrics. Distribution of numeric render forms in tool responses, token-sequence length and identity for the same value under each form, unnecessary-call rate stratified by whether the prior result used a non-canonical form, tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. The token-sequence comparison is MEASURED with the deployed tokenizer version, which must be recorded.

Controlled experiment. Tokenise each observed render form of the same set of values and record where sequences diverge; this requires no model inference. Then run a two-arm replay on one frozen checkpoint that differs only in the response rendering, holding numeric values byte-identical in meaning, and stratify the result by the original form so the mechanism is tested where it predicts an effect.

Confounders. Canonicalising also changes response length, which shifts prompt length and cache behaviour, so a length-matched control is required. Rounding during canonicalisation would change the value itself and turn a formatting change into a correctness change, so the transformation must be lossless and verified as such. Tokenizer upgrades silently invalidate the divergence table.

Rollback criteria. Revert the rendering change if any consumer's parse-failure rate rises or if a value is altered rather than reformatted, verified by an exact numeric equality check across arms. The canonical form must be declared in the response contract and versioned, so downstream consumers can pin it and the revert is a schema version change."""),
 ("STANCE 104 - Scope the result to the serving build and hardware: the same weights on a different region, kernel or accelerator generation do not produce the same rate.",
  """Mechanism. Behaviour depends on the numerics of the execution path. Different accelerator generations, attention kernel implementations, quantisation settings and framework versions produce slightly different logits, and a decision near the boundary can flip. Fleets are heterogeneous by design, so a rate measured in one region is a sample from one build, and a fleet-wide claim built on it is an extrapolation across an unmeasured axis.

Falsifiable hypothesis. H1: unnecessary-call rate differs across regions or serving builds by more than the within-build control-versus-control band, with the same weights and prompt. Falsified if all builds agree within that band, which supports a single fleet-wide rate and simplifies every downstream claim.

Metrics. Unnecessary-call rate per region and per build, with accelerator model, kernel backend, quantisation setting and framework version recorded for each; logit margin at adjudicated decision points per build; and the standard behavioural set of tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. Every rate is reported with its build tuple, and a rate without the tuple is an ESTIMATE of unknown scope.

Controlled experiment. Replay one identical item set with identical seeds and decode configuration against each build in the fleet, comparing outputs token by token as well as comparing rates, since a token-level divergence localises the cause to numerics rather than to traffic. Where builds diverge, bisect on the differing component by holding all others fixed.

Confounders. Regions differ in traffic mix as well as in build, so a rate difference measured on live traffic confounds the two; the replay must use one fixed item set. Load differs by region and changes batch composition, which itself perturbs numerics under some kernels. Autoscaling can move a replay onto a different hardware pool mid-run.

Rollback criteria. Restrict any claim and any rollout to the builds where the effect was measured; treat unmeasured builds as unsupported and gate them separately. If the fleet is mid-migration between accelerator generations, pin the evaluation to a single generation and re-measure after migration rather than pooling across it."""),
 ("STANCE 105 - Examine the constrained-decoding grammar: if tool calls are emitted under a grammar and free text is not, the call path is structurally cheaper and the policy will drift toward it.",
  """Mechanism. Many serving stacks enforce a JSON grammar once a tool-call prefix is emitted. Inside the grammar the sampler is restricted to valid continuations, which raises the probability of completing a well-formed call and lowers the chance of abandoning it midway. Composing a final answer enjoys no such support. The asymmetry is in the decoding machinery rather than in the weights, and it biases the action distribution toward calling in a way that no training-side analysis will reveal.

Falsifiable hypothesis. H1: disabling grammar enforcement, or applying an equivalent constraint to the final-answer path, reduces the unnecessary-call rate by more than the replay noise band. Falsified if the rate is unchanged, which would eliminate the decoding asymmetry as a driver and leave the policy prior as the explanation.

Metrics. Unnecessary-call rate with grammar enforcement enabled and disabled, tool-call parse-failure rate under each setting, abandonment rate of partially emitted calls, tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. Parse-failure rate must be reported because disabling the grammar trades structural bias for malformed calls, and that trade is the point of the measurement. All per-arm rates are MEASURED from the replay; any latency difference attributed to the grammar itself remains an ESTIMATE until the no-op processor arm separates decoding-path effects from behavioural ones.

Controlled experiment. Three arms on one frozen checkpoint over one logged window with identical seeds: grammar as deployed, grammar disabled, and grammar retained with a matched constraint applied to the final-answer path. Include a no-op processor arm to confirm that merely installing a processor, which can disable fast decoding paths, is not itself the source of any observed difference.

Confounders. Enabling or disabling a logits processor can turn off cuda-graph or speculative-decoding fast paths, changing latency independently of behaviour, which the no-op arm isolates. Grammar implementations differ in how they handle multi-token delimiters. A malformed call that the framework silently retries appears as a duplicate rather than as a failure.

Rollback criteria. Do not ship grammar-disabled serving if parse-failure rate rises above zero, since malformed tool calls are a correctness and safety regression. Any grammar change must be revertible by a single serving configuration flag, validated in staging under load, with the prior grammar retained as a versioned artifact."""),
 ("STANCE 106 - Publish the prefix-cache hit rate with every cost claim: measurement environments are warmer than production, and cost savings computed on a warm cache do not transfer.",
  """Mechanism. The dominant cost of an extra tool observation is the prefill of the lengthened prompt on every subsequent turn. Prefix caching removes most of that cost when the prefix is already resident. Replay harnesses run the same items repeatedly and therefore operate at a hit rate far above production, where traffic is diverse and evictions are frequent. A saving computed in that environment can be several times the saving realised in production, in either direction depending on which arm benefits more.

Falsifiable hypothesis. H1: the GPU-second saving attributed to removing redundant calls changes by more than the noise band between a warm-cache replay and a cache-disabled replay of the same items. Falsified if the saving is invariant to cache state, which would allow the replay figure to be quoted directly for production.

Metrics. Prefix cache hit rate in the replay and in production over the comparison window, GPU-seconds per completed task with cache enabled and disabled, prefill token counts, unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool p95 latency and recovery rate. The replay hit rate and the production hit rate are both MEASURED; any production saving projected from a replay is an ESTIMATE and must state both hit rates.

Controlled experiment. Run the same item set twice, once with prefix caching enabled and once disabled, and report the saving under both as a bracket rather than a point. Then compare the production hit rate against the replay hit rate to place the realistic figure inside that bracket, rather than asserting a single number.

Confounders. Disabling the cache changes latency substantially, which alters batch composition under continuous batching and therefore per-request cost, so the bracket is not a pure isolation. Replay item order determines warmth, so order must be randomised as part of this measurement. Production hit rate varies by tenant and by hour and cannot be summarised by a single value without its dispersion.

Rollback criteria. Withdraw any cost or capacity claim quoted from a warm replay without its bracket, and do not resize pools on such a figure. If pools were already resized, revert them to the recorded prior values and re-derive the requirement from the bracketed estimate before making any further change."""),
 ("STANCE 107 - Name an owner and test the alert route before shipping the metric, because an unrouted metric is documentation rather than a control.",
  """Mechanism. A regression is only prevented if some person or system acts on it within the detection window. That requires an owning team, a defined threshold, a route from the alert to that team, and a runbook describing the action. Any missing link converts the whole measurement programme into a dashboard that is read after incidents rather than before them. The links are individually testable, so this is an engineering precondition rather than a process opinion.

Falsifiable hypothesis. H1: a synthetic breach of the redundancy threshold reaches the owning on-call and produces a documented action within the pre-declared response objective. Falsified if the synthetic breach fails to route, routes to an unstaffed destination, or produces no documented action, which means the metric is not yet a control regardless of its statistical quality.

Metrics. Alert route test result with end-to-end delivery time, acknowledgement time, time to documented action, runbook existence and last-review date, alert false-positive rate over a trailing window, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate. The route test outcome and its timings are MEASURED, not asserted.

Controlled experiment. Fire a synthetic breach through the real alerting path, not a test harness, at a time not announced to the on-call, and record delivery, acknowledgement and action. Repeat once during business hours and once out of hours, since routing and staffing differ, and record the false-positive rate over the same trailing window so that alert credibility is measured alongside reachability.

Confounders. Announcing the test inflates responsiveness and invalidates the timing. Alert deduplication can suppress the synthetic breach if a similar alert is already open. Routing configuration often differs between environments, so a successful staging test says nothing about production.

Rollback criteria. Do not promote the metric to a blocking gate until the route test passes in production out of hours; until then it remains advisory and must be labelled as such in the write-up. If the false-positive rate exceeds the pre-declared ceiling, demote the alert to advisory by a single configuration change rather than allowing the on-call to silence it informally."""),
 ("STANCE 108 - Make arm assignment version-aware during a canary: with mixed replica versions and non-sticky routing, a request can be served partly by each arm and the comparison is contaminated.",
  """Mechanism. During a canary the fleet runs two versions simultaneously. If routing is per-request rather than per-session, a multi-turn agent trajectory can have successive turns served by different versions. The trajectory then belongs to neither arm, and because redundancy is a property of a trajectory rather than of a turn, the contaminated trajectories systematically dilute the measured difference toward zero.

Falsifiable hypothesis. H1: a non-trivial share of trajectories in the canary window are served by more than one version, and excluding them changes the measured effect by more than the noise band. Falsified if effectively all trajectories are version-pure, which would mean routing is already sticky enough and no exclusion is needed.

Metrics. Share of trajectories spanning multiple serving versions, effect size computed with and without contaminated trajectories, per-version request share, session stickiness rate, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate. Version purity is MEASURED by stamping the serving version on every turn record.

Controlled experiment. Stamp the serving version per turn, then compute the effect on all trajectories and on the version-pure subset. Where contamination is material, enable session-sticky routing keyed on the trajectory identifier for the canary window only, and re-measure. Verify stickiness by checking that the stamped versions within each trajectory are constant.

Confounders. Sticky routing changes load balance and can concentrate long trajectories on the canary replicas, altering their cache and batch conditions relative to the control. Restricting analysis to version-pure trajectories conditions on an outcome-adjacent property, since trajectories that fail over are more likely to span versions, so the exclusion must be reported alongside the unrestricted figure. Rolling restarts break stickiness silently.

Rollback criteria. Abandon the canary result if version purity cannot be established retrospectively, rather than reporting the diluted figure. Sticky routing enabled for the experiment must be reverted afterwards by a single configuration change, and its effect on load distribution must be watched during the window since concentration can breach the objective independently of the intervention."""),
 ("STANCE 109 - Tag generated trajectories at the source: once agent traffic feeds future training mixtures, an intervention becomes self-perpetuating and its measured effect is no longer independent of the data.",
  """Mechanism. Production trajectories are a common source of supervised and preference data. If the current policy calls redundantly, those trajectories teach the next policy to do the same; if an intervention suppresses calls, the next mixture inherits the suppression including its errors. Without provenance tags distinguishing model-generated from human-authored content, and recording which intervention was active, the training pipeline silently closes a feedback loop that no single experiment can detect.

Falsifiable hypothesis. H1: a measurable share of the current supervised mixture consists of trajectories generated by a prior version of this same policy, establishing an active feedback loop. Falsified if the mixture contains no self-generated trajectories, which would make the loop hypothetical for now and reduce this to a preventive control.

Metrics. Share of the mixture by provenance class covering human-authored, model-generated and synthetic-template, the generating checkpoint and active intervention recorded per model-generated record, redundant-call base rate within each provenance class, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length and recovery rate. Provenance shares are MEASURED from tags, and an untagged record must be counted as unknown rather than assumed human.

Controlled experiment. Audit the current mixture for provenance, then compare the redundant-call base rate across provenance classes on the same adjudication rule. Where self-generated content is present, run a paired fine-tune differing only in whether self-generated trajectories are included, holding mixture size and every other hyperparameter fixed, and compare the resulting rates.

Confounders. Provenance tags added late do not cover historical records, so the untagged share is a floor on the unknown rather than an estimate of it. Model-generated content that has been human-edited belongs to neither class cleanly and needs its own label. Deduplication across the mixture can remove self-generated records disproportionately and confound the paired comparison.

Rollback criteria. Block any mixture whose untagged share exceeds the pre-declared ceiling from being used for a training run, rather than proceeding and noting the caveat. If a run has already used an untagged mixture, treat its behavioural rates as unattributable and retain the prior checkpoint as the revert target until provenance is established."""),
 ("STANCE 110 - Require that every number in the report be regenerable by a committed script from committed inputs, or it is an assertion rather than a measurement.",
  """Mechanism. Analyses of this defect involve many derived quantities: adjudicated rates, noise bands, cost conversions and power calculations. Each passes through notebooks, ad-hoc queries and manual steps. If the path from raw artifact to reported number is not executable end to end, the number cannot be checked, and disagreements become arguments about recollection. Making regeneration mechanical converts the report into an artifact that can be falsified.

Falsifiable hypothesis. H1: re-running the committed analysis scripts against the committed raw artifacts reproduces every reported number exactly, or within a declared tolerance for genuinely stochastic steps. Falsified if any number cannot be regenerated, which invalidates that number and any decision resting on it until the path is repaired.

Metrics. Share of reported numbers regenerable by a committed script, count of manual steps in the pipeline, hash agreement between committed raw artifacts and those used in the original analysis, wall-clock time to regenerate the full report, and the standard behavioural set of unnecessary-call rate, tool success, tool-required-stratum accuracy, trajectory length, tool latency and recovery rate. Regeneration success is MEASURED by execution, not by inspection of the code.

Controlled experiment. Have an engineer who did not perform the original analysis regenerate the report from the committed repository on a clean checkout, with no access to the original working environment. Any step requiring undocumented local state is a defect in the pipeline and is recorded as such. Compare the regenerated numbers to the published ones field by field.

Confounders. Stochastic steps such as sampled adjudication reproduce only within a tolerance, which must be declared in advance rather than negotiated after a mismatch. Data that has since been deleted under retention policy makes regeneration impossible for reasons unrelated to code quality, so the retention horizon bounds this guarantee. Environment drift in library versions changes results subtly and requires a pinned environment.

Rollback criteria. Withdraw any number that fails regeneration from the report rather than annotating it, and block decisions that depended on it until it is reproduced. The pinned environment specification and the raw artifact hashes must be committed alongside the scripts, so that a future regeneration failure is attributable to a specific missing input rather than to general rot."""),
]

DECISIONS = ["rewrite"] * 10

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

CONF = [0.7,0.73,0.71,0.72,0.7,0.69,0.71,0.72,0.7,0.74]

RISKS = [
 ["Source answer does not exclude the deployment transient that follows any rollout.",
  "Warm-up coinciding with a diurnal traffic shift confuses deployment mechanics with load composition.",
  "Autoscaler settings differ between staging and production, so a staging-measured envelope understates production."],
 ["Source answer assumes redundant calls are sequential, so its intervention cannot suppress concurrent duplicates.",
  "A cache keyed without session or tenant scope leaks results across requests, which is an isolation defect rather than an optimisation.",
  "Clock skew between orchestrator and tool service corrupts the concurrent-versus-sequential classification."],
 ["Source answer treats the prior result as recognisable regardless of how the value was rendered.",
  "Canonicalisation that rounds rather than reformats silently converts a formatting change into a correctness change.",
  "Tokenizer upgrades invalidate any previously computed divergence table without any visible signal."],
 ["Source answer states a single rate without scoping it to the serving build and accelerator that produced it.",
  "Regions differ in traffic mix as well as build, so live-traffic comparisons confound the two.",
  "Autoscaling can move a replay onto a different hardware pool mid-run."],
 ["Source answer does not consider that the decoding machinery may make the tool-call path structurally cheaper than a final answer.",
  "Installing or removing a logits processor can disable fast decoding paths and change latency independently of behaviour.",
  "A malformed call silently retried by the framework appears as a duplicate rather than as a failure."],
 ["Source answer quotes cost effects without stating the cache conditions under which they were measured.",
  "Replay harnesses run warmer than production, so a saving computed there does not transfer.",
  "Disabling the cache changes latency and therefore batch composition, so the bracket is not a pure isolation."],
 ["Source answer defines metrics and thresholds without establishing that a breach reaches anyone who can act.",
  "Announcing the route test inflates responsiveness and invalidates the measured timings.",
  "Alert deduplication can suppress a synthetic breach when a similar alert is already open."],
 ["Source answer assumes arm assignment is clean during a staged rollout.",
  "Per-request routing lets a single trajectory be served by both versions, diluting the measured effect toward zero.",
  "Restricting to version-pure trajectories conditions on an outcome-adjacent property and must be reported alongside the unrestricted figure."],
 ["Source answer proposes training-side interventions without controlling whether production traffic re-enters the training mixture.",
  "Provenance tags added late leave a historical untagged share that is a floor on the unknown rather than an estimate of it.",
  "Deduplication can remove self-generated records disproportionately and confound a paired mixture comparison."],
 ["Source answer specifies what to measure but not that the reported numbers must be mechanically regenerable.",
  "Stochastic steps reproduce only within a tolerance, which must be declared before a mismatch is observed.",
  "Retention deletion and library-version drift can make regeneration impossible for reasons unrelated to code quality."],
]

EVID = [
 ["No-op rollout of an identical artifact with cache hit rate, p95 latency, queue depth and replica count tracked against time since rollout.",
  "Pre-declared exclusion window derived from the measured warm-up envelope, with the no-op rollout repeated at comparable times of day."],
 ["Concurrent-versus-sequential duplicate classification from call start and end timestamps taken on a single clock.",
  "Passive-mode single-flight coalescing with byte-for-byte comparison against the independently executed second call, plus a zero cross-request serve count."],
 ["Tokenised comparison of every observed numeric render form for the same values, recorded with the deployed tokenizer version.",
  "Two-arm replay differing only in response rendering, with an exact numeric equality check across arms and a length-matched control."],
 ["Identical item set replayed against each serving build with accelerator model, kernel backend, quantisation and framework version recorded.",
  "Token-level output comparison per build to localise divergence to numerics, with component-wise bisection where builds differ."],
 ["Three-arm replay covering deployed grammar, grammar disabled and matched constraint on the final-answer path, plus a no-op processor arm.",
  "Tool-call parse-failure rate and partial-call abandonment rate under each setting on the same window and seeds."],
 ["Prefix cache hit rate measured in both the replay and the production comparison window, reported with its dispersion.",
  "GPU-seconds per completed task measured with caching enabled and disabled, reported as a bracket with randomised item order."],
 ["Synthetic breach fired through the production alerting path unannounced, with delivery, acknowledgement and action times recorded.",
  "Business-hours and out-of-hours route tests plus the trailing-window false-positive rate and runbook last-review date."],
 ["Serving version stamped on every turn record, with the share of version-spanning trajectories computed for the canary window.",
  "Effect size reported both unrestricted and on the version-pure subset, with stickiness verified by constant stamped version within each trajectory."],
 ["Provenance audit of the training mixture by class, with generating checkpoint and active intervention recorded per model-generated record.",
  "Paired fine-tune differing only in inclusion of self-generated trajectories, with mixture size and all other hyperparameters held fixed."],
 ["Clean-checkout regeneration of the full report by an engineer who did not perform the original analysis, compared field by field.",
  "Committed pinned environment specification and raw artifact hashes, with the count of manual pipeline steps recorded."],
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
