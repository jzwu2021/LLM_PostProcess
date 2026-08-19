import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0248.jsonl"
START, END = 2470, 2480

STANCES = [
[
"STANCE 51 - Start from a stale-cache model of the failure: the agent re-calls because nothing in its context marks a prior result as still valid.",
"""Mechanism. A tool result enters the trajectory as an untyped observation string. Nothing carries a validity scope (which expression, which inputs, until when). At the next decision point the policy has no cheap test for 'this exact quantity is already known', so re-calling is the locally safe action. This is a representation defect at the observation boundary, not a reasoning defect.

Falsifiable hypothesis H1. If tool observations are rewritten to a typed form carrying (canonical_expression, value, computed_at_step), then the unnecessary-call rate on a held-out arithmetic-agent suite drops by at least 30% relative, with no degradation in final-answer accuracy beyond 1 absolute point.

Controlled experiment. Two arms over the same 500 held-out tasks, same model weights, same decoding seed, paired by task id. Arm A: current untyped observation string. Arm B: typed observation. Only the observation serializer differs; system prompt, tool schema and sampling parameters are byte-identical, diffed and hashed before the run.

Measurements. Primary: unnecessary_call_rate, where a call is labeled unnecessary iff its canonicalized argument string exactly matches an earlier call in the same trajectory whose result is still in context. Secondary: final correctness, trajectory length in steps, tool-call success rate, p50/p95 end-to-end latency, and recovery rate after an injected tool error.

Confounders to control. (1) Canonicalization is itself a decision -- '2+3' vs '3+2' -- so publish the canonicalizer and report the metric under both a strict and a commutative-normalized variant. (2) Typed observations are longer, so prompt length changes; report tokens-in per task and re-run Arm B with a length-matched padded control. (3) Truncation: if the prior observation was evicted from the window, the call is not redundant. Log window occupancy per decision.

Numbers. If a redundant call costs one extra prefill of ~1.5k tokens plus ~60 decoded tokens, and redundancy occurs on 20% of tasks, the expected token overhead is ~0.2 * 1.56k ~= 310 tokens/task. ESTIMATE, derived from assumed prompt size and an assumed 20% rate; both must be replaced by MEASURED values from the Arm A logs before any capacity claim.

Evidence required. Paired per-task logs for both arms, the canonicalizer source, window-occupancy traces, and the config diff proving single-variable change.

Rollback gate. Revert to Arm A if final correctness drops more than 1 absolute point, if p95 latency rises more than 10%, or if recovery-after-tool-error drops at all. Rollback is a serializer flag flip, so it is reversible within one deploy cycle.""",
["Canonicalization choice can manufacture or hide the effect","Longer typed observations may displace task-relevant context","A metric that only counts exact-duplicate arguments undercounts semantic redundancy"],
["Paired per-task trajectory logs for both arms","Published canonicalizer implementation and its unit tests","Config and prompt hash diff showing a single changed variable","Context-window occupancy trace at each decision point"],
0.62],

[
"STANCE 52 - Contrarian: measure the counterfactual value of the call before calling it redundant, because 'already known' is an assumption about the model, not an observation.",
"""Objection to the framing. The scenario asserts the answer is already known. That is an outside-view claim. From inside the policy the prior result may be low-confidence, may have come from a tool that has since errored, or may be a value the model never actually attended to. Labeling such calls 'unnecessary' bakes the conclusion into the metric.

Mechanism. Redundancy is only a defect if suppressing the call leaves the final answer unchanged. That makes it a counterfactual quantity, and counterfactuals must be measured by intervention, not inferred from trajectory shape.

Falsifiable hypothesis H2. For at least 25% of calls labeled redundant by argument-matching, forcibly suppressing the call changes the final answer -- i.e. the naive label has a false-positive rate above 25%. If true, any intervention trained on the naive label is optimizing a partly wrong target.

Controlled experiment. Replay harness. For each labeled-redundant call, fork the trajectory: branch S suppresses the call and injects the cached prior result as the observation; branch C proceeds unchanged. Same seed, same weights, deterministic decoding, so the only difference is the suppressed call. Compare final answers pairwise.

Measurements. False-positive rate of the naive label, final-answer agreement between branches, per-branch correctness against ground truth, and the distribution of steps-to-answer after the fork.

Confounders. (1) Deterministic decode is required or branch divergence is confounded by sampling noise -- verify with a null fork that suppresses nothing and must produce identical output. (2) Injecting a cached observation changes token positions and therefore the KV state; measure with a positional-null control. (3) Ground-truth labeling of arithmetic answers must be exact-match on a normalized numeric form.

Numbers. With 500 tasks and ~1.2 redundant calls per affected task, expect ~120 fork pairs at a 20% affected rate. Detecting a false-positive rate of 25% vs a 10% null needs roughly 130 pairs at 80% power. ESTIMATE from a two-proportion power calculation with alpha 0.05; the affected rate is assumed, not measured.

Evidence required. Null-fork determinism proof, fork pair logs with both branch outputs, normalized ground-truth answers, and the power calculation as run.

Rollback gate. Do not ship any suppression policy while the measured label false-positive rate exceeds 10%. If it does, the deliverable of this stage is a corrected label, not an intervention.""",
["Non-deterministic decoding invalidates every fork comparison","Suppression injects a synthetic observation the model never produced","Optimizing the naive label can suppress genuinely load-bearing calls"],
["Null-fork determinism check with byte-identical outputs","Per-pair fork logs (suppressed and control branches)","Normalized exact-match ground truth for every task","The power calculation and its assumed base rates"],
0.6],

[
"STANCE 53 - Treat it as a scheduler and admission-control problem: on a shared serving cluster, redundant tool calls are queueing load, and the first fix is a quota, not a model change.",
"""Mechanism. Each redundant call is a new request against the serving tier: a prefill of the grown trajectory plus a short decode. On a saturated cluster, prefill is the scarce resource and arrives in bursts. Redundancy therefore inflates queue depth and tail latency for every tenant sharing the pool, including ones with no agentic traffic. That is an admission-control defect regardless of whether the model is later fixed.

Falsifiable hypothesis H3. Imposing a per-task tool-call quota at the orchestrator reduces p95 end-to-end latency for the shared pool by at least 15% at fixed offered load, without reducing agent task success by more than 1 absolute point.

Controlled experiment. Load-generator replay of a recorded request mix at a fixed arrival rate, three quota settings (unlimited, 8 calls/task, 4 calls/task), each run 30 minutes after a 5-minute warmup, arms interleaved in randomized blocks to absorb cluster drift. Serving config, model, batch scheduler and GPU allocation held constant and captured in a config hash.

Measurements. p50/p95/p99 end-to-end latency, prefill queue depth, GPU utilization and achieved batch size, tokens/s, tool-call counts per task, and agent task success. Separate the shared-pool latency from the agent's own success so the tradeoff is visible.

Confounders. (1) Cluster drift and noisy neighbors -- hence interleaved blocks and a repeated baseline at the end that must match the opening baseline within noise. (2) A quota changes trajectory length, which changes prompt length, which changes prefill cost non-linearly; report cost per task in tokens as well as calls. (3) Warmup effects on cache hit rates.

Numbers. If 20% of tasks emit one extra call at ~1.5k prefill tokens, offered prefill load falls ~4-6% under a tight quota. ESTIMATE, derived from assumed redundancy rate and prompt size; the p95 improvement is not proportional to load reduction because queueing is non-linear near saturation, so the 15% figure in H3 must be treated as a hypothesis to reject, not a projection.

Evidence required. Serving-side metrics at per-request granularity, config hash equality across arms, the open/close baseline comparison, and per-tenant latency breakdown.

Rollback gate. Raise or remove the quota if agent success drops more than 1 point, if any tenant's error rate rises, or if truncated-by-quota tasks exceed 2%. The quota is an orchestrator config value, revertible without redeploying the model.""",
["A quota can silently truncate legitimately long tasks","Cluster drift can masquerade as a treatment effect","Latency gains may accrue to one tenant while another regresses"],
["Per-request serving metrics with tenant labels","Config hash equality across all arms","Opening and closing baseline runs that agree within noise","Count and inspection of quota-truncated tasks"],
0.58],

[
"STANCE 54 - Probe whether the model can even represent 'I already computed this' before designing any policy fix.",
"""Mechanism. Suppressing a redundant call requires the policy to (a) retrieve the earlier result from context and (b) trust it. If a linear probe on the residual stream at the decision position cannot separate 'result present in context' from 'absent', then no prompt or reward change is operating on a represented feature, and behavioral interventions will be brittle.

Falsifiable hypothesis H4. A linear probe trained on decision-position activations distinguishes present-vs-absent prior results with AUC >= 0.85 on held-out tasks. If AUC <= 0.65, the capability is not linearly represented and the intervention should shift to the scaffold rather than the weights.

Controlled experiment. Construct matched pairs: identical task and identical prefix, differing only in whether the prior tool result is present in the observation. Extract activations at the last token before the tool-call decision across several layers. Train logistic probes per layer on a train split; evaluate AUC on a held-out split with disjoint task templates to prevent template leakage.

Measurements. Per-layer probe AUC, probe AUC on a shuffled-label control (must be ~0.5), correlation between probe score and observed suppression behavior, and generalization AUC across an unseen tool schema.

Confounders. (1) Template leakage -- enforce disjoint templates between splits and report the shuffled-label control. (2) The probe may pick up mere token presence rather than a functional 'known' feature; test by paraphrasing the cached result so surface tokens differ. (3) Layer selection is a multiple-comparisons problem; pre-register the layers or correct for the number tested.

Numbers. Roughly 2000 matched pairs give a probe AUC standard error near 0.01 under standard assumptions. ESTIMATE from a normal approximation for AUC standard error at balanced classes; verify empirically with bootstrap confidence intervals rather than trusting the closed form.

Evidence required. Matched-pair construction script, per-layer AUC with bootstrap CIs, shuffled-label control results, paraphrase-robustness results, and cross-schema generalization.

Rollback gate. If AUC is at or below 0.65 with a clean control, abandon weight-level interventions for this defect in this cycle and route the effort to scaffold-level caching. This gate is analysis-only and carries no production risk.""",
["Probe may detect token presence rather than a functional representation","Template leakage inflates AUC","Layer sweep without correction invites false positives"],
["Matched-pair generation code and split manifest","Per-layer AUC with bootstrap confidence intervals","Shuffled-label control at chance","Paraphrase and cross-schema generalization results"],
0.57],

[
"STANCE 55 - Put the fix in a deterministic memoization layer outside the model, and use the model-side study only to decide whether that layer can ever be removed.",
"""Mechanism. A content-addressed cache keyed on the canonicalized tool arguments makes a redundant call cost a hash lookup instead of a prefill plus decode. This does not change the agent's behavior; it changes the cost of that behavior. It is the cheapest available intervention and it is fully reversible.

Falsifiable hypothesis H5. A memoization layer reduces tool-execution cost per task by at least 15% while leaving final-answer accuracy statistically unchanged (paired difference CI containing zero at 95%).

Controlled experiment. Same 500 tasks, same seeds, cache on vs cache off, paired. The cache returns the byte-identical prior result, so the observation stream is unchanged when the cache hits; that is the key design property making the arms comparable.

Measurements. Cache hit rate, tool-execution time saved, tokens per task, final accuracy paired difference with CI, and a correctness audit that cached and live results agree on a sampled subset.

Confounders. (1) For a pure calculator the cache is trivially safe; for any tool with state or time dependence it is not. Scope the cache by an explicit tool allowlist with declared purity, and treat purity as an assertion to be tested, not assumed. (2) Cache hits shorten wall-clock, which can change downstream timeout behavior. (3) Hit rate is a property of the workload and will not transfer to a different task mix.

Numbers. At a 20% redundancy rate and a calculator latency that is negligible against model latency, the wall-clock gain is small; the real saving is the avoided prefill of the grown trajectory, roughly 310 tokens/task under the earlier assumptions. ESTIMATE with the same assumed inputs; report MEASURED token deltas from the paired run before claiming any cost reduction.

Evidence required. Cache hit/miss logs with keys, the purity allowlist and its tests, paired accuracy difference with CI, and the live-vs-cached agreement audit.

Rollback gate. Disable the cache immediately if any live-vs-cached disagreement is observed on an allowlisted tool, or if accuracy regresses at all. Disabling is a single flag with no model change.

Why the model study still matters. Memoization hides the behavior rather than fixing it, so the redundancy metric must continue to be logged with the cache on. If the underlying rate keeps rising, the cache is masking a regression.""",
["Caching an impure tool returns stale or wrong values","Cache masks a worsening underlying behavior","Hit rates do not transfer across workloads"],
["Cache key and hit/miss logs","Tool purity allowlist with tests","Live-vs-cached agreement audit on a sampled subset","Redundancy metric logged with cache enabled"],
0.64],

[
"STANCE 56 - Any preference or reward signal against redundancy must carry an explicit anti-undercalling invariant, or the fix trades one failure mode for a worse one.",
"""Mechanism. A reward term penalizing tool calls has a degenerate optimum: never call. On arithmetic the model may absorb that penalty and answer from parametric memory, which is exactly where it is least reliable. The observable symptom -- fewer calls -- looks like success on the target metric while accuracy on hard instances collapses.

Falsifiable hypothesis H6. Under a redundancy penalty, accuracy on a hard-arithmetic slice (multi-digit multiplication, nested expressions) degrades by more than 2 absolute points relative to the baseline, i.e. the penalty induces undercalling. The intervention is acceptable only if this hypothesis is rejected.

Controlled experiment. Preference-tuned arm vs baseline, evaluated on two disjoint slices: an easy slice where calling is optional and a hard slice where calling is load-bearing. Both slices held out from tuning data. Report per-slice results separately and never as an aggregate, because the aggregate can hide a slice regression behind an easy-slice gain.

Measurements. Per-slice accuracy, per-slice call rate, abstention rate, unnecessary-call rate, and the joint distribution of (called, correct) to distinguish 'called and still wrong' from 'did not call and wrong'.

Confounders. (1) Tuning data may leak slice templates; verify with n-gram overlap between tuning and eval sets. (2) A loss-masking bug can make the model train on tool-result tokens it should never predict; audit the mask on real batches by decoding the supervised positions before trusting any tuning result. (3) Reward scale interacts with KL regularization; sweep the penalty weight rather than reporting a single point.

Numbers. Detecting a 2-point drop from an assumed 80% baseline needs roughly 1500 examples per slice at 80% power. ESTIMATE from a two-proportion power calculation at alpha 0.05 with an assumed baseline; recompute once the true baseline is MEASURED.

Evidence required. Per-slice accuracy tables with CIs, the decoded loss-mask audit, tuning/eval n-gram overlap report, and the penalty-weight sweep.

Rollback gate. Do not promote any tuned checkpoint that loses more than 1 absolute point on the hard slice, regardless of redundancy improvement. Rollback means serving the prior checkpoint, which must remain resident and traffic-switchable.""",
["Redundancy penalty collapses into a never-call policy","Aggregate metrics hide a hard-slice regression","Loss-mask bugs silently train the wrong tokens"],
["Per-slice accuracy with confidence intervals","Decoded loss-mask audit on real training batches","N-gram overlap report between tuning and eval data","Penalty-weight sweep results","Prior checkpoint resident and traffic-switchable"],
0.63],

[
"STANCE 57 - Order the interventions by reversibility and cost, and forbid any step from starting before its predecessor's gate has passed.",
"""Objection to the source answer. It lists what to measure and what to add, but gives no ordering and no gates. Without ordering, the expensive irreversible intervention (weight change) can begin before the cheap reversible one has been shown insufficient, which is the most common way this class of work wastes a cluster.

Mechanism. Each candidate fix has a different blast radius and a different revert cost: observation serializer (revert = flag), memoization (revert = flag), orchestrator quota (revert = config), system-prompt change (revert = deploy), preference tuning (revert = checkpoint switch plus re-validation). Reversibility should determine order.

Staged plan with gates.
R1 Instrument: land the redundancy metric and the counterfactual replay harness. Gate: label false-positive rate measured and below 10%.
R2 Serializer and memoization: typed observations plus a purity-scoped cache. Gate: measured cost reduction with accuracy paired-difference CI containing zero.
R3 Orchestrator quota: bound calls per task. Gate: quota-truncation rate under 2% and no tenant regression.
R4 Prompt ablation: cheapest behavioral change. Gate: effect exceeds the run-to-run noise band established by two identical baseline runs.
R5 Weight-level tuning: only if R1-R4 leave a residual gap that matters. Gate: hard-slice invariant from the undercalling test holds.

Falsifiable hypothesis H7. The residual redundancy remaining after R1-R4 is small enough that R5 cannot yield more than a 2-point improvement in unnecessary-call rate -- i.e. weight-level work is not justified. Running R1-R4 first is what makes this testable.

Measurements. At every rung: unnecessary-call rate, final accuracy, p95 latency, tokens per task, and the carried invariant (hard-slice accuracy) which must be re-checked at each rung, not only at the end.

Confounders. Rungs interact; report each rung's effect against the immediately preceding configuration, not against the original baseline, and re-establish the noise band whenever the base configuration changes.

Numbers. No numeric projection is offered here deliberately. Any figure at this stage would be an ESTIMATE built on the unmeasured base rate, and the point of R1 is to replace that base rate with a MEASURED one.

Evidence required. A per-rung record with the gate criterion stated before the run, the measured value, and the pass/fail decision, all timestamped and version-controlled.

Rollback gate. Any rung failing its gate stops the program at that rung; the prior configuration remains in production and the failure is written up rather than worked around.""",
["Skipping ahead to weight changes burns cluster time on an unproven need","Rung interactions make late-stage attribution ambiguous","Gates set after seeing results are not gates"],
["Pre-registered gate criteria per rung, committed before each run","Per-rung measurement record against the immediately preceding config","Two identical baseline runs establishing the noise band","Carried hard-slice invariant re-checked at every rung"],
0.66],

[
"STANCE 58 - Report the metric as a rate per decision, not per trajectory, or the intervention will be evaluated on a quantity that moves for the wrong reasons.",
"""Mechanism. 'Unnecessary calls per task' is a ratio whose denominator the intervention also changes. Any fix that shortens trajectories reduces the numerator mechanically. The defensible unit is the decision point: at each step where a tool call was possible, did the policy call redundantly? That yields a per-opportunity rate whose denominator is set by the task, not by the treatment.

Falsifiable hypothesis H8. Per-task and per-decision redundancy rates diverge in sign or magnitude for at least one candidate intervention -- meaning at least one intervention would be accepted under one metric and rejected under the other. If they never diverge, the simpler metric is adequate and this objection is refuted.

Controlled experiment. Recompute both metrics offline from the same stored trajectories for every arm already run. No new inference is required; this is a re-analysis, which makes it cheap and makes disagreement, if any, unambiguous.

Measurements. Per-task rate, per-decision rate, decision-opportunity count per task, and the rank correlation between arms under the two metrics. Report the arm ranking under each metric side by side.

Confounders. (1) Defining a 'decision opportunity' requires a rule for steps where no tool was applicable; publish it and report sensitivity to the two obvious variants. (2) Trajectories truncated by a quota have censored denominators and must be flagged or excluded, with the exclusion count reported. (3) Stored trajectories must come from runs with identical logging schema versions.

Numbers. If mean trajectory length falls from 6.0 to 4.5 steps under an intervention, a per-task count can drop 25% with no change in per-decision behavior at all. ESTIMATE, illustrative arithmetic on assumed step counts; the actual step-length distributions must be MEASURED from the stored logs before the size of this artifact is claimed.

Evidence required. The decision-opportunity rule with sensitivity analysis, both metrics computed for every arm, censored-trajectory counts, and logging-schema version equality across arms.

Rollback gate. Not applicable in production terms; this is an analysis change. The gate is procedural: no arm may be promoted on the per-task metric alone once divergence has been demonstrated.""",
["Treatment-dependent denominators create phantom improvements","Decision-opportunity definition can be tuned to a desired result","Censored trajectories bias the per-decision rate"],
["Published decision-opportunity rule plus sensitivity to variants","Both metrics recomputed for every existing arm","Censored and excluded trajectory counts","Logging schema version equality across arms"],
0.59],

[
"STANCE 59 - Treat the finding as a permanent regression gate, not a one-off study, because behavioral defects of this kind reappear at every checkpoint and prompt change.",
"""Mechanism. Redundant calling is an emergent property of weights, prompt and scaffold jointly. Any of the three can change independently in a normal release cycle, so a fix validated once has no durability guarantee. The only durable artifact is an automated gate that runs on every candidate and fails the release.

Falsifiable hypothesis H9. Without a gate, the unnecessary-call rate regresses by more than 3 absolute points within three release cycles following the fix. Instrumenting the gate is what makes this measurable either way.

Gate design. A fixed, version-pinned suite of tasks with an immutable seed set; the counterfactual replay harness computes the per-decision redundancy rate; the gate fails if the rate exceeds the last accepted value plus the noise band. The noise band is established empirically from repeated runs of an unchanged candidate, not chosen by hand.

Controlled experiment for the gate itself. Run the unchanged production candidate ten times through the gate. The false-failure rate must be at or below 5%. A gate whose false-failure rate is unknown will be disabled by the first team it inconveniences, which is the actual failure mode of most quality gates.

Measurements. Gate pass/fail per candidate, measured rate with CI, false-failure rate from the repeated-run calibration, gate wall-clock and GPU cost per invocation, and the carried hard-slice accuracy invariant.

Confounders. (1) Suite staleness -- a pinned suite eventually stops representing traffic; schedule a review and report suite-vs-traffic distribution drift. (2) Seed sensitivity; the noise band must come from multiple seeds, not one. (3) Optimizing against a fixed suite invites overfitting to it; keep a rotating holdout that is never used for tuning.

Numbers. If the gate runs 500 tasks at ~6 steps and ~1.5k prompt tokens per step, that is on the order of 4.5M prefill tokens per invocation. ESTIMATE from assumed step counts and prompt sizes; measure actual GPU-minutes on the first calibration run and publish the MEASURED cost so the gate's budget is explicit.

Evidence required. Pinned suite manifest with hashes, ten-run calibration output and false-failure rate, measured per-invocation cost, rotating holdout definition, and the gate's failure history over subsequent releases.

Rollback gate. If the gate's false-failure rate exceeds 5%, widen the band or fix the harness before enforcing; an unreliable gate is worse than none because it trains the team to override it.""",
["An uncalibrated gate gets overridden and becomes decorative","Fixed suites go stale and drift from real traffic","Tuning against the gate suite overfits to it"],
["Version-pinned suite manifest with content hashes","Ten-run calibration establishing the false-failure rate","Measured GPU cost per gate invocation","Rotating holdout never used for tuning","Gate outcome history across releases"],
0.61],

[
"STANCE 60 - Synthesis: state the assumptions that the whole program rests on, and name the observation that would invalidate each one.",
"""Assumptions and their killers.
A1 The prior result is actually in context at the decision point. Killer: window-occupancy traces showing it was evicted, in which case the behavior is correct and the problem is context management.
A2 Suppressing the call leaves the answer unchanged. Killer: a counterfactual replay false-positive rate above 10%, in which case the label is wrong and every downstream metric inherits the error.
A3 The behavior is stable enough to measure. Killer: two identical baseline runs differing by more than the claimed effect size, in which case nothing below the noise band is reportable.
A4 The fix generalizes past the calculator. Killer: no improvement on a second, structurally different tool schema, in which case the fix is to one schema, not to the behavior.
A5 The fix does not induce undercalling. Killer: a hard-slice accuracy drop above 1 absolute point.

Consolidated hypothesis H10. The cheapest reversible interventions -- typed observations, purity-scoped memoization, and an orchestrator call quota -- together reduce the per-decision redundancy rate by at least 30% relative while holding hard-slice accuracy within 1 absolute point, making weight-level tuning unnecessary this cycle.

Experiment. Sequential, gated, paired-by-task, single-variable-per-arm, with the baseline re-run at the start and end of the sequence and required to agree within the noise band. Every arm's config is hashed and the hashes are committed alongside the results.

Measurements. Per-decision unnecessary-call rate, final accuracy overall and per slice, tool success rate, trajectory length, p50/p95 latency, tokens per task, and recovery rate after injected tool errors.

Numbers. Every figure quoted across this program so far -- the 20% redundancy rate, the 1.5k-token prompt, the ~310 tokens/task overhead, the 4.5M-token gate cost -- is an ESTIMATE derived from assumed inputs stated at the point of use. None is MEASURED. No capacity, cost or savings claim may be made externally until each is replaced by a logged measurement with its collection method recorded.

Evidence required. The full per-arm record: config hashes, opening and closing baselines, paired per-task logs, counterfactual replay outputs, loss-mask audit if any tuning occurred, per-slice accuracy, and the pre-registered gate criteria with their outcomes.

Rollback gate. Program-level: if the opening and closing baselines disagree beyond the noise band, discard the entire sequence and re-run rather than reporting it, because a drifting substrate makes every within-sequence comparison uninterpretable.""",
["Reporting a sequence whose substrate drifted mid-run","Presenting estimates as measurements in external claims","Declaring success on the easy slice while the hard slice regresses"],
["Config hashes committed with results for every arm","Opening and closing baselines agreeing within the noise band","Paired per-task logs and counterfactual replay outputs","Per-slice accuracy tables","Pre-registered gate criteria and recorded outcomes"],
0.65],
]

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]
assert len(src) == 10
assert len(STANCES) == 10

with open(OUT, "w") as f:
    for i, s in enumerate(src):
        m = {x["role"]: x["content"] for x in s["messages"]}
        head, body, risks, ev, conf = STANCES[i]
        tc, ic, os_ = QD[i]
        rec = {
            "source_id": s["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": m["user"],
            "source_assistant": m["assistant"],
            "corrected_answer": head + "\n\n" + body,
            "quality_dimensions": {
                "technical_correctness": tc,
                "instruction_coverage": ic,
                "operational_safety": os_,
            },
            "risks": risks,
            "evidence_required": ev,
            "confidence": conf,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("WROTE", OUT)
