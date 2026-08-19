STANCES = [
    ("Stance 310 - The unit of analysis must be the trajectory, not the call, because per-call rates hide the fact that redundancy concentrates in a small tail of pathological trajectories.",
     """A redundancy rate computed over calls treats every call as an independent draw. In practice the failure is bursty: most trajectories emit zero or one redundant call, and a small tail loops the same calculator invocation many times until a turn limit stops it. Those two populations demand different fixes - the broad low-rate case is a policy-threshold issue, the tail is a loop-detection issue that no reward shaping will reliably remove.
The boundary condition is the turn cap. A tight cap truncates the tail and makes the per-call rate look moderate while the user-visible symptom is a hung request. Any report must therefore state the turn cap in force and the fraction of trajectories that terminated by hitting it, or the tail is invisible by construction.
Falsifiable hypothesis H310: H310: the distribution of redundant calls per trajectory is heavy-tailed rather than approximately uniform, so a small minority of trajectories contributes a disproportionate share of all redundant calls and a per-call mean misrepresents the typical trajectory (ESTIMATE; derivation: repeated identical calls arise from a self-reinforcing context in which the previous call and its result both remain visible, which makes further repetition more likely rather than less; such positive feedback produces heavy tails. If redundant calls are spread evenly across trajectories, the mean is representative and the claim is refuted).
Controlled experiment: Recompute the same trace's redundancy at three units - per call, per trajectory and per trajectory-tail-decile - and report the share of total redundant calls contributed by the worst decile.
Rollback gate: no intervention is credited with a redundancy improvement until the improvement is shown at the trajectory unit as well as the call unit, because a fix that only trims the tail and a fix that only shifts the broad threshold have different correctness risks."""),

    ("Stance 311 - Loop detection belongs in the serving layer as a hard bound, because a probabilistic policy cannot be relied on to terminate.",
     """Reward shaping changes the expected number of redundant calls; it does not bound the worst case. If the tail matters - and a hung interactive request means it does - the bound must come from a deterministic component outside the model: a per-trajectory counter of identical normalized tool invocations that refuses the call and injects an explicit observation telling the model the value is already available.
This is deliberately not a silent drop. Refusing without telling the model leaves it blind and it will simply re-emit. Returning a structured observation - 'this exact call was already made, result was X' - converts an unbounded loop into a single informative turn and keeps the trajectory interpretable in the logs.
Falsifiable hypothesis H311: H311: a deterministic identical-call refusal with an explicit observation bounds worst-case trajectory length without degrading final-answer correctness, whereas a silent refusal degrades it because the model re-emits or stalls (ESTIMATE; derivation: the model conditions on observations; an informative observation supplies the missing state, while a silent drop leaves the context unchanged and therefore leaves the same call the most likely next action. If the silent variant matches the informative one on correctness, the observation channel is not load-bearing and the claim is refuted).
Controlled experiment: Three arms - no refusal, silent refusal, informative refusal - identical checkpoint and prompts, reporting p99 trajectory length, correctness and recovery.
Rollback gate: the refusal threshold is set above the maximum legitimate repeat count observed on a labelled sample, and is disabled outright if any legitimate workflow is found that requires more repeats than the threshold allows."""),

    ("Stance 312 - Normalization of the call signature is where this measurement quietly breaks, and it must be specified and audited before any rate is quoted.",
     """Whether two calls are 'identical' is a decision about normalization: numeric formatting, argument ordering, units, whitespace, precision. An under-normalized comparator undercounts redundancy because trivially different spellings of the same call look distinct; an over-normalized one overcounts because genuinely different calls collapse together and legitimate work is labelled waste.
The consequence is that every redundancy number is a function of an unpublished normalization rule. Two teams measuring the same system will disagree, and neither will be wrong. The rule must therefore ship as code with the metric, with a labelled audit sample showing its false-merge and false-split rates.
Falsifiable hypothesis H312: H312: plausible variations in the normalization rule move the measured redundancy rate by a margin comparable to the effect size of the interventions under test, so an unpublished rule makes the metric unusable for promotion decisions (ESTIMATE; derivation: model-emitted arguments vary in formatting far more than in semantics, so the fraction of pairs whose classification depends on formatting handling is substantial; interventions in this area produce modest relative changes, making the two magnitudes comparable. If the rate is insensitive across normalization variants, the rule is not load-bearing and the claim is refuted).
Controlled experiment: Hold the trace and agent fixed and recompute redundancy under several pre-specified normalization rules, reporting the spread as a measurement-uncertainty floor beneath every quoted effect.
Rollback gate: no redundancy figure is published without the normalization rule's version hash and its audited false-merge and false-split rates on the labelled sample."""),

    ("Stance 313 - The tool schema is an instruction surface, and its description field is doing more work than the system prompt in deciding whether a call happens.",
     """Engineers debug this problem in the system prompt because that is where they are used to looking, but the tool description sits immediately adjacent to the decision and is often written once, by whoever added the tool, with language like 'use this for any arithmetic'. That single clause can dominate every downstream prompt instruction, and it is rarely under the same review discipline.
The correct edit is to make the description state the precondition rather than the capability: when the tool should be used and, explicitly, when it should not. This is a schema change, deployable and revertible without touching weights, and it should be attempted and measured before any prompt engineering campaign is funded.
Falsifiable hypothesis H313: H313: editing the tool description to include an explicit negative precondition reduces redundant calls by a larger margin than an equivalent-length instruction added to the system prompt, because the description is closer to the decision point in the context (ESTIMATE; derivation: instruction adherence degrades with distance and intervening content between the instruction and the decision, and the tool description is rendered adjacent to the action schema. If the two placements produce equivalent effects, position is not load-bearing and the claim is refuted).
Controlled experiment: Two arms with textually matched instructions placed in the tool description versus the system prompt, plus a no-instruction control, identical checkpoint and decoding.
Rollback gate: the schema version hash is recorded in every run artifact, and a description change that moves the must-call set beyond its noise band is reverted in the same deploy that introduced it."""),

    ("Stance 314 - Latency accounting must be end-to-end and percentile-based, because a redundancy fix that improves the mean while worsening p99 is a regression the mean will conceal.",
     """The economic case for this work is usually made in means: so many fewer calls times so many milliseconds. But the user experience is set by the tail, and interventions that add a decision step - a cache lookup, a refusal check, a longer prompt - add a small fixed cost to every trajectory to remove a large cost from a few. Whether that trade is good depends entirely on the shape of the distribution, not on its mean.
The honest report therefore gives p50 and p99 end-to-end trajectory latency for every arm, decomposed into model turn time, tool time and orchestration overhead, with the per-trajectory count of removed calls alongside. Anything less permits a real p99 regression to ship as a latency win.
Falsifiable hypothesis H314: H314: interventions that add a per-call check improve p99 end-to-end latency while leaving or slightly worsening p50, because the removed cost is concentrated in tail trajectories while the added cost is uniform (ESTIMATE; derivation: the check runs on every call whereas the saving accrues only where a redundant call would have occurred, and redundant calls concentrate in the tail; the two therefore load onto different parts of the distribution. If p50 improves proportionally with p99, redundancy is uniform rather than tail-concentrated and the claim is refuted).
Controlled experiment: Load-matched arms at a fixed request rate, reporting p50 and p99 end-to-end latency with the three-way decomposition and the per-trajectory removed-call count.
Rollback gate: any arm that worsens p99 end-to-end latency beyond the measured noise band is rejected regardless of mean improvement, and the load level at which the measurement was taken is recorded because tail behaviour is load-dependent."""),

    ("Stance 315 - Offline adjudication of redundancy is itself a measuring instrument and needs its own agreement study before its output is trusted.",
     """The whole programme rests on a judge - human or model - deciding post hoc whether each call was necessary. That judge has an error rate, and if its error rate is comparable to the intervention effect being measured, the experiment cannot resolve anything. This is the standard instrument-validation problem and it is routinely skipped because the judge feels like ground truth rather than a measurement.
The minimum discipline is a double-adjudicated sample with inter-rater agreement reported, plus a held-out set of deliberately constructed clear-cut cases in both directions to estimate the judge's directional bias. A model judge additionally needs the checkpoint and prompt of the judge itself pinned and hashed, because a judge upgrade silently rewrites the historical metric.
Falsifiable hypothesis H315: H315: inter-rater agreement on redundancy labels is materially below unity and the residual disagreement concentrates on borderline multi-step cases, so effect sizes smaller than the disagreement band are not resolvable by this instrument (ESTIMATE; derivation: the label requires a counterfactual judgement about what the model could have done without the tool, which is not directly observable, and unobservable counterfactuals produce systematic rater disagreement concentrated where the counterfactual is least clear. If agreement is near-perfect across all buckets, the instrument is sharp and the claim is refuted).
Controlled experiment: Double-adjudicate a stratified sample, report agreement per difficulty bucket, and score both adjudicators on a constructed clear-cut set to estimate directional bias.
Rollback gate: no promotion decision is taken on an effect smaller than the adjudication disagreement band, and any change to the judge model or judge prompt invalidates historical comparisons until the affected runs are re-adjudicated."""),

    ("Stance 316 - Redundant calls may be an artifact of context truncation rather than policy, and that diagnosis changes the intervention completely.",
     """If the earlier tool result has been evicted from the model's visible context by truncation, summarization or a sliding window, then re-calling is not redundant from the model's point of view - it is the only way to recover a value it can no longer see. Every policy-level intervention is then aimed at the wrong layer, and reward shaping will teach the model to guess instead of call, which is strictly worse.
The diagnostic is mechanical and cheap: for each labelled redundant call, check whether the prior result was still inside the context window actually sent to the model at that turn. That requires logging the rendered prompt or at minimum its token count and truncation decisions, which many stacks do not do. The absence of that log is itself the finding.
Falsifiable hypothesis H316: H316: a measurable share of calls labelled redundant occur at turns where the prior result had already been evicted from the rendered context, so those calls are context-management failures and are immune to policy-level interventions (ESTIMATE; derivation: long agent trajectories exceed context budgets and eviction removes oldest observations first, which are exactly the earlier tool results; the share therefore grows with trajectory length. If every labelled redundant call had the prior result in context, the failure is purely policy and the claim is refuted).
Controlled experiment: Join the redundancy labels against per-turn rendered-context logs and report the evicted share by trajectory length decile.
Rollback gate: policy-level interventions are not funded until the evicted share is measured; if it is material, context management is fixed first and the redundancy baseline is recomputed afterwards, because the pre-fix baseline is not comparable."""),

    ("Stance 317 - The cost model must be stated in currency and load, not in call counts, or the programme cannot be prioritized against anything else.",
     """A redundancy reduction is worth funding only if the saved resource is scarce. On a deployment that is memory-bandwidth bound with idle headroom, removing calls saves nothing anyone can spend; on a deployment saturating its accelerators at peak, the same reduction buys queue headroom that translates directly into admitted requests. The identical behavioural change therefore has wildly different value, and the call-count metric is silent on which situation applies.
The translation requires three inputs: tokens added per redundant call, accelerator time per token at the serving batch size, and the utilization headroom at peak. All three are deployment-specific and none are properties of the agent. Publishing a saving without them invites a prioritization decision that the evidence does not support.
Falsifiable hypothesis H317: H317: on a deployment with substantial off-peak headroom, the throughput value of removing redundant calls is negligible outside the peak window, so the programme's economic case rests entirely on peak-window behaviour and must be evaluated there (ESTIMATE; derivation: freed capacity has value only when capacity is the binding constraint; off peak the constraint is demand, so saved accelerator time is simply idle time. If value is realized uniformly across the day, capacity is binding throughout and the claim is refuted).
Controlled experiment: Instrument tokens per redundant call and accelerator time per token at the production batch size, then combine with the measured utilization profile to compute realized savings per hour across a full day.
Rollback gate: no engineering effort is committed on a call-count reduction alone; the cost translation must be produced with its three inputs labelled MEASURED or ESTIMATE individually, and a saving resting entirely on ESTIMATE inputs is treated as a hypothesis rather than a result."""),

    ("Stance 318 - Determinism of the evaluation harness is a precondition, not a nicety, because non-deterministic serving will manufacture and destroy effects at this effect size.",
     """Batching, continuous scheduling and non-deterministic reduction orders mean the same prompt at the same checkpoint can produce different token sequences on different runs. In an agent loop that difference compounds: one divergent token changes whether a call is emitted, which changes the whole trajectory. Effects of the size under discussion here are comfortably inside that noise unless the harness is pinned.
Pinning means fixed seed, fixed batch composition or batch size one, a single replica, and a recorded engine version and kernel configuration. It costs throughput and is therefore skipped, and the resulting variance is then attributed to the intervention. The alternative - running many repeats and reporting a confidence band - is legitimate but must be done explicitly, not implicitly assumed.
Falsifiable hypothesis H318: H318: repeated runs of an unchanged agent under an unpinned harness produce redundancy-rate variation comparable to the intervention effects being claimed, so single-run comparisons under that harness cannot support a promotion decision (ESTIMATE; derivation: agent trajectories amplify single-token divergences into whole-branch differences, and unpinned batched serving admits such divergences; the amplification makes run-to-run spread grow with trajectory length. If repeat runs are tightly clustered, the harness is effectively deterministic and the claim is refuted).
Controlled experiment: Run the identical configuration repeatedly under both pinned and unpinned harnesses and report the run-to-run spread of every headline metric as an explicit noise floor.
Rollback gate: every reported effect must exceed the harness noise floor measured on the same day and configuration; effects below it are reported as null results rather than as small wins."""),

    ("Stance 319 - The closing position for this batch is unchanged and must be stated plainly: the source assistant turn is a rubric, and forty-odd near-identical variants multiply that defect rather than diluting it.",
     """The user turn is a sound agent-engineering prompt asking for metrics, an intervention, a falsifiable hypothesis and a controlled experiment. The assistant turn lists what such an answer would need to contain and stops. Across a long run of variants differing only in an index, the corpus therefore carries the same empty target many times, and the resulting supervision signal is not weak but consistently wrong in a specific direction: it teaches the model to reply with requirements.
The disposition is rewrite with the prompt preserved. Instruction coverage is scored low on the response side alone; the prompt is the scarce and reusable part. The duplication is worth recording as a corpus-level property, because deduplication or down-weighting decisions later depend on knowing how concentrated this pattern is.
Falsifiable hypothesis H319: H319: near-duplicate prompts paired with an identical rubric response contribute redundant gradient in proportion to their count, so deduplicating the slice to a small number of exemplars with substantive responses yields equal or better held-out generative quality than retaining all variants (ESTIMATE; derivation: identical targets on near-identical inputs supply repeated rather than additional information, while consuming training budget; substantive targets supply new information per example. If retaining all variants outperforms the deduplicated slice, the repetition carries useful signal and the claim is refuted).
Controlled experiment: Fine-tune three variants from one base - all rubric variants retained, deduplicated rubric exemplars, deduplicated with substantive rewritten responses - and score a held-out generative evaluation for meta-response rate and content quality.
Rollback gate: no deduplication is applied to the training corpus until the duplicate-cluster sizes are measured and reported per topic, because collapsing a cluster that carries genuine prompt diversity discards coverage rather than redundancy."""),
]

EXTRA_RISKS = [
    ["Per-call redundancy rates hide a heavy tail of looping trajectories that require loop detection rather than threshold tuning.",
     "A turn cap truncates the tail and makes the per-call rate look moderate while users experience hung requests."],
    ["A probabilistic policy changes expected call count but supplies no worst-case bound on trajectory length.",
     "Silently dropping a repeated call leaves the model blind to why, and it re-emits or stalls instead of proceeding."],
    ["Every redundancy figure depends on an unpublished call-signature normalization rule, so independent measurements are not comparable.",
     "Over-normalization labels genuinely distinct calls as redundant and charges legitimate work as waste."],
    ["The tool description field instructs calling behaviour more proximally than the system prompt but is rarely under review discipline.",
     "A description edit that suppresses calls can raise omitted-necessary-call rate on the must-call set without appearing in the redundancy metric."],
    ["Mean-based latency accounting conceals p99 regressions introduced by a per-call check that costs every trajectory to help a few.",
     "Tail latency is load-dependent, so a measurement taken off peak does not bound behaviour at peak."],
    ["The redundancy adjudicator is an unvalidated instrument whose error rate may exceed the effects being measured.",
     "Upgrading a model judge silently rewrites the historical metric and invalidates prior comparisons."],
    ["Calls labelled redundant may follow context eviction, making them context-management failures immune to policy interventions.",
     "Reward shaping applied to eviction-driven calls teaches the model to guess a value it cannot see, which is worse than calling."],
    ["Call-count savings quoted without a currency and utilization translation cannot be prioritized against other work.",
     "Savings realized only inside the peak window are reported as all-day savings when the utilization profile is omitted."],
    ["Unpinned batched serving produces run-to-run trajectory divergence comparable to the claimed effect sizes.",
     "Pinning is skipped for throughput reasons and the resulting variance is then attributed to the intervention."],
    ["Repeating one rubric-shaped target across many near-identical prompts multiplies a specific wrong supervision signal.",
     "Deduplicating without measuring cluster composition discards genuine prompt coverage along with the redundancy."],
]

RISKS_COMMON = [
    "Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
    "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them.",
]

EXTRA_EVIDENCE = [
    ["Redundancy recomputed at per-call, per-trajectory and per-tail-decile units on the same trace.",
     "Share of total redundant calls contributed by the worst trajectory decile.",
     "Turn cap in force and fraction of trajectories terminated by hitting it."],
    ["Three-arm comparison of no refusal, silent refusal and informative refusal on p99 trajectory length, correctness and recovery.",
     "Labelled sample establishing the maximum legitimate repeat count for the tool.",
     "Refusal threshold value and the observation text injected on refusal."],
    ["Normalization rule shipped as versioned code with its content hash recorded beside every quoted rate.",
     "Audited false-merge and false-split rates of the normalization rule on a labelled sample.",
     "Redundancy rate recomputed under several pre-specified normalization variants, reported as a measurement-uncertainty floor."],
    ["Three-arm comparison of instruction placed in tool description, in system prompt, and absent, with textually matched instructions.",
     "Tool schema version hash recorded in every run artifact.",
     "Must-call set results for each arm alongside the redundancy results."],
    ["p50 and p99 end-to-end trajectory latency per arm, decomposed into model turn, tool time and orchestration overhead.",
     "Per-trajectory count of removed calls reported alongside the latency deltas.",
     "Request rate and utilization level at which each latency measurement was taken."],
    ["Inter-rater agreement on redundancy labels from a double-adjudicated stratified sample, reported per difficulty bucket.",
     "Adjudicator scores on a constructed clear-cut set in both directions, estimating directional bias.",
     "Judge model checkpoint and judge prompt hash pinned for the duration of the comparison."],
    ["Join of redundancy labels against per-turn rendered-context logs, reporting the evicted share by trajectory length decile.",
     "Context budget, truncation policy and per-turn rendered token counts.",
     "Redundancy baseline recomputed after any context-management fix, with the pre-fix baseline explicitly marked non-comparable."],
    ["Tokens added per redundant call, measured at the production prompt format.",
     "Accelerator time per token at the production serving batch size, labelled MEASURED or ESTIMATE.",
     "Full-day utilization profile with peak-window boundaries, used to compute realized savings per hour."],
    ["Run-to-run spread of every headline metric under repeated identical runs, on both pinned and unpinned harnesses.",
     "Harness pinning record: seed, batch composition, replica count, engine version and kernel configuration.",
     "Explicit comparison of each claimed effect against the same-day noise floor."],
    ["Duplicate-cluster sizes per topic across the corpus slice, with the near-duplicate definition specified.",
     "Three-way fine-tuning comparison of full variants, deduplicated exemplars and deduplicated substantive rewrites on a held-out generative evaluation.",
     "Meta-response rate and content quality scores for each variant, with the held-out set construction recorded."],
]

QD = [
    (3, 2, 3), (3, 2, 4), (3, 2, 3), (3, 2, 4), (3, 2, 4),
    (3, 2, 3), (3, 2, 4), (3, 2, 3), (3, 2, 4), (3, 2, 3),
]
CONF = [0.79, 0.80, 0.78, 0.79, 0.80, 0.78, 0.81, 0.77, 0.80, 0.79]
