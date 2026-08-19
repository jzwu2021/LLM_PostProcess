COMMON_HEAD = """Review note: the source record's assistant field is a grading rubric ("Answer should state assumptions, ...") rather than an actual answer, so it cannot serve as a training target and is rewritten below into a concrete response that satisfies that rubric.

Assumptions (stated, not inherited): a single-agent tool-calling loop with a deterministic, side-effect-free calculator exposed alongside other tools; tool results are appended into the same context window; evaluation is offline replay on a hash-pinned, production-representative task set; exactly one variable moves per experimental arm (prompt, tool schema, decoding parameters, serving stack and model version are otherwise pinned).

Mechanism, stated plainly: tool invocation is a token-level decision. The model emits a tool-call token sequence when the conditional probability of doing so exceeds that of answering directly. Redundancy is therefore a distributional property of the policy induced by the system prompt, the tool descriptions, the few-shot examples and any tool-use fine-tuning — not a defect in a controller that can be patched. One redundant call costs one extra forward pass over a longer context plus one tool round-trip, and because the result is appended to the transcript, every later turn pays the extra prefill too."""

COMMON_TAIL = """Numeric discipline: every number in this answer that is not read from a trace artifact is labelled ESTIMATE and carries its derivation. No throughput, latency or utilization figure here is MEASURED, because no run was executed for this review. Any figure entering a capacity or cost decision must be replaced by a MEASURED value read from the serving stack's own traces.

Boundary conditions that flip the recommendation: (B1) repeated is not redundant — a call that re-verifies a value after an intervening state change is correct behavior, and suppressing it trades correctness for latency; (B2) if the model's mental arithmetic is unreliable, calling the calculator when the answer is "already known" may still be the right policy, because the model's belief that it knows is itself unreliable; (B3) if redundancy concentrates in a few prompt templates, a targeted prompt edit dominates any model-level intervention on cost-effectiveness.

Scope note: this is an agent-loop, evaluation and observability design. It says nothing about the model's underlying domain knowledge; any metric movement must be attributed to the runtime/system change unless a held-out, model-only evaluation isolates the model contribution. This output is provisional teacher-B review material, not expert gold."""

BASE_RISKS = [
    "source_assistant is a grading rubric rather than an answer; training on it would teach the model to emit meta-commentary about answers instead of solutions",
    "call-count reduction treated as a success metric has a degenerate optimum at zero tool calls and can be reached by strong suppression prompts",
    "single-run metric movements without confidence intervals can be mistaken for capability gains",
]

BASE_EV = [
    "per-call trajectory traces with canonical args hash, result hash, latency, turn index and in-window flag",
    "repeated baseline runs establishing the noise band of task_success_rate and p95 latency before any arm is compared",
]

S = [
 dict(v=91, f="Judge-adjudicated redundancy labels and inter-rater agreement.",
  body="""Automatic detection by args-hash equality is a lower bound on redundancy, not the quantity of interest: it misses semantically identical calls with differently formatted arguments and it over-counts legitimate re-verification. Before any intervention is funded, the metric itself must be adjudicated.

Procedure: draw a stratified sample of at least 300 calls flagged by the hash-based detector plus 300 unflagged calls, have two independent raters label each as REDUNDANT / NECESSARY / AMBIGUOUS against a written adjudication rubric, and report Cohen's kappa. The hash detector is only usable as a shipping metric once its precision and recall against the adjudicated labels are known and stated.""",
  h="H91: the hash-based redundancy detector has precision below 0.9 against human adjudication, because argument formatting variance and legitimate re-verification both fall outside the equality test (ESTIMATE; derivation: the detector tests string-canonical equality only and has no access to intervening state changes, so both error modes are structurally present; refuted if adjudicated precision and recall both exceed 0.9 on the stratified sample).",
  exp="Freeze the detector, draw the stratified sample from a single hash-pinned replay, collect two independent label sets under the written rubric, compute kappa, precision and recall, and publish the confusion matrix with examples of each error cell before any suppression arm is run.",
  gate="Rollback gate: no suppression intervention is promoted while detector precision against adjudicated labels is unknown or below the pre-registered threshold. If precision is below threshold, the programme stops at metric repair; the intervention backlog is not started.",
  risks=["a detector with unknown precision turns every downstream experiment into an uninterpretable measurement of the detector itself",
         "raters who see the detector's flag before labelling will anchor on it, inflating apparent agreement"],
  ev=["confusion matrix of hash-detector flags against two-rater adjudicated labels with Cohen's kappa",
      "the written adjudication rubric, timestamped before labelling began"],
  qd=(2,1,2), conf=0.63),

 dict(v=92, f="Pre-registration and equivalence bands for the accuracy guard.",
  body="""The dominant failure of this programme is not a bad intervention but a moving goalpost: the accuracy equivalence band is widened after results are seen, and a real regression is reclassified as noise. The countermeasure is procedural, not technical.

Before any arm runs, record in a timestamped artifact: the primary metric (end-task accuracy), the guard metrics (p95 latency, tool success rate), the equivalence band on each, the sample size and the stopping rule. Any change to those values after data collection begins invalidates the experiment and requires a fresh replay.""",
  h="H92: without pre-registration, at least one reported cost win in this programme will be accompanied by an accuracy regression that is reclassified as noise post hoc (ESTIMATE; derivation: suppression acts on the same token-level decision that produces necessary calls, so accuracy loss is expected at some suppression strength, and an unfixed band admits reclassification; refuted if every arm's accuracy lower confidence bound sits inside a band that was fixed before results were seen).",
  exp="Run the suppression ladder twice: once with the band fixed in a timestamped pre-registration artifact, once allowing the band to be chosen after the results are visible. Compare which arms would be promoted under each regime on the identical hash-pinned trace.",
  gate="Rollback gate: an arm is promotable only against a band recorded before data collection. Any arm promoted against a post-hoc band is reverted on discovery, and the replay is repeated under pre-registration.",
  risks=["post-hoc band selection converts a real regression into a reported success",
         "underpowered samples make the equivalence test vacuous — a wide confidence interval passes any band"],
  ev=["timestamped pre-registration artifact naming primary metric, guard metrics, bands, sample size and stopping rule",
      "power calculation showing the sample size can detect the smallest accuracy loss the band is meant to exclude"],
  qd=(2,1,2), conf=0.66),

 dict(v=93, f="Stratified accuracy reporting to expose harm concentrated on hard items.",
  body="""Aggregate accuracy is a mixture statistic. Items where the calculator is genuinely needed are typically a minority of traffic, so a policy that wrongly abstains loses only a small fraction of the aggregate mean and hides inside the noise band of a normally powered experiment.

Report accuracy separately per difficulty stratum and per arithmetic-density bucket, with confidence intervals on each stratum. An intervention that improves the aggregate while degrading the arithmetic-heavy stratum is a regression and must be read as one.""",
  h="H93: at least one suppression arm improves aggregate end-task accuracy or leaves it flat while degrading accuracy on the arithmetic-heavy stratum beyond that stratum's equivalence band (ESTIMATE; derivation: suppression removes calls uniformly across the policy while the benefit of a call is concentrated on arithmetic-heavy items, so harm is expected to be stratum-concentrated; refuted if per-stratum accuracy moves are within band for every stratum).",
  exp="Tag every item with a difficulty label and an arithmetic-density bucket before running any arm, then report per-stratum accuracy with intervals alongside the aggregate for every arm on the identical hash-pinned replay. Pre-register the per-stratum bands, not only the aggregate one.",
  gate="Rollback gate: a per-stratum regression beyond that stratum's band triggers rollback even when the aggregate improves. Aggregate-only reporting is not sufficient evidence for promotion.",
  risks=["aggregate-only reporting hides harm concentrated on the minority of items where the tool matters most",
         "strata defined after seeing results allow the analyst to choose a slicing that shows no harm"],
  ev=["per-stratum accuracy table with confidence intervals for every arm, strata defined before the run",
      "the stratum definition artifact with its timestamp and the stratum volumes"],
  qd=(2,1,2), conf=0.68),

 dict(v=94, f="Result-cache injection as the reversible first intervention.",
  body="""Fine-tuning is not cheaply reversible; a context-level change is. The lowest-risk intervention is to make prior results salient rather than to instruct the model to abstain: inject a compact TOOL RESULT CACHE block immediately before the decision point, listing (tool, canonical args, result) for the current trajectory.

This changes the conditioning context without adding a suppression instruction, so it does not push the policy toward the degenerate zero-call optimum. It costs prefill tokens, which must be measured, because a cache block that grows with trajectory length can cost more than the calls it removes.""",
  h="H94: cache injection reduces redundant-call rate without an accuracy loss exceeding the pre-registered band, but its net token cost is positive on long trajectories because the injected block grows faster than the calls it removes (ESTIMATE; derivation: the block grows linearly in distinct calls while removed calls grow sub-linearly once the model has seen the value once; refuted if net tokens per trajectory fall in every trajectory-length bucket).",
  exp="Two arms differing only by presence of the cache block, replayed on an identical hash-pinned trace. Report redundant-call rate, end-task accuracy with intervals, and net tokens per trajectory broken out by trajectory-length bucket, plus the prefill token cost of the block itself.",
  gate="Rollback gate: single-flag revert. Roll back if accuracy drops below band in any stratum, or if net tokens per trajectory increase in the bucket carrying the majority of production traffic.",
  risks=["the injected cache block can itself cost more prefill than the redundant calls it removes on long trajectories",
         "a stale or wrongly-scoped cache block can present a result that an intervening state change has invalidated"],
  ev=["net tokens per trajectory by trajectory-length bucket for both arms",
      "diff showing the two arms differ only in the injected block"],
  qd=(2,1,2), conf=0.65),

 dict(v=95, f="Tool purity declarations as the precondition for any suppression.",
  body="""Automatic suppression of a repeated call is only safe when the tool is pure — same arguments imply same result, no side effects, no dependence on external state. A calculator is plausibly pure; most tools in an agent's toolbox are not, and suppression logic written for the calculator tends to be generalized to them.

Require a signed purity declaration per tool before suppression is enabled for that tool, with impure tools excluded by default rather than by an allow-list omission. Fail closed: an undeclared tool is treated as impure.""",
  h="H95: enabling suppression by default and excluding impure tools by allow-list produces at least one false suppression on a state-dependent tool within the evaluation window, whereas fail-closed default-impure produces none (ESTIMATE; derivation: allow-list omission is a silent failure mode while fail-closed requires an explicit positive act per tool; refuted if shadow-mode compare shows zero false suppressions under both configurations).",
  exp="Run suppression in shadow mode only: the suppression decision is logged but the call still executes, and the suppressed-call result is compared byte-for-byte against the cached one. Count mismatches per tool. Ship only for tools with zero mismatches over the pre-registered window and a signed purity declaration.",
  gate="Rollback gate: any single shadow-mode mismatch on a tool declared PURE reverts that tool's declaration and disables suppression for it immediately, before any user-visible traffic is affected.",
  risks=["suppression logic validated on a pure calculator is routinely generalized to impure tools where it silently corrupts results",
         "allow-list based exclusion fails silently when a new tool is added and nobody updates the list"],
  ev=["signed per-tool purity declarations with reviewer identity and date",
      "shadow-mode compare log with byte-level mismatch counts per tool over the pre-registered window"],
  qd=(2,1,3), conf=0.7),

 dict(v=96, f="Trajectory length as a confounder rather than an outcome.",
  body="""Trajectory length is commonly reported as an outcome of the intervention, but it is jointly determined with redundancy by task difficulty: harder items produce both longer trajectories and more tool calls. Reading a shortened trajectory as evidence of reduced redundancy confuses the confounder with the effect.

The correct treatment is to condition on difficulty: report redundancy per trajectory-length bucket, and report length changes only within a fixed difficulty stratum. A length reduction that appears only in the aggregate and vanishes within every stratum is a mix shift, not an improvement.""",
  h="H96: a measurable part of the aggregate trajectory-length reduction attributed to a suppression arm disappears once difficulty is held fixed, indicating mix shift rather than a causal effect (ESTIMATE; derivation: difficulty drives both length and call count, so any arm that differentially affects hard items shifts the observed mix; refuted if the within-stratum length reduction matches the aggregate reduction across all strata).",
  exp="Report trajectory length both aggregate and within each pre-defined difficulty stratum for every arm on the identical hash-pinned replay, plus the stratum volume per arm so that a mix shift is directly visible.",
  gate="Rollback gate: trajectory length is a diagnostic, never a promotion criterion. No arm is promoted on a length reduction alone; promotion requires the accuracy guard to pass within every stratum.",
  risks=["trajectory length reported as an outcome invites promotion of arms that merely shift the difficulty mix",
         "replay traces that are not hash-pinned allow the item mix itself to drift between arms"],
  ev=["within-stratum and aggregate trajectory-length tables with stratum volumes per arm",
      "trace pin hash recorded per arm, proving both arms replayed the identical item set"],
  qd=(2,1,2), conf=0.64),

 dict(v=97, f="Cost accounting that prices prefill growth, not just call count.",
  body="""The cost of a redundant call is not one tool round-trip. The result is appended to the transcript, so every subsequent turn re-processes it during prefill. A cost model that counts calls understates the true cost on long trajectories and overstates the benefit of removing an early call relative to a late one.

Price each redundant call as: one extra decode of the call tokens, one tool round-trip latency, plus (tokens of call+result) × (number of subsequent turns) of extra prefill. Report cost per trajectory in tokens and in wall-clock, not in calls.""",
  h="H97: the true cost of a redundant call, priced with the carried prefill term, exceeds the naive per-call cost by a factor that grows with the number of subsequent turns, so removing early-trajectory redundancy dominates removing late-trajectory redundancy (ESTIMATE; derivation: the carried term is linear in subsequent turns while the round-trip term is constant; refuted if measured cost per redundant call is flat in turn index, which would indicate prefill caching already absorbs the carried term).",
  exp="Instrument prefill token counts per turn and attribute them to prior tool results by turn index. Compare cost per redundant call at early versus late turn indices on the identical replay, with prefix-cache hit rates reported alongside, since an effective prefix cache changes the conclusion.",
  gate="Rollback gate: cost claims are not accepted without the prefill attribution table and the prefix-cache hit rate. A cost win reported in call counts alone is not sufficient evidence for promotion.",
  risks=["a call-count cost model misprices the intervention and can justify work with no wall-clock or token benefit",
         "an effective prefix cache can absorb the carried prefill term entirely, invalidating cost claims computed without hit-rate data"],
  ev=["per-turn prefill token attribution table with turn index",
      "prefix-cache hit rate over the replay window for each arm"],
  qd=(2,1,2), conf=0.66),

 dict(v=98, f="Per-call provenance as the precondition for attribution.",
  body="""Without per-call provenance, a redundancy metric cannot be attributed to a cause: an observed change could come from the prompt, the tool schema, a model version bump, a decoding parameter change or a serving-stack deploy, and the trace does not distinguish them.

Every logged call must carry: model version and weights hash, system prompt hash, tool schema hash, decoding parameters, serving-stack build id, trace pin hash and turn index. Only with these fields can a redundancy delta be assigned to a single moved variable.""",
  h="H98: without the full provenance field set, at least one observed redundancy delta in the programme's history cannot be attributed to a single variable, because more than one pinned quantity changed between the compared runs (ESTIMATE; derivation: prompts, model versions and serving builds change on independent schedules and nothing forces them to be aligned in the absence of a pin; refuted if a provenance audit shows exactly one field differing between every compared pair of runs).",
  exp="Audit the existing trace store: for each historical pair of runs used to claim an effect, diff the provenance field set and count how many fields differ. Report the number of comparisons where more than one field moved.",
  gate="Rollback gate: an effect claim is withdrawn if its provenance diff shows more than one field changed. Claims cannot be reinstated without a fresh single-variable replay.",
  risks=["missing provenance fields make historical effect claims unfalsifiable and unreproducible",
         "hash fields that are recorded but never asserted on give false confidence that a pin was honored"],
  ev=["provenance field diff for every historical run pair used to support an effect claim",
      "assertion in the harness that all provenance fields except the moved one are byte-identical across arms"],
  qd=(2,1,2), conf=0.67),

 dict(v=100, f="Honest default: no intervention until the metric and its noise band are established.",
  body="""The defensible default in this scenario is inaction on the policy and action on the instrument. Redundancy has not been shown to be a material cost driver until the decomposition exists; the detector's precision is unknown until adjudication exists; and the noise band of the accuracy guard is unknown until the baseline is repeated.

Sequence: (1) repeat the baseline at least three times to establish the noise band of every guard metric; (2) adjudicate the detector; (3) decompose redundancy by template and turn index; (4) only then consider a reversible context-level intervention. Fine-tuning is last, because it is the least reversible step available.""",
  h="H100: repeating the unchanged baseline three times produces a spread in redundant-call rate comparable to the effect size claimed for at least one candidate intervention, meaning that intervention is indistinguishable from run-to-run noise (ESTIMATE; derivation: sampling and decoding stochasticity both perturb the token-level tool-call decision, so a nonzero baseline spread is structurally guaranteed and small claimed effects sit inside it; refuted if the baseline spread is materially smaller than every claimed effect).",
  exp="Run the unchanged configuration three or more times on the identical hash-pinned trace with only the sampling seed varying. Report the spread of redundant-call rate, end-task accuracy and p95 latency. Compare every claimed intervention effect against that spread before it is discussed as real.",
  gate="Rollback gate: no intervention is funded before the baseline noise band is published. Any claimed effect smaller than the baseline spread is reported as not-detected, not as a small win.",
  risks=["intervening before the noise band is known produces effect claims that cannot be distinguished from seed variation",
         "the least reversible intervention (fine-tuning) is often attempted first because it is the most visible piece of work"],
  ev=["three or more repeated baseline runs with only the seed varying, and the resulting spread per metric",
      "explicit comparison of every claimed effect size against the published baseline spread"],
  qd=(2,1,2), conf=0.72),

 dict(v=101, f="Rollout, guardrails and the single-command revert.",
  body="""Whatever intervention survives the evidence gates ships behind a flag with a single-command revert and a staged exposure ramp, because the failure mode — silent over-abstention on the items that most need the tool — degrades correctness rather than availability and will not trigger an availability alert.

Ramp exposure in stages with a hold at each stage long enough to accumulate the pre-registered sample on the arithmetic-heavy stratum, not merely on aggregate traffic. Alert on the stratified accuracy guard and on false-suppression counts from the shadow comparator, not on redundant-call rate, which will improve by construction.""",
  h="H101: an alert configured on aggregate accuracy alone fails to fire during a rollout that degrades the arithmetic-heavy stratum beyond its band, because the stratum is a minority of traffic and its loss is diluted below the aggregate alert threshold (ESTIMATE; derivation: dilution of a minority stratum's loss in an aggregate mean is arithmetic; refuted if the stratum's traffic share is large enough that its in-band loss limit still exceeds the aggregate alert threshold).",
  exp="Replay a deliberately over-suppressing arm through the alerting configuration in shadow and record which alerts fire and after how many items, comparing an aggregate-accuracy alert against a stratified one on the identical trace.",
  gate="Rollback gate: single-command revert, mandatory and non-discretionary, triggered by a stratified accuracy breach, any false suppression on a tool declared PURE, or a p95 latency breach. Revert first, diagnose afterwards.",
  risks=["correctness regressions do not trigger availability alerts and can persist through a full ramp unnoticed",
         "alerting on redundant-call rate rewards the intervention for doing exactly what it was built to do and detects nothing"],
  ev=["alert configuration showing stratified accuracy and false-suppression alerts, with thresholds and evaluation windows",
      "rehearsed revert with the measured wall-clock time from decision to full traffic restoration"],
  qd=(2,1,3), conf=0.69),
]
