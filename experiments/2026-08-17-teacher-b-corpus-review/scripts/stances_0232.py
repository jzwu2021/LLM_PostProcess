STANCES = [
    ("Stance 320 - Redundancy and omission are two sides of one threshold, and neither may be reported without the other.",
     """Every intervention discussed here moves a single latent quantity: the model's willingness to call. Push it down and redundant calls fall while omitted-necessary calls rise; push it up and the reverse. Reporting only the redundancy number is therefore not a partial report, it is a biased one, because the metric that would reveal the cost of the change has been left out of the frame.
The boundary condition is the composition of the evaluation set. A set dominated by items the model can answer unaided will show large redundancy gains and almost no omission cost, simply because there are few must-call items to fail. The two rates must be computed on separately constructed sets - a may-call set and a must-call set - or the trade-off is invisible.
Falsifiable hypothesis H320: interventions that reduce redundant calls also increase omitted-necessary calls on a must-call set, so the two rates move together and no configuration change reduces both simultaneously (ESTIMATE; derivation: prompt, schema and reward edits all act by shifting one decision threshold rather than by improving the model's discrimination between cases; a pure threshold shift cannot improve both error types at once. If some intervention lowers both rates, it improved discrimination rather than the threshold, and the claim is refuted for that intervention).
Controlled experiment: Run each arm against a paired may-call and must-call set built from the same task distribution, and plot redundancy against omission as a single operating curve rather than reporting two isolated numbers.
Rollback gate: any arm whose omitted-necessary-call rate rises beyond the must-call set's measured noise band is reverted regardless of how large its redundancy improvement is."""),

    ("Stance 321 - A result cache converts the redundancy problem from a policy problem into a cost problem, and that reframing should be attempted before any training is funded.",
     """If a repeated call returns from a cache, its accelerator cost is zero and its tool latency collapses to a lookup. What remains is the token cost of the extra turn and the extra derailment opportunity - real, but far smaller than the original framing suggests. Caching is deployable, revertible, and does not touch the checkpoint, which places it below reward shaping on any sane intervention ladder.
The boundary condition is tool purity. A calculator is deterministic and safe to cache indefinitely; a tool that reads mutable state is not, and caching it silently returns stale results. The cache policy must therefore be declared per tool in the schema, not applied globally, and tools default to uncacheable.
Falsifiable hypothesis H321: enabling a per-tool result cache removes the majority of the accelerator and tool-latency cost attributed to redundant calls, leaving a residual dominated by token cost and turn count (ESTIMATE; derivation: redundant calls are by definition repeats of an earlier call in the same trajectory, so they are cache hits by construction whenever the key normalization is correct; what a cache cannot remove is the model turn that consumes the result. If measured savings fall well short of the cached share, the cost was never in the tool execution and the claim is refuted).
Controlled experiment: Identical traffic replayed with the cache disabled and enabled, reporting tool time, accelerator time, token count and trajectory length separately so the residual is attributable.
Rollback gate: the cache is disabled for any tool whose declared purity cannot be demonstrated on a staleness test, and cache hit rate is monitored as a first-class metric so a silent key-normalization change is visible."""),

    ("Stance 322 - Preference training on trajectories is the most expensive rung of the ladder and must not be climbed until the configuration rungs are exhausted and measured.",
     """Collecting preference pairs over trajectories, training a reward signal and producing a new checkpoint costs annotation effort, accelerator hours and a full evaluation gate, and it produces an artifact that cannot be reverted in a deploy. Prompt, schema and cache changes cost hours and revert in minutes. Reaching for the checkpoint first is not ambition, it is a failure to sequence.
The boundary condition is whether the residual after configuration fixes is still material. If cache plus schema editing removes most of the measured cost, the remaining redundancy may not justify a training run at all, and that determination requires the configuration arms to have been measured first.
Falsifiable hypothesis H322: after the configuration interventions are applied, the residual redundancy cost is small enough that a preference-training run cannot be justified on cost grounds alone (ESTIMATE; derivation: the configuration rungs address the three named causes - instructed calling, execution cost, unbounded repetition - leaving only the discrimination failure, which is the narrowest of the three. If the residual remains the dominant term after configuration fixes, training is justified and the claim is refuted).
Controlled experiment: Measure the full cost stack at baseline, after schema edit, after cache, and only then decide whether to fund a training arm, with the decision threshold written down before the numbers arrive.
Rollback gate: a preference-trained checkpoint must clear the full generative evaluation gate including must-call omission and recovery rate, not merely the redundancy metric it was optimized against."""),

    ("Stance 323 - Recovery rate is the metric most likely to be silently destroyed by a redundancy fix, and it must be measured on deliberately perturbed trajectories.",
     """Suppressing calls makes the agent more reliant on its own earlier conclusions. When one of those conclusions is wrong - a mistyped argument, a tool error, a stale value - the redundant call was the mechanism by which the agent noticed and corrected. Remove it and the failure propagates to the final answer instead of being caught mid-trajectory.
This cannot be observed on a clean evaluation set, because clean sets contain few errors to recover from. Recovery must be measured on trajectories where an error is injected deliberately: a tool returning an error code, a tool returning a plausible but wrong value, a truncated observation.
Falsifiable hypothesis H323: arms that most reduce redundant calls show the largest degradation in recovery rate under injected tool errors, even where clean-set final-answer correctness is unchanged (ESTIMATE; derivation: re-checking is the observable behaviour that both produces redundancy and enables error detection; an intervention that cannot distinguish the two cases suppresses both. If recovery holds while redundancy falls, the intervention discriminated correctly and the claim is refuted).
Controlled experiment: Each arm evaluated twice - clean set and error-injected set - with injection type and rate fixed across arms and reported alongside every headline number.
Rollback gate: recovery rate on the injected set is a hard gate; any arm below the baseline's lower noise bound is reverted even if every clean-set metric improved."""),

    ("Stance 324 - The claim 'the answer is already known' is an assertion about the model's internal state that the evaluation cannot observe, and treating it as ground truth imports an unmeasured error.",
     """The framing assumes we can tell that the model knew the answer without calling. We cannot. What is observable is that the value appeared earlier in the visible context, or that a strong reference model answers correctly without the tool. Both are proxies, and both fail in specific directions: context presence overstates knowledge when the value was truncated or ambiguous, reference-model agreement overstates it when the reference is stronger than the model under test.
The consequence is that the redundancy label carries a proxy error that must be quantified before effects smaller than that error are reported as real.
Falsifiable hypothesis H324: the redundancy rate measured by context-presence and the rate measured by reference-model agreement disagree materially on the same trace, establishing a proxy-error floor beneath every quoted effect (ESTIMATE; derivation: the two proxies fail on disjoint item populations - truncation and ambiguity for the first, capability gap for the second - so their disagreement is expected to be non-trivial rather than incidental. If they agree closely, the proxies are interchangeable and the claim is refuted).
Controlled experiment: Label the same trace under both proxies, report the confusion matrix, and adopt the disagreement rate as the declared uncertainty floor for the programme.
Rollback gate: no effect smaller than the declared proxy-disagreement floor is used to authorize a deploy, promote a checkpoint, or close an investigation."""),

    ("Stance 325 - Per-tool and per-task-category reporting is mandatory, because a global redundancy mean can be flat while every constituent segment moves.",
     """A single global rate is a weighted average over a task mix that the team controls and frequently changes. Segments move in opposite directions, cancel, and the headline number reports no change while the system's behaviour has changed substantially. The reverse is equally common: a mix shift produces a headline movement with no behavioural change at all.
The boundary condition is segment sample size. Slicing far enough that individual segments carry few items produces noise that will be read as signal, so the reporting granularity must be fixed in advance against a minimum per-segment count.
Falsifiable hypothesis H325: holding the agent and checkpoint fixed and changing only the task mix moves the global redundancy rate by a margin comparable to the effect sizes attributed to interventions (ESTIMATE; derivation: redundancy propensity varies strongly by task type because arithmetic-heavy tasks offer many more calling opportunities than retrieval-heavy ones; a weighted mean over segments with divergent rates is sensitive to the weights. If the global rate is insensitive to mix, segments are homogeneous and the claim is refuted).
Controlled experiment: Recompute the same trace's global rate under several pre-registered task mixes and report the induced spread beside every intervention effect.
Rollback gate: every reported rate carries its task-mix version hash and per-segment counts; a comparison across differing mixes is rejected at review rather than footnoted."""),

    ("Stance 326 - The turn-limit interaction means redundancy and task-completion rate must be read together, or a fix will be credited for a truncation artifact.",
     """Trajectories that hit a turn cap are terminated, and their unspent redundant calls never occur. Any change that shortens trajectories therefore lets more trajectories finish inside the cap, which changes the population over which redundancy is averaged. A fix can look effective purely because fewer trajectories were truncated, or ineffective because more were.
The boundary condition is the cap's proximity to the trajectory-length distribution. If the cap sits far into the tail, the interaction is negligible; if it sits near the median, it dominates.
Falsifiable hypothesis H326: measured redundancy is sensitive to the turn cap at the caps currently in production use, so cap value must be pinned across arms for any comparison to be valid (ESTIMATE; derivation: caps are typically set close enough to the working distribution to bind on a non-trivial share of trajectories, and every bound trajectory has its call count censored. If varying the cap leaves redundancy flat, the cap is not binding and the claim is refuted).
Controlled experiment: Sweep the turn cap across several values with everything else fixed, reporting redundancy, completion rate and truncation share at each.
Rollback gate: the turn cap and truncation share are recorded in every artifact, and any comparison across differing caps is rejected rather than adjusted."""),

    ("Stance 327 - Multi-tool interference means a calculator-specific fix must be regression-tested against every other registered tool.",
     """Interventions land in shared surfaces: the system prompt, the general calling policy, the decoding configuration. A clause added to discourage calculator calls is read by the model when it considers the retrieval tool, the code tool and the shell tool as well. The intended effect is local; the actual effect is global, and the unmeasured tools are where the damage lands.
The boundary condition is whether the intervention is scoped to the tool schema - which is genuinely local - or to shared prompt text, which is not. This is a further argument for preferring schema edits.
Falsifiable hypothesis H327: a calculator-targeted instruction placed in shared prompt text measurably changes calling rates for unrelated tools, whereas the same instruction placed in the calculator's schema description does not (ESTIMATE; derivation: shared prompt text is in context for every tool decision, while a schema description is rendered adjacent to one tool's action surface; instruction effects generalize across superficially similar decisions. If both placements leave other tools unchanged, scoping is not load-bearing and the claim is refuted).
Controlled experiment: Report per-tool call rates for the full registered tool set in every arm, not only for the tool being targeted.
Rollback gate: any arm that moves a non-target tool's call rate beyond its noise band is reverted, and the intervention is re-scoped to the schema before being retried."""),

    ("Stance 328 - Serving-layer refusal and model-side suppression produce indistinguishable metrics but different failure modes, so the artifact must record which mechanism was active.",
     """A trajectory with no redundant call looks identical in the aggregate whether the model chose not to call or the orchestrator refused the call. The distinction matters enormously in incident response: a serving-layer bound can be disabled in a config push, while a checkpoint's learned behaviour cannot. Aggregating the two into one metric destroys the information needed to respond to a regression.
The consequence is a logging requirement, not an analysis one. Each suppressed call must be attributed at emission time to model choice or orchestrator refusal, and the counts reported separately in every artifact.
Falsifiable hypothesis H328: with attribution logging enabled, a material share of apparent model-side improvement in redundancy is attributable to orchestrator refusals rather than to changed model behaviour (ESTIMATE; derivation: refusal mechanisms are typically deployed alongside prompt and schema changes and their effect is not separated by default, so their contribution is folded into the model-side number. If attribution shows refusals contributed negligibly, the model-side attribution was correct and the claim is refuted).
Rollback gate: no redundancy improvement is attributed to a checkpoint unless attribution logs show the orchestrator refusal count was flat across the compared arms.
Controlled experiment: Enable attribution logging and re-analyse an already-reported comparison, publishing the corrected decomposition beside the original."""),

    ("Stance 329 - Closing position for this batch, restated without softening: the source assistant turn is a rubric, and the near-identical variant family multiplies that defect rather than diluting it.",
     """Across this batch every source item asks a legitimate agent-behaviour question and every source answer replies with a description of what a good answer would contain. Fine-tuning on that pairing teaches the model to produce grading commentary when asked an engineering question - a specific, observable, and unusually sticky failure mode, because the meta-response is fluent and superficially on-topic.
The variant structure makes it worse rather than better. Forty-odd items differing only in a scenario-variant integer carry one training signal repeated forty-odd times at full weight, which is precisely the condition under which a stylistic defect becomes dominant.
Falsifiable hypothesis H329: fine-tuning on the unmodified variant family produces a higher rate of rubric-style meta-responses on held-out engineering prompts than fine-tuning on a deduplicated set of substantive rewrites drawn from the same topic (ESTIMATE; derivation: repeated near-identical exemplars concentrate gradient signal on their shared surface form, and the shared surface form here is the rubric register itself. If meta-response rates match across the two variants, the duplication is not load-bearing and the claim is refuted).
Controlled experiment: Three-way comparison - full variant family, deduplicated exemplars, deduplicated substantive rewrites - on a held-out generative evaluation scored for meta-response rate and content quality.
Rollback gate: this batch's rewrites are provisional teacher-B output, not expert gold; they must clear independent adjudication before any of them is admitted to a training set, and none of them is evidence about any model's domain capability."""),
]

EXTRA_RISKS = [
    ["Redundancy reported without omitted-necessary-call rate is a biased report, because both rates are driven by one threshold.",
     "An evaluation set dominated by may-call items hides the omission cost of any suppression intervention."],
    ["Caching a tool that reads mutable state silently returns stale results and converts a cost fix into a correctness bug.",
     "A change to the cache key normalization alters hit rate invisibly and shifts every downstream cost attribution."],
    ["Preference training produces a checkpoint that cannot be reverted in a deploy, unlike every configuration rung below it.",
     "Optimizing against the redundancy metric directly invites the checkpoint to satisfy the metric without improving discrimination."],
    ["Suppressing re-checks removes the mechanism by which the agent detects wrong intermediate values, degrading recovery.",
     "Clean evaluation sets contain too few errors to expose a recovery regression at all."],
    ["The redundancy label rests on unobservable model knowledge and is approximated by proxies with directional, unquantified error.",
     "Effects smaller than the proxy-disagreement floor are reported as real when they are measurement artifacts."],
    ["A global redundancy mean can be flat while every task segment moves, and can move purely from a task-mix shift.",
     "Slicing to segments with few items manufactures noise that is then read as segment-level signal."],
    ["Turn-cap truncation censors call counts and changes the population over which redundancy is averaged.",
     "A fix that shortens trajectories is credited for a truncation artifact when the cap is not pinned across arms."],
    ["Interventions placed in shared prompt text change calling behaviour for every registered tool, not the targeted one.",
     "Damage lands on tools that are not being measured in the arm and therefore never appears in the report."],
    ["Serving-layer refusals and model-side suppression are indistinguishable in aggregate metrics but differ in revertibility.",
     "Attributing an orchestrator-driven improvement to the checkpoint misdirects incident response during a regression."],
    ["Fine-tuning on rubric-style answers teaches meta-commentary in place of engineering reasoning.",
     "Forty-odd near-identical variants concentrate that defect at full weight rather than diluting it."],
]

RISKS_COMMON = [
    "Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
    "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them.",
]

EXTRA_EVIDENCE = [
    ["Paired may-call and must-call evaluation sets built from one task distribution, with construction procedure recorded.",
     "Redundancy and omitted-necessary-call rates reported jointly as an operating curve per arm.",
     "Measured noise band on the must-call set, used as the revert threshold."],
    ["Per-tool cache purity declarations plus a staleness test result for every tool marked cacheable.",
     "Replayed identical traffic with cache off and on, decomposed into tool time, accelerator time, tokens and turns.",
     "Cache hit rate and key-normalization version hash recorded in every run artifact."],
    ["Full cost stack measured at baseline, after schema edit and after cache, with the funding threshold written before the numbers.",
     "Annotation volume, accelerator hours and wall-clock estimate for the preference-training arm, labelled ESTIMATE with derivation.",
     "Generative evaluation gate results for any trained checkpoint, including must-call omission and recovery rate."],
    ["Error-injected evaluation set with injection type and rate fixed and reported across all arms.",
     "Recovery rate per arm on both clean and injected sets, with baseline noise bounds.",
     "Trajectory samples showing the detection turn that a suppression arm removed."],
    ["Confusion matrix between context-presence labels and reference-model-agreement labels on the same trace.",
     "Declared proxy-disagreement floor, published as the programme's uncertainty floor.",
     "Reference model identity, checkpoint and prompt hash pinned for the labelling run."],
    ["Per-tool and per-task-category redundancy rates with per-segment item counts and a pre-registered minimum count.",
     "Global rate recomputed under several pre-registered task mixes, reporting the induced spread.",
     "Task-mix version hash attached to every quoted rate."],
    ["Turn-cap sweep reporting redundancy, completion rate and truncation share at each cap value.",
     "Trajectory-length distribution relative to the cap, showing whether the cap binds.",
     "Cap value and truncation share recorded in every artifact."],
    ["Per-tool call rates for the full registered tool set in every arm, including non-target tools.",
     "Placement record stating whether each instruction landed in shared prompt text or in a tool schema description.",
     "Noise bands for non-target tool call rates, used as the revert threshold."],
    ["Per-call attribution logs distinguishing model-side non-calling from orchestrator refusal.",
     "Refusal counts reported separately in every arm, verified flat before any checkpoint attribution.",
     "Re-analysis of at least one previously reported comparison with the corrected decomposition published."],
    ["Duplicate-cluster sizes across the variant family, with the near-duplicate definition specified.",
     "Three-way fine-tuning comparison of full variants, deduplicated exemplars and deduplicated substantive rewrites.",
     "Held-out generative evaluation scored for meta-response rate and content quality, with set construction recorded."],
]

QD = [
    (3, 2, 4), (3, 2, 3), (3, 2, 4), (3, 2, 4), (3, 2, 3),
    (3, 2, 3), (3, 2, 4), (3, 2, 4), (3, 2, 3), (3, 2, 4),
]
CONF = [0.80, 0.79, 0.78, 0.81, 0.78, 0.79, 0.80, 0.79, 0.78, 0.80]
