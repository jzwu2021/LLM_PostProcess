import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT = f"{EXP}/results/train-batch-0228.jsonl"
START, END = 2270, 2280

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
rows_src = corpus[START:END]

COMMON = """Common frame (applies to every stance below).
Assumptions (must be restated by the answering engineer, not inherited silently):
A1. The agent is an LLM-driven tool-calling loop serving interactive traffic; a calculator tool is one of several registered tools and returns deterministic results.
A2. The failure under discussion is a redundant call: the model emits a tool call whose result it could have produced directly, or which it already obtained earlier in the same trajectory.
A3. Ground truth for "the answer was already known" is not directly observable at serving time; it must be approximated by an offline judge or by cache-hit analysis, and that approximation is itself a measurement instrument that needs validation.
A4. Exactly one variable moves per experimental arm: prompt, tool schema, decoding policy and model checkpoint are not changed simultaneously.
Mechanism, stated plainly:
- A tool call is a control-flow decision the model makes from its context. Redundant calls arise from three distinct causes that require different interventions: (i) the policy has learned that calling is always safe, because training data rewarded calling and never penalized the extra call; (ii) the model cannot reliably tell that it already has the value, which is a context-attention failure, not a policy failure; (iii) the tool description or system prompt instructs unconditional verification, in which case the model is correct and the specification is wrong.
- Every redundant call costs latency (one extra tool round trip plus one extra model turn to consume the result), tokens, and trajectory length, and it raises the probability of a downstream error because each additional turn is another chance to derail.
Measurement layer, minimum viable set:
- Unnecessary-call rate: redundant calls divided by total calls, where redundancy is adjudicated offline. Report it per tool and per task category, never as a single global mean.
- Tool success rate: fraction of calls that return without error, tracked separately from usefulness. A high success rate with a high redundancy rate is the exact pathology here.
- Final-answer correctness: the only metric that authorizes a change. Call-count reductions that cost correctness are regressions.
- Trajectory length: turns and tool calls per completed task, p50 and p95.
- Tool latency: per-call p50/p95, needed to convert call-count deltas into user-visible latency deltas.
- Recovery rate: fraction of trajectories that still succeed after a tool error or a wrong intermediate value. Suppressing calls can quietly destroy this.
Intervention ladder, cheapest and most reversible first: (1) fix the tool description and system prompt so unconditional calling is not instructed; (2) add a stop/no-tool option and evaluate it explicitly; (3) add a result cache so the redundant call is free rather than forbidden; (4) collect preference pairs over trajectories and train a preference or reward signal that penalizes the extra call while holding correctness fixed. Steps 1-3 are configuration and are revertible in one deploy; step 4 changes the checkpoint and requires the full evaluation gate.
Evidence policy: every number below that was not produced by a run on this hardware is labelled ESTIMATE and carries its derivation. Only values read out of named benchmark artifacts may be labelled MEASURED. This review reports no MEASURED values, because no benchmark was executed for it.
"""

CRITIQUE = """Critique of the source item: the prompt is a legitimate agent-behaviour question and does ask for metrics plus an intervention, with an explicit falsifiable hypothesis and a controlled experiment, but the corpus pair is degenerate - the assistant turn contains only a rubric describing what an answer should contain, not an answer. There is therefore no substantive content to keep, and the item is rewritten into a complete response that supplies the mechanism behind redundant tool calls, the boundary conditions that change the correct intervention, an explicit falsifiable hypothesis, a single-variable controlled experiment, the evidence artifacts required to adjudicate it, and a rollback gate. Every quantitative claim is labelled ESTIMATE and carries its derivation; no value here is MEASURED, because no benchmark run was performed for this review. This output is provisional teacher-B review material, not expert gold, and it is not evidence about any model's domain capability."""

STANCES = [
    ("Stance 280 - The redundancy metric is worthless until the offline redundancy judge is itself validated, because every downstream claim inherits that judge's error rate.",
     """A redundant call is defined by a counterfactual: would the final answer have been correct without this call? That counterfactual is not observable in the production trace, so teams substitute a judge - a rule, a second model, or a human rater. The moment that substitution happens, the reported unnecessary-call rate is a function of two unknowns, the agent's behaviour and the judge's accuracy, and an intervention that appears to cut redundancy by a third may have shifted the judge's failure mode instead.
The correct order of work is therefore inverted from the usual instinct: validate the instrument before optimizing against it. Build a small adjudicated set by replaying trajectories with the call removed and the cached prior value substituted, and check whether the final answer survives. That replay is the closest available approximation to the counterfactual and it produces labels that the cheap judge can be scored against.
Falsifiable hypothesis H280: H280: the offline redundancy judge agrees with replay-based ground truth at a rate high enough that a 30% relative reduction in judged redundancy corresponds to a real reduction rather than a judge artifact (ESTIMATE; derivation: if judge error is uncorrelated with the intervention it inflates variance only, but interventions that change output phrasing are exactly the ones that shift a text-based judge's decisions, so correlation is the default expectation. If judged redundancy falls while replay-labelled redundancy on a held-out sample does not, the claim is refuted).
Controlled experiment: Sample trajectories stratified by task category, produce replay-based labels for each, and score the cheap judge against them before and after the intervention is deployed. Hold the judge prompt and version fixed across both scorings, and report agreement per category rather than pooled, so a judge that fails only on arithmetic-heavy tasks is visible.
Rollback gate: no intervention is promoted on judged redundancy alone; promotion requires that the replay-labelled sample move in the same direction and that judge-versus-replay agreement not degrade after the change, otherwise the measurement is treated as broken and the change is reverted."""),

    ("Stance 281 - The intervention should start with the tool description and system prompt, because in a large fraction of cases the model is faithfully following an instruction to always verify.",
     """Before any training signal is contemplated, read what the agent was actually told. Tool descriptions written defensively - "use this tool for any arithmetic", "always confirm numeric results" - make the observed behaviour compliant rather than pathological. Penalizing it with a reward signal teaches the model to disobey its own specification, which generalizes badly and shows up later as under-calling on tasks where the tool was genuinely needed.
The prompt-level fix is also the only intervention that is fully reversible within a single deploy and carries no checkpoint risk, which is why it belongs first on the ladder. Its limitation is equally clear: if the redundancy persists after the instruction is neutralized, the cause is policy or attention, and the cheap fix has been correctly ruled out rather than wasted.
Falsifiable hypothesis H281: H281: rewriting the tool description to make calling conditional, with no other change, reduces the unnecessary-call rate materially while leaving final-answer correctness statistically unchanged (ESTIMATE; derivation: instruction-following models weight explicit tool-description directives heavily in the call decision, so removing an unconditional directive should move the call rate directly. If the rate is unchanged, the behaviour is not instruction-driven and the hypothesis is refuted, which itself is a useful result).
Controlled experiment: Run two arms on an identical replayed trace with the same checkpoint, decoding seed and tool backend; the only delta is the tool-description text, versioned and diffed in the run artifact. Report unnecessary-call rate, correctness and recovery rate per arm and per task category.
Rollback gate: revert the description if correctness drops on any category or if recovery rate after tool errors degrades, since a conditional instruction can suppress the retry that rescues a failed call."""),

    ("Stance 282 - A result cache makes the redundant call nearly free, which converts a behavioural problem into a cost problem and should be tried before touching the policy.",
     """If the same calculator query is issued twice in a trajectory, a memoized result returns without a network round trip and the marginal cost collapses to a few tokens of context. That does not make the behaviour good - trajectory length still grows and each extra turn is another derailment opportunity - but it removes the latency and tool-load component of the harm at near-zero risk, and it produces an exact, judge-free measurement of within-trajectory duplication as a side effect, because the cache hit rate is ground truth for one specific class of redundancy.
The boundary is that caching only addresses repeats of identical queries. A model that recomputes a value it could have derived from context without calling at all produces a cache miss and remains invisible to this mechanism, so the cache is a floor on the problem, not a solution to it.
Falsifiable hypothesis H282: H282: within-trajectory duplicate calls, measured directly as cache hits on normalized query keys, account for a substantial share of total redundant calls, such that caching alone removes most of the latency cost without any behavioural change (ESTIMATE; derivation: duplicate-key repeats are the mechanically simplest redundancy mode and require no semantic judgement to detect, so they are the natural first slice to quantify. If the cache-hit rate is near zero while the judge still reports high redundancy, the redundancy is semantic rather than literal and the claim is refuted).
Controlled experiment: Deploy the cache in shadow mode first - compute what would have been a hit without serving from it - so the hit rate is measured with zero behavioural risk, then enable serving and compare tool-call latency p95 and tool backend load across the two periods on comparable traffic.
Rollback gate: disable the cache if any staleness incident occurs, meaning a cached value served after the underlying inputs changed; correctness outranks the latency saving, and the cache must carry an explicit key that includes every operand."""),

    ("Stance 283 - Adding an explicit stop or no-tool action and evaluating it as a first-class option is the intervention that most directly targets the decision being made.",
     """The agent's per-step choice is not "call or answer" unless the no-tool path is represented as clearly as the tool paths. Where the action space is a list of tools plus an implicit fallback, the implicit option is under-specified and under-selected. Making abstention explicit - a named action with a description of when it applies - puts the two branches on comparable footing in the model's decision.
This also produces a clean evaluation target: a no-tool held-out set of items where the correct behaviour is to answer directly. Accuracy on that set is a direct measure of the over-calling tendency, independent of any judge, and it can be tracked as a regression gate on every subsequent checkpoint.
Falsifiable hypothesis H283: H283: exposing an explicit abstention action raises accuracy on a no-tool evaluation set without lowering accuracy on a tool-required set, so the change improves discrimination rather than merely shifting the calling threshold (ESTIMATE; derivation: a threshold shift moves both sets in opposite directions, whereas improved discrimination moves one without harming the other; measuring both sets is what separates the two. If the tool-required set degrades by a comparable amount, the intervention is a threshold shift and the claim is refuted).
Controlled experiment: Construct two disjoint evaluation sets - one where the tool is necessary, one where it is not - built from the same task distribution and labelled independently of the agent. Run the same checkpoint with and without the abstention action and report both accuracies with confidence intervals.
Rollback gate: remove the abstention action if the tool-required set regresses at all, since under-calling on tasks that need computation is a more damaging failure than over-calling on tasks that do not."""),

    ("Stance 284 - Preference or reward training is the only intervention that generalizes, but it must hold correctness fixed by construction or it will trade accuracy for brevity.",
     """Prompt and cache fixes address the instances they were written for. A learned signal changes the policy and therefore transfers to tools and task types that were never enumerated. The danger is equally structural: a reward that includes a penalty per tool call and a bonus for correctness has an exchange rate between them, and any exchange rate the designer picks will be exploited. The model discovers that skipping a call is cheap and being wrong is only somewhat expensive, and the redundancy metric improves while accuracy quietly erodes.
The construction that avoids this is to build preference pairs where correctness is held constant by selection: both trajectories in a pair reach the same correct final answer, and the preferred one is simply shorter. Under that construction the signal cannot express "be wrong faster", because no pair ever offers that trade.
Falsifiable hypothesis H284: H284: preference training on correctness-matched pairs reduces tool calls per task while leaving final-answer accuracy within noise, whereas a scalar reward mixing a per-call penalty with a correctness term reduces accuracy measurably at the same call reduction (ESTIMATE; derivation: the matched-pair construction removes accuracy from the comparison by design while the scalar reward leaves it tradeable, so only the latter has a gradient toward being wrong faster. If both preserve accuracy equally, the constructional argument is refuted and the simpler scalar reward is preferable).
Controlled experiment: Train two variants from the identical base checkpoint on the identical trajectory pool, differing only in the objective, and evaluate both on held-out tool-required and no-tool sets plus a recovery-after-error set. Report accuracy, calls per task and recovery rate for each.
Rollback gate: no checkpoint ships on a call-count improvement alone; promotion requires accuracy within a pre-registered noise band on every evaluation slice and no degradation in post-error recovery, with the prior checkpoint kept warm for immediate revert."""),

    ("Stance 285 - Latency is the metric the user actually feels, so call-count reductions must be converted into user-visible time before any of them counts as a win.",
     """A reduction from three calls to two is not a result. The result is the change in end-to-end p95 for a completed task, and the conversion depends on facts the call count does not carry: how long the tool takes, whether calls are serial or parallel, and how many extra model turns each call forces. A cheap local calculator behind a fast path may contribute a few milliseconds, in which case eliminating it changes nothing a user perceives while the extra model turn it triggered dominates the cost.
This reframes where effort should go. If the model turn to consume the tool result is the expensive part, the intervention should target turn count rather than call count, and those two are not the same quantity.
Falsifiable hypothesis H285: H285: the dominant latency contribution of a redundant calculator call is the additional model turn required to consume its result, not the tool round trip itself, so interventions must be evaluated on turns saved rather than calls saved (ESTIMATE; derivation: a local arithmetic tool returns in well under the time of a full model forward pass over an extended context, so the model turn is the larger term; the ordering flips only for tools with genuine network or compute cost. If measured tool latency dominates the added turn time, the claim is refuted for that tool).
Controlled experiment: Instrument each trajectory with per-phase timing - tool wall time, model turn time, orchestration overhead - and attribute the total. Compare arms on end-to-end p95 per completed task rather than on any per-call metric, using an identical replayed trace.
Rollback gate: an intervention that reduces call count without improving end-to-end p95 is not promoted, because it has added policy risk with no user-visible benefit; the pre-registered success criterion is stated in milliseconds before the run, not chosen afterwards."""),

    ("Stance 286 - Recovery after a tool error is the metric most likely to be destroyed by an anti-redundancy intervention, and it is the one teams forget to measure.",
     """The behaviour being suppressed - calling again when the value seems already available - is the same behaviour that rescues a trajectory after a tool returns an error, a truncated result or a value the model has reason to distrust. Push the calling threshold up and both disappear together, because from inside the policy they are the same decision under different context.
This makes recovery rate a mandatory paired metric rather than a nice-to-have. It also implies the evaluation set must contain injected failures; a trace of clean runs cannot reveal the regression, and a team that only replays healthy traffic will ship the regression and discover it during an incident.
Falsifiable hypothesis H286: H286: any intervention that reduces the unnecessary-call rate also reduces the post-error retry rate, and the two move together closely enough that recovery must be gated explicitly rather than assumed safe (ESTIMATE; derivation: both behaviours are produced by the same call-versus-answer decision and differ only in context, so a global threshold shift moves both; only an intervention that improves discrimination rather than shifting the threshold would separate them. If retry rate holds while redundancy falls, the intervention is genuinely discriminative and the claim is refuted in the good direction).
Controlled experiment: Build a fault-injection suite that returns tool errors, timeouts and malformed payloads at a fixed rate, and run every arm through it in addition to the clean trace. Report recovery rate - trajectories that still reach a correct final answer after an injected fault - alongside the redundancy metric for each arm.
Rollback gate: revert on any statistically significant recovery-rate regression regardless of how large the redundancy improvement is, since a silent loss of retry behaviour surfaces as a correctness incident during a tool outage, which is exactly when the system is least able to absorb it."""),

    ("Stance 287 - Aggregate metrics hide the problem, because redundancy is concentrated in specific task categories and a pooled mean lets a large easy segment mask a harmful one.",
     """Traffic is not homogeneous. Simple arithmetic embedded in conversational tasks may show heavy over-calling while multi-step quantitative tasks show none, or the reverse. A pooled unnecessary-call rate averages these into a number that no segment exhibits, and an intervention tuned on that number optimizes for whichever segment happens to be largest, which is a property of the traffic mix rather than of the problem.
Segmentation must be fixed before the experiment, not discovered after it. Post-hoc slicing until a favourable cut appears is the standard way an ineffective intervention gets promoted, and the defence is a pre-registered segmentation with a stated decision rule per segment.
Falsifiable hypothesis H287: H287: the unnecessary-call rate varies substantially across pre-registered task categories, such that the pooled mean misrepresents every individual segment and per-segment reporting changes which intervention is selected (ESTIMATE; derivation: the three redundancy causes - instruction, policy, attention - are triggered by different task shapes, so their prevalence should differ by category; a uniform rate across categories would suggest a single global cause and refute the claim).
Controlled experiment: Pre-register the segmentation and the per-segment decision rule, then report every arm's metrics per segment with sample sizes, refusing to report a pooled number as the headline. Traffic-mix weights are recorded so that a shift in mix cannot be mistaken for a behavioural change.
Rollback gate: promotion requires no segment to regress on correctness or recovery, not merely a favourable weighted average; any segment with insufficient sample size is declared undecided rather than silently folded into the pooled result."""),

    ("Stance 288 - Trajectory length is a proxy that becomes a target the moment it is optimized, so it needs a guard that ties it back to completed useful work.",
     """Turns and calls per task are attractive because they are cheap to compute and correlate with cost. That correlation is exactly what makes them dangerous under optimization pressure: the shortest trajectory is one that gives up immediately, and a policy nudged toward brevity can learn to terminate early with a confident, unsupported answer. Length falls, the redundancy metric improves, and the failure is invisible to both because the abandoned tasks never appear as tool-call errors.
The guard is to report length only over completed and correct trajectories, and to track the abandonment rate as a separate first-class metric. A length reduction accompanied by rising abandonment is a regression wearing a win's clothing.
Falsifiable hypothesis H288: H288: optimizing directly for trajectory length raises the early-termination rate on tasks that genuinely require multiple steps, and this effect is invisible in any metric computed only over completed trajectories (ESTIMATE; derivation: length-conditioned selection pressure has no term distinguishing a task finished efficiently from a task abandoned, so both are rewarded identically; only an abandonment metric computed over attempted rather than completed tasks can separate them. If early-termination rate holds flat under length optimization, the claim is refuted).
Controlled experiment: Report length conditioned on correctness, plus an abandonment rate over all attempted tasks, for each arm on an identical trace containing a known proportion of genuinely multi-step items. Multi-step items are labelled in advance by required tool count, independent of the agent's behaviour.
Rollback gate: revert any change where abandonment rises, even if mean trajectory length and unnecessary-call rate both improve; the pre-registered gate is stated on the abandonment metric before the run so it cannot be renegotiated afterwards."""),

    ("Stance 289 - The rollout mechanism is part of the intervention, because a behavioural change that cannot be reverted per-segment without a redeploy has no usable rollback gate.",
     """Every rollback criterion above presumes that reverting is fast and surgical. If the intervention is a checkpoint change shipped to all traffic at once, the only available revert is a full redeploy, and the gate degrades into a promise rather than a control. That asymmetry should influence which intervention is chosen: a prompt or cache change carries a per-segment kill switch by construction, while a trained policy change does not unless a second checkpoint is kept warm and routing can move traffic between them.
Stated plainly, the operationally correct answer prefers the intervention whose blast radius is bounded and whose revert is measured in seconds, and it accepts a smaller behavioural improvement in exchange for that bound until the measurement layer has been validated on real traffic.
Falsifiable hypothesis H289: H289: interventions deployed behind per-segment routing with a warm previous checkpoint achieve a materially shorter time-to-revert than those requiring redeploy, and that difference dominates the expected cost of the intervention when the regression probability is non-trivial (ESTIMATE; derivation: expected cost is regression probability multiplied by exposure duration; routing collapses the duration term to seconds while redeploy leaves it at minutes plus queueing, so for equal regression probability the routed path is cheaper by the ratio of those durations. If a redeploy path can be shown to revert equally fast in a drill, the claim is refuted).
Controlled experiment: Run a revert drill on a staging replica for each deployment mechanism, measuring wall-clock from alert firing to traffic fully served by the previous behaviour, and record whether any in-flight request was dropped. Do this before the intervention ships, not after.
Rollback gate: no intervention ships until its own revert has been demonstrated in a drill with a recorded timing artifact and zero dropped in-flight requests; the drill result, not the intended design, is what authorizes the rollout."""),
]

RISKS_COMMON = [
    "Source assistant turn is a grading rubric, not an answer; training on it teaches meta-commentary about answers instead of the underlying reasoning.",
    "No falsifiable hypothesis, no confounder list and no rollback gate, despite the prompt explicitly demanding them.",
]

EXTRA_RISKS = [
    ["Judge error correlates with the intervention being measured, so a reported redundancy reduction can be entirely instrument drift.",
     "Replay-based ground truth is expensive, so teams sample too few trajectories and report differences inside the noise band."],
    ["Neutralizing an unconditional verification instruction can suppress the retry that rescues a failed call.",
     "Prompt edits are often shipped untracked, leaving no artifact to attribute a later behavioural change to."],
    ["A cache serving a stale value after operands change converts a latency win into a correctness incident.",
     "Cache-hit rate measures only literal duplicate queries and gives a falsely reassuring picture of semantic redundancy."],
    ["An explicit abstention action can shift the calling threshold rather than improve discrimination, degrading tool-required tasks.",
     "A no-tool evaluation set built from the agent's own traces inherits its bias and cannot detect the failure."],
    ["A scalar reward mixing call penalty and correctness has an exploitable exchange rate that trades accuracy for brevity.",
     "Checkpoint-level interventions cannot be reverted per segment and expand the blast radius of any regression."],
    ["Call-count improvements that do not move end-to-end p95 add policy risk with no user-visible benefit.",
     "Success criteria chosen after seeing the timing data allow any result to be declared a win."],
    ["Anti-redundancy pressure suppresses post-error retries, and clean-traffic replays cannot reveal it.",
     "Recovery regressions surface during tool outages, when the system has the least capacity to absorb them."],
    ["Pooled means let a large easy segment mask a harmful one, so a promoted intervention can regress real traffic.",
     "Post-hoc slicing until a favourable cut appears manufactures significance from noise."],
    ["Optimizing trajectory length rewards early abandonment identically to efficient completion.",
     "Metrics computed only over completed trajectories are structurally blind to abandoned ones."],
    ["A rollback gate without a demonstrated revert path is a promise, not a control.",
     "Reverts that drop in-flight requests convert a behavioural regression into a visible availability event."],
]

EXTRA_EVIDENCE = [
    ["Replay-based redundancy labels on a stratified trajectory sample, with the replay procedure versioned.",
     "Judge-versus-replay agreement rate reported per task category, before and after the intervention.",
     "Judge prompt and model version pinned in the run artifact."],
    ["Versioned diff of the tool description and system prompt between arms.",
     "Unnecessary-call rate, final-answer correctness and post-error recovery rate per arm and per category.",
     "Decoding seed and checkpoint hash confirming only the prompt text differed."],
    ["Shadow-mode cache-hit rate on normalized query keys, measured before the cache serves any traffic.",
     "Tool-call latency p95 and tool backend request rate across the shadow and serving periods.",
     "Cache key definition showing every operand is included, plus an audit of any staleness incident."],
    ["Accuracy on a tool-required evaluation set and a disjoint no-tool set, with confidence intervals.",
     "Provenance of both evaluation sets, showing labels were not derived from the agent's own traces.",
     "Per-step action distribution before and after the abstention action was exposed."],
    ["Preference-pair construction log demonstrating correctness is matched within every pair.",
     "Accuracy, calls per task and recovery rate for both objectives from an identical base checkpoint.",
     "Pre-registered noise band for the accuracy gate, recorded before training."],
    ["Per-phase timing breakdown per trajectory: tool wall time, model turn time, orchestration overhead.",
     "End-to-end p95 per completed task for each arm on an identical replayed trace.",
     "Pre-registered success threshold stated in milliseconds."],
    ["Fault-injection suite definition with injected error, timeout and malformed-payload rates.",
     "Recovery rate per arm measured under injection, alongside the clean-trace redundancy metric.",
     "Statistical test and sample size supporting any claim that recovery is unchanged."],
    ["Pre-registered task segmentation with per-segment decision rules, timestamped before the run.",
     "Per-segment metrics with sample sizes for every arm, plus recorded traffic-mix weights.",
     "Explicit undecided marking for segments below the minimum sample size."],
    ["Trajectory length conditioned on correctness, plus abandonment rate over all attempted tasks.",
     "Independent labels for genuinely multi-step items, derived from required tool count rather than agent behaviour.",
     "Pre-registered abandonment gate recorded before the run."],
    ["Revert drill artifact per deployment mechanism with wall-clock from alert to full previous-behaviour serving.",
     "Count of in-flight requests dropped during the drill.",
     "Routing configuration showing per-segment traffic can be moved without redeploy."],
]

QD = [
    (3, 2, 3), (3, 2, 3), (3, 2, 4), (3, 2, 3), (3, 2, 3),
    (3, 2, 3), (3, 2, 4), (3, 2, 3), (3, 2, 3), (3, 2, 4),
]
CONF = [0.78, 0.80, 0.79, 0.77, 0.76, 0.78, 0.80, 0.79, 0.77, 0.78]

out = []
for i, src in enumerate(rows_src):
    m = {x["role"]: x["content"] for x in src["messages"]}
    head, body = STANCES[i]
    ca = f"Analytical stance under test: {head}\n\n{COMMON}\n{body}\n\n{CRITIQUE}"
    qd = QD[i]
    out.append({
        "source_id": src["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": ca,
        "quality_dimensions": {
            "technical_correctness": qd[0],
            "instruction_coverage": qd[1],
            "operational_safety": qd[2],
        },
        "risks": EXTRA_RISKS[i] + RISKS_COMMON,
        "evidence_required": EXTRA_EVIDENCE[i],
        "confidence": CONF[i],
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(out))
