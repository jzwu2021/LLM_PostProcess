"""Question angles for the v0.4 corpus.

Six task types times five angles gives thirty genuinely different questions per
mechanism. Each angle consumes a different subset of the mechanism's fields, so
two records built from the same mechanism do not restate one another.

Field-usage discipline, enforced by verify.py:
  * sentence fields (symptom, chain, metric, signature, confounders, fixes,
    rollback, wrong_claim, wrong_why, threshold, cost, scaling) are authored as
    complete sentences and may only appear in sentence position;
  * phrase fields (title, options, tradeoff, flip, falsifier) are authored as
    lowercase clauses without terminal punctuation and may be spliced inline.
Never lowercase a sentence field into the middle of a clause.

Angle index also selects the deployment setting, so every mechanism is asked
about across all five settings.
"""
from __future__ import annotations

from core import Mechanism, Setting

TASKS = ("diagnosis", "calculation", "decision", "experiment_design", "review", "runbook")


def sl(s: Setting, k: int) -> str:
    """Six phrasings of the same deployment, so questions do not all open alike."""
    v = (
        f"{s.gpu_count}x {s.accel} {s.mem_gb}GB over {s.interconnect}, serving {s.model} "
        f"({s.params_b}B, {s.dtype}, TP{s.tp}) at context {s.ctx} with a p99 target of "
        f"{s.slo_ms} ms and {s.concurrency} concurrent requests",
        f"a {s.model} deployment on {s.gpu_count}x {s.accel} ({s.mem_gb}GB each, TP{s.tp}, "
        f"{s.dtype}), context {s.ctx}",
        f"{s.gpu_count} {s.accel} accelerators connected by {s.interconnect}, running {s.model} "
        f"at {s.dtype} with {s.concurrency} concurrent requests",
        f"a {s.mem_gb}GB-per-device {s.accel} cluster of {s.gpu_count} running {s.model} "
        f"(TP{s.tp}, context {s.ctx})",
        f"{s.model} served at {s.dtype} across {s.gpu_count}x {s.accel}, p99 objective {s.slo_ms} ms",
        f"the {s.accel} fleet ({s.gpu_count} devices, {s.mem_gb}GB, {s.interconnect}) hosting "
        f"{s.model} with a {s.slo_ms} ms p99 objective",
    )
    return v[k % len(v)]


def _b(items) -> str:
    return "\n".join(f"- {x}" for x in items)


def _cfg(s: Setting) -> str:
    return (f"{s.gpu_count}x {s.accel}, {s.mem_gb}GB, {s.interconnect}, TP{s.tp}, {s.dtype}, "
            f"context {s.ctx}")


# --------------------------------------------------------------------------- #
# diagnosis
# --------------------------------------------------------------------------- #

def d0(m: Mechanism, s: Setting):
    q = (f"On {sl(s, 0)}, operators report the following. {m.symptom} You have the first hour and the "
         f"service is still taking traffic. State what you measure first and what result would "
         f"confirm the cause.")
    a = (f"Measure this before changing anything, because the change destroys the evidence.\n\n"
         f"{m.metric}\n\n"
         f"The hypothesis under test is {m.title}. {m.chain}\n\n"
         f"Confirming observation. {m.signature}\n\n"
         f"First action once confirmed. {m.fixes[0]}\n\n"
         f"Refuted if {m.falsifier}. In that case the leading alternative is this one. "
         f"{m.confounders[0]}")
    return q, a, "diagnosis", "medium"


def d1(m: Mechanism, s: Setting):
    q = (f"A colleague has already concluded that {m.title} explains a production issue on "
         f"{sl(s, 0)}. List the competing explanations and give a specific way to exclude each.")
    a = (f"The conclusion may be right, but three other causes produce the same surface symptom and "
         f"none of them are excluded yet.\n\n"
         f"1. {m.confounders[0]}\n"
         f"2. {m.confounders[1]}\n"
         f"3. {m.confounders[2]}\n\n"
         f"What separates the stated hypothesis from all three:\n\n"
         f"{m.metric}\n\n"
         f"{m.signature}\n\n"
         f"Until that measurement exists, treat the conclusion as unproven. The fix it implies is not "
         f"free. {m.fixes[2]}")
    return q, a, "diagnosis", "hard"


def d2(m: Mechanism, s: Setting):
    q = (f"On {sl(s, 0)} you suspect {m.title} and took the primary measurement. {m.metric} The result "
         f"is ambiguous: it moved, but not enough to decide. What second observation resolves it, and "
         f"why that one?")
    a = (f"An ambiguous primary metric usually means the effect is real but diluted by aggregation, or "
         f"that a second process is moving the same number in the opposite direction.\n\n"
         f"Resolve it by looking for the signature rather than the magnitude.\n\n"
         f"{m.signature}\n\n"
         f"That observation is specific to this mechanism and is not produced by the nearest "
         f"alternative. {m.confounders[1]}\n\n"
         f"Stratify before aggregating. {m.scaling}\n\n"
         f"If the signature is absent while the primary metric still moves, the hypothesis is wrong. "
         f"It is refuted if {m.falsifier}.")
    return q, a, "diagnosis", "hard"


def d3(m: Mechanism, s: Setting):
    q = (f"A suspected case of {m.title} appears only at full production scale on {sl(s, 0)} and "
         f"cannot be reproduced in a smaller test. The alerting rule in place is this. {m.threshold} "
         f"Explain why the small test passes, and how to diagnose this without a reproduction.")
    a = (f"Scale dependence is itself diagnostic information. {m.scaling}\n\n"
         f"That is why the smaller test passes: the mechanism needs contention or volume that only "
         f"production supplies. {m.chain}\n\n"
         f"Diagnose in place rather than reproducing. Collect the discriminating measurement "
         f"continuously and wait for the next occurrence.\n\n"
         f"{m.metric}\n\n"
         f"{m.signature}\n\n"
         f"Passive collection is safe. Forcing the condition in production is not, and it would change "
         f"the state you are trying to observe.\n\n"
         f"Threshold to alert on meanwhile. {m.threshold}")
    return q, a, "diagnosis", "hard"


def d4(m: Mechanism, s: Setting):
    q = (f"An incident on {sl(s, 0)} is over and the evidence was lost when the service was restarted. "
         f"The responder had applied this mitigation. {m.fixes[1]} Write the evidence-capture list "
         f"that would have made {m.title} provable next time.")
    a = (f"Restarting clears exactly the state that separates this cause from its rivals, so capture "
         f"must precede mitigation.\n\n"
         f"Capture in this order:\n\n"
         f"1. {m.metric}\n"
         f"2. {m.signature}\n"
         f"3. {m.confounders[0]}\n"
         f"4. {m.confounders[1]}\n"
         f"5. {m.confounders[2]}\n\n"
         f"Bind every item to a timestamp aligned with the request trace. Without alignment you can "
         f"show two things were both true and still not show which preceded the other.\n\n"
         f"Record the refutation condition as well: the hypothesis fails if {m.falsifier}. A "
         f"post-mortem that collects only supporting evidence is not a diagnosis.")
    return q, a, "diagnosis", "medium"


# --------------------------------------------------------------------------- #
# calculation
# --------------------------------------------------------------------------- #

def c0(m: Mechanism, s: Setting):
    qt = m.quant(s)
    q = (f"For {sl(s, 1)}, compute {qt.label}. Show the derivation and state which inputs the result "
         f"depends on.")
    a = ("Derivation:\n\n" + _b(qt.steps) +
         f"\n\nResult: {qt.value}.\n\n"
         f"Interpretation. {qt.interpretation}\n\n"
         f"This is an ESTIMATE from the stated configuration, not a MEASURED value. It holds only "
         f"within the following bound. {m.threshold}")
    return q, a, "calculation", "medium"


def c1(m: Mechanism, s: Setting):
    qt = m.quant(s)
    q = (f"On {sl(s, 1)}, how much headroom is there before {m.title} becomes the binding constraint, "
         f"and what fails first when it does?")
    a = (f"Headroom is bounded by {qt.label}. The binding steps are these:\n\n" +
         _b(qt.steps[-2:]) +
         f"\n\nThat gives {qt.value}. {qt.interpretation}\n\n"
         f"What fails first. {m.symptom}\n\n"
         f"It appears before any hard error, which is why the bound is crossed silently rather than "
         f"through an out-of-memory or a rejected request.\n\n"
         f"Watch for the signature. {m.signature}\n\n"
         f"Bound to hold. {m.threshold}")
    return q, a, "calculation", "hard"


def c2(m: Mechanism, s: Setting):
    qt = m.quant(s)
    q = (f"Derive the break-even condition for {m.title} on {sl(s, 1)}. Above what point does the "
         f"effect stop being ignorable?")
    a = (f"The break-even follows from the same arithmetic that sizes the deployment:\n\n" +
         _b(qt.steps) +
         f"\n\nAnchor value: {qt.value}.\n\n"
         f"Break-even rule. {m.threshold}\n\n"
         f"Below that point the effect is real but dominated by other terms, and optimising it wastes "
         f"effort. Above it the mechanism drives the behaviour. {m.chain}\n\n"
         f"The rule is an ESTIMATE and must be re-derived per deployment. It reverses when {m.flip}.")
    return q, a, "calculation", "hard"


def c3(m: Mechanism, s: Setting):
    qt = m.quant(s)
    q = (f"For {sl(s, 1)}, which single input dominates {qt.label}? Rank the inputs by sensitivity and "
         f"say which one you would measure rather than assume.")
    a = ("Baseline computation:\n\n" + _b(qt.steps) +
         f"\n\nBaseline: {qt.value}.\n\n"
         f"Sensitivity follows from the structure of that expression. Terms entering multiplicatively "
         f"dominate; reserve and overhead terms are second order. {qt.interpretation}\n\n"
         f"The one input to measure rather than assume:\n\n"
         f"{m.metric}\n\n"
         f"Every other input is read from the model or hardware configuration and is exact, so it "
         f"carries no uncertainty worth propagating.\n\n"
         f"{m.scaling}")
    return q, a, "calculation", "hard"


def c4(m: Mechanism, s: Setting):
    qt = m.quant(s)
    q = (f"Express the cost consequence of {m.title} on {sl(s, 1)}. Give the arithmetic a finance "
         f"reviewer can check.")
    a = ("Physical quantity first:\n\n" + _b(qt.steps) +
         f"\n\n{qt.value}.\n\n"
         f"Cost translation. {m.cost}\n\n"
         f"The figure a reviewer should challenge is not the unit price but the utilisation assumption "
         f"behind it. {qt.interpretation}\n\n"
         f"Label every figure. The device counts and byte sizes above are exact arithmetic; the spend "
         f"implied by them is an ESTIMATE whose error is dominated by achieved utilisation, which is "
         f"MEASURED only by running the workload.")
    return q, a, "calculation", "medium"


# --------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------- #

def e0(m: Mechanism, s: Setting):
    q = (f"On {sl(s, 2)}, choose between {m.options[0]} and {m.options[1]} to address {m.title}. "
         f"Justify the choice with the constraint that actually decides it.")
    a = (f"The deciding constraint is {m.tradeoff}. Everything else is preference.\n\n"
         f"Why that constraint exists here. {m.chain}\n\n"
         f"Take {m.options[0]} first. It is the reversible move on this deployment ({_cfg(s)}), and it "
         f"is sufficient while the following bound holds. {m.threshold}\n\n"
         f"Move to {m.options[1]} only after the measurement confirms the mechanism.\n\n"
         f"{m.metric}\n\n"
         f"{m.signature}\n\n"
         f"Choosing it first means paying for it without knowing the mechanism is present. {m.cost}\n\n"
         f"Expiry on this decision: revisit when {m.flip}.")
    return q, a, "decision", "medium"


def e1(m: Mechanism, s: Setting):
    q = (f"Two engineers disagree about whether {m.title} is occurring on {sl(s, 2)}. What single "
         f"measurement settles it, and what does each possible result imply?")
    a = (f"The measurement:\n\n{m.metric}\n\n"
         f"If it shows the signature, the mechanism is active. {m.signature}\n\n"
         f"Then act. {m.fixes[0]}\n\n"
         f"If instead {m.falsifier}, the hypothesis is refuted and the effort moves to the leading "
         f"alternative. {m.confounders[0]}\n\n"
         f"Agree the result-to-action mapping before the measurement is taken. Deciding what a number "
         f"means after seeing it is how a disagreement survives the evidence meant to end it.")
    return q, a, "decision", "medium"


def e2(m: Mechanism, s: Setting):
    q = (f"You chose {m.options[0]} to handle {m.title} on {sl(s, 2)}. Under what conditions does that "
         f"choice become wrong, and how would you notice?")
    a = (f"The choice is correct only while {m.tradeoff} holds in your favour.\n\n"
         f"It flips when {m.flip}.\n\n"
         f"You would notice through the discriminating metric rather than through end-to-end latency, "
         f"which lags and aggregates.\n\n"
         f"{m.metric}\n\n"
         f"{m.scaling}\n\n"
         f"Set the review trigger now rather than later. {m.threshold}\n\n"
         f"A decision without a re-evaluation condition becomes a permanent default by inertia, and "
         f"this one has a known expiry.")
    return q, a, "decision", "hard"


def e3(m: Mechanism, s: Setting):
    q = (f"Your team rejected {m.options[1]} as a response to {m.title} on {sl(s, 2)}. What did that "
         f"rejection cost, and when should it be revisited?")
    a = (f"The rejection was defensible but not free. The rejected option addresses the part of the "
         f"mechanism that the chosen path does not. {m.chain}\n\n"
         f"What you gave up is headroom against the symptom. {m.symptom}\n\n"
         f"The chosen path holds only while {m.tradeoff} remains true.\n\n"
         f"Revisit when {m.flip}.\n\n"
         f"Record the rejection with its reason and its expiry. An undocumented rejection is "
         f"re-litigated from scratch at the next incident, under time pressure and with worse "
         f"information than you have now.")
    return q, a, "decision", "hard"


def e4(m: Mechanism, s: Setting):
    q = (f"Propose a staged plan for addressing {m.title} on {sl(s, 2)}, where every stage is "
         f"independently reversible.")
    a = (f"Stage 1, reversible by configuration. {m.fixes[0]}\n\n"
         f"Gate it on the discriminating measurement. {m.metric}\n\n"
         f"Stage 2, reversible by redeploy. {m.fixes[1]}\n\n"
         f"Proceed only if stage 1 left the symptom present and the signature confirmed. "
         f"{m.signature}\n\n"
         f"Stage 3, structural and slow to reverse. {m.fixes[2]}\n\n"
         f"This one changes capacity planning, so it requires the stage 1 measurement to still hold at "
         f"the new scale.\n\n"
         f"Rollback at any stage. {m.rollback}\n\n"
         f"Do not compress the stages. Applying two together makes the responsible change "
         f"unidentifiable, and the next deployment inherits both without knowing which was needed.")
    return q, a, "decision", "medium"


# --------------------------------------------------------------------------- #
# experiment_design
# --------------------------------------------------------------------------- #

def f0(m: Mechanism, s: Setting):
    q = (f"Design a falsifiable experiment to test whether {m.title} is the cause of a regression on "
         f"{sl(s, 3)}.")
    a = (f"Hypothesis. H1: {m.chain}\n\n"
         f"Falsified if {m.falsifier}.\n\n"
         f"Primary metric. {m.metric}\n\n"
         f"Design. Replay one fixed request trace with the suspected condition present and absent, "
         f"holding model revision, {s.dtype} precision, TP{s.tp} layout and batching policy fixed. "
         f"Vary only that factor.\n\n"
         f"Confirming observation. {m.signature}\n\n"
         f"Pre-registered decision rule. {m.threshold}")
    return q, a, "experiment_design", "medium"


def f1(m: Mechanism, s: Setting):
    q = (f"An experiment on {sl(s, 3)} appears to confirm {m.title}. Name the confounders that could "
         f"have produced that result and how to control each.")
    a = (f"Three conditions produce a confirming result without the mechanism being present.\n\n"
         f"1. {m.confounders[0]} Control by holding it fixed across arms and reporting its value with "
         f"the result.\n\n"
         f"2. {m.confounders[1]} Control by stratifying rather than pooling, since pooling hides it.\n\n"
         f"3. {m.confounders[2]} Control by randomising arm order and repeating, since this one drifts "
         f"with time rather than with the treatment.\n\n"
         f"After controls, the result is informative only if the signature is present. {m.signature}\n\n"
         f"State the run-to-run variation band before interpreting any difference. {m.scaling}")
    return q, a, "experiment_design", "hard"


def f2(m: Mechanism, s: Setting):
    qt = m.quant(s)
    q = (f"How much data does an experiment about {m.title} on {sl(s, 3)} need before a difference in "
         f"the primary metric can be believed? Justify the sample rather than asserting it.")
    a = (f"Sample size is set by the variation of the metric, not by convention. Establish that band "
         f"first: run the identical configuration against itself several times and report the spread. "
         f"Any effect smaller than that spread is undetectable regardless of sample size.\n\n"
         f"Scale anchor for this deployment:\n\n" + _b(qt.steps) +
         f"\n\n{qt.value}. {qt.interpretation}\n\n"
         f"Requests are also not independent. {m.scaling}\n\n"
         f"Autocorrelated samples inflate the apparent sample size, so the effective count is smaller "
         f"than the request count and must be reported as such.\n\n"
         f"Pre-register the smallest difference worth acting on. Without it, a long enough run makes "
         f"any difference statistically visible and operationally meaningless.")
    return q, a, "experiment_design", "hard"


def f3(m: Mechanism, s: Setting):
    q = (f"Write the pre-registration for an experiment about {m.title} on {sl(s, 3)}: what must be "
         f"declared before the first run.")
    a = (f"Declare before running.\n\n"
         f"Hypothesis. {m.chain}\n\n"
         f"Refutation condition: the hypothesis fails if {m.falsifier}.\n\n"
         f"Primary metric. {m.metric}\n\n"
         f"Decision threshold. {m.threshold}\n\n"
         f"Rollback trigger. {m.rollback}\n\n"
         f"Declare also what will not count as evidence. End-to-end latency alone will not, because "
         f"the leading confounder moves it independently of the mechanism under test. "
         f"{m.confounders[0]}\n\n"
         f"Fix the environment in the record: {_cfg(s)}. A result without its configuration cannot be "
         f"compared with the next one and is therefore not reusable.")
    return q, a, "experiment_design", "medium"


def f4(m: Mechanism, s: Setting):
    q = (f"Describe how an experiment about {m.title} on {sl(s, 3)} could produce a confident and "
         f"wrong conclusion.")
    a = (f"Wrong refutation, from the wrong regime. {m.scaling}\n\n"
         f"An arm run below that point shows no effect, and the absence gets reported as evidence "
         f"against the hypothesis.\n\n"
         f"Wrong confirmation, from a rival cause. {m.confounders[2]}\n\n"
         f"The primary metric moves, the hypothesis is accepted, and the real condition survives the "
         f"experiment untouched.\n\n"
         f"Wrong summary, from a plausible reading. Someone will write it up as: \"{m.wrong_claim}\" "
         f"{m.wrong_why}\n\n"
         f"Guard against all three by requiring the signature rather than a magnitude, and by stating "
         f"in advance that the run is refuted if {m.falsifier}.\n\n"
         f"{m.signature}")
    return q, a, "experiment_design", "hard"


# --------------------------------------------------------------------------- #
# review
# --------------------------------------------------------------------------- #

def g0(m: Mechanism, s: Setting):
    q = (f"Review this claim about {sl(s, 4)}: \"{m.wrong_claim}\" State whether it is correct and why.")
    a = (f"The claim is not correct as stated. {m.wrong_why}\n\n"
         f"What is actually happening. {m.chain}\n\n"
         f"The claim would hold only if {m.flip}. That condition is checkable, so the disagreement is "
         f"resolvable by measurement rather than by argument.\n\n"
         f"{m.metric}\n\n"
         f"Corrected statement. {m.threshold}\n\n"
         f"Acting outside that bound is not supported by anything measured on this deployment.")
    return q, a, "review", "medium"


def g1(m: Mechanism, s: Setting):
    q = (f"A plan to address {m.title} on {sl(s, 4)} proposes going straight to {m.options[1]}. Review "
         f"the plan and identify what is missing.")
    a = (f"The plan may reach a working state, but it skips the step that makes the result "
         f"attributable.\n\n"
         f"Missing: the measurement that establishes the cause. {m.metric}\n\n"
         f"Without it the plan cannot separate success from a confounder changing at the same time. "
         f"{m.confounders[1]}\n\n"
         f"Missing: the cheaper reversible option. {m.fixes[0]}\n\n"
         f"If that suffices, the structural change is avoided entirely.\n\n"
         f"Missing: a rollback gate. {m.rollback}\n\n"
         f"The plan is also unbounded in scope. It holds only while {m.tradeoff}, and that condition "
         f"belongs in the plan as an expiry rather than as an unstated assumption.")
    return q, a, "review", "hard"


def g2(m: Mechanism, s: Setting):
    q = (f"You are the approver for a change addressing {m.title} on {sl(s, 4)}. What evidence do you "
         f"require before approving, and what would you reject on?")
    a = (f"Require as preconditions.\n\n"
         f"1. {m.metric}\n"
         f"2. {m.signature}\n"
         f"3. Explicit exclusion of the two leading alternatives. {m.confounders[0]} "
         f"{m.confounders[1]}\n"
         f"4. A stated rollback. {m.rollback}\n\n"
         f"Reject on any of: a benefit claimed from an ESTIMATE rather than a MEASURED run; a "
         f"comparison whose arms differ in more than the treatment; or a result quoted outside the "
         f"regime it was measured in. {m.scaling}\n\n"
         f"Approval carries an expiry: it lapses when {m.flip}. At that point the change is no longer "
         f"justified by the evidence that approved it.")
    return q, a, "review", "hard"


def g3(m: Mechanism, s: Setting):
    q = (f"A result about {m.title} was measured on a different deployment and is being applied to "
         f"{sl(s, 4)}. Is the transfer valid? Bound it.")
    a = (f"The mechanism transfers. The numbers do not.\n\n"
         f"What transfers. {m.chain}\n\n"
         f"That is a property of how the system works, not of a particular machine.\n\n"
         f"What does not transfer: any threshold or magnitude, because the deciding constraint is "
         f"{m.tradeoff}, and this deployment supplies its own values ({_cfg(s)}).\n\n"
         f"Re-derive rather than re-use. {m.threshold}\n\n"
         f"Treat the imported result as a hypothesis to test here, not as a conclusion. The cheapest "
         f"validation is the discriminating measurement. {m.metric}")
    return q, a, "review", "medium"


def g4(m: Mechanism, s: Setting):
    q = (f"Review this rollback plan for a change addressing {m.title} on {sl(s, 4)}: revert the flag "
         f"and confirm latency recovers. Is that sufficient?")
    a = (f"No, on two counts.\n\n"
         f"Latency recovery is not attribution. {m.confounders[2]}\n\n"
         f"A recovery coinciding with the revert is equally consistent with that, and with the other "
         f"leading alternative. {m.confounders[0]}\n\n"
         f"The change may also not be flag-revertible at all. {m.rollback}\n\n"
         f"A sufficient plan states the confirmation that the revert took effect, the expected "
         f"post-revert value, and a bounded time after which the revert is declared failed.\n\n"
         f"{m.metric}\n\n"
         f"{m.signature}\n\n"
         f"Revert together: any capacity or concurrency setting tuned while the change was live was "
         f"valid only under it. {m.scaling}")
    return q, a, "review", "hard"


# --------------------------------------------------------------------------- #
# runbook
# --------------------------------------------------------------------------- #

def h0(m: Mechanism, s: Setting):
    q = (f"Write the detection rule for {m.title} on {sl(s, 5)}: what to alert on, at what threshold, "
         f"and the reason for that threshold.")
    a = (f"Alert on the discriminating metric. {m.metric}\n\n"
         f"Threshold. {m.threshold}\n\n"
         f"Reason for a derived threshold rather than a round number: below it the effect is dominated "
         f"by other terms and the alert fires on normal variation; above it the mechanism drives the "
         f"behaviour. {m.chain}\n\n"
         f"Do not alert on the symptom alone. {m.symptom}\n\n"
         f"It is produced equally by the leading alternative, so that page would be unactionable. "
         f"{m.confounders[0]}\n\n"
         f"Put the confirming signature in the alert payload so the responder starts from evidence. "
         f"{m.signature}")
    return q, a, "runbook", "medium"


def h1(m: Mechanism, s: Setting):
    q = (f"Write the graded response for {m.title} on {sl(s, 5)}: what the on-call does at first "
         f"signal, at sustained signal, and at customer impact.")
    a = (f"First signal, no impact. Capture evidence before acting, because acting destroys the "
         f"attribution. {m.metric}\n\n"
         f"Sustained signal. Apply the reversible mitigation. {m.fixes[0]}\n\n"
         f"Confirm through the same metric, not through latency, which lags.\n\n"
         f"Customer impact. Apply the stronger mitigation and accept its cost. {m.fixes[1]} {m.cost}\n\n"
         f"Do not apply the structural change during an incident. {m.fixes[2]}\n\n"
         f"It is slow to reverse and its benefit depends on a measurement an incident does not permit.\n\n"
         f"Escalate if {m.falsifier}, because the runbook is then treating the wrong cause.")
    return q, a, "runbook", "hard"


def h2(m: Mechanism, s: Setting):
    q = (f"Define the rollback gate for a mitigation of {m.title} on {sl(s, 5)}: what observation "
         f"triggers reverting, and how quickly.")
    a = (f"Trigger. {m.rollback}\n\n"
         f"Bound the decision in time. Declare the mitigation failed if the confirming signature has "
         f"not changed within an interval fixed before applying it. {m.signature}\n\n"
         f"An unbounded wait becomes a permanent state.\n\n"
         f"Revert together with the mitigation any capacity or concurrency setting tuned while it was "
         f"active, since those were valid only under it.\n\n"
         f"After reverting, re-measure rather than assuming the prior state returned. {m.scaling}\n\n"
         f"Restoring the pre-change configuration does not automatically restore the pre-change "
         f"baseline.")
    return q, a, "runbook", "medium"


def h3(m: Mechanism, s: Setting):
    q = (f"A mitigation for {m.title} was applied on {sl(s, 5)} and the symptom is gone. Write the "
         f"verification that the mitigation, and not something else, is responsible.")
    a = (f"Disappearance of the symptom is the weakest evidence available, because a rival condition "
         f"produces the same relief. {m.confounders[2]}\n\n"
         f"Positive check: the signature should now be absent, and the metric should have moved in the "
         f"predicted direction by roughly the predicted amount.\n\n"
         f"{m.signature}\n\n"
         f"{m.metric}\n\n"
         f"Right direction with the wrong magnitude means the mitigation is not the mechanism.\n\n"
         f"Negative check: removing the mitigation should bring the effect back. If reverting is safe, "
         f"revert once deliberately. If it is not, record the causal claim as unverified rather than "
         f"confirmed. {m.rollback}\n\n"
         f"Bound the claim. {m.threshold}\n\n"
         f"Record the baseline with its configuration ({_cfg(s)}). {m.scaling}")
    return q, a, "runbook", "hard"


def h4(m: Mechanism, s: Setting):
    q = (f"Write the handover note for the next on-call about an unresolved {m.title} issue on "
         f"{sl(s, 5)}.")
    a = (f"State: unresolved.\n\n"
         f"Symptom observed. {m.symptom}\n\n"
         f"Leading hypothesis. {m.chain}\n\n"
         f"Supporting evidence so far. {m.metric}\n\n"
         f"Not yet excluded. {m.confounders[0]} {m.confounders[1]}\n\n"
         f"Applied so far. {m.fixes[0]}\n\n"
         f"The effect was partial, so the hypothesis is neither confirmed nor refuted.\n\n"
         f"What would refute it: the hypothesis fails if {m.falsifier}.\n\n"
         f"Do not apply the structural change without first obtaining the confirming signature, "
         f"because it is slow to reverse and would make the current evidence uninterpretable. "
         f"{m.fixes[2]} {m.signature}\n\n"
         f"Rollback trigger if things worsen. {m.rollback}")
    return q, a, "runbook", "medium"


ANGLES = (
    d0, d1, d2, d3, d4,
    c0, c1, c2, c3, c4,
    e0, e1, e2, e3, e4,
    f0, f1, f2, f3, f4,
    g0, g1, g2, g3, g4,
    h0, h1, h2, h3, h4,
)
assert len(ANGLES) == 30
