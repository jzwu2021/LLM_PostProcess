import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0241.jsonl"
START, END = 2400, 2410

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
 ("STANCE 1 - Redundancy is a decision-boundary defect, not a prompt defect: instrument the abstention gate before touching the policy.",
  """Mechanism. A "calculator called when the answer is already known" event is a failure of the tool-vs-answer decision head, not of the calculator. In a ReAct-style loop the model emits a tool call whenever P(tool | context) exceeds the decoder's sampling threshold; SFT corpora that contain a tool call in nearly every trajectory push that prior close to 1, so the model never learns a competing "answer directly" action. The redundancy is therefore an artifact of the action prior, and prompt edits only shift it temporarily.

Falsifiable hypothesis. H1: unnecessary-call rate (UCR) on a held-out set of arithmetically trivial prompts is driven by the fraction of tool-bearing trajectories in the SFT mix. If we hold the model fixed and only change the decode-time system prompt, UCR drops by less than 5 absolute points; if we retrain with >=25% no-tool trajectories, UCR drops by >=20 absolute points. H1 is falsified if the prompt-only arm matches the retrain arm within noise.

Metrics. Primary: UCR = unnecessary tool calls / total tool calls, where "unnecessary" is labeled by an oracle that can answer the item without tools at >=99% accuracy. Secondary: final-answer exact match, tool success rate, mean trajectory length in steps, p50/p95 wall-clock per task, and recovery rate after a forced tool error. Guardrail: under-calling rate on genuinely hard arithmetic, because any intervention that suppresses calls will trade UCR against correctness.

Controlled experiment. Three arms, same base checkpoint, same eval seed set: (A) unchanged, (B) prompt-only abstention instruction, (C) SFT re-mix with no-tool trajectories. Stratify eval items into trivial / borderline / hard arithmetic and report per-stratum. n >= 800 items per arm gives roughly +/-3.5 absolute points at 95% confidence for a rate near 0.3 (ESTIMATE, from the normal approximation sqrt(0.3*0.7/800) ~= 0.016 SE, doubled for the interval); this is a sample-size derivation, not a measurement.

Confounders. Decode temperature differences across arms; eval items whose triviality label is itself contested; length bias in the judge; and tool latency variation on shared hardware that changes timeout-driven retries.

Evidence and rollback. Ship only if UCR drops >=10 absolute points AND final correctness on the hard stratum does not regress by more than 1 absolute point, both measured on the same fixed eval set with paired bootstrap. Roll back on any hard-stratum regression >1 point, or if p95 latency regresses >15% (ESTIMATE threshold chosen to stay inside typical serving SLO headroom)."""),

 ("STANCE 2 - Treat it as a cost-control problem first: the intervention that pays is a cached deterministic pre-check, not model surgery.",
  """Mechanism. Every redundant calculator call costs one extra decode turn plus one tool round-trip. On a vLLM-style server the decode turn re-prefills the growing context, so the marginal cost is superlinear in trajectory length once KV cache pressure forces preemption. The cheapest intervention is therefore a deterministic guard in the tool router: if the requested expression is already present in the trajectory with an identical result, return the cached result without dispatching, and log the event.

Falsifiable hypothesis. H1: >=40% of unnecessary calls are exact-duplicate expressions already computed in the same trajectory, so a pure memo cache removes them with zero effect on final answer accuracy. Falsified if duplicate share is <40% or if accuracy moves by more than 0.5 absolute points.

Metrics. Duplicate-call share, UCR, tokens per completed task, tool dispatch QPS, p95 end-to-end latency, KV cache utilization and preemption count on the serving side, and final correctness. Cost metric: GPU-seconds per completed task, which is the only number that translates directly to spend.

Controlled experiment. A/B at the router with a deterministic hash split on request id, 50/50, same model, same fleet, minimum one full weekday of traffic to cover diurnal load. Log every call with (expression hash, trajectory id, step index) so duplicate share is computed offline rather than inferred.

Confounders. Traffic mix shifts between arms during the window; cache hits masking a model that is degrading for other reasons; and expression normalization bugs that make textually different but semantically identical calls look non-duplicate, which biases duplicate share downward.

Evidence and rollback. Duplicate share and GPU-seconds-per-task must both be MEASURED from router logs on this fleet; the 40% threshold in H1 is an ESTIMATE derived from the observation that agent loops re-emit the same expression roughly once per two extra steps, and it must be replaced by the measured value before it gates anything. Roll back if final correctness drops >0.5 points, if any cache-returned result is ever observed to disagree with a fresh computation, or if the cache introduces cross-tenant leakage in an audit of 10k sampled entries. The cache must be trajectory-scoped, never global, or it becomes a correctness and isolation hazard."""),

 ("STANCE 3 - Measurement first: without a defensible label for 'already known', every reported UCR number is unfalsifiable.",
  """Mechanism. "The answer is already known" is a claim about the model's internal state, and we cannot observe it directly. Operationally we must replace it with an external proxy: either (a) the value appears verbatim earlier in the trajectory, or (b) a no-tool reference model answers the item correctly at high rate. Proxy (a) is precise but narrow; proxy (b) is broad but imports the reference model's competence as a confound. Publishing a single UCR without stating which proxy was used makes the metric unfalsifiable.

Falsifiable hypothesis. H1: the two proxies disagree on <=15% of flagged calls. If disagreement exceeds 15%, UCR is proxy-dependent and no cross-team comparison of UCR values is valid until the proxy is fixed by convention.

Metrics. Report UCR-a and UCR-b separately, plus their Cohen's kappa. Add per-item human adjudication on a stratified sample of 300 disagreements (ESTIMATE: 300 gives roughly +/-5.7 absolute points at 95% confidence for a rate near 0.5, from sqrt(0.25/300) ~= 0.029 SE doubled). Track annotator agreement so the adjudication itself is auditable.

Controlled experiment. Freeze one model and one eval set. Compute both proxies. Then run the intervention arms and report deltas under both proxies. An intervention that improves UCR-a but not UCR-b is suppressing literal repeats, not teaching abstention, and should be described that way.

Confounders. Reference-model contamination if it shares training data with the eval set; verbatim matching defeated by formatting differences (1,000 vs 1000); and adjudicator drift over a long labeling session.

Evidence and rollback. Do not accept any UCR improvement claim that reports only one proxy. Roll back the metric definition itself if kappa < 0.6, and block the intervention decision until the label is stabilized. This is a gate on the measurement, not on the model."""),

 ("STANCE 4 - The right intervention is a stop/no-tool evaluation added to the eval harness before any training change.",
  """Mechanism. Current harnesses score the final answer, so a trajectory that reaches the right number after three pointless calls scores identically to a clean one. The gradient of the evaluation therefore does not point away from redundancy. Adding an explicit no-tool subset (items solvable without tools) plus a stop-behavior subset (items where the correct action is to answer now) makes redundancy visible and creates a target that later training can optimize against.

Falsifiable hypothesis. H1: current models score >=95% on final answer for the no-tool subset while emitting tool calls on >=50% of those items. That gap, if real, proves the harness is blind to the defect rather than the model being incapable. Falsified if call rate on the no-tool subset is already below 20%.

Metrics. No-tool-subset call rate, stop-accuracy (fraction of items where the model terminates at the correct step), final correctness on each subset, trajectory length distribution, and tool latency contribution to total latency.

Controlled experiment. Build the two subsets by construction, not by filtering production traffic, so triviality is guaranteed by the generator rather than assumed. Hold out a third subset of genuinely tool-requiring items as the guardrail. Evaluate the same checkpoint on all three; no training change in this phase.

Confounders. Constructed items being stylistically distinguishable from real traffic, which lets the model pattern-match on surface form instead of learning abstention; and subset size too small to separate a 5-point change.

Evidence and rollback. The deliverable of this phase is the harness plus a baseline table of MEASURED rates on a frozen checkpoint; nothing in this phase is a model change. The 95% and 50% figures in H1 are ESTIMATE priors from the general observation that final-answer scoring is blind to trajectory shape, and the 2-absolute-point reproducibility band is an ESTIMATE chosen as roughly half the smallest effect we intend to act on. Gate: do not proceed to a preference or reward-signal intervention until the harness reproduces the same UCR within that band on two independent runs of the same checkpoint. If it does not reproduce, the harness is the bug."""),

 ("STANCE 5 - Preference optimization is the correct lever, but only after the reward is shown not to be gameable by silence.",
  """Mechanism. A pairwise preference signal that prefers the shorter tool-free trajectory when both reach the same answer directly targets the decision head. The danger is reward hacking: the model learns that not calling is always safer, so it stops calling on items where the tool is genuinely required, and accuracy collapses on the hard stratum while the headline UCR metric looks excellent.

Falsifiable hypothesis. H1: a DPO-style objective with pairs (tool-free correct, tool-using correct) reduces UCR by >=15 absolute points while hard-stratum accuracy falls by <=1 point. Falsified if hard-stratum accuracy falls more than 1 point, which would show the signal is teaching silence rather than judgment.

Metrics. UCR, hard-stratum accuracy, under-calling rate, KL divergence from the reference policy, mean trajectory length, and reward-margin distribution. Watch the KL term specifically: a large KL with a small UCR gain means the policy moved for reasons unrelated to the target behavior.

Controlled experiment. Three betas spanning an order of magnitude, one seed each for the sweep, then three seeds at the chosen beta to separate seed noise from treatment effect. Same preference data, same eval set, paired comparison against the reference checkpoint.

Confounders. Pair construction leaking length as a shortcut, since tool-free trajectories are systematically shorter; eval items whose stratum label is wrong; and reference-policy drift if the SFT checkpoint is re-trained mid-sweep.

Evidence and rollback. Require both UCR and hard-stratum accuracy MEASURED from the same run and reported paired. The 15-point target, the 1-point accuracy tolerance and the 3-point under-calling ceiling are all ESTIMATE, chosen before the sweep from the size of effect that would justify requalifying a checkpoint rather than from any prior run. Roll back if under-calling rate rises above baseline by more than 3 absolute points, or if KL exceeds the budget set before the sweep. Length-controlled pairs are mandatory; without them the result is not interpretable."""),

 ("STANCE 6 - Serving-side view: quantify what redundancy actually costs on this fleet before authorizing any training spend.",
  """Mechanism. A redundant call adds one prefill of the extended context plus one decode turn plus tool RTT. Under continuous batching the extra sequences occupy KV blocks for their full lifetime, so at high load the cost shows up as reduced admitted concurrency and preemption, not as a clean per-request latency delta. This means the true cost is load-dependent and cannot be extrapolated from a single-stream benchmark.

Falsifiable hypothesis. H1: at 70% of measured peak throughput, eliminating redundant calls increases sustainable QPS by >=8%. Falsified if the gain is under 3%, in which case redundancy is a correctness and trust problem but not a capacity problem, and the training spend should be justified on other grounds.

Metrics. Sustainable QPS at fixed p95 SLO, KV cache utilization, preemption rate, GPU-seconds per completed task, tool RTT distribution, and tokens prefilled per task. All of these must be MEASURED under load; none should be quoted from single-stream runs.

Controlled experiment. Replay a captured production trace against two identical server configs, one with the router-level duplicate guard enabled. Sweep offered load until p95 breaches SLO in each arm and compare the breach points. Pin the same model, same tensor-parallel degree, same batch settings.

Confounders. Trace replay losing the original arrival-time correlation; tool backend being the actual bottleneck rather than the GPU; and thermal or clock variation across a long sweep on the same hardware.

Evidence and rollback. Publish the load-sweep curves, not a single number, and label every figure MEASURED with the trace id and date. Roll back the guard if p99 latency regresses at any load point or if the guard's own lookup adds more than 1ms at p99 (ESTIMATE budget, chosen as roughly 1% of a typical multi-second agent turn)."""),

 ("STANCE 7 - Contrarian: some redundant calls are correct verification behavior, so the target should be net-harmful redundancy.",
  """Mechanism. Recomputing a value the model already emitted is sometimes a legitimate self-check, especially when the earlier value came from a long chain of reasoning where the model's own confidence is poorly calibrated. A blanket UCR target treats verification and waste identically and will delete the useful case along with the useless one. The correct target is redundancy that neither changes the final answer nor catches an error.

Falsifiable hypothesis. H1: among calls labeled unnecessary by the verbatim-repeat proxy, >=5% return a value that differs from the earlier emitted value, and on that subset final accuracy is higher than on trajectories without the repeat. Falsified if the differing-value share is under 5% or if accuracy on that subset is not higher, in which case verification framing is unsupported and aggressive suppression is safe.

Metrics. Split UCR into confirming-redundancy (same value returned) and correcting-redundancy (different value returned). Report final accuracy conditional on each. Also track how often a correcting call actually changes the emitted final answer, since a correction the model ignores is still waste.

Controlled experiment. Retrospective analysis on logged trajectories first, since it needs no model change and can falsify the whole stance cheaply. If the correcting share is material, run an arm that suppresses only confirming redundancy and leaves correcting behavior untouched, which requires the router to compare rather than short-circuit.

Confounders. Nondeterministic tools returning different values for benign reasons; expression normalization errors inflating the differing-value share; and selection effects where hard items both attract repeats and have lower baseline accuracy.

Evidence and rollback. Require the conditional accuracy table MEASURED from real logged trajectories before authorizing any suppression; the 5% differing-value threshold is an ESTIMATE set at the level below which the verification story cannot account for a material share of the cost, not a value observed anywhere. Roll back any suppression that reduces accuracy on the subset where corrections previously fired, even if aggregate accuracy is flat, because that pattern indicates we removed a real safety net."""),

 ("STANCE 8 - Operational safety: the intervention must not be deployed as a silent global default.",
  """Mechanism. Any change that suppresses tool calls alters agent behavior for every tenant simultaneously, including tenants whose workloads depend on the tool being called for auditability rather than for the numeric result. A finance-style tenant may require that every arithmetic step appear as a logged tool invocation; silently removing those calls breaks their audit trail even though correctness is unchanged.

Falsifiable hypothesis. H1: at least one deployed workload treats tool-call presence as an audit artifact, so a global suppression would be a compliance regression independent of accuracy. This is falsified only by an explicit inventory showing no such workload exists; absence of complaints is not evidence.

Metrics. Per-tenant call-rate delta, audit-log completeness, final correctness per tenant, and a change-failure indicator tied to tenant-reported incidents. Aggregate metrics must never be the deployment gate here, because a small tenant's total breakage disappears in a fleet average.

Controlled experiment. Staged rollout: internal traffic, then one opt-in tenant, then 5%, 25%, 100%, with a per-stage bake period long enough to cover that tenant's batch cycle. Provide a per-tenant flag so the guard can be disabled without a redeploy.

Confounders. Low-volume tenants producing statistically silent stages; bake periods shorter than the tenant's own reporting cadence, which delays the signal past the rollout.

Evidence and rollback. Require a written per-tenant inventory and an explicit opt-out mechanism before stage 3, plus MEASURED per-tenant call-rate deltas at every stage. The 5/25/100 percent stage ladder and the 1-point per-tenant accuracy tolerance are ESTIMATE, derived from wanting each stage to carry enough volume to surface a regression within one tenant batch cycle rather than from measured variance. Rollback triggers: any audit-completeness violation, any tenant-reported incident attributable to missing calls, or a per-tenant accuracy regression >1 absolute point. Rollback must be a flag flip, verified in a game day, not a code revert."""),

 ("STANCE 9 - Root-cause the data: redundancy is usually inherited from the trajectory generator that built the SFT set.",
  """Mechanism. If the SFT trajectories were produced by a stronger teacher that was prompted to "always show your work with tools," then every training example demonstrates calling, and the student faithfully reproduces the teacher's tic. In that case the model is behaving correctly with respect to its data, and prompt or decode fixes are fighting the training distribution rather than correcting it.

Falsifiable hypothesis. H1: tool-call density in the SFT corpus on trivially-solvable items exceeds 80%, and student UCR on the matching eval stratum is within 10 absolute points of that corpus rate. Falsified if student UCR is far below corpus density, which would mean the student already partially abstains and the data explanation is incomplete.

Metrics. Corpus-side: fraction of trivially-solvable training items containing a tool call, calls per trajectory, and the distribution across data sources. Model-side: UCR on the matching stratum. The comparison of these two is the actual experiment.

Controlled experiment. Purely offline corpus audit first, then a targeted re-mix that adds no-tool trajectories for trivially-solvable items while holding total token count constant, so the arm is not confounded by data volume. Retrain from the same base with the same schedule and seed.

Confounders. Triviality labels applied to training data by a different rule than to eval data, which breaks the comparison; data-volume differences masquerading as behavior change; and multiple data sources with different tics averaging into an uninformative aggregate.

Evidence and rollback. Report the corpus density table by source as MEASURED counts over the actual SFT files before proposing the re-mix; it is cheap and can kill the hypothesis without a training run. The 80% density and 10-point agreement band in H1 are ESTIMATE, chosen as the level at which the data explanation would be sufficient on its own rather than read off any prior audit. Roll back the re-mix if held-out general capability regresses beyond the pre-registered threshold, since adding no-tool data displaces other data at fixed token budget and that trade must be MEASURED, not assumed."""),

 ("STANCE 10 - Synthesis with an explicit ordering: cheap deterministic guard now, measurement harness next, training change only if both fail.",
  """Mechanism. The interventions differ enormously in cost and reversibility. A router-level duplicate guard is reversible in seconds and touches no weights. An eval-harness addition is reversible and produces the evidence needed to judge anything else. A preference-optimization run costs GPU time and produces a new checkpoint that must be requalified end to end. Ordering by reversibility rather than by expected effect size minimizes the cost of being wrong.

Falsifiable hypothesis. H1: the guard plus harness together account for >=60% of the achievable UCR reduction, making the training run unnecessary at the current tolerance. Falsified if residual UCR after both remains above the target, which is exactly the condition that justifies spending on preference optimization.

Metrics. Stage-wise UCR after each intervention, measured on the same frozen eval set; final correctness per stratum; GPU-seconds per task; and engineering time per stage, which is the cost side of the ordering argument.

Controlled experiment. Sequential with a frozen eval set and a locked baseline checkpoint, reporting deltas cumulatively and marginally. Do not re-tune the eval set between stages; any change to the harness after the baseline invalidates the comparison and forces a re-baseline.

Confounders. Sequential design cannot separate interaction effects, so a stage that appears ineffective may only look that way because the prior stage already captured its gain; and eval-set staleness if traffic shifts during a multi-week sequence.

Evidence and rollback. Each stage carries its own gate: guard rolls back on any correctness or isolation issue, harness rolls back on failure to reproduce within 2 absolute points across runs, training rolls back on hard-stratum regression >1 point or under-calling rise >3 points. Nothing here demonstrates model capability; it is an engineering plan whose claims are provisional until each MEASURED number exists."""),
]

DECISIONS = ["rewrite"] * 10
CONF = [0.62, 0.60, 0.64, 0.63, 0.59, 0.61, 0.57, 0.62, 0.60, 0.63]
QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

RISKS = [
 ["Rubric-style source_assistant lists topics but never commits to a mechanism or a threshold, so it cannot be graded as right or wrong.",
  "No distinction between ESTIMATE and MEASURED numbers, inviting fabricated precision downstream."],
 ["Source answer omits any cost model, so a reader could authorize a training run to fix a problem that a cache would remove.",
  "Trajectory-scoped vs global caching is a tenant-isolation hazard that the source never raises."],
 ["Source treats 'already known' as self-evident; without a labeling convention the headline metric is not comparable across teams.",
  "Proxy choice can be selected after the fact to favor a preferred intervention."],
 ["Source jumps to interventions without checking whether the evaluation can even detect the defect.",
  "Constructed eval subsets can be gamed by surface form rather than by learned abstention."],
 ["Preference signal can be satisfied by never calling tools, collapsing accuracy on genuinely hard items.",
  "Length shortcut in pair construction makes any reported gain uninterpretable."],
 ["Single-stream latency numbers do not transfer to continuous-batching serving; capacity claims made from them would be wrong.",
  "Tool backend may be the real bottleneck, misattributing the win to the model change."],
 ["Blanket suppression of repeats can delete legitimate verification behavior and lower accuracy on the items that most needed it.",
  "Nondeterministic tools can inflate the apparent rate of corrective repeats."],
 ["Global default rollout can break audit-trail expectations for tenants that require every arithmetic step to be logged.",
  "Fleet-average metrics hide total breakage of a small tenant."],
 ["Fixing the student while the teacher-generated corpus still demonstrates the tic guarantees regression on the next retrain.",
  "Adding no-tool data at fixed token budget displaces other data, a trade that must be measured."],
 ["Sequential staging cannot separate interaction effects and may under-credit a later stage.",
  "Changing the eval harness mid-sequence silently invalidates all prior deltas."],
]

EVID = [
 ["Per-stratum UCR and final-correctness table for arms A/B/C from one frozen eval set, with paired bootstrap intervals.",
  "SFT mix composition showing the fraction of tool-bearing trajectories per arm."],
 ["Router logs with (expression hash, trajectory id, step index) sufficient to compute duplicate share offline.",
  "GPU-seconds per completed task and p95 latency for both A/B arms over a full weekday."],
 ["UCR under both proxies plus Cohen's kappa on the same flagged-call set.",
  "Human adjudication results on a stratified sample of at least 300 disagreements, with annotator agreement."],
 ["Baseline table: call rate, stop-accuracy, and final correctness on the no-tool, stop-behavior, and tool-required subsets.",
  "Two independent harness runs on the identical checkpoint showing UCR reproducibility within 2 absolute points."],
 ["Paired UCR and hard-stratum accuracy per beta, plus KL from the reference policy and under-calling rate.",
  "Evidence that preference pairs are length-controlled."],
 ["Load-sweep curves to SLO breach for both server configs, labeled MEASURED with trace id and date.",
  "KV cache utilization and preemption counts at each offered-load point."],
 ["Retrospective table splitting redundant calls into confirming vs correcting, with final accuracy conditional on each.",
  "Rate at which a correcting call actually changes the emitted final answer."],
 ["Written per-tenant inventory of workloads that treat tool-call presence as an audit artifact.",
  "Per-stage rollout metrics with per-tenant call-rate delta and audit-log completeness."],
 ["Corpus audit table: tool-call density on trivially-solvable training items, broken down by data source.",
  "Held-out general-capability scores before and after the re-mix at constant token budget."],
 ["Stage-wise cumulative and marginal UCR deltas on a frozen eval set with a locked baseline checkpoint.",
  "Engineering time and GPU-seconds recorded per stage to support the ordering argument."],
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
