import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0244.jsonl"
START, END = 2430, 2440

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
("STANCE 11 - Decode-time constrained generation: forbid the tool sentinel with a logit mask once a matching result is already in context.",
 """Mechanism. The redundancy is emitted at decode time, so it can be suppressed at decode time. A grammar/logit processor in the serving stack (vLLM logits_processors or an outlines-style FSM) masks the tool-call start token whenever the runtime has already observed a tool-result for the normalized expression in the current trajectory. This is a hard constraint, not a soft nudge: the model cannot emit the call.
Falsifiable hypothesis H11. Masking eliminates >=95% of oracle-labeled redundant calls while final-answer correctness stays within 0.5 absolute points of baseline. If correctness drops more than that, the masked calls were not actually redundant and the oracle definition is wrong.
Metrics. masked_call_count, oracle-redundant recall of the mask, final correctness delta, per-token logits-processor overhead in microseconds, p95 TTFT and ITL under fixed concurrency.
Controlled experiment. Same weights, same seeds, same eval set. Arm A baseline, Arm B mask enabled. Because the mask is deterministic given the trajectory prefix, divergence between arms is attributable solely to the mask, which makes per-item paired analysis valid.
Boundary conditions. A logits processor runs per decode step on the critical path; it must be O(1) amortized against vocabulary size or it will show up as ITL regression. It also breaks if the tool sentinel is multi-token and the FSM only masks the first token.
Confounders. Enabling any logits processor in vLLM can disable some fast paths (e.g. certain speculative-decoding or cuda-graph configurations), so an ITL regression may be caused by the code path rather than by the mask's own cost. Measure with a no-op processor as a third arm to separate the two.
Numbers. 95% and 0.5 points are each an ESTIMATE used as a pre-registered accept threshold, not a MEASURED result. The no-op-processor control arm exists precisely because the overhead attribution is otherwise unfalsifiable.
Rollback. Disable the processor via config flag if ITL p95 regresses >5% or correctness drops >0.5 points. The flag must be runtime-togglable without a model reload.
Scope. Serving-stack claim. It changes what the deployed system emits and says nothing about the model's own knowledge.""",
 ["Logits processors can silently disable cuda-graph or speculative-decoding fast paths, causing ITL regression unrelated to the mask itself",
  "Multi-token tool sentinels defeat a first-token-only mask",
  "A hard mask converts a soft error (extra call) into a hard error (cannot call when genuinely needed)"],
 ["Three-arm ITL/TTFT table: baseline, no-op processor, real mask",
  "Per-item paired correctness diff with the list of items where the mask changed the answer",
  "Tokenizer dump proving the tool sentinel is single-token or the FSM covers the full sequence"],
 0.61),

("STANCE 12 - Contrarian: do not intervene at all until the redundancy is shown to cost something a stakeholder cares about.",
 """Position. 'Agent calls calculator twice' is an aesthetic complaint until it is converted into a number in a budget. Every intervention listed elsewhere consumes engineering time, adds a failure mode, and must be maintained.
Mechanism of the cost conversion. Redundancy cost = redundant_calls_per_trajectory x (tool_latency + round-trip decode tokens to re-read the result) x trajectory_rate. The decode component is the one that consumes GPU; the tool latency component may be free if the calculator is a local CPU function.
Falsifiable hypothesis H12. Removing all redundant calls improves neither p95 end-to-end latency by >=5% nor GPU-hours per 1k trajectories by >=5%. If H12 holds, the correct action is to close the ticket, not to ship a fix.
Metrics. GPU-seconds per trajectory, prefill vs decode token split, p95 end-to-end latency, redundant_calls_per_trajectory, and the cost of the intervention itself in engineer-days.
Controlled experiment. Offline oracle-strip: take existing logged trajectories, delete redundant calls and their results, recompute token counts. This is a paper computation requiring zero GPU and it bounds the maximum achievable win before anyone writes code.
Boundary conditions. The oracle-strip upper bound is only valid if removing the call does not change the model's subsequent tokens, which is false in general. It therefore bounds the win from above, not from below, and must be presented that way.
Numbers. The 5% bars are ESTIMATEs chosen as pre-registered materiality thresholds. The oracle-strip computation is MEASURED-eligible: it runs on existing logs and produces a real token count, and should be labeled MEASURED once run.
Rollback. Not applicable; the recommendation is inaction. The reversal condition is the oracle-strip showing a win above the materiality bar.
Scope. Prioritization and cost analysis. No capability claim of any kind.""",
 ["Upper-bound-only reasoning can be misread as a point estimate of the achievable win",
  "Closing the ticket may hide a correctness problem that happens to be cheap today but scales badly",
  "Engineer-day cost estimates are soft and easily gamed to justify a preferred conclusion"],
 ["Oracle-strip token accounting over logged trajectories, labeled MEASURED",
  "GPU-seconds per 1k trajectories broken into prefill and decode",
  "Written materiality threshold agreed before the analysis is run"],
 0.6),

("STANCE 13 - NCCL/collective framing: if the intervention is evaluated at scale, the eval harness itself becomes the bottleneck and will corrupt the measurement.",
 """Mechanism. Running N>=500 trajectories x multiple arms x a tau sweep is a throughput problem, not a modeling problem. On an 8-GPU single node with tensor parallelism, the eval harness contends with itself: concurrent request streams share the same KV cache pool, so arm B's longer prompts evict arm A's prefixes and the arms are no longer independent.
Falsifiable hypothesis H13. Running arms sequentially on a drained server yields UCR and latency numbers that differ from interleaved execution by more than the run-to-run seed noise. If they do not differ, interleaving is safe and evaluation cost halves.
Metrics. per-arm p95 latency under sequential vs interleaved execution, prefix-cache hit rate per arm, KV-cache utilization, preemption/recompute counts from the server metrics endpoint, and UCR per arm in both regimes.
Controlled experiment. Two regimes (sequential, interleaved) x two arms, same seeds. Report the seed-noise band first by running the baseline arm twice with different seeds; only differences exceeding that band count.
Boundary conditions. This concern applies to latency and throughput metrics. Correctness and UCR are, in principle, invariant to scheduling at temperature 0, so if UCR moves between regimes something is nondeterministic and must be found before any conclusion is drawn.
Confounders. Continuous batching means measured per-request latency depends on what else was in flight; tensor-parallel collectives add a synchronization point so a single slow rank inflates all co-batched requests.
Numbers. No MEASURED values here. The claim that UCR should be scheduling-invariant at T=0 is a structural argument; it is exactly the assertion the experiment is designed to falsify.
Rollback. If UCR is not scheduling-invariant, halt the intervention evaluation entirely and fix determinism first; do not publish arm comparisons from a nondeterministic harness.
Scope. Evaluation-infrastructure claim. It constrains what conclusions the experiment can support; it is not a model capability statement.""",
 ["Interleaved arms share a KV-cache pool and are not statistically independent",
  "Latency differences may reflect batch composition rather than the intervention",
  "Nondeterminism at T=0 invalidates paired analysis but is easy to overlook"],
 ["Server metrics: KV-cache utilization, preemption count, prefix-cache hit rate per arm",
  "Baseline-vs-baseline seed-noise band for every reported metric",
  "Determinism check: same trajectory replayed twice at T=0, byte-compared"],
 0.62),

("STANCE 14 - Treat it as a prompt-contract defect: the system prompt never told the agent that tool results persist.",
 """Mechanism. The cheapest hypothesis is that the policy is underspecified in natural language. If the system prompt does not contain an explicit rule such as 'a value returned by a tool in this conversation remains valid; do not recompute it', the model has no instruction to violate, and calling again is compliant behavior.
Falsifiable hypothesis H14. Adding one explicit persistence clause to the system prompt reduces UCR by >=15% relative, with no change to tool_success_rate. If a single clause moves the metric that much, all heavier interventions (training, decoding constraints, middleware) are premature.
Metrics. UCR, tool_success_rate, final correctness, instruction-conflict rate (cases where the new clause causes the agent to reuse a stale value that should have been recomputed), prompt token delta.
Controlled experiment. Single-variable arm: identical weights, identical eval set, one clause added. Also run an adversarial subset where the correct behavior IS to recall the tool because inputs changed; the clause must not degrade that subset.
Boundary conditions. Prompt fixes are non-durable: they are silently lost on template refactors and they consume context budget on every request. They are a diagnostic, not a permanent fix.
Confounders. Adding tokens to the system prompt changes the prefix and therefore prefix-cache behavior; latency deltas are not attributable to behavior. Also, a longer system prompt can dilute attention on other instructions, so unrelated instruction-following metrics must be re-run.
Numbers. 15% relative is an ESTIMATE serving as a pre-registered decision bar; it is deliberately set high enough that a null result is informative rather than merely noisy.
Rollback. Remove the clause if the adversarial stale-value subset regresses at all; correctness on changed inputs dominates call-count economy.
Scope. Prompt/scaffold claim. A prompt-induced metric move is explicitly not evidence of a model capability change.""",
 ["Prompt fixes silently disappear during template refactors",
  "A persistence clause can cause reuse of genuinely stale values when inputs change",
  "Longer system prompts can degrade unrelated instruction-following"],
 ["Adversarial subset results where recomputation is the correct behavior",
  "Full instruction-following regression suite before and after the clause",
  "Exact diff of the system prompt with token counts"],
 0.63),

("STANCE 15 - Instrument first: the intervention debate is unresolvable without per-call structured telemetry that does not exist yet.",
 """Mechanism. Every metric proposed elsewhere (UCR, recovery rate, tool latency percentiles) requires per-call records with trajectory id, call index, normalized arguments, result, wall-clock timestamps at request and response, and the model's own token offsets. Reconstructing this from prose logs is lossy and produces silently wrong denominators.
Falsifiable hypothesis H15. Once structured telemetry exists, the UCR computed from it will differ from the current prose-log-derived UCR by more than 10 percentage points. If so, every prior UCR number in the discussion is retracted.
Metrics. telemetry completeness (fraction of calls with all required fields), UCR from structured vs prose source, clock skew between agent host and tool host, and log volume in GB per 1k trajectories.
Controlled experiment. Dual-write for one window: emit both prose and structured records for the same traffic, then compare derived metrics on identical trajectories. This is a pure comparison with no behavior change and therefore no rollout risk.
Boundary conditions. Timestamps from different hosts are not comparable without NTP-verified skew bounds; latency percentiles derived across hosts are invalid if skew exceeds the effect size being measured. Structured logging of tool arguments may capture sensitive payloads and must be redacted at emit time, not at query time.
Numbers. The 10-point retraction bar is an ESTIMATE, pre-registered. Log volume must be MEASURED during the dual-write window before enabling fleet-wide, since retention cost is a real budget line.
Rollback. Sampling-rate knob: drop structured telemetry to a sampled fraction if storage or emit-path latency becomes material. Never silently switch back to prose-derived metrics without relabeling the numbers.
Scope. Observability claim. Necessary precondition for any of the other hypotheses; it produces no capability claim.""",
 ["Cross-host timestamps without verified NTP skew make latency percentiles invalid",
  "Structured argument logging can capture sensitive payloads if redaction is deferred to query time",
  "Telemetry volume and emit-path cost can themselves regress serving latency"],
 ["Dual-write comparison table: prose-derived vs structured-derived UCR on identical trajectories",
  "NTP skew measurement between agent and tool hosts",
  "Measured log volume per 1k trajectories and the redaction rule applied at emit"],
 0.66),

("STANCE 16 - If a training-side fix is chosen, mask correctly or the run silently teaches the wrong thing.",
 """Mechanism. Any SFT or preference stage aimed at reducing redundant calls must compute loss only on assistant-generated tokens, including the tool-call payload the model itself emits, and must exclude tool-result tokens returned by the environment. If tool results are included in the loss, the model is trained to predict the calculator's output, which is unrelated to and can actively harm the abstention behavior being targeted.
Falsifiable hypothesis H16. A masking-correctness audit on the tokenized training batches will show that the label tensor is -100 on every tool-result span and on every user span. This is a binary property of the data pipeline and is decided by inspection, not by a metric.
Metrics. fraction of label positions that are non-masked, per-role token counts, and a manual spot-check of decoded non-masked spans for N=20 randomly drawn examples.
Controlled experiment. Not a model experiment. Dump one training batch, decode every position where labels != -100, and verify by eye that only assistant-authored text appears. Any tool-result token in that decode is a hard stop.
Boundary conditions. The audit must run against the exact chat template used at training time. A template applied at data-prep time and a different one applied by the trainer is the classic silent failure, and comparing rendered strings byte-for-byte is the only reliable check.
Numbers. No ESTIMATE and no MEASURED figure is appropriate here: this is a pass/fail structural audit, and expressing it as a percentage would obscure that a single leaked span is a failure.
Rollback. Do not launch the training run. A masking defect is not recoverable after the fact by evaluation; the checkpoint must be discarded and the run repeated.
Scope. Training-pipeline correctness. Passing the audit proves the data is well formed, not that the resulting model is better at anything.""",
 ["Loss on tool-result tokens trains the model to imitate the environment rather than to abstain",
  "Divergent chat templates between data prep and trainer produce silent mask misalignment",
  "A percentage-style report can hide a single catastrophic leaked span"],
 ["Decoded dump of all non-masked label positions for a sampled training batch",
  "Byte-comparison of the rendered template from data prep and from the trainer",
  "Per-role token count table for the training set"],
 0.68),

("STANCE 17 - Model the agent as a control loop and specify the stopping rule explicitly rather than hoping it emerges.",
 """Mechanism. Redundant calls are a missing termination condition. Specify the loop: state = (question, set of resolved subgoals with their evidence); action = call tool | answer | give up; the loop terminates when every subgoal has evidence or the step budget is exhausted. Under that specification, calling a tool for an already-resolved subgoal is a detectable violation of the loop invariant, checkable by the scaffold without any model change.
Falsifiable hypothesis H17. Enforcing the invariant in the scaffold (reject the action, return a structured 'already resolved, value=X' message, let the model continue) reduces UCR to near zero without increasing final error rate. If final error rises, the invariant's notion of 'resolved' is too coarse.
Metrics. invariant-violation count, UCR, final correctness, extra turns consumed by the rejection message, step-budget exhaustion rate.
Controlled experiment. Arm A baseline; Arm B invariant enforced with a rejection message. Track whether rejections cause loops where the model retries the same call repeatedly; cap retries and count cap hits as failures.
Boundary conditions. The rejection message consumes turns and tokens, so a model that ignores it converts one wasted call into several wasted turns. A hard retry cap is mandatory, not optional.
Confounders. Step-budget exhaustion can rise for reasons unrelated to the invariant if the rejection text is verbose; hold the message length fixed and short.
Numbers. 'Near zero' is intentionally qualitative because the invariant is deterministic; the meaningful number is the retry-cap-hit rate, which is MEASURED-eligible from the run.
Rollback. Disable enforcement if retry-cap hits exceed the baseline failure rate, since that converts a cost problem into a correctness problem.
Scope. Scaffold/control-flow claim. Deterministic enforcement is a property of the harness, not of the model.""",
 ["Rejection messages can induce retry loops that cost more than the original redundancy",
  "A coarse definition of 'resolved subgoal' will block genuinely needed recomputation",
  "Step-budget exhaustion can rise from message verbosity rather than from the invariant"],
 ["Retry-cap-hit rate and distribution of retries per rejected action",
  "Per-item correctness diff on the subset where the invariant fired",
  "Written definition of subgoal resolution with the exact matching rule"],
 0.64),

("STANCE 18 - Separate the two failure modes: redundancy under certainty and redundancy under anxiety require different fixes.",
 """Mechanism. Two distinct generators produce the same surface symptom. (1) Certainty redundancy: the model knows the answer and calls anyway, a habit learned from trajectories where every step had a call. (2) Anxiety redundancy: the model does not trust the earlier result, often because the result was rendered ambiguously or a later turn contradicted it. A single UCR number averages these and any single intervention will help one and be neutral or harmful on the other.
Falsifiable hypothesis H18. Segmenting calls by whether the model's pre-call direct-answer distribution is confident (low entropy) separates the two modes, and the intervention effect sizes differ significantly across segments. If effects are identical across segments, the distinction is not real and should be dropped.
Metrics. per-segment UCR, per-segment intervention delta, pre-call answer entropy, and the correlation between entropy and whether the tool result contradicted the model's direct answer.
Controlled experiment. Compute entropy for every call site offline from logged logprobs, split into terciles, and re-analyze the existing arms per tercile. This requires logprob capture in the telemetry, which is a hard dependency on the instrumentation work.
Boundary conditions. Entropy is a proxy for confidence and is miscalibrated in general; a low-entropy wrong answer is exactly the dangerous case and must be reported separately rather than folded into the segment average.
Confounders. Entropy correlates with question length and difficulty, so the segments are not balanced; report per-segment sample sizes and avoid comparing raw segment means without stratification.
Numbers. No MEASURED values. Tercile splits are an ESTIMATE-level design choice made to guarantee equal segment sizes rather than to reflect any known threshold.
Rollback. If per-segment sample sizes fall below the level where the effect-size CI is narrower than the effect itself, report the aggregate only and state that segmentation was underpowered.
Scope. Analysis-design claim. It changes how results are read; it asserts nothing about model competence.""",
 ["Entropy is a miscalibrated confidence proxy and low-entropy wrong answers are the dangerous case",
  "Segment imbalance from difficulty correlation can manufacture spurious per-segment effects",
  "Requires logprob capture, creating a hard dependency on telemetry work"],
 ["Per-segment sample sizes with effect-size confidence intervals",
  "Joint distribution of pre-call entropy and tool-result-contradicts-model events",
  "Logprob capture coverage rate in the telemetry pipeline"],
 0.6),

("STANCE 19 - Operational safety: define the blast radius and the abort criteria before any arm touches production traffic.",
 """Mechanism. Every intervention discussed changes agent behavior for real users. The safe unit of change is a canary keyed on a stable hash of the tenant id, starting at 1% and gated on explicit SLO checks rather than on a human eyeballing a dashboard.
Falsifiable hypothesis H19. The chosen intervention holds all guardrail SLOs at 1% for a full traffic cycle including peak. Guardrails: final-answer correctness on the shadow eval set not worse than baseline by >0.5 absolute points, p95 end-to-end latency not worse by >10%, tool-error rate not worse at all, and zero new classes of unhandled exception in the agent loop.
Metrics. per-guardrail time series at 1/5/25/100%, canary-vs-control paired on the same tenant mix, exception taxonomy counts, and time-to-rollback measured in a drill.
Controlled experiment. Staged rollout with a pre-registered promotion rule; promotion is automatic only if all guardrails hold for a full cycle, and any single breach triggers automatic revert without discussion.
Boundary conditions. A full traffic cycle must include the diurnal peak, because latency guardrails are trivially satisfied at trough. Tenant-hash canaries prevent a single heavy tenant from dominating, but they also mean rare scenario families may not appear at 1% at all, so rare-family coverage must be checked explicitly.
Numbers. 0.5 points, 10%, and the 1/5/25/100 ladder are ESTIMATEs functioning as pre-registered guardrails. The time-to-rollback figure must be MEASURED in an actual drill before the rollout, not assumed.
Rollback. Single feature flag, runtime-togglable, no redeploy, no model reload. If rollback requires a restart of the serving process, the rollout does not start.
Scope. Deployment-safety claim about the system. It says nothing about whether the model got better.""",
 ["Canary at low percentage may not exercise rare scenario families at all",
  "Latency guardrails evaluated off-peak give false confidence",
  "If rollback requires a process restart, the blast radius is larger than the canary suggests"],
 ["Measured time-to-rollback from a live drill",
  "Guardrail time series across a full diurnal cycle at each rollout stage",
  "Rare-scenario-family coverage report for the canary traffic slice"],
 0.67),

("STANCE 20 - Synthesis: a dependency-ordered plan where each step's output is the next step's precondition, with explicit exit ramps.",
 """Ordering and rationale. Step 0, instrumentation (STANCE 15): without per-call structured records every downstream metric has an unverifiable denominator. Exit ramp: if dual-write shows prose-derived UCR was already accurate within the pre-registered band, skip re-deriving history. Step 1, cost materiality (STANCE 12) via zero-GPU oracle-strip: if the upper-bound win is below the materiality bar, stop here and close the ticket. Step 2, metric validity (counterfactual vs proxy estimator): if agreement is poor, no arm comparison is interpretable. Step 3, cheapest intervention first, the prompt clause (STANCE 14), because it is one variable and reversible in seconds. Step 4, deterministic scaffold enforcement (STANCE 17) or middleware caching, which fix the fleet without touching weights. Step 5, decode-time masking (STANCE 11), which requires serving-stack changes and a no-op control arm. Step 6, and only if steps 3 to 5 fail, a training-side fix, gated behind the masking audit (STANCE 16).
Falsifiable hypothesis H20. The ordering is optimal in expected-cost terms: no later step, run first, would have resolved the issue at lower total cost. This is testable retrospectively by recording, at each step, whether it resolved the problem and what it cost.
Metrics. cumulative engineer-days and GPU-hours per step, resolution status per step, and a final re-baseline against the original untouched system.
Boundary conditions. The ordering is only defensible if each step is genuinely reversible. Step 6 is not reversible cheaply, which is exactly why it is last.
Numbers. No MEASURED values here. All step costs are ESTIMATEs until logged; the retrospective is what converts them to MEASURED.
Rollback. Each step carries its own revert as described in its stance. The plan-level rollback is returning to the untouched baseline, which requires that the baseline configuration be pinned and reproducible from the start.
Scope. Engineering-process claim. Completing the plan demonstrates a working system-level workflow and is explicitly not evidence of model domain capability.""",
 ["Cost estimates per step are soft until logged and can be arranged to justify a preferred ordering",
  "Reversibility of intermediate steps must be verified, not assumed",
  "Baseline drift over a multi-step plan invalidates the final re-baseline unless the baseline is pinned"],
 ["Per-step decision log with engineer-days and GPU-hours, marked ESTIMATE or MEASURED",
  "Pinned, reproducible baseline configuration hash captured before step 0",
  "Final re-baseline comparison against the original untouched system"],
 0.65),
]

QD = [
 (2,2,3),(2,2,3),(2,3,3),(2,2,3),(2,3,3),
 (2,2,3),(2,2,3),(2,2,3),(2,3,3),(2,2,3),
]

rows = []
for i, s in enumerate(src):
    m = {x["role"]: x["content"] for x in s["messages"]}
    head, body, risks, ev, conf = STANCES[i]
    tc, ic, os_ = QD[i]
    rows.append({
        "source_id": s["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": head + "\n" + body,
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": risks,
        "evidence_required": ev,
        "confidence": conf,
    })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(rows))
print("IDS", rows[0]["source_id"], "->", rows[-1]["source_id"])
