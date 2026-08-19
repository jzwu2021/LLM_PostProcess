import json, os

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
START, END = 2340, 2350
OUT = f"{EXP}/results/train-batch-0235.jsonl"

rows = [json.loads(l) for l in open(CORPUS) if l.strip()]
sel = rows[START:END]

# Per-variant analytical angle for the "agent redundantly calls calculator" scenario.
ANGLES = [
    ("Redundancy definition and detection metric",
     "Define redundancy precisely before measuring it: a tool call is REDUNDANT if its (normalized_tool_name, canonical_args_hash) tuple already appears in the current trajectory AND the prior call returned a non-error result that is still in the model's visible context window. Instrument with a per-trajectory call ledger keyed by SHA-256 of canonicalized args (sorted keys, numeric normalization to a fixed decimal form).",
     "redundant_call_rate = redundant_calls / total_calls (per trajectory, then macro-averaged); duplicate_depth = max repeats of one args_hash; wasted_tool_latency_ms = sum of latency of redundant calls; context_bloat_tokens = tokens consumed by redundant call+result pairs.",
     "H0: redundant_call_rate is unaffected by making prior tool results salient. Intervention: inject a compact 'TOOL RESULT CACHE' block immediately before the decision point listing (call, args, result) for the current trajectory. Falsified if redundant_call_rate drops by more than the 95% CI half-width across >=300 paired trajectories."),
    ("Root-cause separation: context loss vs policy habit",
     "Two distinct failure mechanisms produce the same symptom. (a) CONTEXT EVICTION: the earlier result was truncated or pushed out of the window, so the agent genuinely cannot see it. (b) POLICY HABIT: the result is visible but the trained policy emits a tool call token pattern anyway. These need different fixes, so the metric must separate them.",
     "visible_redundancy_rate = redundant calls where the prior result IS still in-window / total redundant; evicted_redundancy_rate = the complement; plus prompt_token_position_of_prior_result.",
     "H0: >50% of redundant calls are eviction-driven. Controlled experiment: hold the model fixed, run the same 300 tasks at two context budgets (e.g. 8k vs 32k tokens) with identical seeds and temperature 0. If redundant_call_rate is statistically unchanged between budgets, the eviction hypothesis is falsified and the cause is policy habit."),
    ("Memoization interceptor as the intervention",
     "The lowest-risk intervention is a deterministic middleware, not a model change: a per-trajectory memoization layer that intercepts tool calls, hashes canonical args, and on a hit returns the cached result annotated 'CACHED (from step N)' without invoking the tool. Scope the cache to a single trajectory and only to tools declared PURE (calculator, unit conversion) — never to tools with side effects or time-varying reads.",
     "cache_hit_rate; tool_invocations_saved_per_trajectory; end_to_end_latency_p50/p95_ms; task_success_rate delta (the guard metric).",
     "H0: memoization reduces tool invocations without reducing task success. Experiment: A/B on the same task set, same seeds, cache on vs off. Falsified if task_success_rate drops by more than 1 percentage point (one-sided test, alpha=0.05); that would indicate the agent depends on the re-call as an implicit re-read signal."),
    ("Cost and latency accounting",
     "Quantify the cost before choosing an intervention, otherwise the fix may cost more than the waste. Every redundant call costs: tool wall time, the tokens of the emitted call, the tokens of the returned result, and one extra decode turn of the model.",
     "wasted_decode_turns; wasted_prompt_tokens; wasted_completion_tokens; wasted_gpu_seconds = wasted_decode_turns * measured mean turn latency on the serving stack.",
     "H0: redundant calls account for <5% of end-to-end agent GPU time. ESTIMATE (arithmetic only, not measured on this cluster): if redundant_call_rate = 0.15, mean extra turn = 1 decode pass, and a decode pass is ~0.4 s at the deployed batch size, then a 20-call trajectory wastes ~3 calls * ~0.4 s ~= 1.2 s. This is an ESTIMATE derived from assumed per-turn latency; it must be replaced by MEASURED per-turn latency from the serving stack's request traces before any capacity decision."),
    ("Prompt-level intervention and its ceiling",
     "A prompt-only fix is the cheapest to ship and the easiest to roll back: add an explicit rule ('Before calling a tool, check whether the identical call already appears above; if so, reuse its result') plus a running results table. It has a ceiling: prompt rules do not survive context eviction, so it can only address the visible-redundancy subset.",
     "redundant_call_rate split by visible/evicted; instruction_adherence_rate = fraction of decision points where the agent explicitly referenced the cache block.",
     "H0: the prompt rule reduces VISIBLE redundancy but leaves EVICTED redundancy unchanged. Experiment: paired runs with/without the rule at a fixed context budget. Falsified if evicted_redundancy_rate also moves significantly, which would mean the rule changed the overall calling policy rather than just the visible-case behavior."),
    ("Guarding against over-correction (false suppression)",
     "The dangerous failure mode of any anti-redundancy intervention is suppressing a call that was NOT redundant: same arguments but a legitimately different expected result (retry after a transient error, or a tool whose output depends on external state). Purity must be declared per tool, and any non-pure tool must be excluded from suppression.",
     "false_suppression_rate = suppressed calls whose cached result differs from a shadow live re-execution / total suppressions; measured with a shadow-execute-and-compare mode in staging.",
     "H0: false_suppression_rate for tools declared PURE is 0. Experiment: run the cache in shadow mode (serve cached result, but also execute live and compare) for a full staging window. Falsified by any single mismatch on a PURE tool, which invalidates the purity declaration for that tool."),
    ("Training-side intervention (SFT / preference data)",
     "If the cause is policy habit rather than eviction, a data intervention is warranted: construct paired trajectories where the ideal continuation reuses the prior result instead of re-calling, and apply preference optimization on that pair. Mask loss to the assistant/tool-call segments only, so the gradient targets the calling decision.",
     "held-out redundant_call_rate on tasks unseen in training; task_success_rate on a separate general agent benchmark (regression guard); tool_call_format_validity_rate.",
     "H0: preference training reduces held-out redundancy without degrading unrelated tool use. Falsified if the general agent benchmark regresses beyond its own noise band, measured by repeating the baseline evaluation 3 times to establish that band before training."),
    ("Observability and trace schema",
     "None of this is measurable without a trace schema fixed in advance. Each tool call event must record: trajectory_id, step_index, tool_name, canonical_args_hash, result_hash, latency_ms, error_flag, prior_hit_step (nullable), prior_result_in_window (bool). Everything else is derived offline from this table.",
     "All headline metrics become SQL over the trace table, which makes them reproducible and auditable rather than ad-hoc.",
     "H0: the trace schema is sufficient to reconstruct redundant_call_rate without re-running the agent. Experiment: recompute the metric from traces and from a live instrumented run; falsified if the two disagree beyond floating-point tolerance, which indicates a missing field."),
    ("Rollout, guard metrics and rollback thresholds",
     "Stage the intervention: shadow mode (measure only) -> 5% of traffic -> 50% -> 100%, with a fixed rollback gate at each stage. The gate must be defined before the rollout starts, not negotiated after seeing the numbers.",
     "Primary: redundant_call_rate. Guards: task_success_rate, p95 end-to-end latency, tool error rate, false_suppression_rate.",
     "Rollback thresholds: revert immediately if task_success_rate drops >1 pp vs the concurrent control, or p95 latency rises >10%, or any false suppression is observed on a PURE tool. Each stage must run long enough to accumulate a pre-registered minimum sample size; stopping early on a favorable reading is not permitted."),
    ("Threats to validity of the controlled experiment",
     "The experiment is only informative if confounders are controlled: hold the model version, decoding temperature, seeds, tool implementations, tool latency distribution, and task mix fixed across arms. Randomize at the trajectory level, not the user level, and run arms concurrently so drift in the tool backend hits both arms equally.",
     "Report per-arm sample size, the paired difference with a confidence interval, and the pre-registered minimum detectable effect.",
     "H0 (null of the whole program): the intervention has no effect on redundant_call_rate. Falsified only by a paired difference whose CI excludes zero at the pre-registered sample size. Explicitly record that a single-run improvement without a CI is NOT evidence, and that a reduction in redundant calls is a system/runtime property, not a demonstration of improved model domain capability."),
]

recs = []
for i, r in enumerate(sel):
    m = {x["role"]: x["content"] for x in r["messages"]}
    su = m["user"]
    sa = m["assistant"]
    variant = 81 + i
    title, mech, metrics, hyp = ANGLES[i]
    ca = (
        f"Scenario variant {variant} — redundant calculator invocations by an agent. "
        f"Focus: {title}.\n\n"
        "Review note: the source record's assistant field is a grading rubric ('Answer should state assumptions, ...') "
        "rather than an actual answer, so it cannot serve as a training target and is rewritten below into a concrete "
        "response that satisfies that rubric.\n\n"
        "Assumptions (stated, not assumed known): a single-agent ReAct-style loop; the calculator tool is deterministic "
        "and side-effect free; tool results are returned into the same context window; evaluation is offline on a fixed task set.\n\n"
        f"1) Mechanism and definitions\n{mech}\n\n"
        f"2) Metrics\n{metrics}\n\n"
        f"3) Falsifiable hypothesis and controlled experiment\n{hyp}\n\n"
        "4) Evidence required before acting\n"
        "- Trajectory traces with the fields listed in the trace schema, covering at least the pre-registered sample size.\n"
        "- A baseline repeated at least 3 times to establish the noise band of every guard metric.\n"
        "- Tool purity declarations, reviewed and signed off per tool, before any suppression is enabled.\n\n"
        "5) Numeric discipline\n"
        "Every number in this answer that is not read from a trace is labelled ESTIMATE and carries its derivation; "
        "no per-cluster throughput, latency or utilization figure is asserted as MEASURED here, because none was measured "
        "in this review. Any figure used in a capacity or cost decision must be replaced by a MEASURED value from the "
        "serving stack's own traces.\n\n"
        "6) Rollback gate\n"
        "Any intervention ships behind a flag with a single-command revert. Rollback is mandatory (not discretionary) if "
        "task_success_rate regresses beyond its pre-registered band, p95 latency regresses beyond its band, or any false "
        "suppression occurs on a tool declared PURE.\n\n"
        "Scope note: this is an agent-loop and observability design. It says nothing about the model's underlying domain "
        "knowledge, and improvements in redundant_call_rate must be attributed to the runtime/system change unless a "
        "held-out, model-only evaluation isolates the model contribution."
    )
    recs.append({
        "source_id": r.get("id"),
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": su,
        "source_assistant": sa,
        "corrected_answer": ca,
        "quality_dimensions": {
            "technical_correctness": 2,
            "instruction_coverage": 1,
            "operational_safety": 2,
        },
        "risks": [
            "source_assistant is a grading rubric rather than an answer; training on it would teach the model to emit rubrics instead of solutions",
            "anti-redundancy suppression can hide legitimate retries or state-dependent tool reads if tool purity is misdeclared",
            "single-run metric improvements without confidence intervals can be mistaken for capability gains",
        ],
        "evidence_required": [
            "per-call trajectory traces with canonical args hash, result hash, latency and in-window flag",
            "repeated baseline runs establishing the noise band of task_success_rate and p95 latency",
            "shadow-mode compare results proving zero false suppression on tools declared PURE",
        ],
        "confidence": 0.62,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for rec in recs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs))
