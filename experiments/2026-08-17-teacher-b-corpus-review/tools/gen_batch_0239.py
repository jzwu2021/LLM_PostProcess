import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0239.jsonl"
START, END = 2380, 2390

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

COMMON_TAIL = """
Falsifiable hypothesis (H1): gating redundant calculator calls reduces unnecessary_tool_call_rate by >=50% relative WITHOUT degrading final_answer_accuracy by more than 1.0 absolute point on the held-out eval. Null H0: accuracy drops >1.0 point or unnecessary calls fall <50% relative.

Controlled experiment: freeze model weights, decode params (temperature=0.0, top_p=1.0), tool backend version, and eval set. Randomize per-example assignment to arm A (current policy) and arm B (intervention) with a fixed seed; run >=3 seeds. Compare paired per-example outcomes with McNemar's test on correctness flips and a bootstrap CI on call-rate delta.

Measurements (all MEASURED at eval time, not estimated): unnecessary_tool_call_rate = redundant_calls / total_calls, where a call is redundant if the exact normalized argument tuple was already resolved earlier in the same trajectory OR the answer is derivable from already-emitted tokens by a deterministic checker; tool_success_rate; final_answer_accuracy; mean trajectory length in steps; p50/p95 tool latency; recovery_rate after an injected tool error.

Expected confounders: (a) eval-set arithmetic difficulty shifts which items legitimately need a tool; (b) prompt-template drift changing tool-schema salience; (c) latency noise from a shared tool backend; (d) the redundancy checker itself mislabeling legitimate re-verification. Control by fixing the eval split, pinning the template hash, running arms interleaved on the same backend, and hand-auditing a 50-example sample of redundancy labels.

Evidence needed before rollout: per-arm metric table with CIs, the 50-example redundancy audit, a no-tool ("stop") slice showing the model can answer without the tool, and a latency/cost delta.

Rollback gate: revert if final_answer_accuracy regresses >1.0 absolute point, or recovery_rate after injected tool failure drops >3 points, or p95 end-to-end latency rises >10% at fixed load. All numeric thresholds here are ESTIMATE-level policy choices, derived from the requirement that the guardrail must be cheaper than the error it prevents; they are not measured values.

This is a provisional teacher-B review answer, not expert gold, and it says nothing about any model's domain capability."""

STANCES = [
    ("Stance: treat redundancy as a *decoding-time* problem first — add a deterministic pre-call cache keyed on the normalized argument tuple.",
     "Mechanism: intercept the tool-call at the serving layer (e.g. the vLLM/Dynamo-side tool router), normalize arguments (canonical numeric form, sorted kwargs), and hash them. On a hit within the same trajectory, return the cached result and increment a counter instead of dispatching. Boundary condition: only safe for pure/deterministic tools; a calculator qualifies, a clock or a stateful DB does not. The cache must be trajectory-scoped, never cross-request, or you leak state between users.",
     "rewrite", 4, 4, 4, 0.72),
    ("Stance: the root cause is usually a missing *stop* signal in training data, not a bad tool schema.",
     "Mechanism: if the SFT corpus never shows a trajectory where the model answers directly, the policy has no gradient toward not calling. Fix by mining trajectories where the tool result was ignored or trivially re-derivable and relabeling them as direct answers, then re-running SFT with assistant/tool-call loss masking so the supervision lands on the decision token. Boundary condition: this only helps if the base model can actually do the arithmetic unaided — verify with a no-tool eval slice first, otherwise you are training it to guess.",
     "rewrite", 5, 4, 4, 0.74),
    ("Stance: measure per-trajectory *marginal information gain* of each call, not just a raw call count.",
     "Mechanism: for each call, compute whether the final answer changes when that call's result is ablated (replaced with a null and the trajectory re-rolled from that point with a fixed seed). Calls with zero flip rate across seeds are redundant by construction. Boundary condition: this is O(calls) re-rolls, so it is an offline diagnostic, not an online metric; sample a few hundred trajectories rather than the full set.",
     "rewrite", 4, 5, 4, 0.68),
    ("Stance: add an explicit budget in the prompt/scaffold before touching weights — cheapest reversible intervention.",
     "Mechanism: a hard max_tool_calls counter enforced by the agent loop plus a system-prompt clause that the tool is only for values not already computed. The counter is the safety net; the prompt is the behavioral nudge. Boundary condition: budgets can truncate legitimately long multi-hop tasks, so the budget must be set from the observed p95 of *correct* trajectories, not the mean of all trajectories, and must be per-task-class.",
     "rewrite", 4, 4, 5, 0.75),
    ("Stance: use a preference/reward signal (DPO or a light RM) where the only difference between chosen and rejected is the redundant call.",
     "Mechanism: construct contrastive pairs from the same prompt — chosen = shortest correct trajectory, rejected = same trajectory with an extra provably-redundant call — so the preference gradient isolates call parsimony from correctness. Boundary condition: if pairs are not matched on final answer, the model learns to be terse and wrong; enforce that both members of every pair have identical, correct final answers.",
     "rewrite", 5, 4, 4, 0.70),
    ("Stance: separate 'redundant' from 'defensive re-verification' before you optimize anything.",
     "Mechanism: label each repeat call as (i) identical args, identical result — pure redundancy; (ii) identical args after an observed tool error — legitimate retry; (iii) different args that happen to be derivable — arguably fine. Only class (i) should be penalized. Boundary condition: collapsing (ii) into the penalty will suppress retries and directly damage recovery_rate under a flaky backend, which is why recovery_rate is a rollback gate and not just a nice-to-have.",
     "rewrite", 5, 5, 5, 0.76),
    ("Stance: instrument at the trajectory-graph level so the metric survives scaffold changes.",
     "Mechanism: emit a structured trace per step (step_id, tool_name, arg_hash, result_hash, parent_step) and compute redundancy as repeated (tool_name, arg_hash) nodes within one trace DAG. This decouples the metric from any particular agent framework. Boundary condition: arg_hash must be computed after canonicalization or trivially different formatting ('2+2' vs '2 + 2') will hide redundancy and understate the problem.",
     "rewrite", 4, 4, 4, 0.69),
    ("Stance: treat the intervention as a serving-cost problem and justify it with a cost model, clearly marked as an estimate.",
     "Mechanism: each redundant call costs one extra tool round-trip plus the tokens to emit the call and consume the result. ESTIMATE: if a redundant call adds ~40 emitted tokens plus ~60 consumed result tokens and one round-trip, and redundancy is ~15% of calls, removing it saves roughly 15% of tool round-trips and a low-single-digit percentage of total tokens. Derivation: token counts are typical calculator-call sizes, the 15% is a placeholder pending measurement — none of these are MEASURED and all must be replaced with instrumented numbers before the cost argument is used to justify rollout. Boundary condition: if tool latency is small relative to decode time, the latency win may be within noise and the intervention should be justified on accuracy/robustness grounds instead.",
     "rewrite", 4, 4, 4, 0.66),
    ("Stance: run the experiment with an explicit negative control so you can detect metric gaming.",
     "Mechanism: include a third arm C that reduces tool calls by simply capping them at zero. Arm C should show a large accuracy drop; if it does not, the tool was never load-bearing on this eval and the whole study is measuring nothing. Boundary condition: arm C is a diagnostic only and must never be shipped; gate it behind an eval-only flag so it cannot reach production traffic.",
     "rewrite", 5, 4, 5, 0.73),
    ("Stance: define the rollout as a staged, reversible deployment with per-stage exit criteria.",
     "Mechanism: stage 1 shadow — compute the redundancy gate's decision without acting, log disagreements; stage 2 canary at a small traffic slice with the gate active and a kill switch; stage 3 full. Each stage promotes only if the metric table and the audit sample clear the gates. Boundary condition: the kill switch must be a config flag readable at request time, not a redeploy, or rollback latency exceeds the blast radius you were trying to bound.",
     "rewrite", 4, 5, 5, 0.74),
]

RISKS = [
    ["Trajectory-scoped cache could leak results across requests if scoping is wrong", "Caching a non-deterministic tool returns stale values", "Redundancy counter can be gamed by renaming arguments"],
    ["Relabeling data can teach the model to skip tools it actually needs", "No-tool competence may be absent in the base model", "Loss masking bugs move supervision to the wrong tokens"],
    ["Ablation re-rolls are expensive and can be run on an unrepresentative sample", "Seed sensitivity may make flip-rate unstable", "Offline diagnostic mistaken for an online SLO"],
    ["A hard budget can truncate legitimate multi-hop tasks", "Budget tuned on mixed task classes penalizes the hardest ones", "Prompt-only fix silently regresses when the template changes"],
    ["Unmatched preference pairs teach terseness over correctness", "DPO can degrade general instruction following", "Reward model overfits to call count"],
    ["Penalizing retries suppresses recovery under a flaky tool backend", "Mislabeling defensive re-verification as redundancy", "Class boundaries drift as tools change"],
    ["Uncanonicalized argument hashing hides redundancy", "Trace emission adds serving overhead", "Framework migration breaks trace schema"],
    ["Cost estimates presented as measurements", "Latency win may be inside noise", "Optimizing cost while accuracy quietly regresses"],
    ["Negative-control arm reaching production traffic", "Eval where the tool is not load-bearing invalidates conclusions", "Flag misconfiguration"],
    ["Kill switch requiring redeploy makes rollback too slow", "Canary slice unrepresentative of production traffic mix", "Stage gates skipped under schedule pressure"],
]

EVID = [
    ["Tool purity/determinism audit", "Cache scope unit test showing no cross-request hits", "Before/after redundant-call counts with CIs"],
    ["No-tool eval slice accuracy", "Loss-mask unit test on a known trajectory", "Relabeled-sample human audit (>=50 examples)"],
    ["Ablation flip-rate distribution across >=3 seeds", "Sample-representativeness check vs full eval", "Runtime cost of the diagnostic"],
    ["p95 tool-call count of correct trajectories per task class", "Truncation-rate metric under the budget", "Template hash pinning record"],
    ["Pair-construction audit confirming identical correct final answers", "General instruction-following eval before/after", "Call-count vs accuracy scatter"],
    ["Retry-vs-redundancy label audit", "Injected-tool-failure recovery_rate per arm", "Class definition change log"],
    ["Canonicalization unit tests on formatting variants", "Trace overhead measurement at p95", "Schema version pinning"],
    ["Instrumented token and round-trip counts replacing the estimates", "Decode-time vs tool-latency breakdown", "Cost delta at fixed load"],
    ["Arm C accuracy drop magnitude", "Flag audit proving arm C is eval-only", "Per-arm metric table"],
    ["Shadow-stage disagreement log", "Canary traffic-mix comparison", "Kill-switch activation drill timing"],
]

rows = []
for i, s in enumerate(src):
    m = {x["role"]: x["content"] for x in s["messages"]}
    head, body, dec, tc, ic, ops, conf = STANCES[i]
    ca = head + "\n\n" + body + "\n" + COMMON_TAIL
    rows.append({
        "source_id": s["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": dec,
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": ca,
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": ops,
        },
        "risks": RISKS[i],
        "evidence_required": EVID[i],
        "confidence": conf,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(rows))
