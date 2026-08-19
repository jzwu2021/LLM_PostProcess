import json, os, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
sys.path.insert(0, f"{EXP}/scripts")

CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0232.jsonl"
START, END = 2310, 2320

from stances_0232 import STANCES, EXTRA_RISKS, RISKS_COMMON, EXTRA_EVIDENCE, QD, CONF

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
