import json, sys, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT = f"{EXP}/results/train-batch-0233.jsonl"
START, END = 2320, 2330

sys.path.insert(0, f"{EXP}/scripts")
from tb_stances_0233 import COMMON, CRITIQUE, BASE_RISKS, S

rows = [json.loads(l) for l in open(CORPUS) if l.strip()]
sel = rows[START:END]
assert len(sel) == 10
assert len(S) == 10

out = []
for row, s in zip(sel, S):
    m = {x["role"]: x["content"] for x in row["messages"]}
    ans = (
        f"Analytical stance under test: Stance {s['n']} - {s['t']}\n\n"
        f"{COMMON}\n\n"
        f"{s['body'].strip()}\n"
        f"Falsifiable hypothesis H{s['n']}: {s['h']}\n"
        f"Controlled experiment: {s['exp']}\n"
        f"{s['gate']}\n\n"
        f"{CRITIQUE}"
    )
    qd = s["qd"]
    out.append({
        "source_id": row["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": ans,
        "quality_dimensions": {
            "technical_correctness": qd[0],
            "instruction_coverage": qd[1],
            "operational_safety": qd[2],
        },
        "risks": s["risks"] + BASE_RISKS,
        "evidence_required": s["ev"],
        "confidence": s["conf"],
    })

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(out))
print("IDS", [r["source_id"] for r in out])
