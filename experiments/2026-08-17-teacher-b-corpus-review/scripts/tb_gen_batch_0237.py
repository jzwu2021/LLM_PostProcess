import json, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT = f"{EXP}/results/train-batch-0237.jsonl"
START, END = 2360, 2370

sys.path.insert(0, f"{EXP}/scripts")
from tb_stances_0237 import COMMON_HEAD, COMMON_TAIL, BASE_RISKS, BASE_EV, S

rows = [json.loads(l) for l in open(CORPUS) if l.strip()]
sel = rows[START:END]
assert len(sel) == 10
assert len(S) == 10

out = []
for row, s in zip(sel, S):
    m = {x["role"]: x["content"] for x in row["messages"]}
    ans = (
        f"Scenario variant {s['v']} — redundant calculator invocations by an agent. Focus: {s['f']}\n\n"
        f"{COMMON_HEAD}\n\n"
        f"{s['body'].strip()}\n\n"
        f"Falsifiable hypothesis {s['h']}\n\n"
        f"Controlled experiment: {s['exp']}\n\n"
        f"{s['gate']}\n\n"
        f"{COMMON_TAIL}"
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
        "evidence_required": s["ev"] + BASE_EV,
        "confidence": s["conf"],
    })

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(out))
print("IDS", [r["source_id"] for r in out])
