import json, os, glob
ROOT = "/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
train = [json.loads(l) for l in open(os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl"))]
rec = json.loads(open(sorted(glob.glob(os.path.join(RES,"train-batch-*.jsonl")))[-1]).readline())
sid = rec["source_id"]
c = [d for d in train if d["id"] == sid][0]
print("MSG_ROLES", [m["role"] for m in c["messages"]])
for m in c["messages"]:
    print("--", m["role"], repr(m["content"][:80]))
print("MATCH_USER", rec["source_user"] == c["messages"][0]["content"], repr(rec["source_user"][:80]))
print("MATCH_ASSIST_last", rec["source_assistant"] == c["messages"][-1]["content"])
