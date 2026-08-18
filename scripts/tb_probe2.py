import json, os, glob
ROOT = "/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
train = [json.loads(l) for l in open(os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl"))]
rec = json.loads(open(sorted(glob.glob(os.path.join(RES,"train-batch-*.jsonl")))[-1]).readline())
c = [d for d in train if d["id"] == rec["source_id"]][0]
u = [m for m in c["messages"] if m["role"]=="user"][0]["content"]
print("LEN_rec", len(rec["source_user"]), "LEN_corp", len(u))
print("REC:", repr(rec["source_user"]))
print("CORP:", repr(u))
