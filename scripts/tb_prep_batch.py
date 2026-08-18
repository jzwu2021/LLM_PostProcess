import json, os, glob
ROOT = "/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
train = [json.loads(l) for l in open(os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl"))]
tb = sorted(glob.glob(os.path.join(RES, "train-batch-*.jsonl")))
tn = sum(1 for f in tb for l in open(f) if l.strip())
nxt = train[tn:tn+10]
out = []
for i, d in enumerate(nxt):
    u = [m for m in d["messages"] if m["role"] == "user"][0]["content"]
    a = [m for m in d["messages"] if m["role"] == "assistant"][0]["content"]
    out.append({"idx": tn+i, "id": d["id"], "category": d.get("category"), "task_type": d.get("task_type"),
                "concepts": d.get("concepts"), "difficulty": d.get("difficulty"), "user": u, "assistant": a})
json.dump(out, open("/tmp/tb_next.json", "w"), ensure_ascii=False)
for o in out:
    print("=====", o["idx"], o["id"], o["category"], o["task_type"], o["difficulty"], o["concepts"])
    print("USER:", o["user"])
    print("ASSISTANT:", o["assistant"])
