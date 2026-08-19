import json
ROOT = "/home/johnson/workspace/LLM_PostProcess"
rows = [json.loads(l) for l in open(f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl") if l.strip()]
sel = rows[2330:2340]
for i, r in enumerate(sel):
    m = {x["role"]: x["content"] for x in r["messages"]}
    print("=" * 20, i, r["id"])
    print("[USER]", m["user"][:1400])
    print("[ASSISTANT len=%d]" % len(m["assistant"]), m["assistant"][:700])
