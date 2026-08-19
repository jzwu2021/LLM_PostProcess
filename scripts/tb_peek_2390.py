import json
ROOT = "/home/johnson/workspace/LLM_PostProcess"
rows = [json.loads(l) for l in open(ROOT + "/research/ai-infra-expert/corpus/train.jsonl")]
print("TOTAL", len(rows))
for d in rows[2390:2400]:
    u = [m for m in d["messages"] if m["role"] == "user"][0]["content"]
    a = [m for m in d["messages"] if m["role"] == "assistant"][0]["content"]
    print("=" * 70)
    print("ID:", d["id"])
    print("USER:", u[:1400])
    print("--- ASSISTANT:", a[:900])
