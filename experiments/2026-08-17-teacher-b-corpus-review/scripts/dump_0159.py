import json
p = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows = [json.loads(l) for l in open(p, encoding="utf-8")]
print("TOTAL", len(rows))
for r in rows[1580:1590]:
    print("=====", r.get("id"))
    print(json.dumps(r, ensure_ascii=False)[:2000])
