import json
CORPUS="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows=[json.loads(l) for l in open(CORPUS) if l.strip()]
for r in rows[1830:1840]:
    m={x["role"]:x["content"] for x in r["messages"]}
    print("="*70)
    print("ID", r["id"])
    print("USER:", m["user"][:700])
    print("-- ASSISTANT len", len(m["assistant"]), ":", m["assistant"][:500])
