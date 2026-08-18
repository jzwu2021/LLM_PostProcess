import json
CORPUS = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
with open(CORPUS) as f:
    lines = f.readlines()[2050:2060]
for i, l in enumerate(lines):
    d = json.loads(l)
    m = {x["role"]: x["content"] for x in d["messages"]}
    print("=== idx", 2050+i, "id", d["id"])
    print("USER:", m["user"][:700].replace("\n", " | "))
    print("ASST:", m["assistant"][:400].replace("\n", " | "))
