import json
p='/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl'
rows=[json.loads(l) for l in open(p,encoding='utf-8')]
r=rows[1619]
print(json.dumps(r,ensure_ascii=False)[:1200])
