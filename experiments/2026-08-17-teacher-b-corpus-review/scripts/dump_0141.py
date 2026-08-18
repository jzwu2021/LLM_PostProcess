import json
p='/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl'
rows=[json.loads(l) for l in open(p)]
for r in rows[1400:1410]:
    print(json.dumps(r,ensure_ascii=False)[:4000])
    print('='*80)
