import json
rows=[json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')]
print(len(rows))
for r in rows[1740:1750]:
    print(json.dumps(r,ensure_ascii=False)[:4000])
    print('='*80)
