#!/usr/bin/env python3
import json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).parent
required={'id','category','task_type','difficulty','messages','concepts','verifier','provenance','review_status'}
ids=set(); questions=set(); counts=Counter()
for split in ['train','validation']:
    path=ROOT/'corpus'/f'{split}.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert rows, f'empty {path}'
    for r in rows:
        assert required <= r.keys(), (split,r.get('id'),required-r.keys())
        assert r['id'] not in ids, r['id']
        ids.add(r['id']); counts[r['category']]+=1
        assert len(r['messages'])==3
        assert [m['role'] for m in r['messages']]==['system','user','assistant']
        assert r['messages'][1]['content'] not in questions
        questions.add(r['messages'][1]['content'])
        assert r['review_status']=='needs_domain_expert_review'
for p in ['eval_model_domain.json','eval_runtime_system.json']:
    d=json.loads((ROOT/p).read_text())
    assert d['name'] and d['version']
assert len(ids)==1700, len(ids)
assert sum(counts.values())==1700
assert (ROOT/'benchmark.jsonl').exists()
print(json.dumps({'corpus_records':len(ids),'unique_questions':len(questions),'category_counts':dict(sorted(counts.items())),'eval_specs':['eval_model_domain.json','eval_runtime_system.json'],'status':'PASS'},ensure_ascii=False,indent=2))
