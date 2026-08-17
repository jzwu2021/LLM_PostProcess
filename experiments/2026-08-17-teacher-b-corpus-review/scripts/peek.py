import json, sys
p = sys.argv[1]; s = int(sys.argv[2]); e = int(sys.argv[3])
rows = [json.loads(l) for l in open(p)][s:e]
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
    print('=====')
