import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2])
path = sys.argv[3]
rows = [json.loads(l) for l in open(path)][start:start+n]
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
    print('=====')
print('KEYS', list(rows[0].keys()))
