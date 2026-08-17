import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2])
path = sys.argv[3]
lines = open(path).read().splitlines()
print("TOTAL", len(lines))
out = []
for l in lines[start:start+n]:
    d = json.loads(l)
    out.append(d)
print(json.dumps(out, ensure_ascii=False, indent=1)[:20000])
