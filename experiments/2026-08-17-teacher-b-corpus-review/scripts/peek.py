import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2])
path = sys.argv[3]
lines = open(path).read().splitlines()
print("TOTAL_LINES", len(lines))
for i in range(start, min(start+n, len(lines))):
    d = json.loads(lines[i])
    print("===IDX", i, "KEYS", list(d.keys()))
    print(json.dumps(d, ensure_ascii=False)[:2000])
