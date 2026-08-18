import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2])
path = sys.argv[3]
recs = [json.loads(l) for l in open(path) if l.strip()][start:start+n]
for r in recs:
    print("=== KEYS", list(r.keys()))
    print("ID:", r.get("id"))
    print("USER:", r.get("user"))
    print("ASSISTANT:", (r.get("assistant") or "")[:2000])
    print()
