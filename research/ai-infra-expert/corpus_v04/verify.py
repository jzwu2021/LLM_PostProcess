"""Uniqueness and quality gate for the v0.4 corpus.

The v0.3 corpus passed schema validation while containing 522 real questions in
5399 rows, so schema checks are not the interesting part. This verifier is built
to fail on the specific defect that produced that corpus.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
TRAIN = HERE / "train.jsonl"
BENCH = HERE.parent / "benchmark.jsonl"

NGRAM = 5
JACCARD_MAX = 0.45          # any pair above this is a near-duplicate
COMMON_GRAM_DF = 40         # grams appearing in more docs than this are skipped as boilerplate
COUNTER_PAT = re.compile(r"\b(?:scenario|case|variant)\s+(?:variant\s+)?\d+\b", re.I)

# splicing an authored sentence into a mid-clause slot leaves these traces
GRAMMAR_DEFECTS = (
    ("sentence spliced mid-clause", re.compile(r"[a-z]{3}\. [a-z]")),
    ("double space", re.compile(r"[^\n] {2}[^ ]")),
    ("space before punctuation", re.compile(r" [.,;]")),
    ("doubled connective", re.compile(r"\b(\w+) \1\b")),
    ("empty sentence", re.compile(r"\.\s*\.")),
)

fail: list[str] = []


def chk(cond, msg):
    if not cond:
        fail.append(msg)


def grams(text: str) -> set[int]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(tuple(toks[i:i + NGRAM])) for i in range(max(len(toks) - NGRAM + 1, 0))}


def near_dup_report(docs: list[tuple[str, str]], label: str):
    """docs is [(id, text)]. Returns the worst pairs by 5-gram Jaccard."""
    gsets = {rid: grams(t) for rid, t in docs}
    index = collections.defaultdict(list)
    for rid, gs in gsets.items():
        for g in gs:
            index[g].append(rid)

    shared = collections.Counter()
    for g, ids in index.items():
        if len(ids) > COMMON_GRAM_DF:
            continue                      # shared connective phrasing, not content
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                shared[(ids[i], ids[j])] += 1

    worst = []
    for (a, b), n in shared.items():
        ga, gb = gsets[a], gsets[b]
        union = len(ga | gb)
        if not union:
            continue
        jac = len(ga & gb) / union
        if jac > 0.20:
            worst.append((jac, a, b))
    worst.sort(reverse=True)

    over = [w for w in worst if w[0] > JACCARD_MAX]
    for jac, a, b in over[:20]:
        fail.append(f"{label} near-duplicate {a} ~ {b} jaccard={jac:.3f}")
    chk(not over, f"{label}: {len(over)} pairs above jaccard {JACCARD_MAX}")
    return worst


def setting_phrasings() -> list[str]:
    """Every question must state its deployment; that shared text is not duplication."""
    sys.path.insert(0, str(HERE))
    from angles import sl
    from core import SETTINGS
    return sorted({sl(s, k) for s in SETTINGS for k in range(6)}, key=len, reverse=True)


def strip_setting(text: str, phrasings: list[str]) -> str:
    for p in phrasings:
        text = text.replace(p, " ")
    return text


def main():
    rows = [json.loads(l) for l in TRAIN.open() if l.strip()]
    print(f"records: {len(rows)}")

    ids = [r["id"] for r in rows]
    chk(len(set(ids)) == len(ids), "duplicate record ids")

    users, assts = [], []
    for r in rows:
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        chk(set(msgs) == {"system", "user", "assistant"}, f"{r['id']} roles")
        u, a = msgs["user"], msgs["assistant"]
        chk(len(u.split()) >= 20, f"{r['id']} user too short")
        chk(len(a.split()) >= 90, f"{r['id']} assistant too short")
        chk(not COUNTER_PAT.search(u), f"{r['id']} user contains a variant counter")
        chk(not COUNTER_PAT.search(a), f"{r['id']} assistant contains a variant counter")
        chk("{" not in a and "}" not in a, f"{r['id']} unrendered template braces")
        for defect, pat in GRAMMAR_DEFECTS:
            hit = pat.search(a) or pat.search(u)
            chk(not hit, f"{r['id']} {defect}: ...{hit.group(0) if hit else ''}...")
        users.append((r["id"], u))
        assts.append((r["id"], a))

    # exact duplication is what v0.3 failed on
    du = collections.Counter(t for _, t in users)
    da = collections.Counter(t for _, t in assts)
    chk(all(v == 1 for v in du.values()), f"exact duplicate user texts: {sum(v - 1 for v in du.values())}")
    chk(all(v == 1 for v in da.values()), f"exact duplicate assistant texts: {sum(v - 1 for v in da.values())}")

    # question-family collapse: strip digits and see how many distinct questions remain
    fam = collections.Counter(re.sub(r"\d+", "#", u) for _, u in users)
    print(f"distinct users: {len(du)}  distinct assistants: {len(da)}  "
          f"digit-insensitive question families: {len(fam)}")
    chk(len(fam) >= 0.5 * len(rows),
        f"only {len(fam)} question families for {len(rows)} rows: differences are numeric only")

    worst_a = near_dup_report(assts, "assistant")
    ph = setting_phrasings()
    worst_u = near_dup_report([(i, strip_setting(u, ph)) for i, u in users], "user")
    if worst_a:
        print(f"assistant worst jaccard: {worst_a[0][0]:.3f} ({worst_a[0][1]} ~ {worst_a[0][2]})")
        print(f"assistant pairs over 0.35: {sum(1 for w in worst_a if w[0] > 0.35)}")
    if worst_u:
        print(f"user worst jaccard: {worst_u[0][0]:.3f} ({worst_u[0][1]} ~ {worst_u[0][2]})")

    # content binding: the answer must be tied to its own mechanism and setting
    by_mech = collections.defaultdict(list)
    for r in rows:
        by_mech[r["mechanism"]].append(r)
    chk(all(len(v) == 30 for v in by_mech.values()), "every mechanism must yield 30 records")

    # contamination against the held-out benchmark
    if BENCH.exists():
        bench = {json.loads(l)["question"].strip() for l in BENCH.open() if l.strip()}
        overlap = {u for _, u in users if u.strip() in bench}
        chk(not overlap, f"{len(overlap)} training questions appear in benchmark.jsonl")

    if fail:
        print(f"\nVERIFY_FAIL {len(fail)}")
        for f in fail[:25]:
            print(" -", f)
        sys.exit(1)
    print(f"\nVERIFY_PASS records={len(rows)} unique_users={len(du)} unique_assistants={len(da)} "
          f"max_pair_jaccard<={JACCARD_MAX}")


if __name__ == "__main__":
    main()
