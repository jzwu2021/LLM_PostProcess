"""Compare the three arms on the private holdout, with paired tests on binary outcomes.

Accuracy differences on 63 items are small-sample. Every comparison here is
paired on identical items and reported with an interval, because an unpaired
accuracy gap of a few points on this many items is routinely noise.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
ITEMS = HERE / "data/private_holdout_v1.jsonl"
ARMS = ("base", "exp002_step75", "exp004_step170")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def mcnemar(a_correct, b_correct):
    """Paired binary comparison: counts where the two arms disagree."""
    b_only = sum(1 for k in a_correct if b_correct[k] and not a_correct[k])
    a_only = sum(1 for k in a_correct if a_correct[k] and not b_correct[k])
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, None
    # exact two-sided binomial p under H0: disagreements split evenly
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return a_only, b_only, min(p, 1.0)


def main():
    data = {a: json.loads((RESULTS / f"{a}.json").read_text()) for a in ARMS}
    items = {json.loads(l)["id"]: json.loads(l) for l in ITEMS.open() if l.strip()}
    correct = {a: {r["id"]: r["correct"] for r in data[a]["records"]} for a in ARMS}

    print("=" * 76)
    print("1. OVERALL  (63 items, greedy decoding, identical settings)")
    print("=" * 76)
    print(f"{'arm':<18}{'correct':>9}{'accuracy':>10}{'95% Wilson interval':>26}{'trunc':>7}")
    for a in ARMS:
        s = data[a]["summary"]
        lo, hi = wilson(s["correct"], s["items"])
        print(f"{a:<18}{s['correct']:>9}{s['accuracy']:>10.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>26}{s['truncated']:>7}")

    print()
    print("=" * 76)
    print("2. BY ITEM KIND AND BY v0.4 SCOPE")
    print("=" * 76)
    print(f"{'arm':<18}{'numeric':>12}{'code':>10}{'in-scope':>12}{'OUT-of-scope':>14}")
    for a in ARMS:
        s = data[a]["summary"]
        num = s["by_kind"].get("numeric", {})
        cod = s["by_kind"].get("code", {})
        ins = s["by_scope"].get("in_v04_scope", {})
        out = s["by_scope"].get("out_of_v04_scope", {})
        print(f"{a:<18}"
              f"{num.get("accuracy", 0):>12.3f}"
              f"{f'{cod.get(chr(97)+chr(99)+chr(99)+chr(117)+chr(114)+chr(97)+chr(99)+chr(121), 0):.3f}':>10}"
              f"{f'{ins.get(chr(97)+chr(99)+chr(99)+chr(117)+chr(114)+chr(97)+chr(99)+chr(121), 0):.3f}':>12}"
              f"{f'{out.get(chr(97)+chr(99)+chr(99)+chr(117)+chr(114)+chr(97)+chr(99)+chr(121), 0):.3f}':>14}")
    print("\nin-scope n=49, out-of-scope n=14. The out-of-scope items cover quantities")
    print("no v0.4 mechanism teaches, so they are the closest thing here to a clean test")
    print("for exp-004.")

    print()
    print("=" * 76)
    print("3. PAIRED COMPARISONS  (McNemar on identical items)")
    print("=" * 76)
    pairs = [("base", "exp002_step75"), ("base", "exp004_step170"),
             ("exp002_step75", "exp004_step170")]
    for a, b in pairs:
        a_only, b_only, p = mcnemar(correct[a], correct[b])
        n_dis = a_only + b_only
        print(f"\n  {a}  vs  {b}")
        print(f"    items only {a:<16} correct : {a_only}")
        print(f"    items only {b:<16} correct : {b_only}")
        if p is None:
            print("    identical on every item; no evidence of difference")
        else:
            call = "not distinguishable" if p > 0.05 else (
                f"{b} better" if b_only > a_only else f"{a} better")
            print(f"    disagreements                    : {n_dis}")
            print(f"    exact two-sided p                : {p:.4f}")
            print(f"    verdict                          : {call}")

    print()
    print("=" * 76)
    print("4. PER-QUANTITY ACCURACY")
    print("=" * 76)
    by_q = collections.defaultdict(dict)
    for a in ARMS:
        agg = collections.defaultdict(list)
        for r in data[a]["records"]:
            it = items[r["id"]]
            key = it.get("quantity") or f"code:{it.get('function')}"
            agg[key].append(r["correct"])
        for k, v in agg.items():
            by_q[k][a] = sum(v) / len(v)
    scope = {}
    for it in items.values():
        key = it.get("quantity") or f"code:{it.get('function')}"
        scope[key] = it["in_v04_scope"]
    print(f"{'quantity':<30}{'scope':>8}{'base':>8}{'exp002':>9}{'exp004':>9}")
    for k in sorted(by_q, key=lambda x: (scope[x], x)):
        s = "in" if scope[k] else "OUT"
        print(f"{k:<30}{s:>8}{by_q[k]['base']:>8.2f}"
              f"{by_q[k]['exp002_step75']:>9.2f}{by_q[k]['exp004_step170']:>9.2f}")


if __name__ == "__main__":
    main()
