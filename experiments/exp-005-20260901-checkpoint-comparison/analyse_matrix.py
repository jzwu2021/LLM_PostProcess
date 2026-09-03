"""Analyse the cross-evaluation matrix, per benchmark category, with a paired test.

Two things this script refuses to do:

  * report a difference without the paired uncertainty around it;
  * present the benchmark column as neutral, since it is contaminated for
    exp-002 by authoring and topic-covered for exp-004.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results_full"
BENCH = HERE / "data/benchmark_as_messages.jsonl"

ARMS = ("base", "exp002_step75", "exp004_step170")
SETS = ("exp002_heldout", "exp004_heldout", "benchmark")

BIAS = {
    ("exp002_step75", "exp002_heldout"): "HOME - style-native to this arm",
    ("exp004_step170", "exp004_heldout"): "HOME - style-native to this arm",
    ("exp002_step75", "benchmark"): "CONTAMINATED - repair data authored from benchmark inspection",
    ("exp004_step170", "benchmark"): "topic-covered 38/41, item-clean per exp-003",
}


def load():
    return {a: json.loads((RESULTS / f"{a}.json").read_text()) for a in ARMS}


def paired(a_items, b_items):
    """Paired differences b - a, keyed by item id."""
    da = {i: v for i, v in a_items if i is not None}
    db = {i: v for i, v in b_items if i is not None}
    keys = sorted(set(da) & set(db))
    diffs = [db[k] - da[k] for k in keys]
    if len(diffs) < 2:
        return None
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(len(diffs))
    wins_b = sum(1 for d in diffs if d < 0)
    return {"n": len(diffs), "mean": mean, "sd": sd, "se": se,
            "ci95": (mean - 1.96 * se, mean + 1.96 * se),
            "b_lower_on": wins_b, "a_lower_on": len(diffs) - wins_b}


def main():
    data = load()
    bench_meta = {json.loads(l)["id"]: json.loads(l) for l in BENCH.open() if l.strip()}

    print("=" * 78)
    print("1. FULL MATRIX  (masked loss, lower is a closer fit to that set's target text)")
    print("=" * 78)
    hdr = f"{'model':<16}" + "".join(f"{s:>22}" for s in SETS)
    print(hdr)
    for a in ARMS:
        row = f"{a:<16}"
        for s in SETS:
            r = data[a]["results"][s]
            row += f"{r['loss']:>16.6f} (n={r['examples']})".rjust(22)
        print(row)
    print()
    print("Set sizes: exp002_heldout=100, exp004_heldout=300, benchmark=499 (all 10 categories).")

    print()
    print("=" * 78)
    print("2. CHANGE FROM BASE  (negative = fits that set better than the base model)")
    print("=" * 78)
    for s in SETS:
        print(f"\n  {s}")
        base = data["base"]["results"][s]["loss"]
        for a in ("exp002_step75", "exp004_step170"):
            v = data[a]["results"][s]["loss"]
            note = BIAS.get((a, s), "off-diagonal, no home advantage")
            print(f"    {a:<16} {v:.6f}  delta {v - base:+.6f}  [{note}]")

    print()
    print("=" * 78)
    print("3. BENCHMARK BY CATEGORY  (the seven categories the first run never reached)")
    print("=" * 78)
    per_cat = {a: collections.defaultdict(list) for a in ARMS}
    for a in ARMS:
        for item_id, loss in data[a]["results"]["benchmark"]["per_item"]:
            if item_id in bench_meta:
                per_cat[a][bench_meta[item_id]["category"]].append(loss)
    cats = sorted(per_cat["base"])
    e2_items = {i: v for i, v in data["exp002_step75"]["results"]["benchmark"]["per_item"]}
    e4_items = {i: v for i, v in data["exp004_step170"]["results"]["benchmark"]["per_item"]}
    by_cat_ids = collections.defaultdict(list)
    for item_id, meta in bench_meta.items():
        by_cat_ids[meta["category"]].append(item_id)

    print(f"{'category':<30}{'n':>4}{'base':>9}{'exp002':>9}{'exp004':>9}"
          f"{'diff(e4-e2)':>13}{'95% interval':>24}{'call':>14}")
    for c in cats:
        ids = [i for i in by_cat_ids[c] if i in e2_items and i in e4_items]
        b = statistics.mean(per_cat["base"][c])
        e2 = statistics.mean(per_cat["exp002_step75"][c])
        e4 = statistics.mean(per_cat["exp004_step170"][c])
        diffs = [e4_items[i] - e2_items[i] for i in ids]
        m = statistics.mean(diffs)
        se = statistics.stdev(diffs) / math.sqrt(len(diffs))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        call = "tie" if lo <= 0 <= hi else ("exp004" if hi < 0 else "exp002")
        print(f"{c:<30}{len(ids):>4}{b:>9.4f}{e2:>9.4f}{e4:>9.4f}{m:>+13.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>24}{call:>14}")
    print("\n'tie' means the 95% interval spans zero: the two are not distinguishable")
    print("on that category at this sample size.")

    print()
    print("=" * 78)
    print("4. PAIRED COMPARISON exp002_step75 vs exp004_step170")
    print("=" * 78)
    print("Per-item differences on identical inputs. A difference whose 95% interval")
    print("spans zero is not distinguishable from no difference.")
    for s in SETS:
        st = paired(data["exp002_step75"]["results"][s]["per_item"],
                    data["exp004_step170"]["results"][s]["per_item"])
        if not st:
            print(f"\n  {s}: no item ids, paired test not possible")
            continue
        lo, hi = st["ci95"]
        verdict = "NOT DISTINGUISHABLE" if lo <= 0 <= hi else (
            "exp004 lower" if hi < 0 else "exp002 lower")
        print(f"\n  {s}  (n={st['n']})")
        print(f"    mean difference (exp004 - exp002) : {st['mean']:+.6f}")
        print(f"    95% interval                      : [{lo:+.6f}, {hi:+.6f}]")
        print(f"    items where exp004 is lower       : {st['b_lower_on']} of {st['n']}")
        print(f"    verdict                           : {verdict}")

    print()
    print("=" * 78)
    print("5. WHAT THIS MATRIX CANNOT SETTLE")
    print("=" * 78)
    print("""
Masked loss measures agreement with a specific target text, so it partly measures
writing style rather than knowledge. Each fine-tune was trained toward a different
answer style, so the home-set advantages are expected and carry no information
about capability.

The benchmark column is not a neutral referee either: exp-002's repair records
were authored after inspecting benchmark regression topics, and exp-003 found 38
of 41 benchmark topics correspond to exp-004 training mechanisms. Both arms have
an advantage there, of different kinds.

Deciding which model is better at AI-infrastructure work requires generation
against style-independent verifiers on a set neither arm's authoring process saw.
No such set exists in this repository yet.""")


if __name__ == "__main__":
    main()
