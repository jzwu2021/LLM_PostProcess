#!/usr/bin/env python3
"""
Inter-teacher agreement analysis: teacher-A vs teacher-B.

Pure stdlib on purpose — no numpy/sklearn in this environment, and the metrics
below are exactly reproducible without them.

IMPORTANT INTERPRETATION LIMITS (read before citing any number):
  - Both lanes are PROVISIONAL model output, not expert gold. High agreement
    means two models converged, NOT that either is correct. Two LLMs sharing
    training-data priors can agree confidently and both be wrong.
  - This measures agreement on the REWRITTEN ANSWERS, not domain capability of
    any trained model. It says nothing about post-training success.
  - Lexical overlap (Jaccard / cosine) is a proxy for semantic agreement. Two
    technically equivalent answers using different vocabulary will score low.
    Treat low-overlap pairs as "needs human look", not "teachers disagree".
"""
import json
import glob
import math
import os
import re
import sys
from collections import Counter

ROOT = "/home/johnson/workspace/LLM_PostProcess"
A_DIR = f"{ROOT}/experiments/2026-08-14-teacher-a-corpus-calibration/results"
B_DIR = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review/results"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus"

STOP = set("""a an the and or but if then than that this these those of in on at to for
with by from as is are was were be been being it its into via such not no can may
must should will would when where which who whom what how each per over under
between across during about above below same other more most less least very
you your we our they their he she his her i me my one two also both any all some
do does did done have has had having only just even still yet more use used using
""".split())

WORD = re.compile(r"[a-z0-9_./+-]+")


def tokens(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


def load(d):
    out = {}
    for p in sorted(glob.glob(f"{d}/*.jsonl")):
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["source_id"]] = r
    return out


def jaccard(sa, sb):
    if not sa and not sb:
        return 1.0
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def cosine(ca, cb):
    if not ca or not cb:
        return 0.0
    common = set(ca) & set(cb)
    num = sum(ca[t] * cb[t] for t in common)
    da = math.sqrt(sum(v * v for v in ca.values()))
    db = math.sqrt(sum(v * v for v in cb.values()))
    return num / (da * db) if da and db else 0.0


def char_ngrams(text, n=5):
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return Counter(t[i:i + n] for i in range(max(0, len(t) - n + 1)))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] if lo == hi else s[lo] * (hi - k) + s[hi] * (k - lo)


def describe(name, vals):
    if not vals:
        return f"{name}: n/a"
    return (f"{name}: mean={sum(vals)/len(vals):.3f} p10={pct(vals,.10):.3f} "
            f"p50={pct(vals,.50):.3f} p90={pct(vals,.90):.3f} "
            f"min={min(vals):.3f} max={max(vals):.3f}")


def main():
    a, b = load(A_DIR), load(B_DIR)
    common = [k for k in b if k in a]
    # keep corpus order for reproducibility
    order = {}
    for split in ("train", "validation"):
        p = f"{CORPUS}/{split}.jsonl"
        if os.path.exists(p):
            for i, line in enumerate(open(p)):
                if line.strip():
                    order.setdefault(json.loads(line)["id"], (split, i))
    common.sort(key=lambda k: order.get(k, ("zz", 0)))

    print("=" * 72)
    print("INTER-TEACHER AGREEMENT: teacher-A vs teacher-B")
    print("=" * 72)
    print(f"teacher-A records : {len(a)}")
    print(f"teacher-B records : {len(b)}")
    print(f"overlap analysed  : {len(common)}")
    if not common:
        print("no overlap yet")
        return

    models = (sorted({a[k]['teacher_model'] for k in common}),
              sorted({b[k]['teacher_model'] for k in common}))
    print(f"models            : A={models[0]} B={models[1]}")
    splits = Counter(order.get(k, ('unknown', 0))[0] for k in common)
    print(f"splits            : {dict(splits)}")

    # ---- integrity: both lanes must describe the SAME source record ----
    mismatch = [k for k in common
                if a[k]["source_user"] != b[k]["source_user"]
                or a[k]["source_assistant"] != b[k]["source_assistant"]]
    print(f"source alignment  : {'PASS' if not mismatch else 'FAIL ' + str(mismatch[:5])}")
    if mismatch:
        sys.exit("aborting: lanes are not aligned to identical source records")

    # ---- 1. decision-label agreement ----
    da = Counter(a[k]["decision"] for k in common)
    db_ = Counter(b[k]["decision"] for k in common)
    same = sum(1 for k in common if a[k]["decision"] == b[k]["decision"])
    print("\n" + "-" * 72)
    print("1. DECISION LABEL")
    print("-" * 72)
    print(f"A distribution: {dict(da)}")
    print(f"B distribution: {dict(db_)}")
    print(f"raw agreement : {same}/{len(common)} = {same/len(common):.4f}")
    # Cohen's kappa
    labels = sorted(set(da) | set(db_))
    pe = sum((da[l] / len(common)) * (db_[l] / len(common)) for l in labels)
    po = same / len(common)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"expected agree: {pe:.4f}")
    print(f"cohen's kappa : {kappa if kappa == kappa else float('nan'):.4f}"
          if kappa == kappa else "cohen's kappa : undefined (single label -> no variance)")
    if len(labels) == 1 or pe >= 0.999:
        print("NOTE: both lanes emit a single label; kappa is undefined/degenerate.")
        print("      Decision label carries NO discriminative signal. Use content metrics.")

    # ---- 2. content agreement on corrected_answer ----
    print("\n" + "-" * 72)
    print("2. CORRECTED ANSWER CONTENT")
    print("-" * 72)
    jac, cos5, lenratio, jac_vs_src = [], [], [], []
    per = []
    for k in common:
        ta, tb = tokens(a[k]["corrected_answer"]), tokens(b[k]["corrected_answer"])
        j = jaccard(set(ta), set(tb))
        c = cosine(char_ngrams(a[k]["corrected_answer"]), char_ngrams(b[k]["corrected_answer"]))
        la, lb = len(a[k]["corrected_answer"]), len(b[k]["corrected_answer"])
        lr = min(la, lb) / max(la, lb) if max(la, lb) else 1.0
        # how much does each teacher depart from the ORIGINAL assistant answer?
        ts = set(tokens(a[k]["source_assistant"]))
        jac.append(j); cos5.append(c); lenratio.append(lr)
        jac_vs_src.append((jaccard(set(ta), ts), jaccard(set(tb), ts)))
        per.append((k, j, c, lr, la, lb))
    print(describe("token jaccard   ", jac))
    print(describe("char5 cosine    ", cos5))
    print(describe("length ratio    ", lenratio))
    avg_a = sum(x for x, _ in jac_vs_src) / len(jac_vs_src)
    avg_b = sum(y for _, y in jac_vs_src) / len(jac_vs_src)
    print(f"\ndeparture from ORIGINAL assistant answer (token jaccard vs source):")
    print(f"  A vs source: {avg_a:.3f}   B vs source: {avg_b:.3f}")
    print(f"  A vs B     : {sum(jac)/len(jac):.3f}")
    if sum(jac) / len(jac) > max(avg_a, avg_b):
        print("  -> the two teachers resemble EACH OTHER more than either resembles")
        print("     the original answer: convergent rewriting, shared priors likely.")
    else:
        print("  -> each teacher stays closer to the original than to the other:")
        print("     divergent rewriting, genuine independent signal.")

    # ---- 3. quality dimensions ----
    print("\n" + "-" * 72)
    print("3. QUALITY DIMENSIONS")
    print("-" * 72)
    # teacher-A did NOT use one stable rubric: most records carry a 6-dimension
    # schema (mechanism/boundary_conditions/... ) while a minority use the same
    # 3 dimensions as teacher-B. Comparing across different rubrics is invalid,
    # and .get(d, 0) would silently impute 0 for absent keys and fabricate a
    # bogus mean. So we report the schema split and compare ONLY on records
    # where both lanes genuinely share the dimension.
    dims = ["technical_correctness", "instruction_coverage", "operational_safety"]
    schema_a = Counter(tuple(sorted(a[k]["quality_dimensions"].keys()))
                       if isinstance(a[k]["quality_dimensions"], dict) else ("__list__",)
                       for k in common)
    print("teacher-A rubric schemas in overlap:")
    for ks, n in schema_a.most_common():
        print(f"  n={n:<5} {list(ks)}")
    print("teacher-B rubric schemas in overlap:")
    for ks, n in Counter(tuple(sorted(b[k]["quality_dimensions"].keys()))
                         if isinstance(b[k]["quality_dimensions"], dict) else ("__list__",)
                         for k in common).most_common():
        print(f"  n={n:<5} {list(ks)}")

    qd_common = [k for k in common
                 if isinstance(a[k]["quality_dimensions"], dict)
                 and isinstance(b[k]["quality_dimensions"], dict)
                 and all(d in a[k]["quality_dimensions"] and d in b[k]["quality_dimensions"]
                         for d in dims)]
    print(f"\ncomparable on shared 3-dim rubric: {len(qd_common)}/{len(common)}")
    if not qd_common:
        print("  -> NO comparable records; rubric agreement cannot be computed.")
    else:
        if len(qd_common) < 100:
            print(f"  WARNING: n={len(qd_common)} is small; treat these as indicative only.")
        for d in dims:
            xa = [a[k]["quality_dimensions"][d] for k in qd_common]
            xb = [b[k]["quality_dimensions"][d] for k in qd_common]
            exact = sum(1 for p, q in zip(xa, xb) if p == q) / len(xa)
            within1 = sum(1 for p, q in zip(xa, xb) if abs(p - q) <= 1) / len(xa)
            r = pearson(xa, xb)
            rtxt = f"{r:.3f}" if r == r else "undefined(no variance)"
            print(f"{d:24s} A_mean={sum(xa)/len(xa):.2f} B_mean={sum(xb)/len(xb):.2f} "
                  f"exact={exact:.3f} within1={within1:.3f} pearson={rtxt}")

    # ---- 4. confidence ----
    print("\n" + "-" * 72)
    print("4. CONFIDENCE")
    print("-" * 72)
    ca = [a[k]["confidence"] for k in common]
    cb = [b[k]["confidence"] for k in common]
    r = pearson(ca, cb)
    print(f"A mean={sum(ca)/len(ca):.3f}  B mean={sum(cb)/len(cb):.3f}  "
          f"mean|delta|={sum(abs(x-y) for x,y in zip(ca,cb))/len(ca):.3f}")
    print(f"pearson={r:.3f}" if r == r else "pearson=undefined (no variance)")

    # ---- 5. risks / evidence ----
    print("\n" + "-" * 72)
    print("5. RISKS / EVIDENCE_REQUIRED")
    print("-" * 72)
    for field in ("risks", "evidence_required"):
        ov = []
        na, nb = [], []
        for k in common:
            sa = set(tokens(" ".join(a[k].get(field) or [])))
            sb = set(tokens(" ".join(b[k].get(field) or [])))
            ov.append(jaccard(sa, sb))
            na.append(len(a[k].get(field) or []))
            nb.append(len(b[k].get(field) or []))
        print(f"{field:18s} A_items={sum(na)/len(na):.2f} B_items={sum(nb)/len(nb):.2f} "
              f"token_jaccard_mean={sum(ov)/len(ov):.3f}")

    # ---- 6. divergence hotspots ----
    print("\n" + "-" * 72)
    print("6. LOWEST-AGREEMENT PAIRS (highest human-review value)")
    print("-" * 72)
    per.sort(key=lambda t: t[1])
    print(f"{'source_id':<16}{'jaccard':>9}{'cos5':>8}{'lenR':>7}{'A_len':>8}{'B_len':>8}")
    for k, j, c, lr, la, lb in per[:15]:
        print(f"{k:<16}{j:>9.3f}{c:>8.3f}{lr:>7.3f}{la:>8}{lb:>8}")

    thr = pct(jac, 0.10)
    flagged = [t[0] for t in per if t[1] <= thr]
    print(f"\nflagged at p10 jaccard<={thr:.3f}: {len(flagged)} records")

    out = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review/agreement_report.json"
    json.dump({
        "overlap": len(common),
        "teacher_a_model": models[0],
        "teacher_b_model": models[1],
        "splits": dict(splits),
        "decision_raw_agreement": po,
        "decision_kappa": None if kappa != kappa else kappa,
        "decision_dist_a": dict(da),
        "decision_dist_b": dict(db_),
        "token_jaccard_mean": sum(jac) / len(jac),
        "char5_cosine_mean": sum(cos5) / len(cos5),
        "length_ratio_mean": sum(lenratio) / len(lenratio),
        "a_vs_source_jaccard": avg_a,
        "b_vs_source_jaccard": avg_b,
        "confidence_mean_a": sum(ca) / len(ca),
        "confidence_mean_b": sum(cb) / len(cb),
        "low_agreement_ids": flagged,
        "per_record": [
            {"source_id": k, "jaccard": j, "char5_cosine": c,
             "length_ratio": lr, "a_len": la, "b_len": lb}
            for k, j, c, lr, la, lb in per
        ],
    }, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
