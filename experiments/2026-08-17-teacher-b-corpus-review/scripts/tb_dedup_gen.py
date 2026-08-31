"""Generate teacher-B batches for the deduplicating stage.

Batch 0259 starts at corpus row 2580; every batch covers ten rows except the last.
Exemplar rows take an authored stance from tb_dedup_stances; all other rows are
rejected as duplicates of their family.
"""
import argparse
import json
import os

import tb_dedup_common as C

FIRST_BATCH = 259
FIRST_ROW = C.REVIEWED_PREFIX
TOTAL_ROWS = 5399


def window(batch_no):
    start = FIRST_ROW + (batch_no - FIRST_BATCH) * 10
    return start, min(start + 10, TOTAL_ROWS)


def last_batch():
    n = FIRST_BATCH
    while window(n)[1] < TOTAL_ROWS:
        n += 1
    return n


def build(batch_no, corpus, fam_of, exemplars, plan, stances):
    start, end = window(batch_no)
    rows = []
    for i in range(start, end):
        fam = fam_of[i]
        info = exemplars[fam]
        if plan[i] == "rewrite":
            slot = info["exemplar_indices"].index(i)
            entry = stances[fam][slot]
            rows.append(C.rewrite_record(corpus, i, *entry))
        else:
            rows.append(C.reject_record(corpus, i, fam, info))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-batch", type=int, default=FIRST_BATCH)
    ap.add_argument("--to-batch", type=int, default=None)
    args = ap.parse_args()

    corpus, fam_of, exemplars, plan = C.build_plan()
    try:
        from tb_dedup_stances import STANCES
    except ImportError:
        STANCES = {}

    end_batch = args.to_batch or last_batch()
    written = 0
    for n in range(args.from_batch, end_batch + 1):
        start, stop = window(n)
        if start >= TOTAL_ROWS:
            break
        missing = [fam_of[i] for i in range(start, stop) if plan[i] == "rewrite" and fam_of[i] not in STANCES]
        if missing:
            raise SystemExit(f"batch {n:04d} needs stances for families {sorted(set(missing))}")
        rows = build(n, corpus, fam_of, exemplars, plan, STANCES)
        path = os.path.join(C.RESULTS, f"train-batch-{n:04d}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written += len(rows)
    print(f"WROTE batches {args.from_batch:04d}..{end_batch:04d} rows={written}")


if __name__ == "__main__":
    main()
