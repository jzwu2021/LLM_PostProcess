"""Prepare exp-004 training and held-out splits from corpus_v04.

Two decisions here are taken from mechanisms the corpus itself teaches:

  * The held-out split removes whole mechanisms, not random rows. A random split
    would put the same mechanism in both halves, and held-out loss would then
    fall because the model memorised content present on both sides
    (heldout_shares_repetition).
  * The training file is shuffled, because the trainer consumes records in index
    order and a run shorter than one epoch would otherwise never reach the tail
    (data_ordering_reachability).
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "research/ai-infra-expert/corpus_v04/train.jsonl"
OUT = pathlib.Path(__file__).resolve().parent / "data"
SEED = 20260901
WORLD, GRAD_ACCUM = 8, 4


def main():
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    by_topic = collections.defaultdict(list)
    for r in rows:
        by_topic[r["category"]].append(r["mechanism"])

    rng = random.Random(SEED)
    heldout_mechs = set()
    for topic in sorted(by_topic):
        mechs = sorted(set(by_topic[topic]))
        heldout_mechs.add(rng.choice(mechs))

    train = [r for r in rows if r["mechanism"] not in heldout_mechs]
    held = [r for r in rows if r["mechanism"] in heldout_mechs]

    rng.shuffle(train)
    rng.shuffle(held)

    OUT.mkdir(parents=True, exist_ok=True)
    tp, hp = OUT / "aie_v04_train.jsonl", OUT / "aie_v04_heldout.jsonl"
    for path, data in ((tp, train), (hp, held)):
        with path.open("w") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_m = {r["mechanism"] for r in train}
    assert not (train_m & heldout_mechs), "mechanism leaked across the split"
    tq = {next(m["content"] for m in r["messages"] if m["role"] == "user") for r in train}
    hq = {next(m["content"] for m in r["messages"] if m["role"] == "user") for r in held}
    assert not (tq & hq), "question leaked across the split"

    per_step = WORLD * GRAD_ACCUM
    steps_epoch = -(-len(train) // per_step)

    print(f"train records      : {len(train)}  ({len(train_m)} mechanisms)")
    print(f"heldout records    : {len(held)}  ({len(heldout_mechs)} mechanisms)")
    print(f"shared mechanisms  : 0")
    print(f"shared questions   : 0")
    print(f"records per step   : {per_step} (world {WORLD} x grad_accum {GRAD_ACCUM})")
    print(f"steps for 1 epoch  : {steps_epoch}")
    print("held-out mechanisms:")
    for m in sorted(heldout_mechs):
        print(f"  - {m}")
    for p in (tp, hp):
        print(f"sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
