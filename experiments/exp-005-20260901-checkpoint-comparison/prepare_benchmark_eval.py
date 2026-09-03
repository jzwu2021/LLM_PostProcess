"""Convert benchmark.jsonl into the messages format the loss evaluator expects.

Excludes aiinfra-0001, which exp-003 recorded as residually exposed: it was read
during the contamination audit to determine the benchmark's field schema.

The benchmark's reference answers are short (one or two sentences) while both
fine-tuned models were trained on long-form answers. Loss on this file therefore
measures agreement with a terse reference style and is reported as such.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "research/ai-infra-expert/benchmark.jsonl"
OUT = pathlib.Path(__file__).resolve().parent / "data/benchmark_as_messages.jsonl"
EXCLUDE = {"aiinfra-0001"}

SYSTEM = "You are an AI/LLM infrastructure engineer. Answer precisely and concisely."


def main():
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    kept = [r for r in rows if r["id"] not in EXCLUDE]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for r in kept:
            fh.write(json.dumps({
                "id": r["id"],
                "category": r["category"],
                "topic": r["topic"],
                "verifier": r["verifier"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["reference_answer"]},
                ],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} records (excluded {len(rows) - len(kept)}: {sorted(EXCLUDE)})")
    print(f"path {OUT}")


if __name__ == "__main__":
    main()
