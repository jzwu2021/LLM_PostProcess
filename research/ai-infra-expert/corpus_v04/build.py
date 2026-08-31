"""Build the v0.4 corpus: mechanism x task x angle, no variant counters."""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from angles import ANGLES  # noqa: E402
from core import REGISTRY, SETTINGS  # noqa: E402

MECH_MODULES = sorted(p.stem for p in HERE.glob("mech_*.py"))

SYSTEM = ("You are an AI/LLM infrastructure engineer. Answer with the mechanism, the measurement "
          "that discriminates it, and the condition that would refute it. Label every number as "
          "ESTIMATE or MEASURED.")


def build():
    for name in MECH_MODULES:
        importlib.import_module(name)
    if not REGISTRY:
        raise SystemExit("no mechanisms registered")

    records = []
    for m in REGISTRY:
        for ai, angle in enumerate(ANGLES):
            setting = SETTINGS[ai % len(SETTINGS)]
            q, a, task_type, difficulty = angle(m, setting)
            rid = f"aie4-{len(records) + 1:05d}"
            records.append({
                "id": rid,
                "category": m.topic,
                "concepts": m.concept_list(),
                "task_type": task_type,
                "difficulty": difficulty,
                "mechanism": m.key,
                "setting": setting.key,
                "angle": angle.__name__,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "provenance": "authored_synthetic_v0.4_combinatorial",
                "review_status": "needs_domain_expert_review",
                "verifier": "rubric",
                "contamination_policy": "excluded_from_benchmark",
            })
    return records


def main():
    records = build()
    out = HERE / "train.jsonl"
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"WROTE {out.name} records={len(records)} "
          f"mechanisms={len(REGISTRY)} angles={len(ANGLES)} settings={len(SETTINGS)}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
