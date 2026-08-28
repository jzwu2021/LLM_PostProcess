#!/usr/bin/env python3
"""Build the exp-002 Teacher-B plus repair post-training corpus."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "exp-001-20260827" / "data"
DATA = Path(__file__).parent / "data"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records), encoding="utf-8")


def main():
    base = load_jsonl(PREVIOUS / "teacher_b_native_train.jsonl")
    repair = load_jsonl(PREVIOUS / "regression_repair_v0.1.jsonl")
    if len(base) != 2400 or len(repair) != 80:
        raise RuntimeError(f"unexpected source counts: base={len(base)} repair={len(repair)}")
    if len({record["id"] for record in base + repair}) != len(base) + len(repair):
        raise RuntimeError("duplicate IDs between base and repair records")
    if any(record.get("provenance") != "authored_regression_repair_v0.1" for record in repair):
        raise RuntimeError("repair provenance label mismatch")
    if any(record.get("review_status") != "needs_domain_expert_review" for record in repair):
        raise RuntimeError("repair review-status label mismatch")

    mixed = []
    for index, record in enumerate(repair):
        start = index * 30
        mixed.extend(base[start:start + 30])
        mixed.append(record)
    mixed.extend(base[len(repair) * 30:])
    consumed = mixed[:75 * 8 * 4]
    consumed_repair = sum(record["id"].startswith("repair-v0.1-") for record in consumed)
    if len(mixed) != 2480 or consumed_repair != 77:
        raise RuntimeError(f"invalid mixed schedule: records={len(mixed)} consumed_repair={consumed_repair}")

    output = DATA / "teacher_b_plus_repair_v0.1_train.jsonl"
    write_jsonl(output, mixed)
    manifest = {
        "base_records": len(base),
        "repair_records": len(repair),
        "mixed_records": len(mixed),
        "mixing_policy": "insert one repair record after each consecutive block of 30 base records",
        "repair_records_consumed_at_75_steps": consumed_repair,
        "review_status": "needs_domain_expert_review",
        "source_repair_sha256": hashlib.sha256((PREVIOUS / "regression_repair_v0.1.jsonl").read_bytes()).hexdigest(),
        "mixed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (DATA / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()