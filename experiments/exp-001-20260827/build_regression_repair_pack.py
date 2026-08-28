#!/usr/bin/env python3
"""Build a small, auditable repair corpus for observed benchmark regressions."""
import hashlib
import json
from pathlib import Path


SYSTEM = (
    "You are an AI/LLM Infrastructure engineer. State assumptions, use units, "
    "distinguish measured facts from estimates, and do not invent platform-specific facts."
)

TOPICS = [
    (
        "agent_inference",
        "agent inference service",
        "Design or analyze a production agent inference service. Cover tool-call batching, "
        "per-tool timeout budgets, a cache validity contract, speculative-execution limits, "
        "and loop/stopping safeguards.",
        "State the admission contract before optimizing the model: batch only compatible "
        "tool calls, cap each call with a deadline and cancellation path, and carry a request "
        "budget across retries. Cache only tools declared pure; include tenant scope, normalized "
        "arguments, schema version, freshness window, and invalidation in the key. Bound speculative "
        "execution by a cost and side-effect budget, then cancel losing branches. Detect loops with "
        "a canonical call signature and a per-trajectory call limit; return a structured stop reason "
        "rather than silently suppressing work. Measure TTFT, TPOT, tool latency, calls per task, "
        "cache hit correctness, timeout rate, false suppression, and task success. Roll back when "
        "necessary-tool recall or success drops against a simultaneous control."
    ),
    (
        "benchmark_harness",
        "benchmark harness",
        "Design or analyze a reproducible LLM infrastructure benchmark harness. Cover fixed prompts, "
        "deterministic seeds, raw-output retention, verifier isolation, telemetry, and immutable manifests.",
        "Freeze a versioned case set, system prompt, chat template, decoding configuration, model artifact, "
        "and seed before execution. Persist raw requests, raw responses, finish reasons, token usage, latency, "
        "memory telemetry, environment fingerprint, and verifier version as append-only artifacts. Run verifiers "
        "in an isolated sandbox with no network access to model-serving credentials and record their exit status. "
        "Write a content-addressed manifest after generation; any changed input or output creates a new run rather "
        "than overwriting evidence. Validate reproducibility by replaying a pinned subset and comparing deterministic "
        "outputs. Roll back a reported result when the manifest, protocol hash, or verifier isolation check fails."
    ),
    (
        "distributed_startup",
        "distributed training startup",
        "Give a prioritized diagnosis or design for distributed training startup failures. Cover rank mapping, "
        "rendezvous, NCCL environment, topology/NIC selection, environment capture, and a minimal reproduction.",
        "Start with a one-node, two-rank minimal reproduction using the same container image and launcher, then add "
        "nodes one variable at a time. Capture rank, local rank, world size, hostname, selected interfaces, GPU/NIC "
        "PCIe topology, CUDA/NCCL versions, rendezvous endpoint, ports, and all NCCL environment variables. Verify that "
        "rank-to-device mapping is one-to-one and that every rank reaches the rendezvous within a bounded timeout. Use "
        "NCCL debug logs and a targeted collective test to distinguish bootstrap, topology, transport, and collective "
        "formation failures. Do not tune algorithms until interface and route selection are proven. Roll back a cluster "
        "change if the minimal reproduction regresses or rank membership becomes inconsistent."
    ),
    (
        "system_design_controls",
        "operational system design",
        "Write a production system design or technical analysis that makes control-plane decisions auditable. Cover "
        "components, data flow, observability, failure handling, validation, and rollback.",
        "Define the request path, control path, and evidence path separately. Name ownership for admission, scheduling, "
        "configuration, versioning, and rollback; do not hide these in a generic orchestration box. For every capacity "
        "or latency claim, label it ESTIMATE until a pinned run produces the matching MEASURED artifact. Expose service "
        "metrics and device metrics together, including queue time, TTFT, TPOT, memory headroom, errors, and recovery "
        "time. Use a canary with fixed success, latency, safety, and rollback gates, and preserve the prior artifact for "
        "reversal. Validate one hypothesis at a time against a simultaneous control and retain raw evidence."
    ),
]


def make_record(topic_index, variant):
    key, title, request, answer = TOPICS[topic_index]
    case_id = f"repair-v0.1-{key}-{variant:02d}"
    question = f"{request} Scenario variant {variant}: make the operational boundaries explicit."
    return {
        "id": case_id,
        "provenance": "authored_regression_repair_v0.1",
        "review_status": "needs_domain_expert_review",
        "repair_target": key,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Assumptions. This is a bounded {title} scenario; names and thresholds require verification.\n\n{answer}\n\nScenario focus. Variant {variant} changes the failure surface, not the evidence standard: record the configuration and compare only one changed variable at a time."},
        ],
    }


def main():
    root = Path(__file__).parent
    data_dir = root / "data"
    repair_path = data_dir / "regression_repair_v0.1.jsonl"
    mixed_path = data_dir / "teacher_b_plus_repair_v0.1_train.jsonl"
    base_path = data_dir / "teacher_b_native_train.jsonl"
    records = [make_record(topic_index, variant) for topic_index in range(len(TOPICS)) for variant in range(1, 21)]
    if len(records) != 80 or len({record["id"] for record in records}) != 80:
        raise RuntimeError("repair record construction failed")
    repair_path.write_text("".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records), encoding="utf-8")
    base_records = [json.loads(line) for line in base_path.read_text(encoding="utf-8").splitlines() if line]
    mixed_records = []
    for repair_index, record in enumerate(records):
        start = repair_index * 30
        mixed_records.extend(base_records[start:start + 30])
        mixed_records.append(record)
    mixed_records.extend(base_records[len(records) * 30:])
    if len(base_records) != 2400 or len(mixed_records) != 2480:
        raise RuntimeError("unexpected base or mixed dataset size")
    mixed_path.write_text("".join(json.dumps(record, ensure_ascii=True) + "\n" for record in mixed_records), encoding="utf-8")
    manifest = {
        "base_records": len(base_records),
        "repair_records": len(records),
        "mixed_records": len(mixed_records),
        "mixing_policy": "insert one repair record after each consecutive block of 30 base records",
        "repair_target_counts": {key: 20 for key, _, _, _ in TOPICS},
        "review_status": "needs_domain_expert_review",
        "repair_sha256": hashlib.sha256(repair_path.read_bytes()).hexdigest(),
        "mixed_sha256": hashlib.sha256(mixed_path.read_bytes()).hexdigest(),
    }
    (data_dir / "regression_repair_v0.1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()