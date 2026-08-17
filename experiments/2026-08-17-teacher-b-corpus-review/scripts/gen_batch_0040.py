import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0040.jsonl")
START, N = 390, 10  # 0-indexed: lines 391..400

MISLEAD = """Misleading intuition: "NCCL allreduce bandwidth scales with the number of GPUs, so adding ranks makes collectives faster."

Correction and mechanism: ring allreduce moves a fixed 2*(N-1)/N * S bytes per rank (S = buffer size, N = ranks) and completes in 2*(N-1) steps. Per-step payload is S/N, so as N grows the payload per step shrinks while step count grows linearly. Bus bandwidth (busbw, as reported by nccl-tests) is roughly flat in the bandwidth-bound regime; algorithmic time is bounded below by 2*(N-1)*alpha (per-hop latency alpha) plus 2*(N-1)/N*S/B. Adding ranks therefore adds latency, never throughput, for a fixed S.

Boundary condition: below roughly S/N < ~1 MB per step the collective becomes latency-bound, and NCCL typically switches to tree/CollNet or NVLS algorithms (NCCL_ALGO) where latency grows ~log(N) instead of N. So the "more GPUs is slower" statement is strictly true only in the ring, bandwidth-bound regime; at small message sizes tree algorithms change the scaling law.

Falsifiable prediction: run all_reduce_perf -b 8 -e 1G -f 2 at N=2, 4, 8 on the same fabric. Prediction: busbw within +-10% across N for S >= 256 MB; latency for S = 8 B grows about linearly with N under NCCL_ALGO=Ring and about logarithmically under NCCL_ALGO=Tree.

Evidence required: nccl-tests output (algbw/busbw), NCCL_DEBUG=INFO topology and algorithm/protocol selection lines, nvidia-smi topo -m, and confirmation that no rank crossed a slower tier (PCIe/host-bridge/network) between configurations.

Assumptions: single homogeneous fabric, no other traffic, NCCL >= 2.18. Rollback threshold: if a tuning change (NCCL_ALGO/NCCL_PROTO/NCCL_MIN_NCHANNELS) does not improve end-to-end step time by >=3% over 3 runs, revert to defaults."""

EXPERIMENT = """Goal: decide whether a suspected NCCL slowdown is caused by transport/topology selection rather than by the training workload.

Hypothesis (falsifiable): H1 - allreduce busbw for 256 MB buffers is >=25% below the fabric's expected ceiling because NCCL fell back from NVLink/IB-RDMA to PCIe/TCP sockets. H0 - transport is as intended and the regression lives in the workload (imbalance, CPU-side stalls, data loader).

Design:
1. Isolate the collective: run nccl-tests all_reduce_perf -b 8 -e 1G -f 2 -g 1 with one process per GPU, no training job co-resident. This removes workload variables.
2. Control variable: NCCL transport. Arm A = default. Arm B = NCCL_P2P_DISABLE=1. Arm C = NCCL_IB_DISABLE=1 (multi-node only). Arm D = NCCL_ALGO=Tree.
3. Repetitions: 5 runs per arm, discard the first (warm-up/buffer registration), report median and p10/p90.
4. Fixed factors: same nodes, same GPU IDs, same NCCL and driver version, same CPU affinity/NUMA binding, MPS/MIG off, persistence mode on, clocks unlocked or equally locked.

Mechanism being probed: NCCL picks a transport per peer pair at communicator init (P2P/NVLink > shared memory > IB/RoCE > TCP sockets). A silent downgrade shows up as a large busbw drop with unchanged correctness, and is visible in NCCL_DEBUG=INFO "via" / "NET/IB" / "NET/Socket" lines.

Boundary condition: this experiment only separates transport from workload if message sizes overlap the training job's real ones. If the job's gradient buckets are ~25 MB, results at 1 GB do not transfer; add the actual bucket size to the sweep (torch DDP bucket_cap_mb).

Measurements: algbw/busbw per size, NCCL_DEBUG=INFO transport lines, nvidia-smi topo -m, ib_write_bw baseline (multi-node), and end-to-end step time with torch profiler to attribute residual gap.

Decision rule: if arm A busbw is within 10% of arm B/C ceilings and near hardware expectation, reject H1 and move to workload profiling. If default is >=25% below and NCCL_DEBUG shows Socket where IB was expected, accept H1 and fix routing/GID/driver.

Rollback threshold: any env-var change kept only if it improves median end-to-end step time >=3% with no correctness delta (loss curve match over 200 steps); otherwise revert.

Assumptions: no other tenants on the fabric; measurements are measured facts, expected ceilings are vendor-spec estimates, not measured."""

RUNBOOK = """Runbook: NCCL collective hang or slowdown

Trigger: training job stalls with no progress, or a watchdog raises "Watchdog caught collective operation timeout" / NCCL_ASYNC_ERROR_HANDLING abort.

Scope: this entry covers diagnosis only. It performs no destructive action beyond killing the already-failed job.

Steps:
1. Capture before touching anything: full stderr with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH, the rank that timed out, the collective op and sequence number from the watchdog message, and py-spy dump on every rank. Rank-level divergence (one rank in a different op) means a workload/control-flow bug, not a fabric bug.
2. Mechanism to keep in mind: NCCL collectives are lock-step per communicator. If one rank never enters the call (data loader stall, uneven batch count, an exception swallowed on one rank), all other ranks block until the watchdog timeout fires. The hang location therefore reports the victims, not the culprit.
3. Classify with the failure taxonomy: topology (nvidia-smi topo -m), transport (NET/IB vs NET/Socket in the log), process group (world size, rank map, duplicate device assignment), rank (missing/extra participant), timeout (NCCL_TIMEOUT / TORCH_NCCL_BLOCKING_WAIT settings vs real step time), workload (uneven shards, dynamic control flow).
4. Confirm hardware health only if step 3 points there: nvidia-smi -q -d ECC,CLOCK for XID/ECC/thermal-throttle, dmesg for Xid errors, ibstat / ib_write_bw for link state and RDMA path.
5. Reproduce out-of-band with nccl-tests on the same allocation to separate fabric from workload.

Boundary condition: raising the timeout is never a fix. It is only valid when a step legitimately exceeds the timeout (very large batch, checkpoint barrier), which must be proven by a measured step-time distribution. Otherwise it converts a fast failure into a slow one and hides the culprit rank.

Evidence required before any change: timed-out rank ID, collective name and seq, transport lines, topo matrix, dmesg Xid lines, and a nccl-tests baseline from the same nodes.

Rollback threshold: after a mitigation (node drain, env var, NCCL version pin), require 3 consecutive clean runs of >=500 steps. If a hang recurs, revert the change and escalate with the captured evidence rather than stacking mitigations.

Assumptions: single-tenant allocation; NCCL >= 2.18 with async error handling enabled. Anything about a specific cluster's fabric topology must be measured, not assumed."""

ANSWERS = {
    "corpus-00437": MISLEAD, "corpus-00438": MISLEAD, "corpus-00439": MISLEAD, "corpus-00440": MISLEAD,
    "corpus-00441": EXPERIMENT, "corpus-00442": EXPERIMENT, "corpus-00443": EXPERIMENT,
    "corpus-00444": EXPERIMENT, "corpus-00445": EXPERIMENT,
    "corpus-00446": RUNBOOK,
}

RISKS = [
    "source_assistant is a generic one-line taxonomy stub that does not answer the specific instruction (misleading-intuition / experiment design / runbook)",
    "no mechanism, no boundary condition, no falsifiable prediction, and no evidence list as the prompt explicitly requires",
    "training on the stub would reinforce vague non-actionable answers in incident-response contexts",
]
EVID = [
    "nccl-tests all_reduce_perf algbw/busbw sweep on the target allocation",
    "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH logs showing selected algorithm, protocol and transport",
    "nvidia-smi topo -m and, for multi-node, ibstat / ib_write_bw baselines",
    "dmesg Xid / ECC records and measured per-step time distribution",
]

rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if i < START:
            continue
        if i >= START + N:
            break
        d = json.loads(line)
        msgs = {m["role"]: m["content"] for m in d["messages"]}
        sid = d["id"]
        rows.append({
            "source_id": sid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": msgs["user"],
            "source_assistant": msgs["assistant"],
            "corrected_answer": ANSWERS[sid],
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 1,
                "operational_safety": 2,
            },
            "risks": RISKS,
            "evidence_required": EVID,
            "confidence": 0.82,
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), OUT)
