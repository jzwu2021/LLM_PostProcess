import json, os

CORPUS = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0027.jsonl"
START, END = 270, 280  # 0-indexed slice -> lines 271..280 -> corpus-00290..00299

MECH = """Mechanism (must be stated explicitly): with pipeline degree P and M microbatches per global
batch, a synchronous 1F1B schedule has bubble fraction (P-1)/(M+P-1) of the per-stage compute time,
assuming perfectly balanced stages and zero communication cost. Interleaved (virtual pipeline) 1F1B
with v model chunks per device reduces this to (P-1)/(v*M+P-1) at the cost of v x more stage-boundary
point-to-point transfers per step.

Boundary condition (where the formula stops holding): the bubble formula assumes stage compute is
balanced and that the boundary send/recv is fully overlapped. It breaks down when (a) stages are
imbalanced -- the slowest stage sets the step time and the embedding/LM-head stages are the usual
offenders; (b) the stage boundary crosses a node and the activation tensor per microbatch
(micro_bs * S * H * dtype_bytes) exceeds what the measured inter-node RDMA/RoCE bandwidth can move
inside one stage's compute window, at which point you are communication-bound and adding microbatches
stops helping; (c) memory: peak activation memory scales with the number of in-flight microbatches
(~P for 1F1B), so raising M past that point does not raise memory but raising P does."""

EVID = [
    "Per-stage forward/backward wall time from torch profiler or Megatron-LM timers, to prove or disprove stage balance",
    "Measured stage-boundary point-to-point bandwidth (ib_write_bw for RDMA, nccl-tests sendrecv) rather than nameplate NIC speed",
    "Nsight Systems / torch profiler timeline showing actual bubble width vs the (P-1)/(M+P-1) prediction",
    "Recorded config: P, TP, DP, M, micro_bs, schedule (GPipe / 1F1B / interleaved), activation recomputation on-off, dtype",
    "torch.cuda.max_memory_allocated per stage to confirm the in-flight-microbatch memory model",
]

RISKS_COMMON = [
    "Source answer is a one-line restatement with no bubble math, no schedule named, and no boundary condition -- unusable as a target for an infra copilot",
    "Comparing PP configs while silently changing activation recomputation or TP degree produces non-comparable throughput numbers",
    "Treating nameplate NIC bandwidth as achievable inter-stage bandwidth overstates the microbatch count at which PP stays compute-bound",
    "Raising pipeline degree P to fix a memory problem can convert a compute-bound job into a communication-bound one on inter-node boundaries",
]

def misleading(v):
    return f"""Misleading intuition: "pipeline parallelism gives you near-linear speedup once you use enough
microbatches, so PP is a cheap way to scale past one node."

Why it is wrong, concretely:

1. The bubble never reaches zero in a synchronous schedule. {MECH}

2. Even in the limit of large M, PP does not add FLOPs throughput the way DP does -- it only lets a
model that does not fit on one device run at all, and it *adds* per-step point-to-point traffic. With
P=8 and M=8 the bubble alone is 7/15 = 47% of ideal; you need M >= 4P before the bubble drops under
~6%, and large M inflates the global batch size, which is a training-dynamics change, not a free knob.

3. The second half of the intuition ("cheap way to scale past one node") is the more dangerous half.
PP is usually the *right* parallelism to put on the slow inter-node link precisely because it sends
one activation tensor per microbatch boundary instead of all-reducing gradients, but that only holds
if micro_bs*S*H*dtype_bytes / measured_link_bandwidth is small relative to one stage's compute time.
Cross that ratio and every added microbatch adds exposed communication instead of hiding bubble.

Corrected statement: PP converts a memory constraint into a latency/bubble constraint. Its ceiling is
set by max(slowest stage, exposed boundary communication), and the bubble term (P-1)/(M+P-1) is a
lower bound on the loss, not an estimate of it.

Falsifiable form of the claim: "at P=4, M=32, 1F1B, recompute off, we retain >=85% of the single-stage
per-GPU TFLOP/s." Falsified if measured per-GPU TFLOP/s ratio < 0.85, or if the profiler shows exposed
send/recv > 5% of step time.

Rollback gate: if end-to-end tokens/s regresses >3% vs the current production parallelism plan, or if
per-stage peak memory exceeds 90% of device memory on any rank, revert to the previous P/TP/M triple
before the next checkpoint interval. (Variant {v}.)"""

def experiment(v):
    return f"""Controlled experiment: does increasing microbatch count M actually buy back the pipeline bubble
on this cluster, and where does it stop?

Hypothesis (falsifiable): with stages balanced to within 5%, step time follows
T(M) ~ T_compute * (1 + (P-1)/M) up to the point where exposed stage-boundary communication becomes
the binding term; beyond that, T(M) flattens or worsens.

{MECH}

Design:
- Independent variable: M in {{P, 2P, 4P, 8P}}. Everything else frozen.
- Controls held fixed and recorded: P, TP degree, DP degree, micro_bs, sequence length S, dtype,
  schedule (1F1B), activation recomputation setting, NCCL env (NCCL_ALGO, NCCL_PROTO,
  NCCL_IB_DISABLE), driver/NCCL/framework versions, and which stage boundaries cross a node.
- Arm A: all stage boundaries intra-node (NVLink/PCIe). Arm B: identical config with one boundary
  forced across the RDMA/RoCE fabric. The A-vs-B delta at each M isolates the communication term from
  the bubble term -- this is the whole point of the design, since a single-arm sweep cannot tell a
  bubble from an exposed send.
- Dependent variables: step time, per-GPU TFLOP/s, exposed send/recv time from the profiler, per-stage
  peak memory.
- Repeats: >=5 measured steps after >=20 warmup steps, report median and p90; a single step is noise.

Prediction and refutation: if T(M) does not decrease as 1/M in Arm A, the stages are imbalanced and the
bubble model is not the limiter -- go measure per-stage time before touching M. If Arm B diverges from
Arm A beyond M=4P, you are communication-bound at the node boundary and the fix is to move the boundary
or shrink micro_bs*S*H, not to add microbatches.

Confounds to pre-empt: changing M changes the global batch size, so loss-curve comparisons across arms
are invalid unless the LR schedule is re-tuned -- this experiment measures throughput only, and any
convergence claim needs a separate run.

Evidence to capture: {"; ".join(EVID)}.

Rollback gate: this is a benchmark on a scratch job, not a production config change. Promote a winning
(P, M) only after a full-scale run holds >=3% throughput gain with no per-stage memory above 90% and no
loss-curve deviation beyond the established seed-noise band; otherwise revert. (Variant {v}.)"""

def runbook(v):
    return f"""Runbook: pipeline-parallel throughput or OOM investigation.

Preconditions: read-only triage first. Do not change P, M, or the recomputation flag on a running
production job before step 5.

Step 1 -- pin the config. Record P, TP, DP, ZeRO stage, M, micro_bs, S, dtype, schedule
(GPipe / 1F1B / interleaved-1F1B with v chunks), activation recomputation on/off, and the rank-to-node
map. Any throughput number without these is not comparable to any other number.

Step 2 -- decide whether you have a bubble problem or a balance problem. {MECH}
Compute the predicted bubble (P-1)/(M+P-1) and compare with measured (T_step - T_slowest_stage*M) / T_step.
If measured loss far exceeds predicted, it is imbalance or exposed communication, not the bubble.

Step 3 -- check stage balance. Pull per-stage forward/backward timers. The first stage (embedding) and
last stage (LM head + loss) are the usual outliers; a >10% spread means rebalance layer assignment
before touching anything else. Rebalancing is usually the largest single win and costs no memory.

Step 4 -- check the boundary link. For each stage boundary that crosses a node, measure actual
bandwidth (ib_write_bw / nccl-tests sendrecv), not the nameplate. Compare
micro_bs*S*H*dtype_bytes / measured_BW against one stage's compute time. If the ratio is not << 1, you
are communication-bound: adding microbatches will not help, and the correct moves are to relocate the
boundary inside a node, reduce micro_bs, or confirm GPUDirect RDMA is actually active
(NCCL_DEBUG=INFO should show the IB/GDRDMA path; a silent fallback to a host-staged path is a common
and expensive failure).

Step 5 -- if OOM rather than slow: peak activation memory scales with in-flight microbatches (~P under
1F1B), not with M, so raising M is memory-neutral while raising P is not. Options in order of
increasing cost: enable selective activation recomputation, reduce micro_bs, rebalance layers off the
hot stage, then raise P.

Step 6 -- change one variable, rerun >=20 warmup + >=5 measured steps, report median and p90.

Rollback gate: revert immediately if tokens/s regresses >3%, if any rank's peak memory exceeds 90% of
device memory, or if the loss curve leaves the established seed-noise band within 200 steps. Keep the
previous config in the job spec so rollback is a one-line revert, and never carry a config change past
a checkpoint boundary without a comparison run.

Escalate rather than tune blindly if NCCL_DEBUG shows a transport fallback, if per-stage times are
unstable run-to-run (suspect thermal/clock throttling or a noisy neighbor), or if the fabric shows
retransmits -- those are cluster faults, not tuning targets. (Variant {v}.)"""

BUILDERS = {
    "corpus-00290": ("misleading", misleading, 5),
    "corpus-00291": ("experiment", experiment, 1),
    "corpus-00292": ("experiment", experiment, 2),
    "corpus-00293": ("experiment", experiment, 3),
    "corpus-00294": ("experiment", experiment, 4),
    "corpus-00295": ("experiment", experiment, 5),
    "corpus-00296": ("runbook", runbook, 1),
    "corpus-00297": ("runbook", runbook, 2),
    "corpus-00298": ("runbook", runbook, 3),
    "corpus-00299": ("runbook", runbook, 4),
}

rows = []
with open(CORPUS) as f:
    lines = f.readlines()[START:END]

for line in lines:
    d = json.loads(line)
    sid = d["id"]
    msgs = d["messages"]
    u = [m["content"] for m in msgs if m["role"] == "user"][0]
    a = [m["content"] for m in msgs if m["role"] == "assistant"][0]
    kind, fn, v = BUILDERS[sid]
    extra = []
    if kind == "experiment":
        extra = ["Source answer proposes no experiment at all: no independent variable, no control arm, no repeat count, so nothing about it is falsifiable"]
    elif kind == "runbook":
        extra = ["Source answer has no ordered steps, no read-only triage phase, and no rollback gate, so following it on a production job is unsafe"]
    else:
        extra = ["Source answer states the mechanism as if it were the correction, leaving the misleading near-linear-speedup intuition intact"]
    rows.append({
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": fn(v),
        "quality_dimensions": {
            "technical_correctness": 3,
            "instruction_coverage": 1,
            "operational_safety": 2,
        },
        "risks": RISKS_COMMON + extra,
        "evidence_required": EVID,
        "confidence": 0.79,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), "->", OUT)
