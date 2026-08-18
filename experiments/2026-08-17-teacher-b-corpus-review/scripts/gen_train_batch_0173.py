import json, hashlib, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0173.jsonl"
START, N = 1720, 10

rows = open(CORPUS).read().splitlines()[START:START+N]
items = [json.loads(r) for r in rows]

HEAD = ("Assumptions (state them, and correct me if your environment differs): {asm} "
        "Every number below is labelled ESTIMATE (derived by arithmetic from stated inputs) or MEASURED "
        "(only after you actually collect it on your hardware). I assert no platform-specific fact I have not derived here.\n\n")

# ---- NCCL init-hang lane: distinct primary mechanism per variant ----
HANG_ASM = ("a multi-node PyTorch + NCCL job; the stall occurs at init_process_group or the first collective; "
            "no rank has emitted a Python traceback; and the job ran successfully on some earlier configuration.")

HANG = {
"corpus-01896": dict(
 mech="Rendezvous store asymmetry: a subset of ranks never registers with the TCP/c10d store, so the surviving ranks block in the barrier at the end of init_process_group.",
 hyp="H1: fewer than world_size ranks reach the store. Falsifiable: if a store-key dump at hang time shows exactly world_size entries, H1 is dead and the fault is downstream of rendezvous.",
 exp=("Controlled experiment: keep the launcher, image and env fixed; change only instrumentation. "
      "(1) Wrap init_process_group with a pre-log of RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT and a post-log; "
      "(2) run with TORCH_DISTRIBUTED_DEBUG=DETAIL and NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,ENV; "
      "(3) diff the set of ranks that logged pre vs post. "
      "Control arm: the same command with world_size reduced to 2 ranks on one node."),
 meas=("MEASURED to collect: count of ranks logging pre-init vs post-init; MASTER_ADDR resolution per node (getent hosts); "
       "TCP reachability to MASTER_PORT from every node; py-spy dump of one hung rank showing whether the stack sits in store wait or in ncclCommInitRank."),
 conf=("Confounders: a slow-but-not-stuck rendezvous under high node count can look like a hang. "
       "ESTIMATE: c10d store registration for 64 ranks is single-digit seconds on a healthy control plane, so a >120 s stall is not normal slowness — "
       "derivation: one short TCP round trip per rank, ~<10 ms each on a LAN, times 64 ranks, plus barrier; it is arithmetic on assumed RTT, not a measurement of your fabric."),
 fix="If H1 holds, the fault is scheduler/launcher side (a rank died at import, was OOM-killed by the cgroup, or got the wrong MASTER_ADDR), not NCCL.",
),
"corpus-01897": dict(
 mech="Interface selection divergence: ranks pick different network interfaces (e.g. a docker0/bridge address on some nodes, the RDMA-capable NIC on others), so bootstrap connections are attempted to unroutable addresses and block until timeout.",
 hyp="H1: at least two ranks advertise bootstrap addresses on different subnets. Falsifiable: if NCCL INFO lines show every rank using the same interface and subnet, H1 is refuted.",
 exp=("Controlled experiment: fix everything, then pin the interface explicitly. Arm A: unset selection (current behaviour). "
      "Arm B: NCCL_SOCKET_IFNAME=<the single routable NIC> on all nodes. "
      "If B initialises and A hangs, interface selection is causal. Re-run each arm 3 times to exclude a flaky single trial."),
 meas=("MEASURED to collect: the 'NET/Socket : Using [0]<ifname>:<ip>' line from every rank; `ip -o -4 addr` per node; whether any node exposes a bridge/veth address numerically lower than the real NIC."),
 conf=("Confounders: containers with host networking vs bridged networking differ per node in some clusters; a mixed fleet will falsely acquit the config on the node you happen to inspect. "
       "Inspect all nodes, not a sample."),
 fix="If H1 holds, pin NCCL_SOCKET_IFNAME (and NCCL_IB_HCA if RDMA is intended) in the job spec rather than relying on autodetection.",
),
"corpus-01898": dict(
 mech="RDMA/RoCE path failure with silent fallback removal: the IB/RoCE verbs path is half-configured (GID index or PFC/DSCP mismatch), so ring setup stalls instead of falling back to sockets.",
 hyp="H1: the hang disappears when the RDMA transport is disabled. Falsifiable: if NCCL_IB_DISABLE=1 still hangs at the same point, the RDMA path is not the cause and H1 is refuted.",
 exp=("Controlled experiment, one variable at a time. Arm A: baseline. Arm B: NCCL_IB_DISABLE=1 (force socket transport). "
      "Arm C: keep RDMA but pin NCCL_IB_GID_INDEX to the RoCEv2 GID and NCCL_IB_HCA to the intended HCA. "
      "Prior to all arms, run ib_write_bw between the same node pair to establish whether raw RDMA works at all — that isolates fabric from NCCL."),
 meas=("MEASURED to collect: `show_gids` output and which GID index is RoCEv2 IPv4; ib_write_bw pass/fail and achieved GB/s between node pairs; "
       "PFC counters and pause frames on the switch ports; NCCL INFO transport lines (IB vs Socket) per rank."),
 conf=("Confounders: a lossless-fabric misconfiguration often degrades throughput long before it hangs, so a passing ib_write_bw does not fully acquit RoCE — "
       "check pause/discard counters as well. ESTIMATE only: if ib_write_bw reports well under line rate for the NIC's nominal speed, suspect PFC; that is a comparison rule, not a measurement of your fabric."),
 fix="If H1 holds and Arm B works, run degraded on sockets to unblock the team while fixing GID/PFC, and record the throughput cost of that fallback before accepting it.",
),
"corpus-01899": dict(
 mech="Collective mismatch rather than connectivity: ranks enter different collectives, or the same collective with different shapes/dtypes/group membership, so each side waits for a peer that never posts the matching operation.",
 hyp="H1: the ranks' first collective calls are not identical in op, tensor shape, dtype and process group. Falsifiable: if TORCH_DISTRIBUTED_DEBUG=DETAIL reports fully consistent collective signatures across ranks, H1 is refuted and the fault is transport-level.",
 exp=("Controlled experiment: replace the model step with a minimal all_reduce of a fixed 1-element float32 tensor on the world group, same launcher and env. "
      "If the minimal all_reduce completes and the real step hangs, the fault is in your code's control flow (a rank-conditional branch, an uneven dataloader tail, or a conditional checkpoint/eval path), not in NCCL."),
 meas=("MEASURED to collect: per-rank log of (op, shape, dtype, group ranks) for the first 5 collectives; py-spy dump of two hung ranks to see whether they sit in different call sites; "
       "number of batches each rank has consumed at hang time."),
 conf=("Confounders: an uneven final batch or a rank-dependent early exit reproduces only at a specific dataset size, so a short smoke run can hide it. "
       "Reproduce with the real dataset length or force the uneven tail deliberately."),
 fix="If H1 holds, the fix is code-level (drop_last, a synchronised decision broadcast before conditional collectives), not an env-var change.",
),
}

# ---- weight-only quantization fair-comparison lane ----
Q_ASM = ("a decoder-only LLM served behind an OpenAI-compatible endpoint on a fixed GPU fleet; "
         "'weight-only quantization' means weights stored at reduced precision with activations and accumulation kept at higher precision; "
         "and the business goal is cost per served token at a fixed quality bar, not peak benchmark throughput.")

Q = {
"corpus-01902": dict(
 mech="Define the comparison as a paired A/B on one axis: identical model checkpoint lineage, identical prompts, identical serving stack version, identical GPU type and count; only the weight precision differs.",
 hyp="H1: at a fixed quality bar, the quantized deployment reduces cost per 1M output tokens by a material margin. Falsifiable: if measured quality drops below the bar, or throughput gain is inside the confidence interval of the baseline, H1 fails.",
 exp=("Design: two arms, FP16/BF16 baseline vs weight-only quantized, same commit of the server, same tokenizer, same sampling parameters (fix temperature and seed for the quality arm), same request trace replayed from a captured production log. "
      "Randomise arm order across repeats to absorb drift in the shared cluster. At least 3 independent runs per arm."),
 meas=("MEASURED per arm: TTFT p50/p95, TPOT p50/p95, output tokens/s at each concurrency level, GPU memory high-water mark, and task quality on a held-out set. "
       "Report bootstrap 95% confidence intervals over runs, not a single number."),
 conf=("Confounders: KV-cache size is unchanged by weight-only quantization, so memory savings appear in weights only; "
       "ESTIMATE: moving weights from 16-bit to 4-bit cuts the weight footprint to roughly a quarter plus scale/zero-point overhead — derivation is bits-per-parameter arithmetic, not a measurement. "
       "Freed memory usually buys more KV cache and therefore concurrency, which is where the real throughput gain comes from; attribute it correctly."),
 gate="Rollback gate: revert if quality on the held-out set falls below the agreed bar, or if p95 TTFT regresses beyond the SLO, even if average throughput improves.",
),
"corpus-01903": dict(
 mech="Separate the two effects that get conflated: kernel-level compute change (dequant overhead per matmul) and capacity change (more KV cache from smaller weights). Measure them independently.",
 hyp="H1: at batch size 1 the quantized arm is no faster, and may be slower, while at high concurrency it wins. Falsifiable: if the quantized arm wins at batch 1 by a margin outside the CI, the dequant-overhead model is wrong.",
 exp=("Two-part experiment. Part 1: single-stream decode latency, concurrency=1, fixed prompt length and fixed 256 output tokens — this isolates per-token kernel cost. "
      "Part 2: sweep concurrency (1, 4, 16, 64, 256) with a fixed arrival trace and record the throughput/latency curve for both arms. "
      "Hold max_model_len and any KV-cache limit constant in Part 1, then let it float in Part 2 to expose the capacity effect."),
 meas=("MEASURED: TPOT at concurrency 1; the concurrency at which each arm violates the latency SLO; achievable KV-cache blocks per arm; realised GPU utilisation."),
 conf=("Confounders: many quantization kernels are only fast in a narrow shape range and fall back to a slow path otherwise; "
       "a change in batch shape between arms silently switches kernels. Log which kernel/backend each arm actually selected."),
 gate="Rollback gate: if the quantized arm loses at the concurrency your production trace actually exhibits, the win is theoretical — do not ship it on the strength of a saturated-load benchmark.",
),
"corpus-01904": dict(
 mech="Quality evaluation must be generative and task-matched, not perplexity-only, because weight-only quantization damage is uneven across capabilities (long-context recall and multi-step arithmetic degrade before fluency does).",
 hyp="H1: aggregate quality is preserved within the agreed tolerance. Falsifiable: if any single high-stakes slice degrades beyond tolerance, H1 fails even when the aggregate score looks flat.",
 exp=("Evaluate on a stratified suite: short instruction-following, long-context retrieval at your real context length, code generation with execution-based checking, tool-call/JSON schema validity, and any domain slice you actually sell. "
      "Score per slice with per-slice CIs. Run the baseline twice to establish evaluation noise before comparing arms — a difference smaller than baseline-vs-baseline noise is not a result."),
 meas=("MEASURED: per-slice pass rate with bootstrap CIs; tool-call schema-validity rate; refusal/format-failure rate; baseline-vs-baseline noise floor."),
 conf=("Confounders: greedy decoding hides variance and can flatter one arm; sampling adds noise that swamps small effects. "
       "Report both a greedy run and a sampled run with fixed seeds, and state which one the decision is based on."),
 gate="Rollback gate: any regression on the tool-call validity or long-context slice beyond the pre-agreed tolerance blocks the rollout regardless of cost savings.",
),
"corpus-01905": dict(
 mech="Calibration-set dependence: post-training quantization that uses calibration data has a result that is a function of that data. The comparison is not reproducible unless the calibration corpus, sample count and sequence length are pinned and version-controlled.",
 hyp="H1: quantized quality is stable across calibration seeds. Falsifiable: if two quantizations differing only in calibration sample draw disagree beyond the evaluation noise floor, the method is calibration-sensitive and a single quantized artifact cannot be trusted.",
 exp=("Produce 3 quantized artifacts from the same checkpoint with 3 different calibration draws (same size, same source distribution, different seeds). "
      "Evaluate all 3 on the fixed suite. The spread across the 3 is the method's reproducibility error and must be reported alongside the baseline delta."),
 meas=("MEASURED: per-artifact quality per slice; the max-minus-min spread across the 3 artifacts; the evaluation noise floor from repeated baseline runs; sha256 of each artifact and of each calibration file."),
 conf=("Confounders: calibration data drawn from a distribution unlike production traffic can look fine on your eval and fail in production. "
       "Draw calibration samples from captured production prompts where policy allows, and record that provenance."),
 gate="Rollback gate: if the calibration-seed spread exceeds the quality delta you are trying to defend, the comparison is not decision-grade — do not ship, tighten the method first.",
),
"corpus-01906": dict(
 mech="Cost accounting: the decision variable is currency per 1M served tokens at the SLO, not tokens/s. That requires converting measured throughput into a required replica count under the real arrival pattern.",
 hyp="H1: the quantized arm lowers cost per 1M output tokens at the SLO. Falsifiable: if the SLO-constrained replica count is unchanged (for example because the model still needs the same number of GPUs to fit or to meet TTFT), the throughput gain does not convert into savings and H1 fails.",
 exp=("From each arm's measured latency/throughput curve, find the maximum sustained rate that holds p95 TTFT and p95 TPOT within SLO. "
      "Divide the peak production arrival rate by that number to get replicas, round up, multiply by GPU-hour price. "
      "Do this arithmetic for both arms with the same peak trace and the same headroom factor."),
 meas=("MEASURED: SLO-constrained sustained rate per replica for each arm; peak and p95 arrival rate from production logs; GPUs per replica; GPU-hour price."),
 conf=("Confounders: ESTIMATE-only savings are common here — a 2x throughput gain gives 2x cost reduction only if replicas are the sole cost and rounding does not eat it. "
       "With a small fleet, ceil() can erase most of the benefit: derivation is integer division on the replica formula above, not a measured result. "
       "Also count the one-off quantization and re-validation engineering cost against the first period's savings."),
 gate="Rollback gate: revert if realised savings after one billing period are below the modelled savings by more than the pre-agreed margin, or if incident rate rises.",
),
"corpus-01907": dict(
 mech="Failure-mode and operational comparison: the arms must be compared on tail behaviour and blast radius, not just steady state — kernel coverage gaps, unsupported sequence lengths, and loss of the ability to hot-swap back all count as costs.",
 hyp="H1: the quantized arm has no worse tail failure behaviour than the baseline. Falsifiable: if the quantized arm shows a higher rate of request errors, numerical anomalies (NaN/inf), or unsupported-shape fallbacks under the production trace, H1 fails.",
 exp=("Run both arms against a soak test using the real trace, including the pathological tail: maximum-length prompts, maximum-length generations, high-concurrency bursts, and abrupt cancellations. "
      "Duration long enough to cross at least one full traffic cycle. Inject a rollback drill: switch traffic back to the baseline mid-soak and time it."),
 meas=("MEASURED: request error rate by class; count of NaN/inf or garbage-output incidents; p99 and p99.9 latency; observed rollback time from decision to full traffic restored; peak memory headroom before OOM."),
 conf=("Confounders: short benchmarks never hit the tail; a clean 10-minute run is not evidence about a 24-hour one. "
       "Also, the two arms may hold different amounts of free memory, so an OOM that appears only in one arm may be a capacity-tuning artifact rather than a quantization defect — equalise or record the setting."),
 gate="Rollback gate: keep the baseline weights resident and the routing switch armed for the full bake period; revert on any of — error-rate regression, a numerical-anomaly incident, or rollback drill exceeding the agreed recovery objective.",
),
}

def build_hang(d):
    p = HANG[d["id"]]
    return (HEAD.format(asm=HANG_ASM) +
        "Primary mechanism I am testing first\n" + p["mech"] + "\n\n" +
        "Falsifiable hypothesis\n" + p["hyp"] + "\n\n" +
        "Controlled experiment\n" + p["exp"] + "\n\n" +
        "Evidence to collect\n" + p["meas"] + "\n\n" +
        "Confounders and boundary conditions\n" + p["conf"] + "\n\n" +
        "Decision and rollback\n" + p["fix"] + " Rollback gate: apply one change at a time, keep the last known-good launcher spec pinned by commit, and revert immediately if the change does not clear the hang in a single reproduction — "
        "stacking env-var changes destroys attribution. If two consecutive single-variable arms fail to reproduce or clear the hang, stop and escalate with the collected rank logs rather than continuing to guess.\n")

def build_q(d):
    p = Q[d["id"]]
    return (HEAD.format(asm=Q_ASM) +
        "What makes the comparison fair\n" + p["mech"] + "\n\n" +
        "Falsifiable hypothesis\n" + p["hyp"] + "\n\n" +
        "Controlled experiment\n" + p["exp"] + "\n\n" +
        "Evidence to collect\n" + p["meas"] + "\n\n" +
        "Confounders and boundary conditions\n" + p["conf"] + "\n\n" +
        "Decision and rollback\n" + p["gate"] + " Record the serving-stack commit, quantization toolchain version and artifact sha256 with every result, or the comparison cannot be reproduced later.\n")

HANG_RISK = ["Env-var shotgunning (changing several NCCL variables at once) destroys causal attribution",
             "Disabling RDMA to clear a hang silently ships a large throughput regression if left in place",
             "Diagnosing on a sampled subset of nodes misses a per-node fleet inconsistency"]
HANG_EV = ["Per-rank pre/post init_process_group logs with RANK/WORLD_SIZE/MASTER_ADDR",
           "NCCL_DEBUG=INFO with NCCL_DEBUG_SUBSYS=INIT,ENV from every rank",
           "py-spy dump of at least two hung ranks",
           "Minimal 1-element all_reduce control run at reduced world size"]
Q_RISK = ["Reporting throughput gains without holding the quality bar fixed",
          "Attributing capacity-driven gains to kernel speed",
          "Single-run numbers without confidence intervals or a baseline noise floor",
          "Unpinned calibration data making the quantized artifact irreproducible"]
Q_EV = ["Replayed production request trace with recorded arrival pattern",
        "Per-slice generative quality scores with bootstrap 95% CIs",
        "Baseline-vs-baseline repeat run establishing the evaluation noise floor",
        "Serving-stack commit, quantization toolchain version and artifact sha256"]

out, seen = [], set()
for d in items:
    sid = d["id"]
    su = d["messages"][1]["content"]
    sa = d["messages"][2]["content"]
    if sid in HANG:
        ca = build_hang(d); risks, ev = HANG_RISK, HANG_EV
        qd = {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}
        conf = 0.62
    else:
        ca = build_q(d); risks, ev = Q_RISK, Q_EV
        qd = {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}
        conf = 0.64
    h = hashlib.sha256(ca.encode()).hexdigest()
    assert h not in seen, f"duplicate corrected_answer for {sid}"
    seen.add(h)
    assert ca.strip()
    out.append({
        "source_id": sid, "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional", "decision": "rewrite",
        "source_user": su, "source_assistant": sa,
        "corrected_answer": ca, "quality_dimensions": qd,
        "risks": risks, "evidence_required": ev, "confidence": conf,
    })

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(out), "unique_answers", len(seen))
