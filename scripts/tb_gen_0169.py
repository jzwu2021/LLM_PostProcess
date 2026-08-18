#!/usr/bin/env python3
"""Generate teacher-B provisional blind-review batch 0169 (corpus lines 1681-1690)."""
import json, hashlib, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0169.jsonl"
START, N = 1680, 10  # 0-indexed

HEAD = ("Scope: multi-GPU / multi-node job hangs during collective initialization "
        "(torch.distributed init_process_group or the first NCCL collective never returns).\n"
        "Assumptions: PyTorch + NCCL backend, containerized launch, one process per GPU.\n"
        "All numbers below are ESTIMATE unless a step says MEASURED; ESTIMATEs are order-of-magnitude "
        "planning values derived from the stated mechanism, not from a benchmark on this cluster.\n\n")

TAIL = ("\nGeneric confounders to control: stale processes holding GPUs from a previous run; a second job "
        "sharing the same MASTER_PORT or NCCL_COMM_ID; container restarts that renumber ranks; cgroup or "
        "MIG changes between runs.\n"
        "Rollback gate: if a proposed change (env var, NIC pin, topology change) does not move the "
        "MEASURED rendezvous-completion time or the hang stage within 2 controlled runs, revert it before "
        "stacking the next change; never carry more than one unvalidated NCCL env override into production.\n"
        "Evidence to keep for every run: NCCL_DEBUG=INFO + NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV logs from at "
        "least rank 0 and one remote rank, the launcher's rank/host/local_rank table, and py-spy dump of a "
        "hung rank.")

# Each variant isolates ONE distinct mechanism with its own falsifiable hypothesis + controlled experiment.
MECHS = [
    ("Rendezvous / store reachability",
     "Hypothesis (falsifiable): the hang is BEFORE any NCCL traffic — it is the TCPStore/c10d rendezvous, "
     "because a subset of ranks cannot reach MASTER_ADDR:MASTER_PORT.\n"
     "Mechanism: init_process_group blocks in the store barrier until all world_size ranks check in; a single "
     "unreachable rank makes every other rank wait until the (often 1800 s) timeout, which looks like an NCCL hang.\n"
     "Controlled experiment: (1) py-spy dump every rank — if frames sit in `TCPStore`/`_store_based_barrier` and "
     "not in `ncclCommInitRank`, the hypothesis is supported; (2) from each node run a plain TCP connect to "
     "MASTER_ADDR:MASTER_PORT; (3) re-run with world_size reduced to only the nodes that connected.\n"
     "Falsification: if all ranks are already inside ncclCommInitRank, rendezvous is NOT the cause — stop here.\n"
     "Boundary: this failure mode is independent of NIC/RDMA config; it also fires when MASTER_ADDR resolves to "
     "a container-local address.\n"
     "Expected signal: rendezvous normally completes in <5 s for <=64 ranks (ESTIMATE, from store round-trips "
     "being a handful of RTTs); anything >60 s is pathological."),

    ("World-size / rank-map mismatch",
     "Hypothesis (falsifiable): the launcher published an inconsistent rank map — the sum of local ranks does not "
     "equal world_size, or two processes claim the same global rank.\n"
     "Mechanism: NCCL's init is a collective over exactly world_size unique ranks; a duplicated rank means one "
     "expected participant never arrives and the AllGather of comm IDs never completes.\n"
     "Controlled experiment: have every process log (hostname, pid, RANK, LOCAL_RANK, WORLD_SIZE, "
     "CUDA_VISIBLE_DEVICES) before init; assert the set of RANK values is exactly 0..world_size-1 with no "
     "duplicates. Then re-launch with a deliberately correct explicit rank map.\n"
     "Falsification: if the rank table is already a clean bijection, discard this hypothesis.\n"
     "Boundary: common with hand-rolled mpirun/srun wrappers and with elastic restarts; rare with a single "
     "torchrun invocation per node.\n"
     "Evidence: the rank table itself is the artifact — it is MEASURED, not inferred."),

    ("GPU visibility / device binding",
     "Hypothesis (falsifiable): two ranks on the same node are bound to the same physical GPU (or a rank sees a "
     "device that is already busy/ECC-fenced), so the init collective deadlocks on device ordinal contention.\n"
     "Mechanism: NCCL assumes a one-process-one-device mapping; when two communicators on one device enter "
     "different steps of the init graph they can block each other, and the CUDA context creation itself may stall.\n"
     "Controlled experiment: log torch.cuda.current_device() and the GPU UUID per rank; assert UUIDs are unique "
     "per node. Re-run pinned via CUDA_VISIBLE_DEVICES=$LOCAL_RANK. Cross-check `nvidia-smi --query-compute-apps` "
     "for leftover processes.\n"
     "Falsification: unique UUIDs per node and an idle GPU set kill this hypothesis.\n"
     "Boundary: MIG and cgroup device restriction change the ordinal-to-UUID mapping inside containers, so the "
     "ordinal alone is not evidence — the UUID is."),

    ("NIC / interface selection (NCCL_SOCKET_IFNAME)",
     "Hypothesis (falsifiable): NCCL auto-selected a non-routable interface (docker0, a management NIC, or a "
     "second unconnected port), so the bootstrap ring cannot be closed across nodes.\n"
     "Mechanism: NCCL's bootstrap uses TCP sockets over an interface chosen by heuristic; if different nodes pick "
     "different subnets, the bootstrap AllGather never completes and the job hangs before any data-plane transfer.\n"
     "Controlled experiment: NCCL_DEBUG=INFO and read the `NET/Socket : Using [ifname]` line on every node — if the "
     "chosen ifname differs across nodes or is a bridge, the hypothesis is supported. Then re-run with an explicit "
     "NCCL_SOCKET_IFNAME=<data NIC> and confirm the hang clears.\n"
     "Falsification: identical, routable ifname on all nodes and the hang persists -> not interface selection.\n"
     "Boundary: single-node jobs are immune; this only bites at >=2 nodes.\n"
     "Rollback gate: pin the ifname by exact name, not prefix, and revert if the pin does not change the hang stage."),

    ("IB/RoCE transport fallback and GID/PKey mismatch",
     "Hypothesis (falsifiable): the RDMA path (IB verbs / RoCEv2) is half-configured — devices are visible but the "
     "GID index, PKey, or traffic class does not match across nodes — so NCCL selects NET/IB and then stalls on QP "
     "connection instead of cleanly falling back to sockets.\n"
     "Mechanism: NCCL picks a transport per peer pair during init; a mismatched RoCE GID (v1 vs v2) or a DSCP/PFC "
     "policy that drops the CNP/QP-setup traffic leaves QPs in INIT and the init collective never returns.\n"
     "Controlled experiment: (1) confirm `ibv_devinfo` PORT_ACTIVE and identical GID type on both nodes; "
     "(2) run `ib_write_bw` between the two hosts — if raw RDMA also hangs, the fault is the fabric, not NCCL; "
     "(3) re-run the job with NCCL_IB_DISABLE=1 — if it now starts, the RDMA path is implicated.\n"
     "Falsification: ib_write_bw passes AND NCCL_IB_DISABLE=1 does not change behaviour -> not the IB transport.\n"
     "Boundary: NCCL_IB_DISABLE=1 is a DIAGNOSTIC, not a fix — socket fallback typically costs a large fraction of "
     "collective bandwidth (ESTIMATE: sockets deliver on the order of a few GB/s vs tens of GB/s for RDMA, derived "
     "from link-rate vs kernel-copy overhead), so it must be reverted once the root cause is fixed."),

    ("P2P / topology detection (NVLink, PCIe ACS, GDR)",
     "Hypothesis (falsifiable): intra-node peer-to-peer detection is wrong — PCIe ACS is enabled or IOMMU blocks "
     "P2P — so NCCL's topology graph search stalls or picks an unusable path.\n"
     "Mechanism: during init NCCL builds a topology graph and probes P2P/GDR reachability; a device pair that "
     "advertises P2P but fails at map time can wedge the graph search.\n"
     "Controlled experiment: run `nvidia-smi topo -m` and CUDA's simpleP2P/p2pBandwidthLatencyTest first — that is "
     "MEASURED ground truth. Then re-run the job with NCCL_P2P_DISABLE=1 and, separately, NCCL_DEBUG_SUBSYS=GRAPH to "
     "see where graph search stops.\n"
     "Falsification: p2pBandwidthLatencyTest passes for all pairs and NCCL_P2P_DISABLE=1 changes nothing -> reject.\n"
     "Boundary: on hosts without NVLink the P2P path is PCIe and ACS-off is required for direct P2P; disabling ACS "
     "is a host-wide security-relevant change and needs a change window plus a rollback plan.\n"
     "Note: GPUDirect RDMA (GDR) adds nvidia-peermem/dmabuf as an extra dependency; missing peermem shows as GDR "
     "silently off, which is slow but NOT usually a hang."),

    ("Timeout semantics and asymmetric collective ordering",
     "Hypothesis (falsifiable): it is not an init failure at all — ranks entered DIFFERENT collectives (or different "
     "shapes) because of data-dependent control flow, and the 'init hang' is really the first mismatched collective.\n"
     "Mechanism: NCCL collectives are matched positionally; if rank 3 calls all_reduce while others call broadcast, "
     "every rank blocks until the watchdog timeout. With TORCH_NCCL_BLOCKING_WAIT unset, the error surfaces late and "
     "looks like a hang.\n"
     "Controlled experiment: set TORCH_NCCL_ASYNC_ERROR_HANDLING=1 and a short timeout (e.g. 120 s) so the failure "
     "becomes a fast, attributable exception; enable NCCL flight-recorder / dump on timeout and compare the last "
     "collective sequence number per rank — divergent seq numbers confirm the hypothesis.\n"
     "Falsification: identical collective sequence numbers on all ranks -> ordering is not the cause.\n"
     "Boundary: shortening the timeout is a diagnostic aid; production timeouts must stay above the slowest "
     "legitimate collective (including checkpoint barriers), so revert the short timeout afterwards."),

    ("Minimal reproducer / bisect on world size",
     "Hypothesis (falsifiable): the fault is localized to a specific node or NIC, not to the job configuration.\n"
     "Mechanism: an init collective is all-or-nothing, so one bad participant hangs the whole world; bisecting the "
     "participant set converts a global symptom into a per-host verdict.\n"
     "Controlled experiment: replace the training script with a 20-line all_reduce of one tensor. Run it (a) single "
     "node 8 ranks, (b) each node alone, (c) node pairs. Record pass/fail per configuration — this table is MEASURED. "
     "If every single node passes and exactly the pairs containing node X fail, node X (or its NIC) is the fault "
     "domain.\n"
     "Falsification: if the minimal all_reduce passes on the full world while the real job hangs, the fault is in the "
     "application (ordering, dataloader barrier), not the fabric.\n"
     "Boundary: bisect cost is O(log N) rounds for a single bad node but O(N) if two nodes are bad; cap the search "
     "and escalate to fabric telemetry instead of bisecting forever.\n"
     "Rollback gate: drain and quarantine a suspect node rather than leaving it in the pool with a 'temporary' env "
     "workaround."),

    ("Container / namespace and shared-memory limits",
     "Hypothesis (falsifiable): the container lacks the resources NCCL needs — insufficient /dev/shm, missing "
     "IPC_LOCK capability, or a locked-memory (ulimit -l) cap — so intra-node transport setup or IB memory "
     "registration blocks.\n"
     "Mechanism: NCCL uses shared memory for intra-node transports and pinned/registered memory for IB; a small "
     "/dev/shm (Docker's 64 MB default) or a low ulimit -l makes registration fail or retry in a way that presents "
     "as a hang rather than a clean error.\n"
     "Controlled experiment: inside the container check `df -h /dev/shm`, `ulimit -l`, and capability set. Re-run "
     "with --shm-size=1g (ESTIMATE: >=1 GB is a safe floor for 8 ranks/node, derived from per-peer buffer counts "
     "times NCCL_BUFFSIZE) and --ulimit memlock=-1 --cap-add=IPC_LOCK.\n"
     "Falsification: adequate shm and unlimited memlock with the hang unchanged -> reject.\n"
     "Boundary: raising memlock to unlimited has host memory-pressure implications; scope it to the job's cgroup.\n"
     "Note: also verify the network namespace — host networking vs bridge changes which IPs ranks advertise."),

    ("Serving-side variant: inference engine startup (vLLM/Dynamo/Mooncake) hangs on TP init",
     "Hypothesis (falsifiable): for an inference deployment, the hang is in the tensor-parallel worker group "
     "formation, not in a training launcher — i.e. workers came up but one worker's device/port assignment is wrong.\n"
     "Mechanism: TP engines create an NCCL communicator across TP ranks at startup; a worker that failed to allocate "
     "KV cache, or a disaggregated prefill/decode deployment whose transfer plane (e.g. Mooncake-style KV transfer "
     "over RDMA) is not reachable, leaves the group incomplete and the server never reports ready.\n"
     "Controlled experiment: (1) start with tensor_parallel_size=1 — if it serves, the model/weights path is fine and "
     "the fault is in group formation; (2) increase TP stepwise and record the first size that hangs; (3) for "
     "disaggregated setups, disable the KV-transfer plane and run prefill+decode colocated to isolate the "
     "transfer-plane dependency.\n"
     "Falsification: TP=1 also hangs -> the problem is model loading or GPU health, not collectives.\n"
     "Boundary: readiness must be judged by a real health endpoint plus a token-producing request, never by log lines "
     "alone; a 'startup complete' message is not evidence the group is healthy.\n"
     "Rollback gate: keep the previous single-node non-disaggregated config as the fallback route and cut back to it "
     "if the TP group does not form within the deployment SLO (ESTIMATE: model-load-dominated, tens of seconds to a "
     "few minutes for a ~10B model on local NVMe)."),
]


def main():
    with open(CORPUS) as f:
        lines = f.readlines()[START:START + N]
    recs = [json.loads(l) for l in lines]
    assert len(recs) == N, len(recs)

    out = []
    seen = set()
    for i, d in enumerate(recs):
        msgs = d["messages"]
        u = [m for m in msgs if m["role"] == "user"][0]["content"]
        a = [m for m in msgs if m["role"] == "assistant"][0]["content"]
        title, body = MECHS[i]
        corrected = (f"{HEAD}Primary failure mode examined in this variant: {title}.\n\n{body}\n{TAIL}")
        h = hashlib.sha256(corrected.encode()).hexdigest()
        assert h not in seen, f"duplicate corrected_answer at {d['id']}"
        seen.add(h)
        out.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": u,
            "source_assistant": a,
            "corrected_answer": corrected,
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 2,
            },
            "risks": [
                "source_assistant is a grading rubric, not an answer: training on it teaches meta-commentary instead of diagnosis",
                "no falsifiable hypothesis or controlled experiment is actually stated despite the prompt requiring one",
                "no rollback gate or evidence list, so a model trained on it may propose NCCL env overrides without a revert plan",
                "risk of recommending NCCL_IB_DISABLE/NCCL_P2P_DISABLE as a fix rather than as a diagnostic, silently degrading bandwidth",
            ],
            "evidence_required": [
                "NCCL_DEBUG=INFO with NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV from rank 0 and one remote rank",
                "per-rank table of hostname/pid/RANK/LOCAL_RANK/WORLD_SIZE/GPU UUID captured before init_process_group",
                "py-spy dump of at least one hung rank to distinguish store rendezvous from ncclCommInitRank",
                "ibv_devinfo plus an ib_write_bw run between the two hosts when an RDMA transport is selected",
                "pass/fail matrix of a minimal all_reduce reproducer across single-node, per-node and node-pair configurations",
            ],
            "confidence": 0.62,
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out), "records:", out[0]["source_id"], "->", out[-1]["source_id"])


main()
