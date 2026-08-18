import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
OUT = os.path.join(RES, "train-batch-0146.jsonl")
items = json.load(open("/tmp/tb_next.json"))
assert len(items) == 10

BASE = """Assumptions: PyTorch DDP/FSDP or Megatron-style launch over NCCL, world_size > 1, hang observed at init_process_group or at the first collective, no Python traceback (processes alive, GPU util near 0 or pinned at 100 percent on one rank). Treat "hang during collective initialization" as one of four separable faults: (A) rendezvous never completes, (B) rendezvous completes but NCCL bootstrap/transport negotiation fails, (C) ranks disagree on the collective (mismatched shape/dtype/order), (D) transport works but is silently stalled (RDMA/GDR path down, falling into a blocking retry).

Instrumentation first, before any config change:
- Export NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET. A healthy run prints, per rank, "NCCL INFO Bootstrap : Using <iface>", the ring/tree graph, and "comm 0x... rank N nranks W ... - Init COMPLETE". The last line printed before the stall localizes the fault: no Bootstrap line = (A); Bootstrap but no channel/graph lines = (B); graph complete but no Init COMPLETE = (B/D); Init COMPLETE then stall = (C/D).
- Set TORCH_NCCL_ASYNC_ERROR_HANDLING=1 and TORCH_NCCL_BLOCKING_WAIT=0 with a short timeout (init_process_group(timeout=timedelta(minutes=2))) so the hang converts into a watchdog abort naming the collective and the ranks that did not arrive. This is the single highest-yield change: it turns an unobservable hang into a labelled failure.
- Capture py-spy dump --pid <pid> on every rank. Ranks parked in ncclCommInitRank vs in a user-level barrier vs in a data-loader distinguish (A/B) from (C).

Falsifiable hypothesis H1 (rendezvous): a strict subset of ranks reached init_process_group; the remainder died or never launched, so the store never reaches world_size. Prediction: the TCP/etcd store shows fewer than world_size keys, and py-spy shows the arrived ranks blocked in the store wait, not in NCCL. Controlled experiment: run a NCCL-free rendezvous probe (torch.distributed.init_process_group(backend="gloo") with the same launcher, env, and world size). If gloo also hangs, the fault is orchestration/rendezvous and NCCL is exonerated; if gloo completes, H1 is falsified and the fault is inside NCCL bootstrap or transport.

Falsifiable hypothesis H2 (interface selection): NCCL picked an unroutable interface (docker0, a management VLAN, or a link-local address) for bootstrap or for the data path. Prediction: NCCL_DEBUG=INFO shows Bootstrap or NET using an interface that cannot reach the peer subnet, and the stall is at bootstrap. Controlled experiment: pin NCCL_SOCKET_IFNAME to the known-good interface (and GLOO_SOCKET_IFNAME for the store) and rerun. Init completing proves H2; unchanged behaviour falsifies it. Do not leave a permanent wildcard exclusion in place as the "fix" without recording why the wrong interface was selected.

Falsifiable hypothesis H3 (device visibility / rank-to-GPU mapping): two ranks on a node bound to the same device, or CUDA_VISIBLE_DEVICES was set both by the scheduler and by user code, so a rank owns no GPU. Prediction: nvidia-smi shows a device with two contexts or an idle device; NCCL logs show duplicate "Using device" for one rank. Controlled experiment: assert torch.cuda.set_device(local_rank) early and print (global_rank, local_rank, device_uuid) from every rank; any duplicate UUID confirms H3.

Falsifiable hypothesis H4 (transport/fabric): IB/RoCE path is administratively up but not usable (no GID for the chosen RoCE version, PFC/ECN misconfigured, GDR enabled without nvidia_peermem loaded, or MTU mismatch), so NCCL blocks in connection setup or in the first data exchange. Prediction: NCCL logs show NET/IB selection followed by silence, or a fallback to NET/Socket with a large latency jump. Controlled experiment: run the layered escalation below; each layer isolates one variable.

Layered escalation (each step is cheap, ordered by information gain per minute):
1. Single process, single GPU: does the model even initialize? Removes NCCL entirely.
2. Single node, 2 GPUs, NCCL: if this hangs, the fault is local (H3, driver, or IPC/shm). Check /dev/shm size (containers defaulting to 64 MB break NCCL shm transport) and that peer-to-peer is available (nvidia-smi topo -m; p2pBandwidthLatencyTest).
3. Single node, full local world: exposes NVLink/PCIe topology and PXN issues.
4. Two nodes, 1 GPU each, nccl-tests all_reduce_perf -b 8 -e 8M -f 2 -g 1: the minimal cross-node collective. If this hangs, the fault is fabric/bootstrap, not the training script. If it passes at expected bandwidth, the training code (H1 rendezvous ordering or H2 collective mismatch) is implicated.
5. Two nodes with NCCL_IB_DISABLE=1 (force TCP): if the hang clears, the RDMA path is the fault (H4). If it persists, the fault is above the transport.
6. Full world size: only after 1-5 pass.

Discriminating (C) collective mismatch: ranks disagreeing on shape, dtype, device, or on the ORDER of collectives will hang deterministically at the same iteration, not at init. Detect with TORCH_DISTRIBUTED_DEBUG=DETAIL, which validates shapes across ranks and names the offending parameter. Common real causes: conditional branches that skip an all_reduce on some ranks, unused parameters in DDP without find_unused_parameters, logging/eval on rank 0 only that contains a collective, and uneven dataloader lengths on the last batch.

Expected confounders that produce false conclusions:
- A short NCCL timeout converts a slow-but-healthy init (large world size, cold IB, topology discovery over many nodes) into an abort. Establish a baseline init time at the target scale before treating slow as hung.
- Cluster schedulers may kill a rank for OOM or OOD; the surviving ranks then hang. The root cause is in the dead rank's log, not in NCCL.
- Container/host mismatch: NCCL, driver, and CUDA versions differing across nodes in a heterogeneous pool produces protocol-level stalls. Pin and verify: print nccl version, torch.version.cuda, and driver version from every rank and assert equality.
- Setting many NCCL_* env vars at once. Change one variable per run; otherwise the experiment is uninterpretable.
- Firewall/security groups blocking the NCCL port range while allowing the store port: gloo rendezvous succeeds and NCCL still hangs, which looks like a NCCL bug.

Evidence to collect before declaring root cause: full NCCL_DEBUG=INFO logs from ALL ranks (not just rank 0 - the failing rank is usually not rank 0), py-spy stacks per rank, nvidia-smi topo -m, ibstat/ibv_devinfo and show_gids on each node, the nccl-tests result at the smallest failing scale, and the watchdog abort message naming the unarrived ranks.

Rollback criteria and safety: all diagnostic env changes (NCCL_DEBUG, NCCL_SOCKET_IFNAME, NCCL_IB_DISABLE, shortened timeout) are per-job and must be reverted after diagnosis; NCCL_IB_DISABLE=1 in particular is a diagnostic, not a fix, and will silently cost large throughput if left in a production launcher. Any fabric-level change (PFC, MTU, GID/RoCE version) is a cluster-wide blast radius: stage on a two-node canary, require the canary to hit at least the pre-change all_reduce busbw within 5 percent, and revert if any other job regresses. Declare the fix validated only when the minimal reproducer passes three consecutive cold starts at full world size and busbw is within the expected band for the topology; otherwise treat it as a masked, not fixed, fault."""

VARIANTS = {
 5: ("Variant focus - rendezvous and launcher correctness.\nPrioritize H1: verify MASTER_ADDR/MASTER_PORT resolve identically from every node, that the port is free and reachable (nc -z), and that RANK/WORLD_SIZE/LOCAL_RANK are consistent. A single rank with a stale WORLD_SIZE makes the store wait forever. Replace a hand-rolled launcher with torchrun --rdzv_backend=c10d for one run: if the hang clears, the launcher's env propagation was the fault.", ["NCCL startup", "rendezvous", "torchrun"]),
 6: ("Variant focus - time-to-detect and timeout budgeting.\nMeasure init latency as a function of world size (2, 4, 8, 16 ranks) and fit it; a superlinear curve indicates bootstrap fan-in over a slow socket path rather than a true hang. Set the watchdog timeout to 3x the measured p99 init time at target scale, not to an arbitrary 30 minutes, so real hangs surface in minutes.", ["NCCL startup", "timeout tuning"]),
 7: ("Variant focus - topology and device mapping.\nRun nvidia-smi topo -m on every node and compare; asymmetric NVLink/PCIe topologies across nodes change the ring construction and can expose a broken link only at certain world sizes. Assert one process per GPU with distinct device UUIDs before the first collective.", ["NCCL startup", "topology", "device mapping"]),
 8: ("Variant focus - container and shared-memory constraints.\nContainers commonly ship /dev/shm at 64 MB and drop IPC_LOCK, which breaks NCCL shm transport and pinned-memory registration. Check df -h /dev/shm and ulimit -l on every node; rerun with --shm-size=1g and unlimited locked memory as a single-variable experiment.", ["NCCL startup", "containers", "shared memory"]),
 9: ("Variant focus - network interface and multi-homed hosts.\nOn multi-homed nodes NCCL may bootstrap on one NIC and move data on another. Enumerate interfaces, pin NCCL_SOCKET_IFNAME and NCCL_IB_HCA explicitly, and confirm from the logs that the selected NIC is the one on the training fabric rather than the management network.", ["NCCL startup", "interface selection"]),
 10: ("Variant focus - RDMA/RoCE and GPUDirect RDMA.\nIf the fabric is RoCEv2, verify the GID index matches the RoCE version in use, that PFC/ECN are configured consistently end to end, and that nvidia_peermem (or the dmabuf path) is loaded before enabling GDR. NCCL_IB_DISABLE=1 clearing the hang isolates the RDMA path; NCCL_NET_GDR_LEVEL then separates GDR from plain IB verbs.", ["NCCL startup", "RoCE", "GDR"]),
 11: ("Variant focus - version and image homogeneity.\nHeterogeneous NCCL/CUDA/driver versions across a node pool produce stalls that look like fabric faults. Print and assert equality of driver version, CUDA runtime, torch version, and NCCL version from every rank at startup; fail fast on mismatch rather than hanging.", ["NCCL startup", "version skew"]),
 12: ("Variant focus - collective ordering and rank divergence.\nEnable TORCH_DISTRIBUTED_DEBUG=DETAIL and audit every conditional path that contains a collective: rank-0-only logging, early exit on NaN, uneven dataloader length, and DDP unused parameters. A deterministic hang at the same step, rather than at init, is strong evidence for ordering divergence rather than fabric failure.", ["NCCL startup", "collective ordering"]),
 13: ("Variant focus - partial rank failure and scheduler interaction.\nWhen one rank is killed (OOM, preemption, node drain) the survivors hang in the collective. Correlate the hang timestamp with scheduler events and dmesg OOM kills on every node; the true root cause lives in the dead rank's log. Add a liveness sidecar that aborts the whole job when any rank exits non-zero.", ["NCCL startup", "fault isolation"]),
 14: ("Variant focus - minimal reproducer and regression gating.\nReduce to the smallest hanging configuration (fewest nodes and GPUs) and keep it as a cluster smoke test: nccl-tests all_reduce_perf at 2 nodes must pass and hit the expected busbw before any large job is scheduled. This converts a recurring hang into a pre-flight gate with an explicit pass threshold.", ["NCCL startup", "regression gating"]),
}

RISKS = [
 "Diagnostic env vars such as NCCL_IB_DISABLE=1 mask the fault and silently destroy interconnect bandwidth if left in a production launcher",
 "Shortening the NCCL/watchdog timeout can abort healthy but slow large-scale initialization and be misread as a hang",
 "Fabric-level changes (PFC, MTU, RoCE GID/version) have cluster-wide blast radius and can regress unrelated jobs",
 "Changing multiple NCCL_* variables in one run makes the experiment uninterpretable and can hide the real cause",
 "Reading only rank-0 logs hides the actually failing rank and leads to a wrong root cause",
]

EVIDENCE = [
 "NCCL_DEBUG=INFO with NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET logs from all ranks, not only rank 0",
 "py-spy dump stacks from every rank showing where each process is parked",
 "Watchdog abort message naming the collective and the ranks that did not arrive",
 "gloo-backend rendezvous control run to separate orchestration failure from NCCL failure",
 "nccl-tests all_reduce_perf result at the smallest failing scale, with busbw compared against topology expectation",
 "nvidia-smi topo -m, ibstat/ibv_devinfo, show_gids, /dev/shm size and ulimit -l from every node",
 "Driver, CUDA, torch and NCCL versions asserted equal across all ranks",
 "Three consecutive clean cold starts at full world size after the fix",
]

recs = []
for it in items:
    v = int(it["user"].split("Scenario variant ")[1].split(":")[0])
    focus, _c = VARIANTS[v]
    ans = BASE + "\n\n" + focus
    recs.append({
        "source_id": it["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": it["user"],
        "source_assistant": it["assistant"],
        "corrected_answer": ans,
        "quality_dimensions": {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3},
        "risks": RISKS,
        "evidence_required": EVIDENCE,
        "confidence": 0.79,
    })

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(recs))
