import json, hashlib, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0153.jsonl")
START, N = 1520, 10  # 0-indexed offset; lines 1521..1530

rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if START <= i < START + N:
            rows.append(json.loads(line))
assert len(rows) == N

COMMON_TAIL = (
    "\n\nRollback gate: change exactly ONE variable per trial and keep an explicit bounded init timeout "
    "(init_process_group(timeout=timedelta(minutes=10))) plus NCCL_ASYNC_ERROR_HANDLING=1, so a failed trial "
    "aborts with a stack trace instead of occupying the queue. If a trial does not move the phase boundary "
    "within two runs, revert it. NCCL_P2P_DISABLE / NCCL_IB_DISABLE / NCCL_SHM_DISABLE are bisection "
    "instruments, not fixes: leaving them set silently demotes NVLink->PCIe or RDMA->TCP on every later job."
)

BODIES = [
    # 1679 - rendezvous store
    ("Mechanism: an init-time hang is almost always a rendezvous problem, not a collective problem. "
     "Before any ring is built, every rank must register with the TCPStore/etcd endpoint at MASTER_ADDR:MASTER_PORT; "
     "init_process_group blocks until store.wait() sees WORLD_SIZE keys.\n\n"
     "Falsifiable hypothesis H1: the hang is a rendezvous shortfall — fewer than WORLD_SIZE ranks ever reach the "
     "store — not a transport failure. Prediction: the number of established TCP sessions on MASTER_PORT is strictly "
     "less than WORLD_SIZE, and no rank ever emits an 'NCCL INFO Bootstrap' line.\n\n"
     "Controlled experiment: keep the launcher, topology and node set fixed; replace only the backend with 'gloo' "
     "and run a 1-element all_reduce. Gloo uses the same store but no CUDA/NIC path. If gloo also hangs, the fault is "
     "in rendezvous/launcher/env; if gloo succeeds, rendezvous is healthy and the fault is downstream in NCCL "
     "bootstrap or transport.\n\n"
     "Boundary conditions: this test is only valid when MASTER_ADDR resolves identically on every node and the port "
     "is not reused by a zombie job; verify both first or the control is confounded.\n\n"
     "Evidence to collect: `ss -tanp | grep <MASTER_PORT>` on all nodes (count sessions), the actual RANK/WORLD_SIZE/"
     "LOCAL_RANK read from /proc/<pid>/environ (never trust the job script), `getent hosts $MASTER_ADDR` on every node, "
     "and the scheduler's node allocation vs. the launcher's --nnodes.\n\n"
     "Confounders: a rank that crashed at import time (bad CUDA driver, OOM in dataset init) looks identical to a "
     "network partition from rank 0's side — check every rank's exit status before blaming the fabric."),

    # 1680 - stuck-rank identification via stacks
    ("Mechanism: PyTorch's init barrier is symmetric — all surviving ranks block, so the visible symptom is uniform "
     "even though the causal rank is usually a single one. Diagnosis must therefore start by locating the odd rank "
     "out, not by tuning NCCL.\n\n"
     "Falsifiable hypothesis H1: exactly one rank is in a different call frame from the rest (still in CUDA context "
     "creation, dataset build, or already dead), and the others are merely waiting on it. Prediction: py-spy stacks "
     "partition into a large uniform class and one singleton.\n\n"
     "Controlled experiment: `py-spy dump --pid <pid>` (or gdb -p, thread apply all bt) for every local rank on every "
     "node at the same wall clock, plus `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`. Then re-run "
     "with WORLD_SIZE reduced to exclude the singleton's node. If the reduced-world run initialises, the singleton "
     "rank/node is causal; if it still hangs, the fault is global (env, store, image mismatch).\n\n"
     "Boundary conditions: stacks must be taken while the job is hung, not after the timeout fires; after abort the "
     "evidence is gone. py-spy needs matching CAP_SYS_PTRACE inside the container.\n\n"
     "Evidence to collect: per-rank stack dumps, per-rank exit codes from the launcher, dmesg on each node (Xid errors, "
     "OOM killer), container image digest per node.\n\n"
     "Confounders: an unhealthy GPU that fails cudaSetDevice will present as a hang rather than an error if the "
     "launcher swallows stderr — capture per-rank logs to separate files."),

    # 1681 - interface selection
    ("Mechanism: NCCL bootstrap picks a socket interface by scanning /sys/class/net and applying NCCL_SOCKET_IFNAME "
     "filters. On hosts with docker0, cni0, or multiple management NICs, different nodes can select different, "
     "mutually unroutable subnets — the ring then never completes and the job hangs in bootstrap.\n\n"
     "Falsifiable hypothesis H1: ranks selected non-matching interfaces. Prediction: NCCL_DEBUG=INFO "
     "NCCL_DEBUG_SUBSYS=INIT,NET logs show different 'NET/Socket : Using [0]<ifname>' values across nodes, and the "
     "chosen address is in an RFC1918 bridge range (172.17.x, 10.42.x) rather than the cluster data subnet.\n\n"
     "Controlled experiment: pin NCCL_SOCKET_IFNAME to the known data-plane interface (and NCCL_IB_HCA if RDMA is "
     "used) on all nodes, change nothing else, re-run. Success under pinning with failure without it confirms H1; "
     "failure under pinning falsifies it and moves suspicion to the store or to link state.\n\n"
     "Boundary conditions: the pin must name an interface that exists on every node; a typo produces a clean, fast "
     "'no usable interface' error rather than a hang, which is itself a useful discriminator.\n\n"
     "Evidence to collect: `ip -br addr` from every node, NCCL INIT/NET logs from rank 0 and one remote rank, and a "
     "raw reachability check (nc -z or ping) between the selected addresses.\n\n"
     "Confounders: a correct IFNAME with a down link or a MTU mismatch (9000 vs 1500) fails later, during the first "
     "large transfer, not at init — do not conflate the two phases."),

    # 1682 - minimal reproducer / world-size bisection
    ("Mechanism: initialisation cost and failure surface scale with world size and with the number of distinct "
     "transports in play. A minimal reproducer collapses the search space before any variable is changed.\n\n"
     "Falsifiable hypothesis H1: the hang is scale- or topology-dependent — it does not occur within a single node. "
     "Prediction: a 2-rank single-node all_reduce succeeds; the same script at 2 ranks across 2 nodes hangs.\n\n"
     "Controlled experiment: a ~20-line script that only does init_process_group + all_reduce(torch.ones(1).cuda()), "
     "run in this order: (a) 2 ranks, 1 node; (b) 8 ranks, 1 node; (c) 2 ranks, 2 nodes; (d) full world. The first "
     "configuration that hangs names the failing dimension — intra-node P2P/SHM, or inter-node transport, or scale.\n\n"
     "Boundary conditions: the reproducer must use the same container image, launcher and env as the real job; if it "
     "does not reproduce, the difference between reproducer and job (dataset init, custom process groups, a second "
     "init_process_group call) is itself the lead.\n\n"
     "Evidence to collect: pass/fail matrix over (a)-(d) with timestamps, NCCL INFO logs per configuration, and "
     "`nvidia-smi topo -m` per node type.\n\n"
     "Confounders: heterogeneous nodes in the allocation — one node with a different driver or a missing NIC — make "
     "(c) pass or fail depending on which pair the scheduler picked; repeat (c) on several pairs before concluding."),

    # 1683 - P2P / IOMMU / ACS intra-node
    ("Mechanism: intra-node NCCL prefers P2P over NVLink or PCIe. When IOMMU is enabled without passthrough, or PCIe "
     "ACS redirection is on, peer-to-peer DMA between GPUs under different root ports is silently broken; NCCL's P2P "
     "capability probe can then block rather than fall back cleanly.\n\n"
     "Falsifiable hypothesis H1: the hang is in intra-node P2P setup, not in the network. Prediction: the job hangs "
     "even at single-node/8-rank scale, and NCCL_P2P_DISABLE=1 makes that same run complete.\n\n"
     "Controlled experiment: run the single-node 8-rank minimal all_reduce three times — baseline, with "
     "NCCL_P2P_DISABLE=1, with NCCL_SHM_DISABLE=1 — changing nothing else. A pass only under P2P_DISABLE isolates the "
     "P2P path; a pass only under SHM_DISABLE points at /dev/shm sizing (a container run with the 64MB default) "
     "instead.\n\n"
     "Boundary conditions: this only applies when the hang reproduces on one node. If single-node passes, P2P is "
     "exonerated and the variables must not be carried forward into the multi-node trials.\n\n"
     "Evidence to collect: `nvidia-smi topo -m`, `lspci -vvv | grep -i acs` on the host, IOMMU state from "
     "/proc/cmdline, `df -h /dev/shm`, and cuda-samples p2pBandwidthLatencyTest output.\n\n"
     "Confounders: P2P_DISABLE 'fixing' the hang is a diagnosis, not a remedy — it costs real intra-node bandwidth; "
     "the durable fix is host-level (iommu=pt / ACS disabled / larger --shm-size), applied and then re-verified with "
     "P2P re-enabled."),

    # 1684 - RDMA/GID/RoCE
    ("Mechanism: on RoCE fabrics NCCL must select an RDMA HCA, a port, and a GID index. A wrong GID index (v1 vs v2, "
     "or an IPv4-mapped entry that does not match the configured DSCP/PFC domain) leaves the QP unable to complete "
     "its handshake, and the connection attempt hangs rather than erroring.\n\n"
     "Falsifiable hypothesis H1: the RDMA transport, not the rendezvous, is the blocked phase. Prediction: bootstrap "
     "lines appear in NCCL INFO logs, then output stops at 'NET/IB : Using [0]mlx5_x:1/RoCE'; and setting "
     "NCCL_IB_DISABLE=1 (TCP fallback) lets the identical job initialise, at lower bandwidth.\n\n"
     "Controlled experiment: (1) run `ib_write_bw` between the two nodes using the same HCA and GID index NCCL "
     "reported — a point-to-point RDMA test independent of NCCL; (2) rerun the job with NCCL_IB_DISABLE=1. If "
     "ib_write_bw also hangs/fails, the fault is fabric-level and NCCL is a victim; if ib_write_bw passes but NCCL "
     "hangs, the fault is NCCL's HCA/GID selection and NCCL_IB_HCA/NCCL_IB_GID_INDEX pinning is the next trial.\n\n"
     "Boundary conditions: valid only on RoCE/IB hosts with a live ibstat port (state ACTIVE, correct link layer); a "
     "port in INIT or DOWN state makes every downstream inference meaningless.\n\n"
     "Evidence to collect: `ibstat`, `show_gids`, PFC/ECN counters from the switch, `ethtool -S` pause and discard "
     "counters, NCCL NET logs from both ends, ib_write_bw results.\n\n"
     "Confounders: an asymmetric fabric config — PFC enabled on one leaf and not another — produces an intermittent "
     "hang that depends on which nodes the scheduler allocated, so record the node list with every trial."),

    # 1685 - timeout/version/image skew
    ("Mechanism: many 'hangs' are really 'not finished yet': CUDA context creation, JIT/PTX recompilation for an "
     "unmatched arch, and lazy module loading can each add minutes at first collective, and NCCL versions differ in "
     "handshake protocol details across ranks.\n\n"
     "Falsifiable hypothesis H1: the job is progressing slowly or the ranks run mismatched NCCL/CUDA builds, rather "
     "than being deadlocked. Prediction: with a 30-minute timeout the job eventually initialises (slow-start case), "
     "OR the NCCL INFO banner reports different NCCL versions or different CUDA driver versions across ranks "
     "(skew case).\n\n"
     "Controlled experiment: re-run once with an enlarged init timeout and NCCL_DEBUG=INFO, and in parallel collect "
     "`python -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'` plus the "
     "container image digest on every node. Slow-but-successful init falsifies the deadlock hypothesis entirely and "
     "redirects effort to startup cost (CUDA_MODULE_LOADING, TORCH_CUDA_ARCH_LIST, warm image cache).\n\n"
     "Boundary conditions: raising the timeout is a diagnostic, not a production setting — a permanently large "
     "timeout converts fast failures into multi-hour queue occupancy.\n\n"
     "Evidence to collect: timestamped NCCL banners per rank, image digests, driver versions from `nvidia-smi`, and "
     "the wall-clock delta between process start and first NCCL log line.\n\n"
     "Confounders: a shared filesystem under load makes image/module loading slow on some nodes only, producing an "
     "apparent single-rank straggler that is not a hardware fault."),

    # 1686 - mismatched collectives / process group ordering
    ("Mechanism: NCCL collectives are matched positionally. If ranks issue a different sequence of collectives — a "
     "conditional all_reduce, a barrier only on rank 0, or subgroups created in a different order — the ranks block "
     "on mismatched operations and the job hangs with no error. Multiple process groups must be constructed in the "
     "same order on every rank.\n\n"
     "Falsifiable hypothesis H1: the hang is a program-logic desync, not an infrastructure failure. Prediction: the "
     "minimal single-collective reproducer passes at the same scale and on the same nodes where the real job hangs, "
     "and py-spy stacks show ranks in *different* collective call sites.\n\n"
     "Controlled experiment: run the minimal reproducer at full world size on the identical allocation. If it passes, "
     "infrastructure is exonerated; then enable TORCH_DISTRIBUTED_DEBUG=DETAIL and NCCL_ASYNC_ERROR_HANDLING=1 on the "
     "real job, which turns collective-shape and ordering mismatches into explicit assertions instead of a hang.\n\n"
     "Boundary conditions: TORCH_DISTRIBUTED_DEBUG=DETAIL adds per-collective validation overhead and is not safe for "
     "throughput runs; use it only for the diagnostic run.\n\n"
     "Evidence to collect: per-rank stack traces annotated with the collective call site, the code path guarded by "
     "any rank-conditional branch, the order of new_group() calls, and DETAIL-mode assertion output.\n\n"
     "Confounders: data-dependent branches (a rank whose shard is empty skips a step) reproduce only on certain "
     "datasets or seeds — record the input shard map with each trial."),

    # 1687 - phase boundary / instrumentation
    ("Mechanism: 'init hang' is not one phase. The sequence is: process start -> CUDA context -> store rendezvous -> "
     "NCCL bootstrap (unique-id broadcast) -> topology detection -> transport setup (SHM/P2P/NET) -> first collective. "
     "Every remedy targets exactly one of these, so the first task is to localise the boundary, not to try fixes.\n\n"
     "Falsifiable hypothesis H1: the last completed phase is identifiable from logs and is the same on every run. "
     "Prediction: across three repeated runs, the final NCCL_DEBUG=INFO log line is from the same subsystem "
     "(INIT vs GRAPH vs NET), i.e. the failure is deterministic in phase.\n\n"
     "Controlled experiment: run three times unchanged with NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET, "
     "per-rank log files, and timestamps. A stable phase boundary licenses targeted bisection; a wandering boundary "
     "falsifies H1 and indicates a race or a load-dependent timeout, which needs a different (statistical) approach.\n\n"
     "Boundary conditions: log capture must be per-rank to separate files — merged stdout interleaves and destroys "
     "ordering evidence at scale.\n\n"
     "Evidence to collect: three sets of per-rank NCCL logs, wall-clock timestamps per phase, and a table of "
     "last-line-subsystem by run.\n\n"
     "Confounders: log buffering can truncate the true last line at kill time; use unbuffered output (PYTHONUNBUFFERED=1) "
     "and prefer SIGQUIT/py-spy over SIGKILL when ending a hung trial."),

    # 1688 - end-to-end plan + safety/queue governance
    ("Mechanism: at cluster scale the expensive resource is not the fix but the debugging loop itself — each full-world "
     "trial burns N GPUs for its whole duration. The plan must therefore be ordered by information-per-GPU-hour and "
     "bounded by explicit stopping rules.\n\n"
     "Falsifiable hypothesis H1: the fault is reproducible in a cheap configuration (<= 2 nodes) and does not require "
     "full-world runs to diagnose. Prediction: the 2-node minimal all_reduce reproduces the hang.\n\n"
     "Controlled experiment, in cost order: (1) inspect env/logs from the already-hung job — zero extra GPUs; "
     "(2) 2-rank single-node reproducer; (3) 2-node reproducer; (4) only if all pass, one instrumented full-world run. "
     "Each step has a bounded timeout and one changed variable. If (3) reproduces, H1 holds and full-world runs are "
     "forbidden for the rest of the investigation.\n\n"
     "Boundary conditions: the small configurations must use the same image, launcher and fabric path as production; "
     "a reproducer that quietly falls back to TCP proves nothing about an RDMA hang.\n\n"
     "Evidence to collect: for each step — config, changed variable, wall-clock to failure, per-rank NCCL logs, "
     "pass/fail — recorded in one table so the bisection is auditable and reversible.\n\n"
     "Confounders: concurrent cluster events (a firmware rollout, a switch reconfiguration, another job saturating the "
     "same leaf) can flip results between trials; timestamp every trial and cross-check the cluster change log before "
     "attributing a pass to your own variable."),
]

RISKS = [
    ["Blaming the fabric while a single rank crashed at import time misroutes the investigation",
     "MASTER_PORT reuse by a zombie job produces a hang that survives every NCCL change"],
    ["Post-abort stack collection loses the only decisive evidence",
     "Swallowed per-rank stderr hides a hard CUDA error behind an apparent hang"],
    ["Pinning NCCL_SOCKET_IFNAME to a name that is absent on some nodes converts a hang into a partial failure",
     "Selecting a docker bridge interface causes silent cross-node unreachability"],
    ["A reproducer that diverges from the real job (different image or launcher) yields a false negative",
     "Heterogeneous node allocation makes 2-node results non-reproducible"],
    ["Leaving NCCL_P2P_DISABLE on as a 'fix' permanently demotes NVLink to PCIe bandwidth",
     "Container default 64MB /dev/shm silently breaks the SHM transport"],
    ["Leaving NCCL_IB_DISABLE on falls back to TCP and destroys inter-node throughput",
     "Wrong GID index on RoCE produces an indefinite QP handshake block rather than an error"],
    ["A permanently enlarged init timeout converts fast failures into hours of wasted queue occupancy",
     "Mixed container image digests across nodes yield protocol skew that looks like a network fault"],
    ["TORCH_DISTRIBUTED_DEBUG=DETAIL left on in production adds per-collective validation overhead",
     "Rank-conditional branches make the desync data-dependent and non-reproducible"],
    ["Merged stdout at scale destroys the per-rank ordering evidence needed to localise the phase",
     "SIGKILL on a hung trial truncates buffered logs and erases the last-phase marker"],
    ["Repeated full-world debug runs consume N-GPU-hours per bit of information",
     "Concurrent cluster maintenance can be mistaken for the effect of your changed variable"],
]

EVIDENCE = [
    ["ss -tanp on MASTER_PORT across all nodes", "RANK/WORLD_SIZE read from /proc/<pid>/environ",
     "getent hosts $MASTER_ADDR per node", "gloo-backend control run result"],
    ["py-spy dump per rank while hung", "per-rank launcher exit codes", "dmesg Xid/OOM lines per node",
     "container image digest per node"],
    ["ip -br addr from every node", "NCCL_DEBUG_SUBSYS=INIT,NET logs from rank 0 and a remote rank",
     "nc -z reachability between selected addresses"],
    ["pass/fail matrix over 1-node/2-node/full-world reproducer", "nvidia-smi topo -m per node type",
     "NCCL INFO logs per configuration"],
    ["nvidia-smi topo -m", "lspci ACS state and /proc/cmdline IOMMU flags", "df -h /dev/shm",
     "p2pBandwidthLatencyTest output"],
    ["ibstat and show_gids on both nodes", "ib_write_bw point-to-point result",
     "ethtool -S pause/discard counters", "switch PFC/ECN counters"],
    ["timestamped NCCL banner per rank", "torch/CUDA/NCCL versions per rank", "container image digests",
     "wall-clock delta from process start to first NCCL log"],
    ["per-rank stacks annotated with collective call site", "order of new_group() calls per rank",
     "TORCH_DISTRIBUTED_DEBUG=DETAIL assertion output", "input shard map per rank"],
    ["three repeated runs of per-rank NCCL logs", "last-log-subsystem table by run",
     "per-phase wall-clock timestamps"],
    ["ordered trial table: config, changed variable, time-to-failure, result",
     "per-rank NCCL logs per trial", "cluster change log for the trial window"],
]

QD = [
    (3, 2, 3), (3, 2, 3), (3, 2, 4), (3, 2, 3), (3, 2, 4),
    (3, 2, 4), (3, 2, 3), (3, 2, 3), (3, 2, 3), (3, 2, 4),
]
CONF = [0.82, 0.80, 0.83, 0.81, 0.84, 0.83, 0.80, 0.82, 0.81, 0.84]

out = []
seen = set()
for k, d in enumerate(rows):
    msgs = d["messages"]
    u = [m for m in msgs if m["role"] == "user"][0]["content"]
    a = [m for m in msgs if m["role"] == "assistant"][0]["content"]
    ca = BODIES[k] + COMMON_TAIL
    h = hashlib.sha256(ca.encode()).hexdigest()
    assert h not in seen, "duplicate corrected_answer"
    seen.add(h)
    tc, ic, os_ = QD[k]
    out.append({
        "source_id": d["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": ca,
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": RISKS[k],
        "evidence_required": EVIDENCE[k],
        "confidence": CONF[k],
    })

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(out), "unique_answers", len(seen))
