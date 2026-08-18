import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0152.jsonl")
START, END = 1511, 1520  # 1-indexed inclusive

COMMON_TAIL = (
    "\n\nEvidence to collect before any fix is declared: py-spy dump (or gdb bt) of one hung rank per node, "
    "NCCL_DEBUG=INFO + NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH logs from rank 0 and one non-zero rank, `ss -tanp | grep <MASTER_PORT>` "
    "on every node, `nvidia-smi topo -m`, `ibstat`/`ethtool -S` counters if RDMA is in play, and the exact "
    "WORLD_SIZE/RANK/LOCAL_RANK/CUDA_VISIBLE_DEVICES the launcher actually exported (read /proc/<pid>/environ, do not trust the job script).\n\n"
    "Rollback gate: apply exactly one variable per trial, keep NCCL_ASYNC_ERROR_HANDLING=1 and a bounded "
    "init timeout (e.g. timeout=timedelta(minutes=10)) so a failed trial aborts with a stack instead of hanging the queue. "
    "If a change does not move the phase boundary within two runs, revert it — accumulating NCCL env vars is how a "
    "reproducible hang turns into an unreproducible one. Do not permanently set NCCL_P2P_DISABLE / NCCL_IB_DISABLE / "
    "NCCL_SHM_DISABLE as a 'fix': they are diagnostic bisection tools, and leaving them on silently costs bandwidth "
    "(NVLink -> PCIe, or RDMA -> TCP) on every subsequent job."
)

HEADS = [
    # 1668 Troubleshooting
    ("Treat a collective-init hang as a three-phase problem and localize the phase before changing any knob.\n\n"
     "Mechanism. init_process_group runs: (1) store rendezvous — all ranks connect to the TCPStore at "
     "MASTER_ADDR:MASTER_PORT (or c10d/etcd); (2) unique-id exchange — rank 0 creates ncclUniqueId and broadcasts it "
     "through that store; (3) bootstrap + transport setup — every rank opens bootstrap sockets to every other rank and "
     "then selects P2P/NVLink, SHM, or NET (IB/RoCE, else sockets). All three fail as an identical silent stall.\n\n"
     "Falsifiable hypothesis H1: the stall is in phase 1/2 because at least one rank never reached the store. "
     "Predicted observation: on the master node `ss -tan state established '( sport = :MASTER_PORT )' | wc -l` is "
     "strictly less than WORLD_SIZE, and the missing ranks' stacks sit in TCPStore::wait, not in ncclCommInitRank.\n\n"
     "Controlled experiment: run a 20-line script that only does init_process_group(backend='gloo') with the same "
     "launcher, env, and world size. Gloo uses the same store but no NCCL transport. If gloo also hangs, the fault is "
     "rendezvous/launcher (wrong MASTER_ADDR, duplicated RANK, firewall on MASTER_PORT, a rank that died before init and "
     "left the group short). If gloo completes and NCCL hangs, phase 1/2 is exonerated and the fault is in phase 3.\n\n"
     "Confounders to control: a rank that OOMed or crashed at import time looks exactly like a network partition; "
     "container DNS resolving MASTER_ADDR to a per-namespace loopback; a stale process from the previous job still "
     "holding MASTER_PORT; SLURM/torchrun restarting one node so RANK is duplicated."),
    # 1669 Performance Analysis
    ("Frame it as latency-to-first-collective and measure where the wall-clock is spent; a 'hang' is often a timeout "
     "you have not waited out.\n\n"
     "Mechanism. Phase-3 transport negotiation cost scales with world size and with how many candidate paths NCCL must "
     "probe. Bootstrap is O(N) ring plus all-gather of peer info; a single unresponsive interface makes each connect "
     "attempt block for the TCP SYN retry budget (~127 s with default tcp_syn_retries=6), so a 64-rank job can appear "
     "dead for many minutes while it is only serially timing out.\n\n"
     "Falsifiable hypothesis H1: this is not a deadlock but serialized connect timeouts on a wrong-subnet interface. "
     "Predicted observation: NCCL_DEBUG=INFO shows 'NET/Socket : Using [0]<some unexpected NIC>' and the elapsed time "
     "between successive INIT log lines is a multiple of ~2 s/4 s/8 s backoff steps, not uniform.\n\n"
     "Controlled experiment: hold everything fixed and pin the interface — NCCL_SOCKET_IFNAME=<known-good NIC> "
     "(and NCCL_IB_HCA=<hca> if RDMA). Re-run and record time-to-init at world sizes 2, 8, and full N. If pinning turns "
     "an unbounded stall into a bounded init time that grows sub-linearly in N, the timeout hypothesis is confirmed; if "
     "time-to-init still diverges at some N, the fault is topology/resource, not interface selection.\n\n"
     "Confounders: docker0/veth/tailscale interfaces advertised to NCCL; GID index mismatch on RoCE making "
     "connections succeed on one node pair and stall on another; NUMA-imbalanced ranks making measured init time noisy."),
    # 1671 Troubleshooting
    ("Bisect by world size and by transport; do not read the code path from the log alone.\n\n"
     "Mechanism. NCCL picks a transport per peer pair: NVLink/PCIe P2P intra-node, SHM as fallback intra-node, NET "
     "inter-node. Any single unusable pair blocks the whole communicator, so the failing pair — not the failing node — "
     "is the unit of diagnosis.\n\n"
     "Falsifiable hypothesis H1: exactly one node (or one GPU pair) is bad. Predicted observation: a 2-rank run on that "
     "node pair hangs while every other pair of the same size completes.\n\n"
     "Controlled experiment: (a) 1 node, 2 ranks -> exercises P2P/SHM only; (b) 1 node, all local GPUs; (c) 2 nodes, 1 "
     "rank each, sweeping node pairs; (d) full world. Run the same minimal all_reduce of a 4 MiB tensor in each case and "
     "record pass/fail plus time-to-first-result. The first failing configuration names the culprit pair. If (a) fails, "
     "confirm with NCCL_P2P_DISABLE=1: if that unblocks it, the fault is P2P/IOMMU/ACS or a wedged nvidia-peermem path, "
     "not the network.\n\n"
     "Confounders: CUDA_VISIBLE_DEVICES remapping so 'GPU 3' in the log is not GPU 3 in nvidia-smi; MIG or exclusive "
     "compute mode making a device unavailable to a second rank; a leftover process pinning the GPU (check "
     "`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before blaming the fabric)."),
    # 1672 Performance Analysis
    ("Separate 'never completes' from 'completes far slower than the topology allows' — they need different evidence.\n\n"
     "Mechanism. Even a successful init can be pathologically slow when NCCL falls back from NVLink to PCIe, from RDMA "
     "to TCP sockets, or when it builds rings across a mis-detected topology. The same fallbacks, when the fallback path "
     "is also broken, present as a hang.\n\n"
     "Falsifiable hypothesis H1: the chosen transport is a degraded fallback. Predicted observation: NCCL_DEBUG=INFO "
     "reports 'via NET/Socket' or 'via P2P/direct pointer' where the topology (nvidia-smi topo -m) says NVLink/IB should "
     "be available.\n\n"
     "Controlled experiment: after init succeeds under a bisected configuration, run nccl-tests all_reduce_perf "
     "(-b 8 -e 1G -f 2 -g <ngpu>) and compare busbw against the paper bound for the detected path (NVLink vs PCIe Gen4 "
     "x16 ~26 GB/s effective, 200 Gb/s IB ~22-24 GB/s busbw). A busbw an order of magnitude below the path's bound "
     "confirms fallback; matching busbw falsifies it and moves suspicion to the application's first collective (shape "
     "mismatch, one rank calling a different collective).\n\n"
     "Confounders: measuring on a shared node; ECC/row-remap events throttling a GPU; PCIe ASPM; a busy NIC serving "
     "checkpoint I/O concurrently."),
    # 1673 System Design
    ("Design the job so that this class of hang is observable by construction, then diagnose within that frame.\n\n"
     "Mechanism. A hang is only mysterious when init is unbounded and unlogged. Bounded timeouts plus a pre-flight "
     "collective convert a silent stall into a typed failure with a rank list.\n\n"
     "Falsifiable hypothesis H1: the current job has no bounded init, so the observed 'hang' is an unbounded wait on a "
     "condition already known to be false. Predicted observation: setting timeout=timedelta(minutes=5) and "
     "NCCL_ASYNC_ERROR_HANDLING=1 turns the hang into an exception naming the ranks that never joined.\n\n"
     "Controlled experiment: add a pre-flight stage to the launcher — every rank logs (hostname, RANK, LOCAL_RANK, "
     "CUDA_VISIBLE_DEVICES, nccl version) to a shared path, then performs a 4-byte all_reduce with a 60 s timeout, then "
     "the real job starts. Re-run the failing job. If the pre-flight all_reduce fails, the problem is infrastructure and "
     "the rank list is the answer; if pre-flight passes and the job still hangs later, the fault is application-level "
     "collective ordering, and the design change has already falsified the infra hypothesis.\n\n"
     "Confounders: shared log path on a slow NFS mount adding its own stall; a pre-flight that uses gloo and therefore "
     "does not exercise the NCCL transport you care about."),
    # 1674 Troubleshooting
    ("Start from the rank inventory: most 'collective init hangs' are an arithmetic mismatch, not a fabric fault.\n\n"
     "Mechanism. init_process_group blocks until exactly WORLD_SIZE distinct ranks have registered. If the launcher "
     "starts WORLD_SIZE-1 processes, or two processes claim the same RANK, or one node's container sees fewer GPUs than "
     "assumed, the group never closes and every healthy rank waits forever — with zero errors.\n\n"
     "Falsifiable hypothesis H1: the set of registered ranks is not {0..WORLD_SIZE-1}. Predicted observation: collecting "
     "each process's RANK from /proc/<pid>/environ across all nodes yields a set with a gap or a duplicate.\n\n"
     "Controlled experiment: before init, have every rank append '<hostname> <pid> <RANK> <LOCAL_RANK> <WORLD_SIZE> "
     "<visible GPU UUIDs>' to a shared file; run the failing job; sort and diff against the expected set. If a gap exists, "
     "inspect that node's launcher stderr for an import-time or CUDA-init crash (a rank that dies before init is "
     "indistinguishable from a network hang at the group level). If the set is complete and correct, this hypothesis is "
     "falsified and you move to transport bisection.\n\n"
     "Confounders: SLURM --ntasks vs --gpus-per-node disagreement; torchrun --nproc_per_node exceeding visible GPUs; "
     "an elastic agent restarting a worker mid-rendezvous."),
    # 1675 Performance Analysis
    ("Quantify the stall: a hang with no time axis cannot be analyzed.\n\n"
     "Mechanism. Each init sub-step has a characteristic duration. Store rendezvous is sub-second on a healthy LAN; "
     "unique-id broadcast is milliseconds; transport setup grows with N and with the number of probed paths. Timestamped "
     "logs turn 'it hangs' into 'it stops after step k'.\n\n"
     "Falsifiable hypothesis H1: the stall is concentrated in one sub-step and is deterministic across runs. Predicted "
     "observation: with NCCL_DEBUG=INFO and NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log, the last log line before the stall is "
     "identical across three consecutive runs, and its timestamp delta from the previous line exceeds 60 s.\n\n"
     "Controlled experiment: run the job three times capturing per-rank NCCL logs; compute for each rank the delta "
     "between consecutive INIT lines; build a histogram. A single dominant gap at the same line in all runs confirms a "
     "deterministic sub-step fault (interface probe, IB queue-pair creation). Varying stall points across runs falsifies "
     "determinism and points to a race — contention on MASTER_PORT, or a shared filesystem-based init_method with stale "
     "state.\n\n"
     "Confounders: log buffering hiding the true last line (use unbuffered/per-rank files); clock skew across nodes "
     "when comparing timestamps — normalize per rank, not across ranks."),
    # 1676 System Design
    ("Make the failure domain explicit in the topology design, then test the design's assumption.\n\n"
     "Mechanism. Multi-node init depends on a specific control plane (rendezvous over the management network) and a "
     "specific data plane (IB/RoCE or high-speed Ethernet). Systems that conflate the two — MASTER_ADDR on the fast "
     "fabric, or NCCL_SOCKET_IFNAME left to auto-detect on a node with many interfaces — hang whenever either plane "
     "degrades.\n\n"
     "Falsifiable hypothesis H1: NCCL is auto-selecting a non-routable interface for inter-node traffic. Predicted "
     "observation: `ip -o addr` shows multiple UP interfaces per node, and NCCL_DEBUG=INFO names one that is not the "
     "cluster's data-plane NIC.\n\n"
     "Controlled experiment: fix the design in one variable — set NCCL_SOCKET_IFNAME to the explicit data-plane prefix "
     "and keep MASTER_ADDR on the management interface — and re-run the 2-node minimal all_reduce. If init succeeds "
     "and busbw matches the fabric bound, the auto-detection hypothesis holds; if it still hangs, the control plane is "
     "exonerated only after you separately verify MASTER_PORT reachability with nc/ss from every node.\n\n"
     "Confounders: RoCE requiring a matching GID index and PFC/ECN configuration on both endpoints and the switch — a "
     "correct NIC choice with a wrong GID still stalls at queue-pair connect; MTU mismatch between nodes."),
    # 1677 Troubleshooting
    ("Get the stack traces first; everything else is inference.\n\n"
     "Mechanism. A Python-level stack tells you unambiguously which phase each rank is in: TCPStore::wait (rendezvous), "
     "ncclCommInitRank/bootstrapAllGather (bootstrap), or inside the first collective (application-level mismatch). "
     "Guessing between these three wastes the reproduction.\n\n"
     "Falsifiable hypothesis H1: the ranks are not all in the same phase, i.e. a minority is stuck earlier and the "
     "majority is waiting on them. Predicted observation: `py-spy dump --pid <pid>` on every rank yields at least two "
     "distinct top frames, and the minority frame is upstream of the majority frame in init order.\n\n"
     "Controlled experiment: reproduce with a bounded timeout, and at T+120 s run py-spy dump on all ranks (or gdb "
     "`thread apply all bt` for the C++ frames). Classify ranks by top frame. If all ranks share one frame, the fault is "
     "symmetric — environment/topology — and you bisect transports. If frames differ, attack only the minority ranks' "
     "nodes; the majority is healthy by construction. This prediction is falsifiable: a uniform frame set kills the "
     "'one bad rank' hypothesis outright.\n\n"
     "Confounders: py-spy needing ptrace permission inside containers (--cap-add SYS_PTRACE); a rank blocked in a CUDA "
     "call showing a misleadingly shallow Python frame — cross-check with nvidia-smi utilization near 0% and no compute "
     "app listed."),
    # 1678 Performance Analysis
    ("Compare against a known-good baseline on the same hardware; absolute numbers without a control prove nothing.\n\n"
     "Mechanism. Init behaviour is a function of (driver, NCCL version, container image, topology, env). Changing any "
     "one can flip a working job into a hang — most commonly a driver/NCCL/kernel-module skew after a node reimage, or a "
     "new image with a NCCL built without IB support.\n\n"
     "Falsifiable hypothesis H1: the regression is version/image skew, not the cluster. Predicted observation: the same "
     "minimal all_reduce, same nodes, same env, run under the previously working image, completes; under the new image "
     "it hangs.\n\n"
     "Controlled experiment: run the identical 2-node minimal all_reduce under (a) last known-good image, (b) current "
     "image, on the same node pair, back to back, recording `python -c 'import torch;print(torch.cuda.nccl.version())'`, "
     "`nvidia-smi --query-gpu=driver_version --format=csv`, `modinfo nvidia_peermem`, and whether NCCL logs 'NET/IB' or "
     "'NET/Socket'. A clean A-passes/B-hangs split confirms skew and the fix is image pinning, not fabric work. If both "
     "images hang on that pair but both pass on another pair, skew is falsified and the pair is faulty.\n\n"
     "Confounders: node reimage also changing IB firmware or GID config; container missing /dev/infiniband bind mounts "
     "so NCCL silently degrades to sockets; host and container NCCL both present with LD_LIBRARY_PATH deciding the winner."),
]

RISKS = [
    ["Blind NCCL env-var stacking hides the root cause and permanently degrades bandwidth",
     "A crashed rank misread as a network partition sends the investigation to the wrong team"],
    ["Declaring a hang before the TCP retry budget elapses; the job may only be slow",
     "Pinning the wrong NIC moves traffic onto a low-bandwidth management link"],
    ["Bisection runs on a shared cluster can be preempted, producing false 'pass' results",
     "NCCL_P2P_DISABLE left set after diagnosis silently costs NVLink bandwidth"],
    ["Benchmarking on a busy node yields busbw numbers that falsely confirm 'fallback'",
     "Comparing busbw against the wrong theoretical bound leads to chasing a non-problem"],
    ["Pre-flight collective writing to slow shared storage introduces a new stall",
     "A gloo-based pre-flight passes while the real NCCL transport is still broken"],
    ["Reading RANK from the job script instead of /proc/<pid>/environ misses launcher-level overrides",
     "Duplicate ranks from an elastic-agent restart look like a fabric hang"],
    ["Cross-node timestamp comparison without clock-skew correction produces spurious stall attribution",
     "Buffered logs truncate the decisive last line"],
    ["Changing NIC selection and GID index together makes the result uninterpretable",
     "RoCE without matching PFC/ECN on the switch stalls even with correct NIC and GID"],
    ["py-spy without SYS_PTRACE inside containers returns nothing and wastes the reproduction",
     "Attaching a debugger to all ranks simultaneously can perturb timing-sensitive races"],
    ["Rolling back to the old image without recording the delta loses the actual root cause",
     "Host/container NCCL library shadowing makes the 'version' you print not the one in use"],
]

EVID = [
    ["py-spy/gdb stack of one hung rank per node", "NCCL_DEBUG=INFO INIT/NET logs", "ss -tan on MASTER_PORT", "gloo-only init control run"],
    ["Per-line timestamp deltas in NCCL INIT logs", "ip -o addr per node", "time-to-init at N=2/8/full", "nccl-tests busbw after pinning"],
    ["Pass/fail matrix over node pairs", "nvidia-smi topo -m", "nvidia-smi compute-apps listing", "NCCL_P2P_DISABLE A/B result"],
    ["all_reduce_perf busbw sweep 8B-1GiB", "NCCL transport line from INFO log", "topo -m expected path", "GPU clock/throttle reasons"],
    ["Pre-flight rank inventory file", "Exception text after bounded timeout", "NCCL_ASYNC_ERROR_HANDLING abort stack"],
    ["Sorted rank inventory from /proc/<pid>/environ", "Launcher stderr per node", "GPU UUID list per rank", "WORLD_SIZE vs allocation"],
    ["Per-rank NCCL_DEBUG_FILE logs across 3 runs", "Histogram of inter-line deltas", "Identity of last log line per run"],
    ["ip -o addr and route table", "NCCL INFO NIC selection line", "ibstat / show_gids output", "2-node busbw after pinning"],
    ["py-spy dump from every rank at T+120s", "Frame classification table", "nvidia-smi utilization during stall"],
    ["A/B image run results on same node pair", "nccl version, driver version, peermem modinfo", "presence of /dev/infiniband in container", "NET/IB vs NET/Socket log line"],
]

DIMS = [
    (3, 2, 3), (3, 2, 3), (3, 2, 3), (3, 2, 3), (3, 2, 3),
    (3, 2, 3), (3, 2, 3), (3, 2, 3), (3, 2, 3), (3, 2, 3),
]
CONF = [0.82, 0.80, 0.81, 0.79, 0.80, 0.83, 0.78, 0.79, 0.82, 0.80]

rows = []
with open(CORPUS, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if START <= i <= END:
            rows.append(json.loads(line))
        elif i > END:
            break
assert len(rows) == 10, len(rows)

out = []
for k, d in enumerate(rows):
    msgs = d["messages"]
    su = next(m["content"] for m in msgs if m["role"] == "user")
    sa = next(m["content"] for m in msgs if m["role"] == "assistant")
    tc, ic, os_ = DIMS[k]
    out.append({
        "source_id": d["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": su,
        "source_assistant": sa,
        "corrected_answer": HEADS[k][0] + COMMON_TAIL,
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": RISKS[k],
        "evidence_required": EVID[k],
        "confidence": CONF[k],
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(out))
print("ids", out[0]["source_id"], "->", out[-1]["source_id"])
