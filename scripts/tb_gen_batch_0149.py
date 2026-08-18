import json

CORPUS = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0149.jsonl"
START, N = 1480, 10

# per-record: (variant_focus, hypothesis_id, body_dict)
SPECS = [
 dict(
  focus="stuck-rank identification via per-rank arrival timestamps rather than whole-job log scraping",
  h="H36", hyp="the hang is caused by a strict subset of ranks (>=1, <world_size) never reaching ncclCommInitRank, not by a symmetric transport failure; if true, per-rank arrival timestamps show a bimodal distribution with a non-empty 'never arrived' set that is stable across three repeats",
  exp="Instrument every rank to append (global_rank, hostname, local_rank, monotonic_ts) to a shared file immediately before and after the init call. Run the same job 3 times with NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,ENV and TORCH_NCCL_BLOCKING_WAIT=1 with a 300 s watchdog. Falsified if all ranks log 'before' and none log 'after' (symmetric failure) or if the missing set is random across repeats (then suspect scheduler/host flakiness, not topology).",
  meas="count of ranks with pre-init but no post-init record; wall-clock spread of pre-init timestamps (p50/p99, seconds); py-spy dump or gdb backtrace of one stuck rank showing whether it blocks in bootstrap (TCP rendezvous) or in transport setup",
  conf="a slow rank that merely arrives late (data loader, checkpoint load) looks identical to a dead rank until the timeout is raised; a rank killed by OOM leaves the same gap as a rank blocked on the network",
  rb="do not restart or drain nodes until the stuck-rank set is captured; abort the diagnostic run at 300 s; revert watchdog/debug env after collection since NCCL_DEBUG=INFO adds startup latency and log volume",
  risks=["killing the job before capturing per-rank state destroys the only evidence","raising the NCCL/watchdog timeout as a 'fix' converts a fast hang into a slow silent stall"],
  ev=["per-rank pre/post-init timestamp table for 3 repeats","stack dump of at least one non-arriving rank","scheduler/OOM-killer logs for hosts in the missing set"],
  dec="rewrite", qd=(2,2,2), c=0.66),
 dict(
  focus="device visibility and duplicate GPU assignment as the failure mechanism",
  h="H37", hyp="two or more ranks on the same host bind to the same CUDA device (or a rank sees fewer devices than local_world_size), so NCCL builds an inconsistent communicator and blocks; if true, the (hostname, cuda_device_uuid) pairs collected across all ranks contain a duplicate or a missing entry",
  exp="Before init, each rank logs CUDA_VISIBLE_DEVICES, local_rank, torch.cuda.device_count() and the UUID of the device it selected. Assert globally that UUIDs are unique per host and that count == local ranks per host. Then rerun with an explicit, container-aware mapping. Falsified if UUIDs are already unique and complete yet the hang persists — then the mechanism is transport, not device binding.",
  meas="unique-UUID assertion result; nvidia-smi -L per host vs. container view; nvidia-smi topo -m to confirm expected NVLink/PCIe paths on an 8x A30 node",
  conf="launchers differ: torchrun sets LOCAL_RANK while MPI sets OMPI_COMM_WORLD_LOCAL_RANK; a device-masking layer in the container can renumber devices so that logs look consistent inside the container but collide on the host",
  rb="binding changes are per-job env only; do not edit cluster-wide launcher defaults until one job proves the fix; roll back if the corrected mapping does not change time-to-init within 2 runs",
  risks=["assuming CUDA_VISIBLE_DEVICES semantics are identical between torchrun, MPI and the container runtime","changing global launcher defaults from a single-job observation"],
  ev=["global (host, gpu_uuid, rank) table with the uniqueness assertion output","nvidia-smi -L inside and outside the container","init latency before and after the mapping fix"],
  dec="rewrite", qd=(2,2,2), c=0.66),
 dict(
  focus="rendezvous/bootstrap layer (store, master address, port) separated from the NCCL transport layer",
  h="H39", hyp="the block is in the bootstrap rendezvous (TCPStore / master_addr:master_port reachability), not in NCCL transport setup; if true, replacing the collective with a pure torch.distributed.gloo barrier over the same store also hangs",
  exp="Run the identical launcher and world size but initialize with backend='gloo' and perform a barrier only. If gloo hangs too, the fault is in rendezvous (DNS for MASTER_ADDR, port blocked, stale store key, mismatched job id). If gloo passes and nccl hangs, the fault is below, in transport/interface selection. Falsifiable in one run each way.",
  meas="gloo barrier completion time vs. nccl init time; TCP connectivity to MASTER_ADDR:MASTER_PORT from every host (ss/nc); resolution of MASTER_ADDR on each host; store key namespace/job id equality across ranks",
  conf="gloo and nccl may select different network interfaces (GLOO_SOCKET_IFNAME vs NCCL_SOCKET_IFNAME), so a gloo pass does not prove the data-plane NIC is healthy; a stale rendezvous key from a previous run can make an otherwise healthy job hang",
  rb="diagnostic gloo run uses a distinct rendezvous id so it cannot collide with the production job; do not leave the job running on gloo as a workaround since it silently loses NVLink/RDMA bandwidth",
  risks=["shipping gloo as a permanent workaround and absorbing a large throughput regression","reusing the same store key for the diagnostic run and corrupting the production rendezvous"],
  ev=["gloo-barrier vs nccl-init outcome pair","per-host reachability and DNS resolution results for MASTER_ADDR:MASTER_PORT","store key / job id captured from every rank"],
  dec="rewrite", qd=(2,2,2), c=0.66),
 dict(
  focus="minimal reproducer: smallest all-reduce that still hangs, as the experimental unit",
  h="H40", hyp="the hang reproduces with a 4-byte all-reduce and no framework code, i.e. it is a communicator/topology problem and independent of model, data pipeline and message size; if true, a ~30-line script hangs at the same world size while the full training job is irrelevant",
  exp="Write a minimal script that only does init_process_group + one all_reduce of a single float. Sweep world size 2, 4, 8, then multi-node 16. Record the smallest configuration that hangs. Falsified if the minimal script always succeeds at every world size while the real job hangs — then look at framework-level ordering (e.g. ranks entering different collectives) rather than transport.",
  meas="hang/pass matrix over world sizes and node counts; time-to-completion for passing configurations (milliseconds); which hop (intra-node NVLink vs inter-node fabric) first fails",
  conf="the minimal script may not reproduce a race that needs concurrent CUDA work or memory pressure; a passing minimal run can hide an ordering bug where ranks call different collectives in different order",
  rb="keep the reproducer job small (<=16 GPUs) and time-boxed at 120 s so it does not occupy the cluster; do not use its pass result to declare the fabric healthy for the full job",
  risks=["declaring the fabric healthy from a toy reproducer that never exercises the failing code path","confusing a mismatched collective order bug with a transport failure"],
  ev=["world-size vs hang/pass matrix","source of the minimal reproducer","NCCL_DEBUG=INFO transport lines for the smallest hanging configuration"],
  dec="rewrite", qd=(2,2,2), c=0.66),
 dict(
  focus="interface and transport selection: which NIC/IB device NCCL actually chose",
  h="H41", hyp="NCCL selected a wrong or unroutable interface (docker0/management NIC instead of the RDMA-capable data NIC), so bootstrap succeeds but ring/tree connection setup blocks; if true, NCCL_DEBUG=INFO shows a NET/Socket or NET/IB line naming an unexpected device, and pinning NCCL_SOCKET_IFNAME / NCCL_IB_HCA to the intended device removes the hang",
  exp="Collect the NET selection line from every rank. Then rerun with NCCL_SOCKET_IFNAME and, if RoCE/IB is intended, NCCL_IB_HCA and NCCL_IB_GID_INDEX pinned explicitly, changing nothing else. Falsified if the selected interface was already the intended one, or if pinning it leaves the hang unchanged.",
  meas="per-rank NET selection line; ip addr / ibv_devinfo per host; ib_write_bw or nccl-tests bus bandwidth (GB/s) on the intended path once init succeeds; time-to-init before vs after pinning",
  conf="on RoCE the GID index differs per fabric and a wrong GID gives a hang that looks identical to a wrong HCA; asymmetric NIC naming across hosts makes a single IFNAME value work on some nodes and fail on others",
  rb="pin interfaces via job env only; require one clean multi-node run plus a bandwidth check at >=80% of the previously observed baseline before promoting the setting to the cluster default; roll back immediately if any node's init regresses",
  risks=["setting a global NCCL_SOCKET_IFNAME that is invalid on a heterogeneous subset of nodes and breaking healthy jobs","silently falling back to TCP over the management NIC and losing most of the fabric bandwidth without noticing"],
  ev=["per-rank NET/IB or NET/Socket selection lines","ibv_devinfo and ip addr from every participating host","bus-bandwidth measurement on the pinned path vs baseline"],
  dec="rewrite", qd=(2,2,2), c=0.68),
 dict(
  focus="timeout semantics: distinguishing a true deadlock from an extremely slow init",
  h="H42", hyp="the job is not deadlocked but slow — init completes if given more time — because of serialized rendezvous or per-connection setup cost that grows with world size; if true, init latency scales superlinearly with world size and eventually completes rather than tripping the watchdog",
  exp="Raise the process-group timeout to 30 minutes purely as an observation instrument (not a fix) and measure init latency at world sizes 8, 16, 32, 64. Fit latency vs world size. Falsified if it never completes at any timeout at a fixed world size — that is a true deadlock and the timeout dimension is a dead end.",
  meas="init wall-clock (seconds) per world size; slope of latency vs ranks; whether the curve is linear, quadratic, or a step at a specific node count; concurrent DNS/store request rate",
  conf="a large timeout masks real deadlocks in production and delays failure detection; shared filesystem or DNS contention from other jobs makes the same world size slow only at certain times of day",
  rb="the raised timeout is diagnostic-only and must be reverted to the production value (documented, typically 10-30 minutes for large jobs but explicitly chosen) after measurement; do not ship an unbounded timeout",
  risks=["shipping the raised timeout as the fix, converting fast visible failures into long invisible stalls","attributing slowness to world size when the real cause is transient cluster-wide contention"],
  ev=["init latency vs world size table with repeats","timestamped record that the diagnostic timeout was reverted","contention metrics (store QPS, DNS latency) during the slow runs"],
  dec="rewrite", qd=(2,2,2), c=0.66),
 dict(
  focus="bisection over the node set to localize a single bad host or NIC",
  h="H43", hyp="a single node (or one NIC on it) is responsible; if true, binary search over the node set converges to one host whose inclusion is necessary and sufficient for the hang, reproducible across 3 trials",
  exp="Fix world size per node and run the minimal all-reduce over halves of the node set, then quarters, keeping every other variable constant. Each configuration runs 3 times with a 120 s bound. Falsified if hangs appear in configurations that share no common node, which points to a fabric/switch-level or configuration-level cause instead.",
  meas="per-configuration hang rate (hangs/3); the minimal node set that still hangs; switch/port counters (link errors, symbol errors, discards) for the implicated host; ibv_devinfo state PORT_ACTIVE",
  conf="intermittent faults give false negatives, so a single passing run does not clear a node; a bad top-of-rack switch port produces a node-shaped signature even though the node is healthy",
  rb="do not drain nodes automatically from the bisection result; require the implicated node to fail a standalone loopback/pairwise bandwidth test before draining, and re-admit it only after a passing pairwise test",
  risks=["draining healthy nodes on the basis of a flaky single-trial bisection and shrinking cluster capacity","missing an intermittent fault because each configuration was tried only once"],
  ev=["bisection matrix with 3 trials per configuration","pairwise bandwidth test result for the implicated node","switch port error counters over the incident window"],
  dec="rewrite", qd=(2,2,2), c=0.66),
 dict(
  focus="version and configuration skew across ranks as the mechanism",
  h="H44", hyp="ranks are not running identical software or environment (NCCL/CUDA driver/torch version, container image digest, or a subset of NCCL_* variables set on only some hosts), and the resulting protocol/algorithm mismatch blocks init; if true, a global diff of the collected environment fingerprints is non-empty",
  exp="Each rank emits a fingerprint: image digest, torch.__version__, torch.cuda.nccl.version(), driver version, and the sorted list of NCCL_*/CUDA_* variables. Hash it and compare across ranks. Rerun with a single pinned image digest on all hosts. Falsified if all fingerprint hashes are already identical yet the hang persists.",
  meas="number of distinct fingerprint hashes across the world; the exact differing keys; init outcome after pinning one digest",
  conf="an image tag such as :latest can resolve to different digests per host and hide skew behind an identical-looking tag; environment injected by the scheduler prologue may differ per partition and not appear in the job spec",
  rb="pin by digest, not tag, and roll the pinned image to one job before any fleet-wide change; roll back to the previous digest if throughput regresses beyond a preset threshold",
  risks=["trusting image tags instead of digests and concluding 'no skew' incorrectly","fleet-wide image or driver rollout triggered by one job's evidence"],
  ev=["per-rank environment fingerprint hashes and the diff of differing keys","resolved image digest per host","init outcome with a single pinned digest"],
  dec="rewrite", qd=(2,2,2), c=0.67),
 dict(
  focus="separating intra-node (NVLink/PCIe/P2P) setup from inter-node fabric setup",
  h="H45", hyp="the failure is intra-node: P2P/shared-memory transport setup fails (IOMMU/ACS settings, restricted /dev/shm, or container IPC isolation), not inter-node; if true, a single-node 8-GPU all-reduce hangs on its own, with no fabric involved",
  exp="Run the minimal all-reduce on one node with 8 ranks. If it hangs, rerun with NCCL_P2P_DISABLE=1 and separately with NCCL_SHM_DISABLE=1 to identify which transport is the blocker. Falsified if single-node always passes and only multi-node hangs — then the intra-node hypothesis is dead and the inter-node path owns the fault.",
  meas="single-node hang/pass; which disable flag restores progress; /dev/shm size and container --ipc mode; nvidia-smi topo -m P2P matrix; single-node bus bandwidth (GB/s) with and without P2P",
  conf="disabling P2P or SHM can hide the fault while imposing a large bandwidth penalty, which looks like a fix in CI and a regression in production; on some A30/PCIe topologies P2P is legitimately unavailable and is not the bug",
  rb="NCCL_P2P_DISABLE/NCCL_SHM_DISABLE are diagnostic switches only; keeping either requires a documented bandwidth measurement and explicit acceptance; revert if single-node bus bandwidth drops below 80% of baseline",
  risks=["leaving P2P or SHM disabled permanently and silently losing intra-node bandwidth","changing host IOMMU/ACS settings cluster-wide from a single-node observation"],
  ev=["single-node hang/pass result","bus bandwidth with and without each disable flag vs baseline","container IPC mode, /dev/shm size, and nvidia-smi topo -m output"],
  dec="rewrite", qd=(2,2,2), c=0.67),
 dict(
  focus="a permanent preflight gate so the class of hang is detected before the job starts",
  h="H46", hyp="the majority of these hangs are detectable by a <60 s preflight collective check run under the same launcher, image and node set as the job; if true, replaying the preflight against archived incidents flags most of them before a full job is scheduled",
  exp="Add a preflight stage that runs the minimal all-reduce with a hard 60 s bound and asserts rank arrival, GPU UUID uniqueness, environment fingerprint equality and NET selection. Replay it against archived hang incidents to estimate detection rate, then run it in shadow mode on live jobs for a defined window before it is allowed to block anything. Falsified if the detection rate is low or the false-positive rate makes it unusable.",
  meas="detection rate on archived incidents; false-positive rate in shadow mode; added startup latency (seconds) and GPU-seconds cost per job; number of full-scale hangs avoided per week",
  conf="the preflight uses a smaller world size than the job, so it cannot catch scale-dependent faults; passing preflight can create false confidence and slow down triage when a real hang occurs anyway",
  rb="ship in shadow mode first; only allow it to block job submission after a defined shadow window with a false-positive rate under an agreed ceiling; a preflight failure quarantines the job for human review rather than auto-draining nodes",
  risks=["auto-blocking or auto-draining on preflight failures before false-positive rate is measured","preflight passing at small scale and creating false confidence for large-scale jobs"],
  ev=["archived-incident replay detection rate","shadow-mode false-positive rate and added latency per job","policy document stating the blocking threshold and the quarantine (not auto-drain) action"],
  dec="rewrite", qd=(2,2,2), c=0.65),
]

def build(spec):
    return (
f"""Failure class: multi-GPU collective initialization hang. Variant focus: {spec['focus']}.

Assumptions: the job previously ran on this cluster; hardware is treated as nominally healthy until measured otherwise; no platform-specific defaults are assumed beyond what the collected logs show. All numbers below are to be measured, not quoted from memory.

Falsifiable hypothesis {spec['h']}: {spec['hyp']}.

Controlled experiment: {spec['exp']}

Measurements to collect: {spec['meas']}.

Expected confounders: {spec['conf']}.

Rollback criteria and safety: {spec['rb']}.

Reporting: record the hypothesis, the configuration under test, the raw measurement and the accept/reject decision for each run, so the conclusion can be reproduced by another engineer without re-deriving the plan."""
    )

rows = []
with open(CORPUS) as f:
    for i, line in enumerate(f):
        if START <= i < START + N:
            rows.append(json.loads(line))
        elif i >= START + N:
            break
assert len(rows) == N, len(rows)

out = []
for r, spec in zip(rows, SPECS):
    msgs = {m["role"]: m["content"] for m in r["messages"]}
    tc, ic, os_ = spec["qd"]
    out.append({
        "source_id": r["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": spec["dec"],
        "source_user": msgs["user"],
        "source_assistant": msgs["assistant"],
        "corrected_answer": build(spec),
        "quality_dimensions": {"technical_correctness": tc, "instruction_coverage": ic, "operational_safety": os_},
        "risks": spec["risks"],
        "evidence_required": spec["ev"],
        "confidence": spec["c"],
    })

with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(out), [o["source_id"] for o in out])
