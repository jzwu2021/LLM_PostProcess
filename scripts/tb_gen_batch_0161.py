import json, hashlib, os

SRC = 'research/ai-infra-expert/corpus/train.jsonl'
START, N = 1600, 10  # zero-based positional start
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0161.jsonl'

rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if START <= i < START + N:
            rows.append(json.loads(line))
        elif i >= START + N:
            break
assert len(rows) == N, len(rows)

# Per-variant distinct root-cause mechanism, so the ten answers are not templated.
MECH = [
    dict(
        key="rendezvous_store_unreachable",
        title="TCPStore rendezvous never completes: MASTER_ADDR resolves to a non-routable interface",
        hyp="H1: the hang is in torch.distributed rendezvous (before any NCCL comm is built), because non-rank-0 processes cannot open a TCP connection to MASTER_ADDR:MASTER_PORT.",
        exp="Controlled experiment: keep the job identical but replace the training body with a script that only calls dist.init_process_group(backend='gloo') and prints rank/world_size. Gloo uses the same TCPStore rendezvous but no GPU/NCCL path. If gloo also hangs, the fault is rendezvous, not NCCL. Then run `python -c \"import socket;socket.create_connection((MASTER_ADDR,MASTER_PORT),5)\"` from every node.",
        meas="Per rank: MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK from the env; `ss -ltnp | grep <port>` on the rank-0 node; `getent hosts $MASTER_ADDR` on every node; py-spy dump on each hung PID (expect the stack parked in TCPStore::wait / _store_based_barrier).",
        conf="A firewall that drops rather than rejects makes this look like a NCCL timeout; container network namespaces can make MASTER_ADDR resolve locally on each node so rank 0's own connect succeeds and masks the failure.",
        fix="Pin MASTER_ADDR to the routable management IP of the rank-0 node and open the port; set TORCH_DISTRIBUTED_DEBUG=DETAIL for the retry.",
        rb="Rollback gate: if a 2-node gloo-only init does not complete within 120 s after the change, revert the addressing change and escalate to the network team; do not proceed to a full-scale run.",
        risks=["Opening MASTER_PORT broadly on a shared cluster widens the attack surface; scope it to the job's node set.",
               "py-spy dump requires ptrace on the container; enabling CAP_SYS_PTRACE cluster-wide is a security regression."],
        ev=["Per-rank env dump (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK)",
            "py-spy dump stack of every hung rank showing whether it is in TCPStore rendezvous or ncclCommInitRank",
            "TCP reachability test from each node to MASTER_ADDR:MASTER_PORT",
            "gloo-only init_process_group control run result"],
        dims=(5, 5, 5), conf_v=0.62,
    ),
    dict(
        key="world_size_rank_mismatch",
        title="World-size / rank-set mismatch: fewer ranks join than WORLD_SIZE declares",
        hyp="H2: the collective blocks because the process group is waiting on ranks that were never started — the launcher's nnodes*nproc_per_node does not equal the WORLD_SIZE the ranks were told, or one node's agent died before joining.",
        exp="Controlled experiment: instrument rank 0 to log the store key count after rendezvous, and run the identical job at world sizes 2, 4, then 8. If it completes at 2 and 4 and hangs only at 8, the missing member is on the node added at 8. Cross-check by having every rank write /tmp/joined.$RANK and diffing the observed rank set against range(WORLD_SIZE).",
        meas="torchrun agent logs per node (elastic agent 'rendezvous complete' line with the member list); the set of ranks that wrote their heartbeat file; exit codes of any already-dead worker PIDs; dmesg for OOM-killer kills on the missing node.",
        conf="A rank that segfaults or is OOM-killed right after rendezvous looks identical to a rank that never joined; SLURM's --ntasks disagreeing with --nproc-per-node silently produces the same symptom.",
        fix="Make the launcher the single source of truth for WORLD_SIZE (do not also set it in the environment), and enable elastic health-check so a dead member aborts the group instead of hanging it.",
        rb="Rollback gate: if the reduced-world 4-rank control also hangs, this hypothesis is falsified — revert to H1/rendezvous and stop bisecting world size.",
        risks=["Silently shrinking world size to 'make it run' changes the effective batch size and invalidates any loss curve comparison.",
               "OOM-killed ranks may leave stale shared-memory segments that break the next launch."],
        ev=["Launcher/agent logs from every node with the rendezvous member list",
            "Observed joined-rank set vs range(WORLD_SIZE)",
            "dmesg / OOM-killer and worker exit codes on the suspected node",
            "Results of the 2 / 4 / 8 world-size ladder"],
        dims=(5, 5, 4), conf_v=0.6,
    ),
    dict(
        key="nic_interface_selection",
        title="NCCL picks the wrong network interface (docker0 / loopback) for its bootstrap",
        hyp="H3: ncclCommInitRank hangs because NCCL's socket bootstrap auto-selected a non-routable interface (docker0, virbr0, or a second NIC on an unreachable subnet), so out-of-band bootstrap packets never arrive.",
        exp="Controlled experiment: run the same 2-node minimal all-reduce twice — once unchanged, once with NCCL_SOCKET_IFNAME pinned to the known data-plane NIC. Identical code, one variable changed. Completion in the pinned run and a hang in the unpinned run confirms interface selection.",
        meas="NCCL_DEBUG=INFO + NCCL_DEBUG_SUBSYS=INIT,NET on all ranks — read the 'NET/Socket : Using [0]<ifname>' line per rank and check every rank chose the same routable NIC; `ip -br addr` and `ip route get <peer>` on each node.",
        conf="Interface names differ between nodes (eth0 vs ens5f0), so a single pinned name can be right on one node and wrong on another; NCCL_SOCKET_IFNAME accepts a prefix and '^' exclusion, which is easy to invert.",
        fix="Pin NCCL_SOCKET_IFNAME (and NCCL_IB_HCA if IB is used) via the job template, using a prefix that matches on every node, and exclude docker0/lo explicitly.",
        rb="Rollback gate: if the pinned run still hangs at the same NCCL init line, revert the pin (it may exclude the only working NIC) and move to the transport/IB hypothesis.",
        risks=["Pinning to a management NIC 'fixes' the hang but silently drops throughput to 1-10 GbE, which will later be misread as a model-side slowdown.",
               "Excluding interfaces wholesale can break other jobs sharing the same node template."],
        ev=["NCCL_DEBUG=INFO INIT/NET lines from every rank showing the selected interface",
            "`ip -br addr` and `ip route get <peer-ip>` on each node",
            "Paired minimal all-reduce runs with and without NCCL_SOCKET_IFNAME pinned",
            "Achieved busbw from nccl-tests after the fix, to confirm the fast path was not lost"],
        dims=(5, 5, 4), conf_v=0.66,
    ),
    dict(
        key="gpu_visibility_duplicate_device",
        title="Two ranks on one node map to the same GPU (CUDA_VISIBLE_DEVICES / local-rank binding bug)",
        hyp="H4: initialization deadlocks because more than one rank called cudaSetDevice on the same physical GPU — NCCL requires a unique device per rank in a communicator and will block (or abort) during init.",
        exp="Controlled experiment: before init_process_group, have each rank print hostname, LOCAL_RANK, torch.cuda.current_device() and the GPU UUID from pynvml; assert the (hostname, UUID) pairs are unique. Then rerun with one rank per node (nproc_per_node=1). If single-rank-per-node succeeds and multi-rank per node hangs, device binding is the fault.",
        meas="Per-rank GPU UUID and PCI bus id; `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv` on each node to see two PIDs on one UUID; the value of CUDA_VISIBLE_DEVICES inside each container.",
        conf="A container that already sets CUDA_VISIBLE_DEVICES to a single GPU makes every rank see 'device 0', so local_rank-based cudaSetDevice collapses all ranks onto the same card; MPS or MIG changes what 'unique device' means.",
        fix="Set the device from LOCAL_RANK only when the container exposes all GPUs; when the runtime already partitions GPUs, always use device 0 and let the runtime do the binding. Assert uniqueness at startup and fail fast instead of hanging.",
        rb="Rollback gate: if UUIDs are already unique across ranks, H4 is falsified — do not change the binding code, move to the transport hypothesis.",
        risks=["Changing device binding while a checkpoint/resume path assumes a fixed mapping can silently load the wrong shard.",
               "Fail-fast assertions added at startup must not abort long-running production jobs on a transient NVML read error."],
        ev=["Per-rank (hostname, LOCAL_RANK, GPU UUID, PCI bus id) table proving uniqueness or collision",
            "nvidia-smi compute-apps listing per node",
            "CUDA_VISIBLE_DEVICES as seen inside each container",
            "nproc_per_node=1 control run outcome"],
        dims=(5, 5, 5), conf_v=0.63,
    ),
    dict(
        key="ib_rdma_transport_down",
        title="RDMA/RoCE transport unusable: bootstrap succeeds, IB/RoCE connection setup stalls",
        hyp="H5: TCP bootstrap completes but the ring never forms because the InfiniBand/RoCE path is broken — HCA port down, wrong GID index for RoCEv2, or a PFC/DSCP mismatch on the switch — and NCCL blocks in transport setup rather than falling back.",
        exp="Controlled experiment: run the same two nodes with NCCL_IB_DISABLE=1 (forcing TCP sockets). If the job initializes and trains at low bandwidth, the fault is isolated to the IB/RoCE transport, not to rendezvous or device binding. Separately run `ib_write_bw` between the same two HCAs to test the fabric without any NCCL code.",
        meas="`ibstat` / `ibv_devinfo` port state and rate; `show_gids` to confirm the RoCEv2 GID index matches NCCL_IB_GID_INDEX; per-port counters (PortXmitDiscards, symbol errors) before and after; NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH logs showing the chosen transport per channel.",
        conf="A single degraded link can make only some rank pairs hang, so a 2-node control may pass while the full job fails — bisect over the actual failing pair; RoCE without correct PFC works at low load and deadlocks under the init burst.",
        fix="Correct GID index / HCA selection via NCCL_IB_HCA and NCCL_IB_GID_INDEX, or take the bad port out of the job's node set; only use NCCL_IB_DISABLE=1 as a diagnostic, never as the production fix.",
        rb="Rollback gate: if ib_write_bw between the pair reaches line rate and NCCL still hangs, H5 is falsified — the fabric is healthy and the fault is in NCCL configuration, so revert any fabric changes before touching more nodes.",
        risks=["NCCL_IB_DISABLE=1 left in a production launcher silently costs an order of magnitude of bandwidth and will be misattributed to the model.",
               "Changing PFC/DSCP on a shared switch affects every tenant and needs a maintenance window and an explicit revert plan."],
        ev=["ibstat/ibv_devinfo port state and link rate on both nodes",
            "ib_write_bw point-to-point result between the suspected HCAs",
            "show_gids output vs the configured NCCL_IB_GID_INDEX",
            "Paired runs with and without NCCL_IB_DISABLE=1"],
        dims=(5, 5, 4), conf_v=0.6,
    ),
    dict(
        key="mismatched_collective_order",
        title="Rank-divergent control flow: ranks call different collectives (or different shapes) in different order",
        hyp="H6: this is not an init fault at all — rendezvous succeeded and the hang is the first collective, because ranks issue a different sequence of collectives (e.g. a conditional all_reduce on a metric that is only non-zero on rank 0, or an uneven last batch producing different tensor shapes).",
        exp="Controlled experiment: set TORCH_DISTRIBUTED_DEBUG=DETAIL, which makes torch validate collective shape/op consistency across ranks and raise instead of hang. Rerun with a fixed synthetic dataset of identical length on all ranks (drop_last=True). If the hang disappears with drop_last and reappears without it, uneven sharding is confirmed.",
        meas="py-spy dump on two hung ranks — compare the Python stacks; if they sit in different call sites, control flow diverged. Also log per-rank (collective name, dtype, shape, group) sequence numbers and diff them; check the NCCL watchdog message naming the mismatched sequence number.",
        conf="Data-dependent branches (early stopping, skipping a step on NaN loss) only diverge on some inputs, so the failure looks intermittent; DDP's internal bucket all-reduce can mask a user-level mismatch.",
        fix="Make every collective unconditional and identical across ranks; guard any rank-conditional logic with an all-reduced flag so all ranks take the same branch; use drop_last or pad the final batch.",
        rb="Rollback gate: if TORCH_DISTRIBUTED_DEBUG=DETAIL reports no mismatch and py-spy shows all ranks at the same call site inside ncclCommInitRank, H6 is falsified — return to the transport/rendezvous hypotheses.",
        risks=["drop_last silently discards samples and changes epoch semantics; record it as an experiment variable, not a fix.",
               "TORCH_DISTRIBUTED_DEBUG=DETAIL adds per-collective overhead and must not stay on in production."],
        ev=["py-spy stacks from at least two hung ranks showing same or different call sites",
            "TORCH_DISTRIBUTED_DEBUG=DETAIL output naming any shape/op mismatch",
            "Per-rank collective sequence log (op, dtype, shape, group)",
            "Paired runs with drop_last True vs False"],
        dims=(5, 5, 5), conf_v=0.61,
    ),
    dict(
        key="timeout_and_watchdog_semantics",
        title="Apparent 'hang' is a too-long (or infinite) collective timeout hiding a fast failure",
        hyp="H7: the job is not deadlocked forever — it is inside the process-group timeout window (default 30 min for NCCL in recent torch, or effectively infinite if TORCH_NCCL_BLOCKING_WAIT/async error handling is disabled), so a real error is being swallowed and only looks like a hang.",
        exp="Controlled experiment: rerun with timeout=timedelta(seconds=120) on init_process_group plus TORCH_NCCL_ASYNC_ERROR_HANDLING=1. If the run now aborts in ~2 minutes with a watchdog message naming the stuck collective and rank, the 'hang' was a masked error, and the message identifies the true fault.",
        meas="Wall-clock time from launch to first observable stall; the watchdog log line (collective name, sequence number, ranks that did not complete); whether any rank exited non-zero while the rest waited.",
        conf="Shortening the timeout can also abort legitimately slow first-iteration work (cuDNN autotune, JIT compile, large checkpoint load), producing a false positive; keep a warm-up allowance.",
        fix="Set an explicit, finite process-group timeout and enable async error handling so a dead rank aborts the group; keep a separate longer timeout only for the known-slow first step.",
        rb="Rollback gate: if the shortened timeout aborts during a phase that is provably just slow (verified by profiling the same step single-node), restore the original timeout — do not keep tightening it.",
        risks=["Aggressive timeouts on a healthy but bursty cluster cause spurious job kills and wasted GPU-hours.",
               "Async error handling turns a hang into an abort, which will destroy in-memory state; confirm checkpointing cadence before enabling it on a long run."],
        ev=["Watchdog / timeout log line naming the stuck collective and non-completing ranks",
            "Wall-clock timeline from launch to stall",
            "Exit codes of all ranks",
            "Single-node profile of the same step to distinguish 'slow' from 'stuck'"],
        dims=(5, 4, 5), conf_v=0.58,
    ),
    dict(
        key="p2p_shm_topology",
        title="Intra-node path broken: P2P/NVLink or /dev/shm restrictions stall the single-node ring",
        hyp="H8: the hang reproduces on a single node with 8 GPUs, so the inter-node fabric is irrelevant — the intra-node transport (P2P over NVLink/PCIe, or the shared-memory transport) is failing, typically a too-small /dev/shm in the container or IOMMU blocking peer access.",
        exp="Controlled experiment: run nccl-tests all_reduce_perf on one node with 8 GPUs — first as-is, then with NCCL_P2P_DISABLE=1, then with NCCL_SHM_DISABLE=1. The variant that completes identifies the broken transport. Also rerun the container with --shm-size=16g and compare.",
        meas="`nvidia-smi topo -m` for the NVLink/PCIe matrix; `df -h /dev/shm` inside the container; NCCL_DEBUG_SUBSYS=GRAPH log showing the chosen ring/tree and per-channel transport; busbw from all_reduce_perf.",
        conf="NCCL_P2P_DISABLE=1 makes the job run but through SHM/host memory, so a 'fix' here is a large silent performance regression; IOMMU on some platforms allows P2P init but corrupts data rather than hanging.",
        fix="Raise the container's /dev/shm to at least a few GB, and correct the IOMMU / ACS settings on the host rather than leaving P2P disabled.",
        rb="Rollback gate: if the single-node 8-GPU nccl-test passes at expected busbw, H8 is falsified — the intra-node path is healthy, so stop changing host BIOS/IOMMU settings and go back to the inter-node hypotheses.",
        risks=["Disabling ACS or changing IOMMU settings has host-wide security implications and requires a reboot plus an explicit revert plan.",
               "Leaving NCCL_P2P_DISABLE=1 in place turns a correctness fix into a permanent throughput loss."],
        ev=["nvidia-smi topo -m output",
            "df -h /dev/shm inside the container",
            "all_reduce_perf busbw for baseline / P2P-disabled / SHM-disabled variants",
            "NCCL GRAPH debug log showing per-channel transport selection"],
        dims=(5, 5, 4), conf_v=0.62,
    ),
    dict(
        key="version_abi_skew",
        title="Heterogeneous software stack: NCCL / CUDA / driver version skew across nodes",
        hyp="H9: nodes are not identical — one node runs a different NCCL, CUDA runtime, or driver version, and the version-sensitive init handshake (protocol/algorithm negotiation) stalls instead of erroring cleanly.",
        exp="Controlled experiment: collect (driver, CUDA runtime, torch, NCCL) versions from every node and partition nodes into version-homogeneous sets. Run the same 2-node job inside one homogeneous set, then across the version boundary. A hang only across the boundary confirms skew.",
        meas="Per node: `nvidia-smi --query-gpu=driver_version --format=csv`, torch.version.cuda, torch.cuda.nccl.version(), and the image digest actually running (not the tag); NCCL_DEBUG=VERSION line from each rank.",
        conf="A mutable image tag (:latest) can differ per node while every manifest claims the same tag — compare image digests, not tags; a matching NCCL version can still be linked against a different CUDA minor.",
        fix="Pin the container image by digest across the whole job, and add a startup assertion that all ranks report identical (driver, CUDA, NCCL) tuples, failing fast with a clear message.",
        rb="Rollback gate: if all nodes already report identical version tuples and identical image digests, H9 is falsified — revert any image changes and return to the transport hypotheses.",
        risks=["Rolling a driver upgrade to homogenize the fleet requires draining nodes and can break other tenants' pinned CUDA builds.",
               "Fail-fast version assertions will block emergency runs on a partially upgraded fleet; make the assertion overridable with an explicit, logged flag."],
        ev=["Per-node (driver, CUDA runtime, torch, NCCL) version table",
            "Container image digests actually running on each node",
            "NCCL_DEBUG VERSION line from every rank",
            "Within-set vs across-boundary 2-node run results"],
        dims=(5, 5, 4), conf_v=0.57,
    ),
    dict(
        key="disaggregated_serving_control_plane",
        title="Disaggregated serving stack (Dynamo / Mooncake): the stalled 'collective' is a KV-transfer handshake, not a training all-reduce",
        hyp="H10: in a prefill/decode-disaggregated deployment the stall is in the KV-cache transfer plane — the prefill worker's registered memory regions never pair with the decode worker's, because the transfer engine's metadata store (etcd/Redis) is unreachable or the workers registered mismatched buffer layouts — so it presents as a 'collective init hang' while no NCCL collective is actually pending.",
        exp="Controlled experiment: bring up one prefill and one decode worker with the KV transfer plane forced to a loopback/TCP fallback path instead of RDMA. If the handshake completes, the fault is the RDMA registration path; if it still stalls, the fault is the metadata/discovery store. Verify independently by querying the metadata store for the registered worker entries.",
        meas="Worker-side logs at the point of buffer registration (region count, size, rkey exchange); metadata-store keys listing registered prefill/decode endpoints and their heartbeat age; per-request timing split into queue / prefill / KV-transfer / decode; RDMA counters on both sides during the stalled handshake.",
        conf="A stalled KV transfer and a stalled NCCL init look the same from the outside (idle GPUs, no progress); autoscaling can add a decode worker with a different tensor-parallel degree, making buffer layouts incompatible in a way that only shows up on the first cross-worker request.",
        fix="Make worker registration validate the KV layout tuple (dtype, num_layers, heads, head_dim, TP degree, page size) and reject mismatches loudly at registration; add a bounded timeout on the handshake with a fallback to co-located execution.",
        rb="Rollback gate: if the loopback/TCP fallback path also stalls, the RDMA hypothesis is falsified — stop touching the fabric and inspect the discovery store. If the fix does not restore end-to-end p99 TTFT to the pre-incident baseline within one canary window, roll back to the co-located (non-disaggregated) deployment.",
        risks=["Rolling back to co-located serving changes the capacity model and can drop throughput below the SLO under the same traffic — pre-compute headroom before flipping.",
               "KV buffers are registered pinned memory; repeated failed handshakes can leak registrations and exhaust the HCA's MR table, escalating a stall into a node-wide failure."],
        ev=["Prefill/decode worker registration logs with buffer layout tuples from both sides",
            "Metadata-store (etcd/Redis) contents listing live worker endpoints and heartbeat ages",
            "Per-stage latency breakdown (queue / prefill / KV-transfer / decode) during the stall",
            "Paired handshake test over RDMA vs TCP-fallback transport",
            "HCA memory-region and RDMA error counters on both workers"],
        dims=(5, 5, 5), conf_v=0.55,
    ),
]

def build(m):
    return (
        m["title"] + "\n\n"
        "Assumptions. The job previously ran, so this is a regression against a known-good state; I have shell access to every node and can rerun at reduced scale; the corpus prompt gives no logs, so every claim below is a hypothesis to be tested, not a conclusion.\n\n"
        "Falsifiable hypothesis. " + m["hyp"] + "\n"
        "This is falsifiable: it predicts a specific, observable outcome, and the control run below can prove it wrong.\n\n"
        "Controlled experiment. " + m["exp"] + " Change exactly one variable between the paired runs; keep the launcher, image digest, node set and dataset fixed so the comparison is attributable.\n\n"
        "Measurements to collect before changing anything. " + m["meas"] + " Collect these from the hung job first — once you kill it the evidence is gone.\n\n"
        "Expected confounders. " + m["conf"] + " Also note that any first-run stall can be plain slowness (checkpoint load, kernel autotune) rather than a deadlock; distinguish the two by checking whether any GPU shows non-zero SM utilization during the stall.\n\n"
        "If confirmed. " + m["fix"] + "\n\n"
        "Rollback criteria. " + m["rb"] + " Apply changes to a 2-node canary first; only widen to the full node set after the canary completes a full step and nccl-tests busbw is within 10 percent of the recorded baseline. Any change that cannot be reverted by a single config edit (BIOS, driver, switch QoS) requires a scheduled window and a written revert procedure before it is attempted."
    )

recs = []
for row, m in zip(rows, MECH):
    msgs = row['messages']
    u = [x for x in msgs if x['role'] == 'user'][0]['content']
    a = [x for x in msgs if x['role'] == 'assistant'][0]['content']
    tc, ic, os_ = m["dims"]
    recs.append({
        "source_id": row["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": build(m),
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": m["risks"],
        "evidence_required": m["ev"],
        "confidence": m["conf_v"],
    })

# anti-template assertion: all corrected_answer distinct
h = [hashlib.sha256(r["corrected_answer"].encode()).hexdigest() for r in recs]
assert len(set(h)) == len(recs), "duplicate corrected_answer"
for r in recs:
    assert r["corrected_answer"] != r["source_assistant"]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs), "ids", recs[0]["source_id"], "->", recs[-1]["source_id"])
