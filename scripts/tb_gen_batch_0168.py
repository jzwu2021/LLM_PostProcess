import json, hashlib, os, glob

SRC = 'research/ai-infra-expert/corpus/train.jsonl'
START, N = 1670, 10  # zero-based positional start
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0168.jsonl'

rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if START <= i < START + N:
            rows.append(json.loads(line))
        elif i >= START + N:
            break
assert len(rows) == N, len(rows)

# Ten distinct root-cause mechanisms for "hang during collective initialization",
# deliberately different from the mechanism sets used in earlier batches.
MECH = [
    dict(
        title="Rendezvous backend split-brain: two TCPStore servers under an elastic restart",
        hyp="H1: the hang is a rendezvous split-brain — an elastic agent restarted after a worker failure and a second TCPStore server bound the same port on a different node, so subsets of ranks completed rendezvous against different stores and each subset waits forever for the other half.",
        exp="Controlled experiment: disable elastic restart (max_restarts=0) and relaunch from a clean state after confirming no stale agent processes remain. Same code, same node set, one variable changed. If a cold launch with max_restarts=0 initializes, split-brain on restart is confirmed; then reproduce deliberately by killing one worker mid-rendezvous.",
        meas="On every node: `ss -ltnp` for listeners on MASTER_PORT and the PIDs owning them; agent logs with the rendezvous run_id and the store address each worker used; process start timestamps to spot a second generation of agents; the number of keys present in each store after rendezvous.",
        conf="A stale agent from a previous job can hold the port and answer rendezvous, so the symptom appears on a job that was never restarted; run_id collisions across concurrently scheduled jobs produce the same split.",
        fix="Use a unique rendezvous run_id per job and a rendezvous backend with real membership semantics (etcd/c10d with a scoped endpoint) rather than a bare host:port; make the agent refuse to start when the port is already owned by a foreign PID.",
        rb="Rollback gate: if `ss -ltnp` shows exactly one listener owned by this job's rank-0 agent and all workers logged the same store address, H1 is falsified — revert the launcher change and move to the transport hypotheses before touching scheduler configuration.",
        risks=["Killing 'stale' processes on a shared node can terminate another tenant's job; match by cgroup/job id before killing anything.",
               "max_restarts=0 removes fault tolerance; it is a diagnostic setting, not a production fix, and must be reverted."],
        ev=["Per-node `ss -ltnp` listener/PID table for MASTER_PORT during the hang",
            "Agent logs from every node showing run_id and store endpoint per worker",
            "Process start timestamps proving one or two agent generations",
            "Outcome of the cold launch with max_restarts=0"],
        dims=(5, 5, 4), conf_v=0.58,
    ),
    dict(
        title="cgroup / pinned-memory limit: NCCL blocks registering buffers it cannot lock",
        hyp="H2: init stalls while NCCL allocates and pins its per-channel buffers, because the container's RLIMIT_MEMLOCK or cgroup memory limit is too low — the allocation path retries/stalls instead of returning a clean error, particularly on the IB registration path.",
        exp="Controlled experiment: rerun the identical job in the identical image with `--ulimit memlock=-1` and a raised cgroup memory limit, changing nothing else. If init completes, the limit is the cause. Independently, run a standalone `ibv_reg_mr` sizing probe (or `ib_write_bw -s` at increasing sizes) under the original limits to find the ceiling without any NCCL code.",
        meas="`cat /proc/<pid>/limits` for Max locked memory on a hung rank; cgroup memory.max and memory.current for the job; NCCL_DEBUG=INFO lines around buffer/channel allocation; `dmesg` for mlock or cgroup OOM messages; per-rank RSS at the moment of the stall.",
        conf="Raising memlock also changes the IB path's behaviour, so a pass in the relaxed run does not by itself prove the limit was the only fault; a node with more free page cache can pass while an identical node fails, making it look node-specific.",
        fix="Set memlock unlimited for GPU jobs in the container runtime defaults, and size the cgroup limit to model+activation+NCCL buffer footprint with explicit headroom rather than a copied constant.",
        rb="Rollback gate: if /proc/<pid>/limits already shows unlimited locked memory and no cgroup pressure is recorded, H2 is falsified — revert the runtime limit change (it weakens isolation) and return to transport hypotheses.",
        risks=["memlock=unlimited lets a buggy job pin enough memory to destabilise the whole node; scope it to the GPU job class.",
               "Raising cgroup limits hides a genuine memory regression that will resurface as a host OOM later."],
        ev=["/proc/<pid>/limits for a hung rank",
            "cgroup memory.max / memory.current for the job",
            "dmesg mlock or OOM entries on the affected node",
            "Paired runs under original vs raised memlock/cgroup limits"],
        dims=(5, 4, 5), conf_v=0.55,
    ),
    dict(
        title="DNS/hostname resolution asymmetry: ranks agree on a name that resolves differently per node",
        hyp="H3: every rank was given the same MASTER_ADDR hostname, but resolution differs per node (search-domain, /etc/hosts injection, or a headless-service record with multiple A records), so ranks connect to different endpoints and the rendezvous barrier never reaches quorum.",
        exp="Controlled experiment: resolve MASTER_ADDR to a literal IP once, and relaunch with that IP substituted on every rank — one variable changed. If the run initializes with the literal IP but hangs with the hostname, resolution asymmetry is confirmed. Verify the mechanism directly with `getent hosts $MASTER_ADDR` executed on every node and diffed.",
        meas="`getent hosts $MASTER_ADDR` and `dig +short $MASTER_ADDR` from each node; /etc/resolv.conf and /etc/hosts inside each container; the peer address each rank actually connected to (`ss -tnp | grep <port>` on rank 0 shows the client IPs that arrived).",
        conf="A headless service with several backing pods returns a rotating record, so the same command run twice on one node can give different answers — sample repeatedly; caching resolvers make the fault intermittent across relaunches.",
        fix="Have the launcher resolve the rendezvous endpoint once and propagate a literal IP (or use a stable, single-record service), and log the resolved address from every rank at startup.",
        rb="Rollback gate: if all nodes resolve MASTER_ADDR to the same single address and rank 0 sees exactly WORLD_SIZE-1 inbound connections, H3 is falsified — revert to the hostname form and investigate device binding or transport instead.",
        risks=["Pinning a literal IP defeats failover if the rendezvous node is rescheduled; acceptable for a diagnostic run, not as a permanent pattern.",
               "Editing /etc/hosts inside images creates drift that will silently break future scheduling."],
        ev=["Per-node getent/dig resolution of MASTER_ADDR, sampled more than once",
            "resolv.conf and /etc/hosts contents from each container",
            "Rank-0 inbound connection list during the hang",
            "Paired hostname vs literal-IP launch outcomes"],
        dims=(5, 5, 4), conf_v=0.6,
    ),
    dict(
        title="Communicator construction order: nested/overlapping subgroups created in different orders per rank",
        hyp="H4: rendezvous succeeded, but the job builds several process groups (data-parallel, tensor-parallel, pipeline) and the ranks call new_group in different orders or with different member lists, so two ranks are blocked constructing different communicators — a classic ordering deadlock, not a network fault.",
        exp="Controlled experiment: reduce the parallelism plan to data-parallel only (TP=1, PP=1) with everything else fixed. If init completes, reintroduce one group type at a time; the first reintroduction that reproduces the hang names the offending group. Additionally, log a global (rank, group_name, sorted member list, creation index) table and diff it across ranks.",
        meas="Per-rank creation log of every process group with its ordinal and member list; py-spy stacks on two hung ranks showing which new_group call each is inside; NCCL_DEBUG=INFO comm-init lines with commHash per rank.",
        conf="A framework that lazily creates groups on first use makes the creation order data-dependent, so the deadlock appears only on some configurations; a group created on a subset of ranks must still be created (collectively) by all ranks in the parent group, which is easy to get wrong.",
        fix="Construct every process group eagerly, unconditionally, and in a single deterministic order on all ranks at startup; assert the (name, member list, ordinal) table is identical across ranks before training begins.",
        rb="Rollback gate: if the TP=1/PP=1 control also hangs at the same point, H4 is falsified — the fault is below the parallelism plan; revert the plan change and go back to rendezvous/transport analysis.",
        risks=["Changing the parallelism plan changes memory footprint and can turn a hang into an OOM, masking the original signal.",
               "Eager group creation costs startup time and extra communicators; on large world sizes this can hit NCCL communicator limits."],
        ev=["Per-rank process-group creation table (name, ordinal, sorted members)",
            "py-spy stacks from at least two hung ranks",
            "NCCL comm-init lines with commHash per rank",
            "Results of the TP=1/PP=1 control and the incremental reintroduction"],
        dims=(5, 5, 5), conf_v=0.62,
    ),
    dict(
        title="RoCE congestion-control mismatch: PFC/ECN misconfiguration deadlocks the init burst",
        hyp="H5: the fabric is RoCEv2 and link-level flow control is only half configured — PFC is enabled on the host but not on the switch port (or on a different priority than the DSCP the HCA marks) — so the dense all-to-all bootstrap burst triggers pause-storm or drop-induced stall and connection setup never completes.",
        exp="Controlled experiment: run `ib_write_bw` and then `all_reduce_perf` between the same pair at increasing message sizes, first on the suspect priority, then with traffic remapped to a known-good priority class. A stall that appears only above a size threshold and only on one priority isolates congestion control from basic reachability.",
        meas="`ethtool -S <nic>` pause frame counters (rx_pause/tx_pause) and drops before/after; switch port PFC counters and configured priority map; `mlnx_qos -i <nic>` host-side priority/DSCP mapping; per-port RDMA error counters (out_of_sequence, packet_seq_err); MEASURED busbw vs the recorded baseline.",
        conf="A pause storm on a shared fabric degrades unrelated tenants, so the observed counters may be caused by a neighbour's job rather than this one; small-message tests pass while the init burst fails, so a green ib_write_bw at default size proves very little.",
        fix="Align host DSCP/priority with the switch PFC-enabled priority end to end, enable ECN with a congestion profile matched to the fabric, and re-baseline nccl-tests busbw afterwards.",
        rb="Rollback gate: any switch QoS change requires a maintenance window and a written revert; if post-change busbw is not within 10 percent of the recorded MEASURED baseline, or pause counters keep rising, revert the QoS profile immediately. If the low-priority remap also stalls, H5 is falsified — stop touching switch config.",
        risks=["PFC changes are fabric-wide and can deadlock other tenants; never apply them outside a window with a tested revert.",
               "Disabling PFC 'to test' converts a stall into silent packet loss and retransmission, which will be misread as a model-side slowdown."],
        ev=["Host pause-frame and RDMA error counters before/after the burst",
            "Switch port PFC/ECN configuration and counters for the involved ports",
            "mlnx_qos host priority/DSCP mapping on both nodes",
            "Message-size sweep of ib_write_bw and all_reduce_perf, with MEASURED busbw vs baseline"],
        dims=(5, 5, 5), conf_v=0.52,
    ),
    dict(
        title="GPUDirect RDMA path unavailable: nvidia-peermem/dmabuf missing so NCCL falls into a stalled staging path",
        hyp="H6: GDR is expected but the peer-memory module is not loaded on one node, so NCCL cannot register GPU memory with the HCA; instead of erroring it negotiates a host-staged path whose setup blocks against a bounce-buffer/pinning constraint, presenting as an init hang.",
        exp="Controlled experiment: run the identical two-node job twice, once with NCCL_NET_GDR_LEVEL forced off (host staging explicit) and once with GDR required. If the explicit-staging run initializes and the GDR run hangs, the GDR registration path is implicated. Confirm independently with `ib_write_bw --use_cuda` between the two HCAs, which exercises GDR without NCCL.",
        meas="`lsmod | grep -E 'nvidia_peermem|nv_peer_mem'` and dmesg on both nodes; NCCL_DEBUG_SUBSYS=NET lines reporting 'GPU Direct RDMA Enabled/Disabled' per rank; PCIe topology from `nvidia-smi topo -m` showing whether GPU and HCA share a PCIe switch; MEASURED ib_write_bw --use_cuda bandwidth vs the host-memory number.",
        conf="GDR silently disabled is normally a performance issue, not a hang, so this hypothesis is weaker than the pure-transport ones; a GPU and HCA under different PCIe root complexes can make GDR technically enabled but pathologically slow, which resembles a stall.",
        fix="Load and pin the peer-memory module in the node image, assert GDR status at job startup and fail fast with a clear message rather than silently falling back, and place ranks so each GPU uses the HCA under its own PCIe switch.",
        rb="Rollback gate: if `ib_write_bw --use_cuda` succeeds at expected bandwidth on both nodes and NCCL logs report GDR enabled, H6 is falsified — revert any module/driver change and return to the rendezvous and communicator-order hypotheses.",
        risks=["Loading kernel modules on production nodes requires a change window; a mismatched peermem/driver pair can panic the host.",
               "Forcing GDR off as a workaround costs bandwidth and increases host memory pressure; it is diagnostic only."],
        ev=["lsmod/dmesg peer-memory module status on every node",
            "NCCL NET debug lines reporting per-rank GDR enable state",
            "nvidia-smi topo -m PCIe/NVLink matrix for GPU-HCA affinity",
            "MEASURED ib_write_bw --use_cuda vs host-memory bandwidth on both nodes"],
        dims=(5, 5, 4), conf_v=0.5,
    ),
    dict(
        title="Scheduler-level partial allocation: some ranks are still queued or on a drained node",
        hyp="H7: this is not a networking or NCCL fault — the scheduler granted the job fewer usable nodes than requested (a node drained, a pod stuck Pending on image pull or a missing GPU resource), so the started ranks wait at the barrier for peers that the cluster never launched.",
        exp="Controlled experiment: query the scheduler for the job's allocation and pod/task states before touching anything, then relaunch with a world size equal to the number of actually-Running tasks. If the reduced-world run initializes immediately, the fault is allocation, not communication. Reproduce deliberately by cordoning one node and relaunching at full size.",
        meas="`squeue`/`scontrol show job` or `kubectl get pods -o wide` with phase and node for every task; container image pull events and timestamps; node conditions (Ready, cordoned, drained); count of Running tasks vs requested WORLD_SIZE; GPU allocatable vs requested on each node.",
        conf="A slow image pull makes a task appear 'missing' for minutes then join, so a hang observed early can resolve by itself — timestamp the observation; a pod that crash-loops can flip between Running and Error and be counted inconsistently.",
        fix="Gate training start on an explicit all-tasks-Running precondition enforced by the launcher, with a bounded wait and a clear failure message; pre-pull images to the node pool so pull latency is not on the critical path.",
        rb="Rollback gate: if the scheduler shows all WORLD_SIZE tasks Running on Ready nodes before the barrier, H7 is falsified — revert the reduced-world experiment (it changes effective batch size and invalidates loss comparisons) and return to network hypotheses.",
        risks=["Running at a reduced world size to 'get results' silently changes global batch size and makes any curve non-comparable; record it as an experiment variable.",
               "Cordoning nodes to reproduce the fault removes capacity from other tenants; do it on a scratch pool."],
        ev=["Scheduler listing of every task with phase, node and start time",
            "Node conditions and GPU allocatable/requested per node",
            "Image pull event timeline for the late/missing tasks",
            "Count of Running tasks vs WORLD_SIZE at the moment of the stall, with timestamp"],
        dims=(5, 5, 5), conf_v=0.6,
    ),
    dict(
        title="Clock/ordering trap: the 'hang' is a slow first-touch (checkpoint load, JIT, autotune) misdiagnosed as a deadlock",
        hyp="H8: nothing is deadlocked — one rank is doing genuinely slow first-iteration work (loading a sharded checkpoint from a cold object store, cuDNN/Triton autotune, or CUDA context creation across many GPUs) and the others are correctly blocked at the barrier waiting for it; the job would complete if given more time.",
        exp="Controlled experiment: leave the job running and sample progress instead of killing it. Take `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5` and py-spy dumps every 60 s on the slowest rank for 15 minutes. A rank whose stack advances between samples, or whose GPU memory keeps growing, is slow, not stuck. Then rerun with a warm page cache / pre-staged checkpoint and compare wall-clock to first step.",
        meas="MEASURED time from launch to first completed step in the last known-good run (baseline); ESTIMATE of checkpoint load time = checkpoint_bytes / observed_storage_read_bandwidth — state both terms and replace with a MEASURED read throughput before concluding; per-rank GPU utilization and memory growth over time; repeated py-spy stacks showing movement.",
        conf="A single slow rank and a genuinely deadlocked rank are indistinguishable from one sample — the diagnosis depends on the time series, not a snapshot; a network filesystem under contention makes the slow phase non-reproducible.",
        fix="Pre-stage checkpoints to local NVMe, cache autotune results across runs, and emit a periodic startup heartbeat per rank so 'slow' is visibly different from 'stuck' in the logs.",
        rb="Rollback gate: if two py-spy samples 10 minutes apart show identical stacks with zero SM utilization and no memory growth, H8 is falsified — it is a real deadlock; stop waiting and escalate to the transport and communicator-order hypotheses.",
        risks=["Waiting on a genuinely deadlocked job burns GPU-hours at full allocation cost; bound the observation window explicitly.",
               "Pre-staging checkpoints to local disk creates a second copy that can go stale and be silently loaded on resume."],
        ev=["Time series of per-rank GPU utilization and memory used during the stall",
            "Repeated py-spy stacks from the slowest rank at spaced intervals",
            "Baseline MEASURED launch-to-first-step time from the last known-good run",
            "Storage read throughput MEASURED during the load phase, with the checkpoint size used in the estimate"],
        dims=(5, 4, 5), conf_v=0.55,
    ),
    dict(
        title="Multi-tenant port and device contention: another job on the same node holds the resources this one needs",
        hyp="H9: a co-scheduled job on the same node is holding the rendezvous port, the same GPUs (leaked processes from a previous run), or saturating the HCA, so this job's init blocks on contended resources rather than on any defect in its own configuration.",
        exp="Controlled experiment: run the identical job on an exclusively allocated node set (or after draining co-tenants) with no other change. If it initializes on exclusive nodes and hangs on shared ones, contention is confirmed. Reproduce by deliberately co-scheduling a second job that binds the same port and repeating.",
        meas="`nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv` on each node listing foreign PIDs; `ss -ltnp` ownership of the rendezvous port with cgroup/job attribution; HCA counters showing traffic not attributable to this job; free GPU memory per device at launch.",
        conf="Leaked processes from this job's own earlier attempt look identical to a foreign tenant — attribute PIDs to cgroups before blaming neighbours; exclusive allocation also changes cache and NUMA conditions, so a pass is not proof of contention alone.",
        fix="Allocate the rendezvous port from an ephemeral range scoped to the job, require exclusive GPU allocation for multi-node training, and add a pre-flight check that fails fast when foreign compute processes occupy the assigned GPUs.",
        rb="Rollback gate: if the pre-flight check reports the assigned GPUs idle and the port free, H9 is falsified — release the exclusive allocation (it wastes capacity) and return to the configuration hypotheses.",
        risks=["Demanding exclusive nodes reduces cluster utilization for everyone; justify it with the measured contention evidence, not by default.",
               "Killing foreign PIDs to free GPUs can destroy another team's multi-day run; require explicit ownership attribution first."],
        ev=["Per-node compute-apps listing with PID, GPU UUID and cgroup/job attribution",
            "Rendezvous port ownership with owning PID and job id",
            "Free GPU memory per device immediately before launch",
            "Paired shared-node vs exclusive-node launch outcomes"],
        dims=(5, 5, 4), conf_v=0.57,
    ),
    dict(
        title="Inference-side analogue: a Dynamo/Mooncake worker fleet stalls forming its TP group after a rolling update",
        hyp="H10: in a disaggregated LLM serving fleet the 'collective init hang' is a replica that cannot form its tensor-parallel communicator after a rolling update — the new replica's TP world contains a pod from the previous version with a different KV layout or TP degree, so the group never closes and the replica stays unready while the router keeps sending it no traffic and capacity silently drops.",
        exp="Controlled experiment: bring up one isolated replica set at the new version only, with the router excluded, and observe whether the TP group forms. If an all-new replica set initializes and a mixed-version one does not, version-mixed group membership is confirmed. Then reproduce by pinning one pod to the old image inside an otherwise new replica set.",
        meas="Per-pod image digest and TP rank assignment; readiness-probe transition timestamps; router-side count of ready replicas vs desired; the KV layout tuple (dtype, num_layers, kv heads, head_dim, page size, TP degree) each pod registers; MEASURED p99 TTFT and queue depth during the degraded window against the pre-update baseline.",
        conf="A shrinking ready-replica pool raises latency on the surviving replicas, so the incident presents as a performance regression rather than an init failure; autoscaling can replace the stuck pod and mask the mechanism between observations.",
        fix="Make replica sets version-atomic (a TP group must be formed only from pods with the same image digest and layout tuple), validate the layout tuple at group formation and fail readiness loudly, and hold the router at the previous version until the new set is fully ready.",
        rb="Rollback gate: roll back the deployment to the previous image digest if ready replicas stay below the capacity floor for more than one canary window, or if MEASURED p99 TTFT exceeds the pre-update baseline by more than the agreed SLO margin. If an all-new isolated replica set also fails to form its group, H10 is falsified — the fault is in the new image itself, so halt the rollout entirely rather than mixing versions.",
        risks=["Rolling back mid-update leaves a mixed fleet for the duration of the rollback; ensure the router can drain the failing version first.",
               "Failing readiness loudly on layout mismatch will take capacity offline immediately — confirm the remaining fleet can carry the load before enabling it."],
        ev=["Per-pod image digest, TP rank and registered KV layout tuple",
            "Readiness-probe transition timeline and ready-vs-desired replica counts",
            "Router-side MEASURED p99 TTFT and queue depth vs the pre-update baseline",
            "Outcome of the isolated all-new-version replica set experiment"],
        dims=(5, 5, 5), conf_v=0.52,
    ),
]


def build(m):
    return (
        m["title"] + "\n\n"
        "Assumptions stated up front. PyTorch + NCCL, one process per GPU, torchrun/elastic launcher, homogeneous GPUs unless proven otherwise, and shell access to every node. No logs were supplied with the prompt, so everything below is a hypothesis with a test attached, not a conclusion. Every number is tagged ESTIMATE or MEASURED; no ESTIMATE may be used as a rollout gate until it has been replaced by a MEASURED value.\n\n"
        "Falsifiable hypothesis. " + m["hyp"] + " It is falsifiable because the control run below predicts a specific observable outcome that would disprove it.\n\n"
        "Controlled experiment. " + m["exp"] + " Hold the launcher, image digest, node set, dataset and seed fixed and change exactly one variable, so any difference is attributable.\n\n"
        "Evidence to capture before changing anything. " + m["meas"] + " Capture this from the live hung job first: killing it destroys the only direct evidence of the failing state.\n\n"
        "Expected confounders. " + m["conf"] + " In addition, a first-run stall can be plain slowness rather than deadlock; separate the two by checking whether any rank shows non-zero SM utilization or growing GPU memory during the stall.\n\n"
        "If the hypothesis survives. " + m["fix"] + " Apply the change to a two-node canary before the full node set.\n\n"
        "Rollback criteria. " + m["rb"] + " Independently of the hypothesis: widen beyond the canary only after it completes a full training step and nccl-tests busbw is within 10 percent of the recorded MEASURED baseline; any change that cannot be reverted by a single config edit (BIOS, kernel module, driver, switch QoS) requires a scheduled window and a written revert procedure agreed before it is attempted."
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

# anti-template assertions: distinct within batch AND globally unseen
h = [hashlib.sha256(r["corrected_answer"].encode()).hexdigest() for r in recs]
assert len(set(h)) == len(recs), "duplicate corrected_answer within batch"
seen = set()
for f in glob.glob('experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-*.jsonl'):
    for l in open(f):
        seen.add(hashlib.sha256(json.loads(l)["corrected_answer"].encode()).hexdigest())
assert not (set(h) & seen), "corrected_answer collides with an existing batch"
for r in recs:
    assert r["corrected_answer"] != r["source_assistant"]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs), "ids", recs[0]["source_id"], "->", recs[-1]["source_id"])
