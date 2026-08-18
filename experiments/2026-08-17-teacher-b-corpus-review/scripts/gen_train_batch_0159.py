import json, os

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
CORP = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT = os.path.join(EXP, "results", "train-batch-0159.jsonl")
START, N = 1580, 10

rows = [json.loads(l) for l in open(CORP, encoding="utf-8")]
sel = rows[START:START + N]

ASSUM = ("Assumptions: the hang is reproducible; launcher, world size and node set are held constant "
         "across runs; no code change is made between control and treatment runs; measured facts are "
         "separated from estimates and no vendor-specific behaviour is asserted without a log line or "
         "command output to back it.")

TRIAGE = ("Ordered triage (do this before touching any tunable): (1) record the full rank census - "
          "hostname, LOCAL_RANK, PID, CUDA device UUID, WORLD_SIZE, MASTER_ADDR/PORT and image digest "
          "for every rank; (2) capture NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH plus "
          "TORCH_DISTRIBUTED_DEBUG=DETAIL and identify the LAST stage each rank reached, since the set "
          "of ranks that never printed a stage localises the fault; (3) reproduce with a minimal "
          "standalone all-reduce at the same world size, outside the training script; (4) shrink the "
          "world - single node, then 2 ranks, then 2 nodes x 1 rank - to separate intra-node transports "
          "(SHM, P2P) from inter-node transports (socket, RDMA).")

CONF = ("Expected confounders. Scheduler-injected environment variables that silently override "
        "CUDA_VISIBLE_DEVICES, NCCL_SOCKET_IFNAME or NCCL_IB_HCA; a node drained, rebooted or "
        "re-imaged between runs; residual debug env vars left in the launch template from an earlier "
        "incident; and node-to-node heterogeneity in driver, firmware or NUMA layout. Record a full "
        "environment diff between control and treatment runs, otherwise the comparison is not "
        "controlled and any conclusion is anecdote.")

ROLLBACK_TAIL = ("General rollback criteria: if two consecutive controlled runs fail to move the stall "
                 "to a different stage, stop tuning, restore the last known-good launch template "
                 "verbatim, and hand off with the collected evidence bundle rather than stacking "
                 "further environment overrides. Never widen NCCL timeouts as a fix - a longer timeout "
                 "converts a fast, diagnosable failure into a slow, expensive one.")

M = [
 dict(
  mech="NUMA / CPU affinity misbinding starving the bootstrap thread",
  why=("Ranks pinned to a socket remote from their GPU and NIC serialise bootstrap work on a "
       "contended core, so communicator creation crawls and appears hung rather than failing."),
  hyp=("H1: the stall is CPU-affinity-induced slowness, not a lost connection. Falsifiable prediction: "
       "every rank's bootstrap thread shows >90% runqueue wait on a single shared core, and running "
       "with one rank per socket (explicit numactl binding) reduces init wall time by a measurable "
       "margin instead of leaving it unbounded."),
  exp=("Controlled experiment: keep world size, image and network path fixed; run arm A with the "
       "current implicit affinity and arm B with explicit numactl --cpunodebind matching each GPU's "
       "PCIe root complex. If arm B completes init while arm A stalls, affinity is causal; if both "
       "stall identically, affinity is excluded. One variable, 3 repeats."),
  meas=["nvidia-smi topo -m to map each GPU to its NUMA node and NIC",
        "per-rank `taskset -pc <pid>` and /proc/<pid>/status Cpus_allowed_list captured at hang time",
        "pidstat -t 1 showing bootstrap-thread %wait and the core it is queued on",
        "init wall-clock time for arm A vs arm B, 3 repeats each, with variance reported"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric never mentions host-side CPU/NUMA effects, so a model imitating it will look only at GPUs and the network and miss affinity-induced stalls.",
         "Applying numactl bindings blindly on a shared node can collide with the scheduler's own cgroup cpuset and degrade co-tenant jobs."],
  rollback=("Rollback gate: if explicit affinity does not cut init wall time by at least a factor of two "
            "in 3 of 3 repeats, revert the binding change - keeping it adds scheduler coupling with no "
            "measured benefit."),
  qd=(3, 2, 2), conf=0.60),
 dict(
  mech="Container /dev/shm undersized, blocking the SHM transport",
  why=("Default 64 MB /dev/shm in a container cannot hold NCCL's intra-node shared-memory buffers, so "
       "rank-local peers block during transport setup with no network error emitted."),
  hyp=("H2: the fault is intra-node shared-memory capacity, not inter-node reachability. Falsifiable "
       "prediction: the hang reproduces with all ranks on ONE node (no network involved), and "
       "NCCL_DEBUG=INFO shows the last line inside SHM/transport setup rather than NET."),
  exp=("Controlled experiment: single node, same rank count; arm A with the current --shm-size, arm B "
       "with --shm-size=16g (or a tmpfs mount of equal size). If arm B initialises and arm A hangs, "
       "shared-memory capacity is causal. Do not simultaneously change NCCL_SHM_DISABLE - that would "
       "confound capacity with transport selection."),
  meas=["df -h /dev/shm inside the container on every node",
        "container runtime spec: --shm-size / --ipc=host setting, captured from the actual launch command",
        "NCCL_DEBUG=INFO last emitted line per rank, showing SHM vs NET stage",
        "single-node reproduction result at identical rank count"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric says nothing about container isolation, so a model imitating it will chase network causes for a purely host-local resource limit.",
         "Setting --ipc=host to work around the limit removes IPC isolation between co-tenant containers and is a security regression, not a fix."],
  rollback=("Rollback gate: if raising /dev/shm to 16 GB does not clear the stall in 2 of 2 repeats, "
            "revert to the original shm size and stop pursuing this branch; do not leave --ipc=host "
            "in place as a residual workaround."),
  qd=(3, 2, 2), conf=0.63),
 dict(
  mech="ECMP / LAG hash collapsing all GPU flows onto one physical link",
  why=("Bootstrap succeeds but the first sizeable collective saturates a single member of a bonded or "
       "ECMP path, so init appears to hang while one link is at line rate and the rest are idle."),
  hyp=("H3: this is a path-imbalance throughput problem, not a connectivity failure. Falsifiable "
       "prediction: the stall disappears at tiny message sizes and returns above a threshold, and "
       "per-interface counters show one member carrying >90% of bytes while peers carry near zero."),
  exp=("Controlled experiment: hold topology fixed and sweep message size (1 KiB -> 1 GiB) in a "
       "standalone all-reduce. If completion time is flat then cliffs at a size boundary, the fault is "
       "bandwidth/path, not rendezvous. Second arm: change only the flow entropy (source port / "
       "NCCL_IB_QPS_PER_CONNECTION or bond xmit_hash_policy) and re-measure."),
  meas=["per-interface rx/tx byte counters sampled at 1 s during the stall on every node",
        "bond/LAG xmit_hash_policy and ECMP hash configuration from the switch and host",
        "all-reduce busbw (GB/s) vs message size curve, 3 repeats",
        "switch port utilisation for the involved uplinks over the same window"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric treats the network as binary up/down and offers no notion of path imbalance, so a model imitating it will declare the network healthy when link counters are non-zero.",
         "Changing hash policy on a production bond affects every tenant on that fabric and must be scheduled, not applied live during triage."],
  rollback=("Rollback gate: if the measured busbw with the new hash policy is not at least 1.5x the "
            "baseline at the failing message size, revert the policy within the same maintenance window; "
            "an unproven fabric change is a standing outage risk."),
  qd=(3, 2, 2), conf=0.58),
 dict(
  mech="GPUDirect RDMA silently disabled, falling back to a staged host-bounce path",
  why=("Without the nvidia-peermem / dmabuf path registered, NCCL stages every transfer through host "
       "memory; init still completes but the first large collective takes long enough to look like a hang."),
  hyp=("H4: transfers are traversing a host bounce buffer rather than GDR. Falsifiable prediction: "
       "NCCL_DEBUG_SUBSYS=NET reports GDRDMA disabled for the relevant NIC, and observed inter-node "
       "bandwidth is bounded well below the NIC line rate while host memory bandwidth and CPU copy "
       "time scale with message size."),
  exp=("Controlled experiment: same job, arm A as-is, arm B with the peer-memory module loaded and "
       "confirmed. If arm B's busbw rises and the stall clears, GDR availability is causal. Setting "
       "NCCL_NET_GDR_LEVEL alone is a probe, not a fix - it cannot enable a path whose kernel module is absent."),
  meas=["lsmod | grep -E 'nvidia_peermem|nv_peer_mem' and dmesg lines at module load on every node",
        "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET lines reporting GDRDMA enabled/disabled per NIC",
        "nvidia-smi topo -m confirming GPU and HCA share a PCIe root complex",
        "inter-node busbw (GB/s) with and without GDR, 3 repeats, against NIC line rate"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric has no concept of a degraded-but-working data path, so a model imitating it will call the job healthy because bytes are moving.",
         "Loading peer-memory modules on a live node can require a driver reload and will kill co-tenant GPU jobs if done without draining."],
  rollback=("Rollback gate: if enabling GDR does not raise measured busbw by at least 30% over the "
            "host-bounce baseline, unload the module and restore the prior driver state during the same "
            "drain window rather than leaving a half-configured node in the pool."),
  qd=(3, 2, 2), conf=0.61),
 dict(
  mech="MTU mismatch between hosts and fabric causing PMTU black-holing",
  why=("Small bootstrap packets pass while large collective payloads exceed a mid-path MTU and are "
       "dropped without ICMP feedback, so the job stalls after rendezvous rather than at connect."),
  hyp=("H5: the failure is size-dependent packet loss, not a down link. Falsifiable prediction: ping "
       "with DF set succeeds at 1500 B and fails at 9000 B between the same pair of nodes, and the "
       "collective completes below a payload threshold and hangs above it."),
  exp=("Controlled experiment: fix the node pair and sweep only payload size with DF-set probes, then "
       "repeat the all-reduce sweep. If both cliff at the same boundary, MTU is causal. Alternatively "
       "set all endpoints to the lowest common MTU and re-run - a completion at reduced MTU confirms it."),
  meas=["ip link show for MTU on every participating interface, host and bond members",
        "ping -M do -s <size> matrix between all node pairs at 1472/8972 payload bytes",
        "switch port MTU configuration for the same ports",
        "tcpdump or NIC drop counters showing large-frame discards during the stall"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric's interface-selection step checks which NIC is used but never whether that NIC's MTU is consistent end to end, so the model will stop one step short.",
         "Lowering MTU cluster-wide to force progress silently costs throughput on every other job and must be recorded as a temporary mitigation, not a fix."],
  rollback=("Rollback gate: if aligning MTU end to end does not clear the stall in 2 of 2 repeats, "
            "restore the original MTU immediately - leaving a reduced MTU in place degrades unrelated "
            "workloads with no established benefit."),
  qd=(3, 2, 2), conf=0.62),
 dict(
  mech="PFC / ECN misconfiguration on RoCE causing pause-storm deadlock",
  why=("Lossless RoCEv2 requires matched PFC priority and ECN marking on host and switch; a mismatch "
       "makes one direction pause indefinitely, freezing the collective with no error surfaced."),
  hyp=("H6: the stall is fabric flow-control deadlock, not application logic. Falsifiable prediction: "
       "PFC pause-frame counters on the involved ports increment monotonically during the stall while "
       "useful byte counters stay flat, and switching the same job to TCP sockets (NCCL_IB_DISABLE=1) "
       "completes, slowly, instead of hanging."),
  exp=("Controlled experiment: hold the job identical and change only the transport, RoCE vs TCP "
       "sockets. Completion under TCP plus rising pause counters under RoCE localises the fault to "
       "flow control. Then verify DSCP/priority mapping symmetry host-vs-switch before re-enabling RoCE."),
  meas=["per-port PFC rx/tx pause frame counters, sampled at 1 s across the stall",
        "host DSCP-to-priority mapping (mlnx_qos or equivalent) vs switch QoS map for the same class",
        "ECN marking and congestion-notification counters on the involved queues",
        "same-job completion time under NCCL_IB_DISABLE=1 as the TCP control arm"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric has no RoCE/lossless-fabric awareness, so a model imitating it will never inspect PFC state and will misattribute a fabric deadlock to NCCL.",
         "Editing PFC or QoS maps on a shared switch can trigger a cluster-wide pause storm; changes require a maintenance window and a pre-agreed revert command."],
  rollback=("Rollback gate: if pause-frame rate does not fall to near zero within 60 s of the QoS "
            "correction, revert the switch configuration from the saved pre-change snapshot immediately "
            "and run the workload on TCP until the fabric owner signs off."),
  qd=(3, 2, 2), conf=0.57),
 dict(
  mech="Mixed collective call order across ranks (non-symmetric communicator usage)",
  why=("If ranks enter collectives in different orders or with different shapes, NCCL blocks waiting "
       "for peers that will never arrive - a pure application-logic deadlock that mimics a network hang."),
  hyp=("H7: the deadlock is caused by asymmetric collective invocation, not by transport. Falsifiable "
       "prediction: TORCH_DISTRIBUTED_DEBUG=DETAIL reports a collective-mismatch or differing op "
       "sequence numbers across ranks, and the standalone all-reduce with identical calls on every "
       "rank completes on the same nodes and same fabric."),
  exp=("Controlled experiment: run the minimal symmetric all-reduce on the exact node set and world "
       "size. Success there plus failure in the real job isolates the fault to application call "
       "structure. Then bisect the training step by logging (op, shape, dtype, seq) per rank and diff "
       "the sequences to find the first divergence."),
  meas=["TORCH_DISTRIBUTED_DEBUG=DETAIL mismatch warnings and per-rank op sequence numbers",
        "per-rank log of (collective op, tensor shape, dtype, sequence index) up to the stall",
        "standalone symmetric all-reduce result on the identical node set and world size",
        "any data-dependent branch in the training loop (dynamic batching, early exit, uneven last batch)"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric frames the problem as an infrastructure fault only, so a model imitating it will escalate to the fabric team for what is an application bug.",
         "Restarting the job to 'clear' the hang destroys the per-rank call-sequence evidence needed to find the divergence; capture logs before any restart."],
  rollback=("Rollback gate: if the symmetric standalone all-reduce passes on the same node set, stop all "
            "infrastructure changes immediately and revert any env-var tuning already applied - further "
            "fabric edits cannot fix an application-side asymmetry and only add variables."),
  qd=(3, 2, 2), conf=0.64),
 dict(
  mech="cgroup / MIG device slicing exposing fewer GPUs than the world size assumes",
  why=("A MIG-partitioned or cgroup-restricted node presents a device set that does not match the "
       "launcher's assumed local rank count, so some ranks bind to a device another rank already owns "
       "and communicator creation never completes."),
  hyp=("H8: the fault is a device-inventory mismatch, not a transport failure. Falsifiable prediction: "
       "the count of visible CUDA devices inside the container is strictly less than nproc_per_node on "
       "at least one node, and two ranks report the same GPU UUID in the rank census."),
  exp=("Controlled experiment: hold everything fixed and set nproc_per_node equal to the actually "
       "visible device count on the most constrained node. If init completes, the inventory mismatch is "
       "causal. Do not change CUDA_VISIBLE_DEVICES and rank count in the same run - that confounds two variables."),
  meas=["nvidia-smi -L and nvidia-smi --query-gpu=uuid,mig.mode.current --format=csv inside the container per node",
        "per-rank CUDA device UUID from the rank census, checked for duplicates",
        "cgroup device allowlist (/sys/fs/cgroup/devices or the scheduler's cgroup spec) per node",
        "launcher nproc_per_node vs visible device count, per node, as an explicit table"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric's 'inspect GPU visibility' step is one line with no duplicate-UUID check, so the most common concrete symptom is left undetected.",
         "Changing MIG mode requires exclusive access to the GPU and will terminate any co-tenant workload; it must never be done as a live triage step."],
  rollback=("Rollback gate: if matching nproc_per_node to visible devices does not clear the stall, "
            "restore the original launcher settings before testing the next hypothesis, so the baseline "
            "stays comparable across arms."),
  qd=(3, 2, 2), conf=0.63),
 dict(
  mech="Clock skew / expired credentials breaking the rendezvous backend mid-handshake",
  why=("A rendezvous backend fronted by an authenticated store (etcd, Redis, object store) can accept "
       "the first ranks and then reject later ones once a token expires or node clocks drift beyond the "
       "allowed window, leaving the world permanently short of members."),
  hyp=("H9: a subset of ranks was rejected by the rendezvous backend, so the world never reaches "
       "quorum. Falsifiable prediction: the store's server-side log shows authentication or lease "
       "errors timestamped inside the stall window, and the number of successfully registered ranks is "
       "strictly less than WORLD_SIZE while every rank process is alive."),
  exp=("Controlled experiment: switch only the rendezvous backend to a static host:port TCPStore with "
       "no authentication, keeping world size, nodes and transport identical. Completion under the "
       "static store isolates the fault to the authenticated backend rather than to NCCL."),
  meas=["registered-member count in the rendezvous store vs WORLD_SIZE, sampled during the stall",
        "rendezvous backend server-side logs filtered to the stall window (auth failures, lease expiry)",
        "chronyc tracking / timedatectl offset on every node, reported in milliseconds",
        "per-rank process liveness (ps) proving no rank crashed while the world stayed short"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric's rendezvous check assumes an unauthenticated store, so a model imitating it has no path to diagnose credential or lease expiry.",
         "Substituting an unauthenticated store is a diagnostic arm only; leaving it in place would remove access control from the cluster's rendezvous path."],
  rollback=("Rollback gate: the unauthenticated static store must be torn down as soon as the arm "
            "completes, in the same session; if the authenticated backend still fails after credential "
            "renewal, escalate to the store owner rather than running production on the open store."),
  qd=(3, 2, 2), conf=0.59),
 dict(
  mech="Silent single-GPU fault (ECC/Xid/fallen-off-the-bus) freezing one rank",
  why=("One unhealthy GPU stops responding while its process stays alive; every other rank blocks in "
       "the collective waiting for a peer that can never complete, so the symptom is a whole-job hang "
       "from a single-device hardware fault."),
  hyp=("H10: exactly one device is faulted and the hang is a downstream symptom. Falsifiable "
       "prediction: dmesg on one node shows an Xid or ECC event timestamped before the stall, and "
       "excluding that single GPU from the node set lets the same job initialise at reduced world size."),
  exp=("Controlled experiment: rerun with the suspect GPU excluded (world size reduced by one, "
       "everything else identical). Success identifies the device; failure exonerates it. Also run a "
       "device-local stress (single-GPU GEMM plus memory test) on the suspect while the rest of the "
       "cluster is idle, so the result is not confounded by fabric load."),
  meas=["dmesg -T | grep -i xid and nvidia-smi -q -d ECC per node, timestamps aligned to the stall",
        "nvidia-smi --query-gpu=uuid,pstate,clocks_throttle_reasons.active,temperature.gpu sampled during the stall",
        "per-rank stack traces (py-spy dump or gdb) showing which rank is inside the collective vs stuck in the driver",
        "reduced-world rerun result with the suspect GPU excluded, 2 repeats"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
         "Rubric's reduced-world step is framed as a scale test, not a device-isolation test, so a model imitating it will not think to exclude a specific GPU.",
         "Repeatedly restarting a job on a GPU throwing Xid errors risks propagating memory corruption into checkpoints; quarantine the device before any further training run."],
  rollback=("Rollback gate: if dmesg shows any Xid on a node, drain that node from the pool immediately "
            "rather than retrying - and do not resume training from a checkpoint written after the first "
            "Xid timestamp until it has been validated against a known-good replica."),
  qd=(3, 2, 2), conf=0.66),
]

assert len(sel) == len(M) == N


def build(rec, m):
    msgs = {x["role"]: x["content"] for x in rec["messages"]}
    u = [x["content"] for x in rec["messages"] if x["role"] == "user"][0]
    a = [x["content"] for x in rec["messages"] if x["role"] == "assistant"][0]
    ca = "\n\n".join([
        ASSUM,
        f"Primary mechanism under test: {m['mech']}. {m['why']}",
        TRIAGE,
        f"Falsifiable hypothesis. {m['hyp']}",
        f"Controlled experiment. {m['exp']}",
        CONF,
        "Measurements to collect: " + "; ".join(m["meas"]) + ".",
        f"{m['rollback']} {ROLLBACK_TAIL}",
    ])
    tc, ic, os_ = m["qd"]
    return {
        "source_id": rec["id"],
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
        "risks": m["risks"],
        "evidence_required": [
            "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH logs from every rank with the last stage identified",
            "TORCH_DISTRIBUTED_DEBUG=DETAIL output including any collective-mismatch warnings",
            "Full rank census: hostname, LOCAL_RANK, PID, CUDA device UUID, image digest",
        ] + m["meas"][:2] + [
            "Control-vs-treatment environment diff proving exactly one variable changed",
        ],
        "confidence": m["conf"],
    }


recs = [build(r, m) for r, m in zip(sel, M)]
import hashlib
h = {hashlib.sha256(r["corrected_answer"].encode()).hexdigest() for r in recs}
assert len(h) == N, "duplicate corrected_answer in batch"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(recs), [r["source_id"] for r in recs])
