#!/usr/bin/env python3
"""Build teacher-B train-batch-0170 (blind provisional review, corpus rows 1691-1700, 0-based 1690..1699)."""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(EXP, "results/train-batch-0170.jsonl")
START, N = 1690, 10

rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")][START:START + N]

HDR = ("Scope: a multi-GPU / multi-node job hangs during collective initialization. Stated assumptions: "
       "PyTorch + NCCL, one process per GPU, c10d/TCPStore rendezvous, homogeneous accelerators, no MIG "
       "unless named. Every number below is tagged ESTIMATE or MEASURED; an ESTIMATE carries its derivation "
       "and must be replaced by a MEASURED value before any rollout or config change is approved.\n\n")

FOOT = ("\nRollback gate: change exactly one variable per iteration and record the baseline as MEASURED first. "
        "If two iterations of the controlled experiment do not move the hang boundary, revert every override "
        "and escalate with the collected artifacts instead of stacking more NCCL_* flags. Debug logging, "
        "NCCL_BLOCKING_WAIT and reduced timeouts alter timing and throughput and must not be left on in "
        "production.\n")

M = [
 ("Asymmetric rendezvous timeout vs. slow container image pull",
  "Hypothesis (falsifiable): no component is broken; ranks simply arrive outside the rendezvous window because "
  "image/layer pull time varies per node, so early ranks time out or block while late ranks are still starting. "
  "Mechanism: TCPStore's barrier waits for WORLD_SIZE check-ins; the default c10d timeout (1800 s ESTIMATE, read "
  "the actual value from the launcher config as MEASURED) is consumed by pull + CUDA context creation, not by any "
  "network fault. Controlled experiment: emit a MEASURED per-rank wall-clock timestamp immediately before "
  "init_process_group, then re-launch with all images pre-pulled and warm page cache on every node. "
  "Discriminating evidence: the spread between earliest and latest pre-init timestamp exceeds the store timeout "
  "on failing runs and collapses to a few seconds on the warm run. Boundary condition: the hang correlates with "
  "cold nodes only and disappears on repeat launches on the same allocation — which no static-config hypothesis "
  "predicts.",
  ["Pre-pulling images cluster-wide can saturate the registry and disrupt other tenants",
   "Raising the rendezvous timeout masks genuine dead ranks and turns a fast failure into a long stall"],
  ["Per-rank MEASURED timestamp just before init_process_group, all ranks",
   "Container runtime pull duration per node",
   "Cold-start vs warm-start launch outcomes on the identical allocation"], 4, 4, 4, 0.6),

 ("GPU clock/ECC state transition blocking CUDA context creation",
  "Hypothesis (falsifiable): the stall is before any NCCL bootstrap — CUDA context creation on one GPU blocks "
  "because the device is in a pending ECC-reset or persistence-mode transition, so that rank never reaches the "
  "collective. Mechanism: nvidia-smi reports 'Pending' ECC or a device in a retired-page/row-remap state; the "
  "driver serializes context creation until the state settles or the GPU is reset. Controlled experiment: run a "
  "one-line torch.zeros(1, device='cuda') probe on every rank with a 60 s watchdog, before any distributed call. "
  "Discriminating evidence: the probe hangs or errors on exactly the ranks that never appear in the NCCL log, and "
  "nvidia-smi -q shows pending retired pages or remap failure on the same device. Boundary condition: the failure "
  "is pinned to a specific physical GPU and follows that GPU across job placements, unlike any topology or "
  "software-version hypothesis.",
  ["Resetting a GPU or toggling persistence mode kills co-resident jobs on that node",
   "Draining a node for ECC remediation reduces capacity and must be scheduled, not done ad hoc"],
  ["nvidia-smi -q ECC and remapped-rows output per device on failing nodes",
   "Per-rank CUDA-context probe exit status with watchdog",
   "Mapping of failure occurrences to physical GPU serial numbers across runs"], 4, 4, 5, 0.58),

 ("Oversubscribed shared memory / IPC namespace limiting NCCL intra-node transport",
  "Hypothesis (falsifiable): NCCL bootstrap starts but the intra-node SHM transport cannot allocate its buffers, "
  "so ring setup never completes. Mechanism: containers commonly default to a 64 MB /dev/shm (ESTIMATE for a "
  "default Docker run; confirm as MEASURED with df -h /dev/shm inside the failing container). NCCL SHM segments "
  "plus dataloader workers exceed it and the allocation blocks or fails silently under some builds. Controlled "
  "experiment: re-launch identical code with --shm-size raised (or --ipc=host) on one node only, and separately "
  "with NCCL_SHM_DISABLE=1 to force the P2P/NET path. Discriminating evidence: the run progresses past init with "
  "SHM disabled or with a larger /dev/shm, while every other variable is unchanged. Boundary condition: this "
  "hypothesis predicts single-node failure at world size 2 on one host; it predicts nothing about cross-node "
  "traffic, so a two-node-one-GPU-each run should succeed.",
  ["NCCL_SHM_DISABLE is a diagnostic, not a fix — it silently degrades intra-node bandwidth if left on",
   "--ipc=host weakens container isolation and should not be a standing production setting"],
  ["df -h /dev/shm MEASURED inside the failing container",
   "NCCL INFO log lines showing which transport was selected per channel",
   "Outcome matrix: default vs enlarged shm vs NCCL_SHM_DISABLE=1"], 4, 4, 4, 0.6),

 ("Routable-but-asymmetric RoCE configuration (PFC/ECN mismatch on one leaf switch)",
  "Hypothesis (falsifiable): the fabric is up and ping/ib_write_bw pass pairwise, yet init stalls because "
  "priority-flow-control settings differ on one leaf, so lossless traffic for the NCCL bootstrap ring is dropped "
  "in one direction. Mechanism: RoCEv2 needs consistent DSCP/PCP-to-priority mapping and PFC enabled on the same "
  "priority end-to-end; a single mismatched port turns congestion into silent drops rather than pause frames, and "
  "the ring closure step waits forever. Controlled experiment: run the all-pairs ib_write_bw matrix with a "
  "sustained (not burst) duration long enough to build congestion, and read per-port PFC and discard counters "
  "before and after. Discriminating evidence: pause-frame counters increment on all ports except the suspect one, "
  "whose discard counters rise instead. Boundary condition: short-burst bandwidth tests pass, so any test shorter "
  "than the congestion onset cannot falsify this hypothesis and must not be used as evidence of health.",
  ["Changing switch QoS affects every tenant on that leaf and needs a change window",
   "Falling back to NCCL_IB_DISABLE=1 hides the fabric defect and collapses throughput"],
  ["Per-port PFC pause and discard counters before/after a sustained test",
   "DSCP/PCP-to-priority map exported from every switch in the path",
   "All-pairs sustained ib_write_bw matrix, MEASURED"], 4, 4, 4, 0.55),

 ("Hostname/DNS resolution asymmetry making rank 0 unreachable by name",
  "Hypothesis (falsifiable): MASTER_ADDR is a hostname that resolves differently (or not at all) on a subset of "
  "nodes, so those ranks connect to the wrong host or block in DNS. Mechanism: c10d resolves MASTER_ADDR per "
  "rank; a stale /etc/hosts entry or a split-horizon DNS view sends some clients to a management-plane address "
  "with no listener, producing an indefinite connect retry with no NCCL output. Controlled experiment: re-launch "
  "with MASTER_ADDR set to rank 0's literal IP on the data-plane interface, changing nothing else. "
  "Discriminating evidence: getent hosts output for the same name differs across nodes, and the literal-IP launch "
  "succeeds. Boundary condition: the failing node set is stable across runs and independent of world size, which "
  "distinguishes it from timeout or straggler hypotheses.",
  ["Hardcoding IPs breaks on rescheduling and must be a diagnostic step, not the permanent fix",
   "Editing /etc/hosts on shared nodes can break other services"],
  ["getent hosts MASTER_ADDR collected on every allocated node",
   "Literal-IP launch outcome vs hostname launch outcome",
   "ss -ltnp on rank 0 confirming the listener's bound interface"], 5, 4, 4, 0.63),

 ("Rank-to-GPU binding collision under a scheduler with partial device isolation",
  "Hypothesis (falsifiable): two local ranks bind to the same CUDA device, so NCCL sees a duplicated device in the "
  "communicator and never completes init. Mechanism: LOCAL_RANK-based torch.cuda.set_device assumes each process "
  "sees all local GPUs; if the scheduler already narrows CUDA_VISIBLE_DEVICES per process, indexing by LOCAL_RANK "
  "maps several ranks onto ordinal 0. Controlled experiment: log the tuple (RANK, LOCAL_RANK, CUDA_VISIBLE_DEVICES, "
  "device UUID from nvidia-smi) on every rank before init and assert UUID uniqueness within each node. "
  "Discriminating evidence: duplicate GPU UUIDs across ranks on the same host. Boundary condition: the failure "
  "appears only when the scheduler's device-isolation mode is on; the same code on a bare-metal launch with all "
  "GPUs visible succeeds, which no fabric hypothesis explains.",
  ["Changing device-binding logic globally can break other jobs that rely on the current convention",
   "Asserting on UUID uniqueness must fail fast and not silently remap devices in production"],
  ["Per-rank (RANK, LOCAL_RANK, CUDA_VISIBLE_DEVICES, GPU UUID) table",
   "Scheduler device-isolation setting for the failing allocation",
   "Bare-metal control launch outcome"], 5, 5, 4, 0.66),

 ("NCCL network plugin absent or version-skewed on a subset of nodes",
  "Hypothesis (falsifiable): nodes disagree on the loaded NCCL net plugin, so one side negotiates a transport the "
  "other cannot serve and bootstrap stalls. Mechanism: an external plugin (libnccl-net.so from a vendor or "
  "SHARP/aws-ofi style stack) is picked up via LD_LIBRARY_PATH; if only some nodes have it, plugin and non-plugin "
  "peers advertise incompatible capabilities. Controlled experiment: dump NCCL INFO NET/Plugin lines from all "
  "ranks and re-launch with the plugin path removed uniformly. Discriminating evidence: the INFO log names a "
  "plugin on some ranks and the built-in socket/IB path on others; the uniform-no-plugin run completes init. "
  "Boundary condition: the failure follows node identity, not GPU or job size, and vanishes when the allocation "
  "is confined to one homogeneous image.",
  ["Removing the plugin can cost real bandwidth on nodes where it is correct — treat as diagnosis only",
   "LD_LIBRARY_PATH edits leak into unrelated jobs sharing the environment module"],
  ["NCCL INFO plugin/transport lines from every rank",
   "ldd / plugin file inventory per node with checksums",
   "Uniform-image control launch outcome"], 4, 4, 4, 0.57),

 ("Cgroup memory limit killing or freezing one rank during pinned-memory registration",
  "Hypothesis (falsifiable): a rank is not hung by NCCL but frozen or reaped by the cgroup while registering "
  "pinned host memory, and the survivors block waiting for it. Mechanism: GDR/IB paths register page-locked host "
  "buffers; the pinned pages count against the container's memory limit, and hitting it triggers reclaim stalls "
  "or the OOM killer. Controlled experiment: watch memory.current / memory.max and memory.events for every rank's "
  "cgroup during init, and re-run with the limit raised on one node only. Discriminating evidence: memory.events "
  "shows oom or high counts on exactly the stuck rank's cgroup, and dmesg records the kill. Boundary condition: "
  "the hang is sensitive to host-memory limit, not to fabric or world size — raising the limit alone must change "
  "the outcome or this hypothesis is falsified.",
  ["Raising memory limits can push the node into system-wide OOM affecting co-tenants",
   "A rank killed by OOM may leave orphaned processes holding GPUs and ports"],
  ["cgroup memory.current/memory.max/memory.events sampled during init",
   "dmesg OOM records correlated with the stuck rank's PID",
   "Raised-limit single-node control run outcome"], 4, 4, 5, 0.56),

 ("Topology-detection stall from an unresponsive PCIe/NVML query path",
  "Hypothesis (falsifiable): NCCL blocks inside topology discovery, not in any peer communication, because an "
  "NVML or sysfs query on a degraded device does not return. Mechanism: NCCL walks /sys PCI topology and queries "
  "NVML for links; a GPU that has fallen off the bus or a driver in an error state makes those reads hang, so no "
  "Bootstrap or Channel lines are ever emitted. Controlled experiment: run nvidia-smi topo -m and a bare "
  "nvidia-smi -q with a timeout on every node before launch, and py-spy/gdb the stuck rank to locate the frame. "
  "Discriminating evidence: nvidia-smi itself times out on the failing node, and the stack shows NCCL inside "
  "topology init rather than a socket wait. Boundary condition: the node fails even for a single-process, "
  "single-GPU NCCL job — which no multi-rank coordination hypothesis predicts.",
  ["Driver reload or node reboot as a remedy destroys evidence needed for the RMA case",
   "Repeatedly polling a wedged NVML can worsen driver state"],
  ["Timed nvidia-smi topo -m and nvidia-smi -q results per node",
   "py-spy/gdb stack of the stuck rank showing the blocking frame",
   "Single-process single-GPU NCCL control result on the suspect node",
   "dmesg Xid entries around the failure window"], 4, 4, 5, 0.59),

 ("Clock skew / expired credential breaking the launcher's control plane mid-rendezvous",
  "Hypothesis (falsifiable): the ranks are healthy but the launcher's coordination layer (etcd-style rendezvous "
  "backend or an authenticated store) rejects late joiners because node clocks are skewed or a token expired, so "
  "the group never forms. Mechanism: TLS/token validation and lease TTLs are time-sensitive; a node minutes off "
  "NTP has its lease considered expired on arrival and silently drops out of the membership set. Controlled "
  "experiment: collect chronyc tracking / timedatectl offsets from every node and re-launch after forcing NTP "
  "sync, plus one run with a freshly issued short-lived credential. Discriminating evidence: the rendezvous "
  "backend logs a rejected or expired lease for exactly the missing rank, and offsets exceed the lease TTL. "
  "Boundary condition: the failing rank varies with which node is most skewed, not with GPU or fabric identity.",
  ["Force-stepping the clock on a running node can disturb other time-sensitive services",
   "Credentials must never be printed into shared job logs while debugging this"],
  ["Per-node NTP offset MEASURED at launch time",
   "Rendezvous backend server-side membership/lease logs for the run id",
   "Post-sync control launch outcome"], 4, 4, 5, 0.54),
]

assert len(M) == len(rows)

recs = []
for row, (topic, body, risks, evid, tc, ic, ops, conf) in zip(rows, M):
    u = next(m["content"] for m in row["messages"] if m["role"] == "user")
    a = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    ca = HDR + "Primary root-cause hypothesis under test: " + topic + ".\n\n" + body + "\n" + FOOT
    recs.append({
        "source_id": row["id"],
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
            "operational_safety": ops,
        },
        "risks": risks,
        "evidence_required": evid,
        "confidence": conf,
    })

import hashlib
h = [hashlib.sha256(r["corrected_answer"].encode()).hexdigest() for r in recs]
assert len(set(h)) == len(h), "duplicate corrected_answer in batch"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs))
