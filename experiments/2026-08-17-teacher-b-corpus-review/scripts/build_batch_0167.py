#!/usr/bin/env python3
"""Build teacher-B train-batch-0167 (blind provisional review, rows 1661-1670)."""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(EXP, "results/train-batch-0167.jsonl")
START, N = 1660, 10  # 0-based positional start

rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")][START:START + N]

HDR = ("Assumptions (stated, not assumed silently): single job, PyTorch + NCCL, one process per GPU, "
       "rendezvous via c10d/TCPStore, homogeneous GPUs, no MIG. All numbers below are ESTIMATE unless "
       "explicitly tagged MEASURED; ESTIMATEs are derived from the stated formula and must be replaced by "
       "MEASURED values before any rollout decision.\n\n")

FOOT = ("\nRollback gate: apply at most one variable change per experiment iteration; if the controlled "
        "experiment does not move the hang boundary within two iterations, revert every env override to the "
        "known-good baseline (recorded as MEASURED before the first change) and escalate with the collected "
        "artifacts rather than stacking further NCCL_* flags. Safety: never leave debug-level logging or "
        "NCCL_BLOCKING_WAIT enabled in production — both change timing and throughput.\n")

# Ten distinct root-cause mechanisms; one per variant, so the batch is not a template.
M = [
 ("Rendezvous / TCPStore mismatch",
  "Hypothesis (falsifiable): the hang is at c10d rendezvous, not inside NCCL — ranks disagree on "
  "MASTER_ADDR/MASTER_PORT/WORLD_SIZE, so init_process_group never returns on at least one rank. "
  "Mechanism: TCPStore has one server (rank 0) and N-1 clients; a client pointed at the wrong host or a "
  "port blocked by a firewall blocks in a store barrier with no NCCL traffic ever emitted. "
  "Controlled experiment: run the identical launcher but replace the model step with dist.init_process_group "
  "followed immediately by dist.barrier() and exit; add a 60 s store timeout. If the barrier-only job also "
  "hangs, NCCL is exonerated. Discriminating evidence: py-spy dump on the stuck rank shows the frame inside "
  "TCPStore/PrefixStore rather than ncclCommInitRank; no NCCL INFO Bootstrap lines are printed at all. "
  "Boundary condition: this failure is world-size independent — it reproduces at world size 2 across two nodes "
  "but never on a single node.",
  ["Firewall/port changes touch other tenants of the node",
   "Lowering the store timeout can turn a slow-but-healthy startup into a spurious crash"],
  ["Per-rank env dump (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) captured before init",
   "py-spy dump of every stuck PID",
   "ss -ltnp on rank-0 host showing the store port listening",
   "Barrier-only reproducer exit codes for all ranks"], 4, 4, 4, 0.62),

 ("Mismatched collective order / shape across ranks",
  "Hypothesis (falsifiable): all ranks reached NCCL, but they enqueued different collectives (or the same "
  "collective with different element counts/dtypes), so each rank waits for peers that will never post the "
  "matching operation. Mechanism: NCCL collectives are matched positionally per communicator; there is no "
  "tag, so an extra all_reduce on rank 0 (e.g. a metric logged only on rank 0, or a conditional branch on "
  "data-dependent batch shape) desynchronises the stream permanently. "
  "Controlled experiment: set TORCH_NCCL_DESYNC_DEBUG=1 (or the legacy NCCL_DESYNC_DEBUG) and "
  "TORCH_NCCL_ASYNC_ERROR_HANDLING=1 with a 300 s watchdog; the watchdog then reports the last enqueued "
  "collective per rank. Rerun with all rank-conditional logging removed. Discriminating evidence: the watchdog "
  "trace shows different opCount or different numel per rank at the point of the hang. "
  "Boundary condition: hang time is data dependent — it survives node/topology changes but disappears when "
  "the same fixed batch is replayed on every rank.",
  ["DESYNC_DEBUG adds per-collective bookkeeping and perturbs timing",
   "Watchdog aborts kill the job, so checkpoint before enabling it"],
  ["Watchdog desync report listing last collective + opCount per rank",
   "Diff of the enqueued-collective sequence between rank 0 and a non-zero rank",
   "Deterministic fixed-batch replay result"], 5, 4, 4, 0.66),

 ("Wrong NIC / interface selection",
  "Hypothesis (falsifiable): NCCL selected an unusable interface (docker0, a loopback alias, or a management "
  "NIC without a route between nodes) for bootstrap or for the data path, so ring construction never completes. "
  "Mechanism: NCCL enumerates interfaces and picks by its own preference order; a container bridge that exists "
  "on every node with the same 172.17.0.0/16 subnet looks routable locally but cannot reach a peer. "
  "Controlled experiment: run once with NCCL_SOCKET_IFNAME set explicitly to the known data NIC and "
  "NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,NET; compare against the unpinned baseline. "
  "Discriminating evidence: the INIT log line naming the chosen interface differs between the hanging run and "
  "the healthy run; the hanging run names a bridge/management device. "
  "Boundary condition: single-node runs pass because loopback works; the hang appears only when ranks span "
  "hosts.",
  ["Pinning IFNAME wrongly can silently push traffic onto a 1 GbE management link, converting a hang into a "
   "large but non-obvious slowdown",
   "IFNAME must be re-validated after any host re-imaging"],
  ["NCCL INIT/NET log lines showing selected interface on every rank",
   "ip -br addr and ip route from two different nodes",
   "MEASURED all-reduce busbw (GB/s) for pinned vs unpinned interface"], 5, 4, 4, 0.68),

 ("IB/RoCE fabric down, silent fallback or stall",
  "Hypothesis (falsifiable): the RDMA path is unhealthy (port DOWN, wrong PKey, GID index or ECN/PFC "
  "misconfiguration on RoCEv2) and NCCL neither succeeds nor cleanly falls back, so the transport connect "
  "phase stalls. Mechanism: NCCL's net plugin probes IB devices at init; if a device is present but the port "
  "is not ACTIVE, or the RoCE GID chosen does not match the peer's L3 subnet, connection establishment blocks "
  "until the (long) transport timeout. "
  "Controlled experiment: A/B a two-rank two-node job with NCCL_IB_DISABLE=1 (TCP path) versus the default. "
  "If TCP completes and IB hangs, the fabric is implicated with no ambiguity. Then test NCCL_IB_HCA and "
  "NCCL_IB_GID_INDEX pinned to the values reported by show_gids. "
  "Discriminating evidence: ibstat port state, and a two-node ib_write_bw run that fails or reports far below "
  "line rate. Baseline for comparison must be MEASURED on a known-good node pair, not assumed. "
  "Boundary condition: disabling IB should make the job slow but alive; if it still hangs, this hypothesis is "
  "refuted.",
  ["NCCL_IB_DISABLE=1 in production silently costs most of the interconnect bandwidth",
   "Changing PFC/ECN settings affects the whole switch domain, not just this job"],
  ["ibstat / ibv_devinfo port state and rate per node",
   "show_gids output and the GID index actually used",
   "ib_write_bw MEASURED bandwidth between the two nodes",
   "Switch counters for pause frames and ECN marks over the hang window"], 5, 5, 4, 0.63),

 ("GPU visibility / device-ordinal collision",
  "Hypothesis (falsifiable): two or more local ranks bound to the same GPU, or a rank saw zero GPUs, because "
  "CUDA_VISIBLE_DEVICES was set both by the scheduler and by the launcher. Mechanism: when a rank calls "
  "cudaSetDevice(local_rank) inside an already-masked visibility set, local_rank 1 can resolve to the same "
  "physical device as local_rank 0; NCCL then builds a communicator with duplicate devices and deadlocks in "
  "init. "
  "Controlled experiment: print (RANK, LOCAL_RANK, CUDA_VISIBLE_DEVICES, torch.cuda.current_device(), GPU UUID) "
  "on every rank before init_process_group, then rerun with the launcher's own masking removed so only the "
  "scheduler sets visibility. Discriminating evidence: duplicate GPU UUIDs across ranks in the pre-init dump. "
  "Boundary condition: purely local — reproduces on a single node with 2 GPUs and is unaffected by any network "
  "setting.",
  ["Clearing CUDA_VISIBLE_DEVICES can let a job escape its cgroup GPU allocation and disturb co-tenants"],
  ["Per-rank GPU UUID table from nvidia-smi --query-gpu=uuid",
   "Pre-init dump of visibility variables and current_device per rank",
   "nvidia-smi process list showing one process per physical GPU"], 5, 4, 5, 0.7),

 ("P2P / IOMMU / ACS blocking intra-node transport",
  "Hypothesis (falsifiable): intra-node peer-to-peer is advertised but not functional (PCIe ACS enabled, or "
  "IOMMU in strict mode), so NCCL selects a P2P path that never completes handshake. Mechanism: with ACS "
  "redirect on, GPU-to-GPU DMA is forced up to the root complex and can fail or serialise; NCCL's P2P "
  "capability probe may still report peer access as available. "
  "Controlled experiment: run with NCCL_P2P_DISABLE=1. If init completes, the P2P path is the cause. Confirm "
  "independently with CUDA samples p2pBandwidthLatencyTest and with nvidia-smi topo -m. "
  "Discriminating evidence: p2pBandwidthLatencyTest hangs or reports ~PCIe-host bandwidth where NVLink is "
  "expected; lspci shows ACSCtl with SrcValid+ on the upstream ports. "
  "Boundary condition: single-GPU runs always pass; the hang appears at 2 GPUs on one node, which cleanly "
  "separates it from any inter-node hypothesis.",
  ["Disabling ACS weakens PCIe isolation and is a host-wide, security-relevant change requiring reboot",
   "NCCL_P2P_DISABLE=1 is a diagnostic, not a fix — it costs intra-node bandwidth"],
  ["nvidia-smi topo -m matrix",
   "lspci -vvv ACSCtl state on upstream switch ports",
   "p2pBandwidthLatencyTest MEASURED matrix",
   "A/B init outcome with and without NCCL_P2P_DISABLE"], 4, 4, 4, 0.6),

 ("Straggler rank: one process never launched or died before init",
  "Hypothesis (falsifiable): the job is not deadlocked at all — N-1 ranks are correctly blocked waiting for one "
  "rank that was never scheduled, was OOM-killed, or crashed before reaching init_process_group. Mechanism: "
  "collective init is an all-or-nothing barrier; a missing rank is indistinguishable from a hang unless you "
  "count live processes. "
  "Controlled experiment: count processes per node against the expected world size, then rerun with an "
  "init_process_group timeout of 120 s so the launcher surfaces which rank is absent instead of waiting the "
  "default 30 min. Discriminating evidence: process count < world size; dmesg shows an oom-kill for the missing "
  "PID; the scheduler shows fewer allocated tasks than requested. "
  "Boundary condition: reducing world size to fit within confirmed-healthy nodes makes the job start normally, "
  "which refutes every transport-level hypothesis.",
  ["Short init timeouts convert transient scheduler slowness into false failures on large clusters",
   "Re-running to reproduce consumes the same allocation that may itself be the constrained resource"],
  ["Per-node process count vs expected world size",
   "dmesg / journalctl oom-kill records over the launch window",
   "Scheduler task state for every rank",
   "Exit codes of all ranks after the shortened timeout"], 5, 5, 5, 0.72),

 ("Version / ABI skew across nodes",
  "Hypothesis (falsifiable): nodes run different NCCL, CUDA driver, or container image versions, so ranks "
  "negotiate incompatible protocols and stall during bootstrap. Mechanism: NCCL requires compatible versions "
  "across a communicator; a mixed 2.x minor set can still connect at bootstrap but disagree on algorithm or "
  "protocol selection, producing a hang rather than an error. "
  "Controlled experiment: collect torch.cuda.nccl.version(), nvidia-smi driver version, and the image digest "
  "from every rank; then constrain the job to a set of nodes with byte-identical image digests. "
  "Discriminating evidence: a version/digest table with any disagreement, plus a successful run on the "
  "homogeneous subset. Boundary condition: the hang should follow the specific node, not the rank index — pin "
  "the suspect node to a different rank and confirm the failure moves with the node.",
  ["Draining nodes to homogenise images reduces cluster capacity",
   "Driver upgrades require host reboot and coordination with other tenants"],
  ["Per-rank table of NCCL version, CUDA driver, container image digest",
   "Successful run confined to the homogeneous node subset",
   "Node-pinning experiment showing the failure follows the node"], 5, 4, 4, 0.64),

 ("CPU/GPU affinity and cgroup starvation during bootstrap",
  "Hypothesis (falsifiable): bootstrap is not deadlocked but starved — the cgroup CPU quota or a bad NUMA "
  "binding leaves the NCCL bootstrap and proxy threads without CPU, so init exceeds the observation window and "
  "looks like a hang. Mechanism: NCCL bootstrap is CPU-bound socket work; with, say, 4 cores shared by 8 ranks "
  "plus dataloader workers, per-rank bootstrap time scales roughly linearly with oversubscription. ESTIMATE: at "
  "8x oversubscription a normally ~5 s bootstrap becomes ~40 s (derivation: bootstrap_time ≈ base_time × "
  "oversubscription_factor); this must be replaced with a MEASURED init duration before being believed. "
  "Controlled experiment: time init_process_group explicitly, then rerun with dataloader workers set to 0 and "
  "the cgroup quota raised. Discriminating evidence: init eventually completes given enough time — that alone "
  "refutes every true-deadlock hypothesis. Boundary condition: progress is monotonic, so cross-node NCCL "
  "counters keep advancing during the stall.",
  ["Raising cgroup quota can starve co-tenant jobs on the same node",
   "Aggressive CPU pinning can hurt dataloader throughput in steady state"],
  ["MEASURED wall-clock duration of init_process_group per rank",
   "cgroup cpu.max / cpu.stat throttling counters during the stall",
   "numactl --hardware and the actual thread affinity mask per rank",
   "Whether the job eventually starts when left running for 30 min"], 4, 4, 4, 0.58),

 ("Stale communicator state after a prior failed run",
  "Hypothesis (falsifiable): the current job inherits residue from a previous crashed run — orphaned processes "
  "still holding the GPUs and the store port, or a stale rendezvous key in the shared backing store — so new "
  "ranks join a half-dead group. Mechanism: NCCL communicators and the TCPStore are not garbage collected when "
  "a job is SIGKILLed; a leftover rank-0 process keeps the port bound and answers rendezvous with an obsolete "
  "world. "
  "Controlled experiment: on every allocated node, enumerate GPU-holding processes and store-port listeners "
  "before launch; then launch on a freshly rebooted or fully drained node set with a unique run_id in the "
  "rendezvous endpoint. Discriminating evidence: nvidia-smi shows compute processes with no corresponding "
  "scheduler task; the clean-slate launch succeeds while the reuse launch hangs. "
  "Boundary condition: the failure is not reproducible on first-boot nodes, which distinguishes it from any "
  "static configuration hypothesis.",
  ["Killing orphaned PIDs may terminate another tenant's legitimate job — verify ownership first",
   "Node reboots as a routine remedy hide the real leak and inflate queue times"],
  ["nvidia-smi compute process list cross-referenced against scheduler task IDs",
   "ss -ltnp showing who holds the store port before launch",
   "Clean-slate vs reuse launch outcomes with a unique run_id"], 4, 4, 5, 0.61),
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

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs))
