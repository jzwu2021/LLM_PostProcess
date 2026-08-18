#!/usr/bin/env python3
"""teacher-B blind review generator for train-batch-0166 (corpus rows 1651-1660)."""
import json, hashlib, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0166.jsonl")
START, COUNT = 1650, 10

MECH = [
 dict(
  name="Topology detection: NCCL_TOPO/ACS misconfiguration forces a PCIe path that stalls P2P setup",
  hyp="H1: intra-node transport setup blocks because PCIe ACS is enabled (or an injected NCCL_TOPO_FILE misdescribes the machine), so cudaDeviceEnablePeerAccess is refused for some GPU pairs and NCCL retries the P2P path indefinitely. Falsifiable: `lspci -vvv | grep -i 'ACSCtl'` shows SrcValid+ on upstream ports, p2pBandwidthLatencyTest reports 0 GB/s for the affected pairs, and NCCL_DEBUG=INFO stops after 'Channel .. via P2P'.",
  exp="Controlled experiment: (a) run CUDA sample p2pBandwidthLatencyTest and record the peer-access matrix; (b) rerun the minimal all-reduce with NCCL_P2P_DISABLE=1 and, separately, with ACS disabled on the root ports. Prediction under H1: the peer matrix has zeros exactly on the pairs where NCCL stalls, the P2P_DISABLE run initializes (slower), and disabling ACS makes the default path work.",
  meas="Peer-access matrix (bool) and MEASURED P2P bandwidth GB/s per pair; ACSCtl bits per upstream port; NCCL last log line; nvidia-smi topo -m output.",
  conf="Virtualized/passthrough hosts require ACS for isolation and will show the same matrix legitimately; NCCL_P2P_DISABLE=1 'fixes' the hang while hiding the topology fault.",
  risk=["Disabling ACS weakens IOMMU isolation between PCIe devices and is unacceptable on multi-tenant hosts", "NCCL_P2P_DISABLE=1 can cut intra-node all-reduce bandwidth by a large factor"],
  ev=["p2pBandwidthLatencyTest full matrix", "lspci ACSCtl dump per root port", "nvidia-smi topo -m", "host tenancy/isolation policy statement"],
  rb="Rollback gate: treat NCCL_P2P_DISABLE=1 as diagnostic only. If intra-node all-reduce bus bandwidth after any topology change is below 80% of the MEASURED pre-incident nccl-tests baseline (ESTIMATE tolerance; baseline must be measured first), revert and escalate to the host owner."),
 dict(
  name="cgroup/pinned-memory limit: cudaHostRegister blocks under a memlock or memory.max ceiling",
  hyp="H1: the process is not deadlocked on the network but blocked/thrashing while pinning host memory for NCCL buffers: RLIMIT_MEMLOCK is low (default 64 KB in some container runtimes) or the cgroup memory.max is near the working set, so pinned allocation stalls or triggers reclaim. Falsifiable: `cat /proc/<pid>/limits` shows a small Max locked memory, and cgroup memory.events shows rising 'high'/'max' counters exactly during init.",
  exp="Controlled experiment: (a) record limits and cgroup counters during the hang; (b) rerun with --ulimit memlock=-1 (and IPC unchanged) on one node only. Prediction under H1: the raised-memlock node initializes while a control node with the original limit still stalls.",
  meas="RLIMIT_MEMLOCK value per rank; cgroup memory.current / memory.max / memory.events deltas; /proc/meminfo Mlocked during init; time spent in cudaHostRegister from a py-spy sample.",
  conf="A network stall can coexist and dominate; raising memlock also changes allocation timing and can mask a slow-storage checkpoint load.",
  risk=["Unlimited memlock lets one job pin enough host RAM to destabilize co-tenants", "Approaching cgroup memory.max risks OOM-kill of the trainer mid-run"],
  ev=["/proc/<pid>/limits for every rank", "cgroup v2 memory.events before/after", "py-spy stacks showing the blocking frame", "host free memory timeline"],
  rb="Rollback gate: raise memlock stepwise (e.g. 16 GB) rather than unlimited; if host available memory falls below 10% (MEASURED from /proc/meminfo) revert to the previous limit immediately."),
 dict(
  name="MTU / PFC mismatch on the RoCE fabric: connection setup completes but the first large message never lands",
  hyp="H1: the apparent init hang is really the first sizeable NCCL message being silently dropped: end-host MTU differs from the switch fabric MTU, or PFC is enabled on one side only, so small control packets pass and larger ones are dropped. Falsifiable: `ping -M do -s 8972` between the pair fails while a small ping succeeds, and switch/NIC counters show rx_discards or pause-frame asymmetry incrementing during the attempt.",
  exp="Controlled experiment: (a) size-swept ping / ib_write_bw across message sizes between two hanging nodes; (b) set both endpoints to the switch-configured MTU and rerun. Prediction under H1: there is a sharp size threshold at which transfers fail, and equalizing MTU removes it.",
  meas="Largest successful ping payload per node pair; ib_write_bw MEASURED bandwidth by size; ethtool -S rx_discards/pause counters delta; switch port MTU and PFC config.",
  conf="A congested fabric drops large messages intermittently without any MTU error; ECN marking can throttle rather than drop and look like a slow hang.",
  risk=["Changing MTU on a live NIC resets the link and drops other traffic on that host", "Enabling PFC incorrectly can cause head-of-line blocking or fabric-wide pause storms"],
  ev=["size-swept reachability results", "ethtool -S counter deltas both endpoints", "switch port configuration export", "change window approval for any NIC/switch edit"],
  rb="Rollback gate: apply the MTU change to a single node pair first; if pairwise nccl-tests all-reduce does not complete or bandwidth is below 50% of the MEASURED baseline, revert the MTU and escalate to the network owner instead of changing the fleet."),
 dict(
  name="Storage stall masquerading as a hang: ranks block loading weights/dataset from a saturated shared filesystem",
  hyp="H1: no collective fault exists; ranks are blocked in filesystem reads (NFS/Lustre/S3-FUSE) before or during init, and because the collective has no timeout the run appears hung. Falsifiable: py-spy shows ranks in read/mmap frames, `cat /proc/<pid>/stack` or wchan shows uninterruptible IO wait, and per-rank progress correlates with storage client throughput rather than with node or NIC identity.",
  exp="Controlled experiment: (a) sample py-spy and /proc/<pid>/io byte counters every 10 s for 2 minutes; (b) rerun with weights pre-staged on node-local NVMe for two ranks only. Prediction under H1: read_bytes advances slowly during the 'hang' and the locally-staged ranks reach init far sooner.",
  meas="/proc/<pid>/io read_bytes rate per rank; storage client stats (nfsstat / lfs df / mount latency); time-to-init per rank; py-spy stack census.",
  conf="A single slow rank due to a degraded GPU produces a similar per-rank asymmetry; page cache warmth makes a re-run look artificially fast.",
  risk=["Pre-staging to local NVMe can exhaust node disk and fail other jobs", "Adding read parallelism can further saturate a shared filesystem used by the whole cluster"],
  ev=["/proc/<pid>/io time series per rank", "storage client latency metrics", "py-spy stacks", "node-local disk free space before staging"],
  rb="Rollback gate: if staging locally does not reduce time-to-init by at least 2x (ESTIMATE; compare against the MEASURED shared-FS load time), stop staging, free the local copies, and move to the collective-side hypotheses."),
 dict(
  name="Mixed process groups / non-uniform collective call order across ranks",
  hyp="H1: ranks do not execute the same sequence of collective-creating calls (a conditional branch creates a subgroup on some ranks only, or ranks initialize sub-process-groups in different orders), so the communicators pair up incorrectly and init blocks. Falsifiable: instrumenting new_group/init calls with an ordered per-rank log shows divergent sequences; TORCH_DISTRIBUTED_DEBUG=DETAIL reports a collective mismatch.",
  exp="Controlled experiment: emit a per-rank ordered trace of every process-group creation and collective entry, then rerun with all conditional group creation removed (single global group only). Prediction under H1: traces diverge at a specific call index, and the single-group run initializes cleanly.",
  meas="Per-rank ordered call trace with timestamps; TORCH_DISTRIBUTED_DEBUG=DETAIL mismatch reports; index of first divergence; pass/fail of the single-group control.",
  conf="A rank that is merely slow produces a trace that looks truncated rather than divergent; logging itself perturbs timing.",
  risk=["Removing subgroup creation changes the parallelism strategy and therefore throughput and memory footprint", "Instrumentation left in production adds per-collective overhead"],
  ev=["ordered per-rank traces", "DEBUG=DETAIL output", "the code path/condition that gates group creation", "diff of the control configuration"],
  rb="Rollback gate: the single-group control is a diagnostic configuration only. Revert to the intended parallelism before any production run, and re-verify step-time is within 10% (ESTIMATE) of the MEASURED baseline."),
 dict(
  name="Zombie processes from a prior run holding GPUs and the bootstrap port",
  hyp="H1: leftover processes from a previous failed launch still hold GPU contexts and/or bind MASTER_PORT, so new ranks either cannot bind the store or cannot get device memory, and init blocks. Falsifiable: `nvidia-smi --query-compute-apps` lists PIDs older than this launch, and `ss -tlnp sport = :29500` shows an owner whose start time predates the job.",
  exp="Controlled experiment: census processes and port owners before launching; then relaunch after cleanly terminating only the confirmed stale PIDs from the previous job id. Prediction under H1: stale PIDs exist, GPU memory is non-zero before launch, and the post-cleanup launch initializes.",
  meas="Process start times vs launch time; per-GPU used memory before launch (expect ~0); port owner PID; count of stale PIDs per node.",
  conf="A co-tenant job legitimately occupies GPUs and ports; killing by port owner alone can hit an unrelated service.",
  risk=["Killing the wrong PID terminates another tenant's training run", "Force-killing a process mid-cudaMalloc can leave the GPU in a state requiring a reset"],
  ev=["ps -o pid,lstart,cmd for candidate PIDs", "nvidia-smi compute-apps before launch", "scheduler record linking PIDs to the previous job id", "confirmation that no co-tenant owns those PIDs"],
  rb="Rollback gate: never issue a blanket pkill or `nvidia-smi --gpu-reset` on a shared node. If ownership of a stale PID cannot be attributed to the previous job id with scheduler evidence, stop and escalate rather than killing."),
 dict(
  name="Time skew / expired credentials break the rendezvous backend (etcd, Kubernetes API, or TLS)",
  hyp="H1: the rendezvous backend rejects or silently drops registrations because node clocks are skewed beyond the TLS/lease tolerance or a service-account token expired, so ranks never all register. Falsifiable: `chronyc tracking` shows offset beyond the backend's tolerance on the failing nodes, and the backend logs TLS 'certificate is not yet valid' / lease-expiry errors correlated with those ranks.",
  exp="Controlled experiment: (a) collect clock offset and token expiry from all nodes; (b) resync NTP on the outlier node and relaunch. Prediction under H1: exactly the non-registering ranks are the skewed/expired ones and they register after resync.",
  meas="chronyc offset per node (ms); token/cert notBefore-notAfter vs current time; backend registration count vs world_size; backend server-side error log lines.",
  conf="A network partition to the backend produces identical symptoms with correct clocks; token refresh may happen automatically and mask the window.",
  risk=["Stepping the clock on a running node can disrupt other time-sensitive services and log ordering", "Rotating credentials affects every workload using that service account"],
  ev=["clock offset table for all nodes", "credential validity window", "rendezvous backend server logs", "registration count timeline"],
  rb="Rollback gate: prefer slewing over stepping the clock on production nodes; if a step is required, drain the node first. If registration still fails after resync, revert credential changes and treat the backend network path as the next hypothesis."),
 dict(
  name="Power/thermal capping or a GPU stuck in a bad state delays CUDA context creation on one node",
  hyp="H1: one node's GPUs are in a degraded state (persistence mode off plus slow context creation, clocks capped by HW power brake, or a pending Xid-triggered reset), so its ranks take minutes to reach init and the rest wait unboundedly. Falsifiable: `nvidia-smi -q -d PERFORMANCE` shows active throttle reasons (HW Power Brake / SW Thermal Slowdown) on that node, dmesg contains Xid entries timestamped near launch, and time-to-CUDA-context on that node exceeds the fleet median by a large margin.",
  exp="Controlled experiment: (a) time a trivial `torch.zeros(1).cuda()` on every node and rank the results; (b) exclude the slowest node and relaunch at reduced world size. Prediction under H1: one node is a clear outlier and the reduced-world run initializes normally.",
  meas="MEASURED time-to-first-CUDA-context per node (seconds); throttle reasons and current vs max SM clocks; dmesg Xid census; inlet temperature and power draw per GPU.",
  conf="Cold caches make the first node timed look slow regardless; a datacenter-wide thermal event affects many nodes and is not a per-node fault.",
  risk=["Continuing to train on a GPU throwing Xid errors risks silent numerical corruption of the checkpoint", "Excluding nodes reduces world size and changes global batch size"],
  ev=["per-node context-creation timing table", "nvidia-smi -q performance and ECC sections", "dmesg Xid lines with timestamps", "facility thermal/power telemetry"],
  rb="Rollback gate: if Xid errors indicating uncorrectable faults are present, cordon the node and do not return it to the pool until it passes a burn-in; if excluding the node changes global batch size, restore the intended size by adjusting gradient accumulation and record the change rather than silently altering training dynamics."),
 dict(
  name="Disaggregated serving stack (Mooncake / NVIDIA Dynamo) confuses transfer-engine handshake with training-collective init",
  hyp="H1: the hanging component is not the training process group but the KV-transfer / disaggregation control plane sharing the same host: a Mooncake-style transfer engine or Dynamo worker registration is blocking on RDMA device or metadata-service discovery, and its stall is being reported as 'collective initialization'. Falsifiable: the training ranks' NCCL logs reach 'Connected all rings' while the stall appears in the transfer-engine/registration logs; a standalone nccl-tests all-reduce on the same nodes passes.",
  exp="Controlled experiment: run the minimal all-reduce probe alone on the same allocation with the serving stack stopped, then start the serving stack alone. Prediction under H1: the bare all-reduce passes, and the stall reproduces only when the transfer engine / worker registration runs.",
  meas="Component-attributed last log line; nccl-tests pass/fail and MEASURED bus bandwidth without the serving stack; metadata-service (etcd) registration count; RDMA device availability seen by each component.",
  conf="Both components can be blocked by one shared cause (e.g. the same RDMA device misconfiguration), so a passing all-reduce over TCP does not exonerate the fabric.",
  risk=["Stopping the serving stack drops live inference traffic if the node is serving production requests", "Sharing one RDMA device between trainer and transfer engine can starve either of queue-pair resources"],
  ev=["separate logs per component with timestamps", "nccl-tests result on the clean allocation", "metadata-service registration dump", "traffic-drain confirmation before stopping any serving component"],
  rb="Rollback gate: drain serving traffic before any stop/start test, and restore the serving stack immediately afterwards; if disabling RDMA for the transfer engine is used as a probe, revert it in the same session because the TCP fallback path is not a supported production configuration."),
 dict(
  name="Observability gap: no timeout, no debug logging — the true failure is unobservable and every hypothesis is unfalsifiable",
  hyp="H1 (meta-hypothesis, test first when logs are absent): the run provides no evidence because NCCL_DEBUG is unset, no init timeout is configured, and stdout is buffered, so the actual fault cannot be localized at all. Falsifiable: relaunching with NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH, TORCH_DISTRIBUTED_DEBUG=DETAIL, PYTHONUNBUFFERED=1 and an explicit init timeout produces a concrete last-line and a named rank within the timeout window.",
  exp="Controlled experiment: relaunch unchanged except the observability variables and a bounded init timeout (e.g. 10 minutes). Prediction under H1: the run now fails fast with a specific rank and phase, converting an unfalsifiable hang into a testable fault; if it instead succeeds, suspect a timing-sensitive race and record that as new information.",
  meas="Presence and content of the last NCCL log line per rank; rank named by the timeout exception; wall time to failure; whether the run succeeds with logging enabled (race indicator).",
  conf="Verbose logging perturbs timing and can hide a race; a run that succeeds under logging proves nothing about the original fault.",
  risk=["NCCL_DEBUG=INFO at large world size generates high log volume and can fill node disks", "A short init timeout can abort healthy runs whose checkpoint load legitimately exceeds it"],
  ev=["per-rank logs with the last emitted line", "timeout exception text naming the rank", "disk headroom check before enabling verbose logging", "record of whether the fault reproduced under logging"],
  rb="Rollback gate: this step adds observability, not a fix — do not declare resolution because the logged run happened to pass. Remove verbose logging before production runs, keep the explicit timeout, and if the fault did not reproduce, re-run the original configuration at least twice to establish the intermittency rate before closing."),
]

HEADER = ("Assumptions (state and verify before acting): multi-node, multi-GPU PyTorch + NCCL job; the reported hang "
          "is at process-group / communicator initialization, before the first optimizer step; no rank has exited "
          "with an error; you have shell access on all nodes and permission to relaunch with modified environment "
          "variables. Every numeric threshold below is an ESTIMATE unless the text says a baseline must be MEASURED "
          "first; ESTIMATE values are starting points to be replaced by your own measurements.\n\n"
          "Step 0 - freeze evidence before mutating anything. Per rank capture: hostname, PID, RANK, LOCAL_RANK, "
          "WORLD_SIZE, MASTER_ADDR/PORT, CUDA_VISIBLE_DEVICES, all NCCL_* variables, NCCL and driver versions, the "
          "full NCCL_DEBUG=INFO log up to its last line, a `py-spy dump`, and `nvidia-smi`. Rationale: the last NCCL "
          "log line alone partitions the hypothesis space into bootstrap vs transport setup vs ring connection, and "
          "any environment change made before this capture destroys evidence that cannot be recovered.\n\n")

FOOTER = ("\n\nGeneral procedure and boundary conditions.\n"
          "1. Bisect scale before bisecting configuration: one process -> one node all GPUs -> two nodes -> full "
          "world. The smallest reproducing configuration bounds the fault to intra-node (device, SHM, P2P, PCIe) or "
          "inter-node (bootstrap, fabric, ACL, rendezvous).\n"
          "2. Probe with a minimal collective, never the training job: nccl-tests `all_reduce_perf -b 8 -e 128M -f 2 "
          "-g <gpus>`, or a ten-line script doing init_process_group plus one all_reduce. A probe costs seconds, so "
          "a wrong hypothesis is cheap.\n"
          "3. Change exactly one variable per trial and keep a written log of (trial id, variable changed, outcome, "
          "duration, evidence file). Compound changes make outcomes uninterpretable and are the most common reason "
          "these investigations stall.\n"
          "4. Distinguish observability fixes from root-cause fixes. Adding a timeout or debug logging converts a "
          "hang into a diagnosable failure; it does not resolve the fault and must not be reported as a fix.\n"
          "5. Confounders that apply to every hypothesis above: slow checkpoint or dataset loading, a co-tenant "
          "saturating fabric or storage, launcher retry logic silently re-spawning ranks, and page-cache warmth "
          "making re-runs faster. Control for these by recording per-rank wall-clock timestamps plus fabric and "
          "storage counters during every trial.\n"
          "6. Global rollback gate: if two consecutive trials fail to narrow the fault domain, revert every "
          "environment change to the last known-good configuration, re-verify with the minimal all-reduce probe, and "
          "escalate with the frozen evidence rather than continuing to mutate production configuration.\n"
          "7. Exit criterion: the minimal all-reduce probe passes at full world size, and the training job reaches "
          "step 1 with MEASURED all-reduce bus bandwidth within 10% (ESTIMATE tolerance) of the MEASURED "
          "pre-incident baseline. Absent a recorded baseline, the fix is unverified and must be labelled as such.")


def build(mech):
    return (HEADER
            + "Primary hypothesis under test: " + mech["name"] + ".\n\n"
            + "Falsifiable hypothesis. " + mech["hyp"] + "\n\n"
            + "Controlled experiment. " + mech["exp"] + "\n\n"
            + "Measurements to collect. " + mech["meas"] + "\n\n"
            + "Expected confounders. " + mech["conf"] + "\n\n"
            + "Operational risks. " + "; ".join(mech["risk"]) + ".\n\n"
            + "Evidence required before declaring root cause. " + "; ".join(mech["ev"]) + ".\n\n"
            + "Rollback criteria. " + mech["rb"]
            + FOOTER)


def main():
    with open(CORPUS, encoding="utf-8") as f:
        rows = [json.loads(l) for i, l in enumerate(f) if START <= i < START + COUNT]
    assert len(rows) == COUNT, len(rows)
    out, seen = [], set()
    for row, mech in zip(rows, MECH):
        msgs = row["messages"]
        su = next(m["content"] for m in msgs if m["role"] == "user")
        sa = next(m["content"] for m in msgs if m["role"] == "assistant")
        ans = build(mech)
        h = hashlib.sha256(ans.encode()).hexdigest()
        assert h not in seen, "duplicate corrected_answer"
        seen.add(h)
        out.append({
            "source_id": row["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": ans,
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 2,
            },
            "risks": mech["risk"] + [
                "Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis",
                "No rollback gate, evidence list, or ESTIMATE/MEASURED labelling in the source text",
            ],
            "evidence_required": mech["ev"] + [
                "MEASURED pre-incident all-reduce bandwidth baseline",
                "Frozen per-rank NCCL_DEBUG=INFO logs captured before any configuration change",
            ],
            "confidence": 0.62,
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out), "records", rows[0]["id"], "->", rows[-1]["id"])


main()
