import json, os, hashlib

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
CORP = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT = os.path.join(EXP, "results", "train-batch-0163.jsonl")
START, N = 1620, 10

rows = [json.loads(l) for l in open(CORP, encoding="utf-8")]
sel = rows[START:START + N]
assert len(sel) == N

ASSUM = ("Assumptions: the stall is reproducible on the same node set within a 30-minute window; launcher, "
         "world size, container image digest, driver/firmware level and fabric path are pinned identically "
         "across control and treatment arms; exactly one variable changes per arm; every number below is "
         "explicitly tagged ESTIMATE or MEASURED, and no vendor-specific behaviour is asserted without a "
         "log line, counter, or command output to back it.")

TRIAGE = ("Ordered triage before touching any tunable: (1) freeze the failing state - do NOT restart; "
          "capture per-rank Python and native stacks (py-spy dump --pid, gdb -p, plus /proc/<pid>/stack) "
          "because a restart destroys the only evidence separating 'blocked in the driver/ioctl' from "
          "'blocked inside the collective bootstrap'; (2) build a rank census - hostname, LOCAL_RANK, "
          "global RANK, PID, GPU UUID, WORLD_SIZE, MASTER_ADDR:PORT, NCCL version, image digest; "
          "(3) re-run with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV and "
          "TORCH_DISTRIBUTED_DEBUG=DETAIL, recording the LAST init stage each rank printed - the ranks "
          "that never print a stage localise the fault; (4) reproduce with a minimal standalone all-reduce "
          "outside the training script to exclude application-level ordering bugs; (5) shrink the world "
          "(2 ranks/1 node, then 1 rank on each of 2 nodes) to separate intra-node transports (SHM, "
          "P2P/NVLink) from inter-node transports (TCP, IB/RoCE).")

CONF = ("Expected confounders: scheduler- or image-injected environment variables that silently override "
        "CUDA_VISIBLE_DEVICES, NCCL_SOCKET_IFNAME, NCCL_IB_HCA or the collective timeout; a node drained, "
        "rebooted or re-imaged between arms; stale debug env vars left in the launch template from a prior "
        "incident; host-to-host heterogeneity in driver, firmware, NUMA layout or kernel; and other tenants "
        "sharing the fabric. Capture a full environment and topology diff between arms - without it the "
        "comparison is uncontrolled and any conclusion is anecdote, not evidence.")

ROLLBACK_TAIL = ("General rollback criteria: if two consecutive controlled arms fail to move the stall to a "
                 "different init stage, stop tuning, restore the last known-good launch template verbatim, "
                 "and hand off the evidence bundle rather than stacking further environment overrides. "
                 "Never 'fix' this by raising the process-group/NCCL timeout: that converts a fast, "
                 "diagnosable failure into a slow, expensive one and destroys the signal.")

M = [
 dict(mech="Rendezvous store deadlock from a partially reused MASTER_PORT / stale TCPStore",
      why=("A previous job left a listener or half-open connections on MASTER_ADDR:MASTER_PORT. New ranks "
           "connect successfully at TCP level but exchange keys with the wrong (orphaned) store, so the "
           "barrier never counts to WORLD_SIZE and no rank raises an error."),
      hyp=("H1: quorum fails because ranks attach to a stale store, not because of any GPU or fabric fault. "
           "Falsifiable prediction: on the master, ss -ltnp shows a listener on MASTER_PORT owned by a PID "
           "that is not in this job's rank census, and re-running on a freshly allocated, unused port lets "
           "the identical job reach quorum on the same node set."),
      exp=("Arm A: today's port. Arm B: a port verified unused on every node (ss -ltn | grep -c :PORT == 0) "
           "with nodes, image, world size and transport held fixed. 3 repeats per arm. Only the port changes."),
      meas=["ss -ltnp and ss -tnp on the master before launch and at the moment of the stall",
            "count of distinct peer addresses connected to MASTER_PORT vs WORLD_SIZE (MEASURED)",
            "rank census cross-referenced against the owning PID of the listener",
            "time-to-quorum for arm A vs arm B, 3 repeats each, with min/median/max reported"],
      risks=["The corpus assistant text is a grading rubric, not an answer; training on it teaches the model to describe what a good answer contains instead of performing the diagnosis.",
             "'Verify process-group rendezvous' in the rubric has no notion of a stale store, so a model imitating it will see a reachable master and wrongly clear the control plane.",
             "Killing the orphaned listener may belong to another tenant's live job; ownership must be confirmed against the scheduler before any kill."],
      rollback=("Rollback gate: if the fresh-port arm still stalls in 2 of 2 repeats, the store hypothesis "
                "is refuted - restore the original port and move to transport-level bisection.")),
 dict(mech="Interface selection picks a non-routable management NIC (NCCL_SOCKET_IFNAME unset)",
      why=("With no explicit interface, the bootstrap can select the first non-loopback interface, which on "
           "many clusters is a management or docker0/cni bridge that is not routable between compute nodes. "
           "The TCP connect neither succeeds nor fails fast, so the init hangs until timeout."),
      hyp=("H2: the bootstrap chose a non-routable interface. Falsifiable prediction: NCCL_DEBUG=INFO "
           "'NET/Socket : Using [0]<iface>' names a bridge/management device, and pinning "
           "NCCL_SOCKET_IFNAME to the verified data-plane interface reaches quorum unchanged otherwise."),
      exp=("Arm A: unset NCCL_SOCKET_IFNAME. Arm B: NCCL_SOCKET_IFNAME=<data-plane iface> confirmed by a "
           "point-to-point iperf3 between two compute nodes on that interface first. Nothing else changes."),
      meas=["NCCL_DEBUG=INFO 'Using [0]' line captured per rank (MEASURED)",
            "ip -br addr and ip route get <peer-data-plane-IP> on every node",
            "iperf3 single-stream throughput on the candidate interface, Gbit/s, 10 s run (MEASURED)",
            "time-to-quorum per arm, 3 repeats"],
      risks=["Pinning an interface name is brittle across heterogeneous nodes where the same fabric has different device names; prefer a prefix match and assert it resolves on every node.",
             "A rubric-trained model may recommend interface pinning as a blanket fix without first proving the current selection is wrong, hiding a real cabling fault.",
             "Excluding the wrong interface can silently downgrade a job from RDMA to TCP with a large throughput loss that no error message reports."],
      rollback=("Rollback gate: if the pinned-interface arm still stalls, revert to unset and escalate to "
                "fabric/link-layer checks; do not accumulate further NCCL_* overrides.")),
 dict(mech="GPU/device visibility mismatch - two ranks bound to the same physical GPU",
      why=("A LOCAL_RANK-to-device mapping bug, or a CUDA_VISIBLE_DEVICES injected by the scheduler that the "
           "training script re-applies, can leave two ranks on one GPU and one GPU unused. The intra-node "
           "P2P/topology setup then cannot form a consistent ring and blocks."),
      hyp=("H3: device binding is not a bijection. Falsifiable prediction: the rank census shows a duplicate "
           "GPU UUID across two ranks on the same host, and nvidia-smi shows a device with zero job "
           "processes while another hosts two."),
      exp=("Arm A: current launch. Arm B: identical launch with an explicit assertion that "
           "len(set(gpu_uuid_per_rank)) == world_size_per_node, failing fast if violated. Compare which "
           "stage the job reaches."),
      meas=["per-rank torch.cuda.get_device_properties + device UUID table (MEASURED)",
            "nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv at stall time",
            "effective CUDA_VISIBLE_DEVICES as seen inside each rank process (/proc/<pid>/environ)",
            "nvidia-smi topo -m to confirm the expected P2P matrix"],
      risks=["The rubric mentions 'GPU visibility' but not the duplicate-binding failure mode, so an imitating model may only print nvidia-smi and declare the GPUs healthy.",
             "Adding a hard assertion changes failure timing and can mask a slower, unrelated init race; keep it as a diagnostic arm, not a silent permanent change.",
             "Reassigning devices mid-incident on a shared node can evict another tenant's process."],
      rollback=("Rollback gate: if the binding is already a bijection, H3 is refuted - stop investigating "
                "device mapping and return to transport bisection.")),
 dict(mech="Cgroup / container limit blocks shared-memory and pinned-host-memory setup",
      why=("An undersized /dev/shm or a restrictive memlock ulimit inside the container makes the intra-node "
           "SHM transport and pinned-buffer registration stall or silently retry rather than error, so init "
           "never completes even though the network is healthy."),
      hyp=("H4: the stall is a host-resource limit, not a network fault. Falsifiable prediction: "
           "df -h /dev/shm reports a small limit (e.g. the 64 MiB container default - ESTIMATE of the common "
           "default, to be confirmed by MEASURED df output) and/or ulimit -l is not unlimited, and raising "
           "both on an otherwise identical container reaches quorum."),
      exp=("Arm A: current container settings. Arm B: same image and node set with --shm-size raised and "
           "memlock unlimited. Network configuration untouched in both arms."),
      meas=["df -h /dev/shm and ulimit -l -n inside each rank container (MEASURED)",
            "cgroup memory.max / memory.current at the moment of the stall",
            "dmesg for OOM or mlock denials on each host",
            "time-to-quorum per arm, 3 repeats"],
      risks=["Granting unlimited memlock broadly weakens isolation on a shared node; scope it to the job and record the change.",
             "A rubric-trained model will not consider host resource limits at all, since the rubric lists only network and topology steps.",
             "Raising shm can mask a genuine leak in the data loader; check steady-state shm usage after the fix."],
      rollback=("Rollback gate: if the raised-limits arm still stalls, revert the container settings "
                "immediately - do not leave elevated privileges in the template for an unproven hypothesis.")),
 dict(mech="Partial world - one rank never launched (scheduler/placement failure)",
      why=("If the launcher starts WORLD_SIZE-1 processes because one node failed to pull the image or was "
           "preempted, the surviving ranks block at the barrier forever with no error, because a missing "
           "peer is indistinguishable from a slow peer until timeout."),
      hyp=("H5: the world is incomplete. Falsifiable prediction: the sum of job-owned processes across all "
           "allocated hosts is strictly less than WORLD_SIZE, and one host shows an image-pull or container "
           "start failure in the scheduler/kubelet log at the job start timestamp."),
      exp=("No tuning arm is needed first: count processes per host and compare to the expected placement. "
           "Then arm B relaunches on a node set excluding the failing host with world size reduced "
           "accordingly; quorum there confirms the missing-peer explanation."),
      meas=["per-host process count owned by the job vs expected ranks per host (MEASURED)",
            "scheduler event log / kubelet log lines within the job start window",
            "container image pull status and digest on every allocated host",
            "whether the reduced-world relaunch reaches quorum, 2 repeats"],
      risks=["This is the cheapest hypothesis and is frequently skipped; the rubric's step list starts at rendezvous, so an imitating model burns hours on NCCL flags for a missing process.",
             "Excluding a host permanently without a root cause hides a recurring node fault; file the node for repair rather than silently shrinking the pool.",
             "Reduced-world runs change per-GPU batch size and are not performance-comparable to the original job."],
      rollback=("Rollback gate: if all WORLD_SIZE processes are present, H5 is refuted immediately and no "
                "config change is made.")),
 dict(mech="RoCE misconfiguration - PFC/DSCP or GID index mismatch across nodes",
      why=("On RoCEv2 the RDMA queue pairs come up only if both ends agree on the GID index (v1 vs v2) and "
           "the lossless priority is consistently configured end to end. A mismatch lets the connection "
           "manager retry indefinitely during init instead of failing loudly."),
      hyp=("H6: the RDMA path is misconfigured, not down. Falsifiable prediction: show_gids reports "
           "different RoCEv2 GID indices in use across nodes, or switch/NIC PFC counters increment without "
           "a matching configured priority; forcing NCCL_IB_GID_INDEX to the verified RoCEv2 index reaches "
           "quorum, and forcing NCCL_IB_DISABLE=1 (TCP fallback) also reaches quorum but slower."),
      exp=("Three arms with the node set fixed: A = today; B = explicit NCCL_IB_GID_INDEX; "
           "C = NCCL_IB_DISABLE=1. C succeeding while A hangs localises the fault to the RDMA path; "
           "B succeeding localises it further to GID selection."),
      meas=["show_gids / ibv_devinfo per node, recording GID index and RoCE version (MEASURED)",
            "ethtool -S and NIC PFC/pause counters before and after each arm",
            "ib_write_bw point-to-point between two nodes, Gbit/s (MEASURED, not estimated)",
            "time-to-quorum for arms A/B/C, 3 repeats each"],
      risks=["NCCL_IB_DISABLE=1 is a diagnostic, not a fix: it can cut effective inter-node bandwidth by roughly an order of magnitude (ESTIMATE, derived from typical 100-400 Gbit/s RDMA vs a single TCP stream; must be replaced by a MEASURED iperf3 vs ib_write_bw comparison on this cluster).",
             "PFC changes are switch-wide and can destabilise other tenants; they require a network-owner change window, not an ad-hoc override.",
             "The rubric never mentions RoCE/GID at all, so a model trained on it will not reach this hypothesis."],
      rollback=("Rollback gate: revert NCCL_IB_* overrides if arm B does not change the stall stage; do not "
                "ship a job with NCCL_IB_DISABLE=1 as a permanent workaround.")),
 dict(mech="Mixed NCCL / PyTorch / CUDA versions across nodes from a drifted image tag",
      why=("A mutable image tag re-resolved to a different digest on one node means ranks negotiate a "
           "bootstrap protocol with incompatible expectations; the mismatch typically manifests as a silent "
           "hang in init rather than a clean version error."),
      hyp=("H7: the ranks are not running identical binaries. Falsifiable prediction: the image digest, "
           "torch.__version__ and torch.cuda.nccl.version() are not identical across all ranks; pinning "
           "every node to one digest reaches quorum with nothing else changed."),
      exp=("Arm A: current mutable tag. Arm B: the same job pinned to a single sha256 image digest on every "
           "node. Node set, world size, network config unchanged."),
      meas=["per-rank image digest, torch version, NCCL version, CUDA runtime and driver version table (MEASURED)",
            "container runtime pull log per host showing which digest was resolved",
            "nvidia-smi driver version per host",
            "time-to-quorum per arm, 3 repeats"],
      risks=["Version drift is invisible to every network-level check in the rubric, so a rubric-imitating model will keep testing the fabric.",
             "Pinning a digest without an update process leaves the fleet on an old, possibly vulnerable image; pair the pin with a controlled upgrade plan.",
             "Rolling nodes to a new digest mid-incident restarts the failing state and destroys evidence if stacks were not captured first."],
      rollback=("Rollback gate: if digests were already identical, H7 is refuted; revert to the tag and "
                "continue bisection elsewhere.")),
 dict(mech="Firewall / security-group blocking the ephemeral port range used after bootstrap",
      why=("Rendezvous on MASTER_PORT succeeds because that single port is allowed, but the subsequent "
           "peer-to-peer connections use ephemeral ports that a host firewall or security group drops. The "
           "connect is silently dropped (DROP, not REJECT), so ranks wait rather than fail."),
      hyp=("H8: packets are dropped, not lost. Falsifiable prediction: firewall counters on the receiving "
           "node increment for the peer's source address during the stall, and a plain nc/iperf3 on an "
           "ephemeral port between the same two nodes also hangs, while MASTER_PORT connects fine."),
      exp=("Arm A: current policy. Arm B: same nodes with the required inter-node port range explicitly "
           "permitted between compute hosts only. Verify with nc before relaunching the job."),
      meas=["iptables/nftables counters (or cloud flow logs) for dropped packets between the two hosts (MEASURED)",
            "nc -zv peer <ephemeral-port> result and a tcpdump SYN-without-SYNACK capture",
            "MASTER_PORT connectivity as a positive control",
            "time-to-quorum per arm, 3 repeats"],
      risks=["Opening a wide port range between hosts is a real security change; scope it to the compute subnet and get it reviewed, do not disable the firewall.",
             "A model imitating the rubric checks that the master is reachable and concludes the network is fine, missing the ephemeral-port asymmetry.",
             "tcpdump on a production node adds load and may require capabilities the container lacks; plan the capture window."],
      rollback=("Rollback gate: if traffic on ephemeral ports flows freely, H8 is refuted and the firewall "
                "change must be reverted rather than left in place.")),
 dict(mech="Application-level collective ordering divergence (ranks call different collectives)",
      why=("A conditional branch - for example only rank 0 loading a checkpoint, or an uneven dataloader "
           "shard triggering an extra all-reduce - makes ranks enqueue different collectives. NCCL matches "
           "operations by call order, so mismatched sequences block permanently with no error."),
      hyp=("H9: the hang is caused by the training script, not by the platform. Falsifiable prediction: a "
           "minimal standalone all-reduce on the identical node set, image and env completes normally, "
           "while the training script stalls, and TORCH_DISTRIBUTED_DEBUG=DETAIL reports a collective "
           "mismatch or divergent op sequence across ranks."),
      exp=("Arm A: full training script. Arm B: minimal all-reduce harness, everything else identical. "
           "B succeeding while A hangs moves the investigation into application code, not infrastructure."),
      meas=["per-rank last-enqueued collective name, shape and dtype from TORCH_DISTRIBUTED_DEBUG=DETAIL (MEASURED)",
            "per-rank Python stack at the stall (py-spy dump), diffed across ranks",
            "dataloader shard sizes per rank to detect an uneven final batch",
            "whether the minimal harness completes, 3 repeats"],
      risks=["Misattributing an application bug to the fabric wastes a network-team escalation and delays the real fix.",
             "The rubric's 'run a minimal all-reduce' step exists but the rubric does not say what conclusion to draw from either outcome, so an imitating model runs the test without using its result.",
             "Adding barriers to 'fix' the divergence can hide the real branch and cause a deadlock elsewhere later."],
      rollback=("Rollback gate: if the minimal harness also hangs, H9 is refuted and the investigation "
                "returns to the platform; revert any application-side barrier patches added for testing.")),
 dict(mech="Clock/topology asymmetry - one node's PCIe/NVLink topology differs, breaking ring construction",
      why=("A node with a different PCIe layout, a degraded NVLink, or GPUs behind different root complexes "
           "produces a topology graph that cannot be reconciled with the other nodes' graphs, so the "
           "graph-search phase of init stalls or falls back unpredictably."),
      hyp=("H10: one node is topologically different from the rest. Falsifiable prediction: nvidia-smi "
           "topo -m and the NCCL GRAPH debug output differ on exactly one host, and excluding that host "
           "(same world size, substitute node) reaches quorum."),
      exp=("Arm A: current node set. Arm B: identical job with the suspect host swapped for a healthy node "
           "of the same SKU. World size, image and network unchanged."),
      meas=["nvidia-smi topo -m from every host, diffed (MEASURED)",
            "NCCL_DEBUG_SUBSYS=GRAPH output showing the chosen rings/trees per node",
            "nvidia-smi nvlink -s error and bandwidth counters on the suspect host",
            "lspci -tv root-complex layout comparison across hosts"],
      risks=["Swapping a node changes two things at once if the replacement has a different firmware level; verify SKU, driver and firmware equality before calling the arm controlled.",
             "The rubric says 'inspect topology' but gives no comparison baseline, so a model imitating it prints topo -m and stops.",
             "A degraded NVLink may still pass init and only show up as a throughput regression later; record baseline bandwidth even if the hang resolves."],
      rollback=("Rollback gate: if the node swap does not change the stall, H10 is refuted - return the node "
                "to the pool rather than leaving it drained on a disproven suspicion.")),
]

def build(mech):
    return "\n\n".join([
        ASSUM,
        TRIAGE,
        "Primary mechanism under test: " + mech["mech"] + ". " + mech["why"],
        "Falsifiable hypothesis. " + mech["hyp"],
        mech["exp"],
        "Required measurements and evidence: " + "; ".join(mech["meas"]) + ".",
        CONF,
        mech["rollback"] + " " + ROLLBACK_TAIL,
    ])

recs, seen = [], set()
for row, mech in zip(sel, M):
    msgs = row["messages"]
    su = [m for m in msgs if m["role"] == "user"][0]["content"]
    sa = [m for m in msgs if m["role"] == "assistant"][0]["content"]
    ca = build(mech)
    h = hashlib.sha256(ca.encode("utf-8")).hexdigest()
    assert h not in seen, "duplicate corrected_answer"
    seen.add(h)
    recs.append({
        "source_id": row["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": su,
        "source_assistant": sa,
        "corrected_answer": ca,
        "quality_dimensions": {"technical_correctness": 2, "instruction_coverage": 2, "operational_safety": 2},
        "risks": mech["risks"],
        "evidence_required": mech["meas"],
        "confidence": 0.62,
    })

assert len(seen) == N
with open(OUT, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs), "distinct", len(seen))
