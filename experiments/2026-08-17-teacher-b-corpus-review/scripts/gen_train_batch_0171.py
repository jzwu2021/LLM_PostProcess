#!/usr/bin/env python3
"""teacher-B blind review generator for train-batch-0171 (corpus rows 1701-1710)."""
import json, hashlib, os, glob

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
RESDIR = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
OUT = os.path.join(RESDIR, "train-batch-0171.jsonl")
START, COUNT = 1700, 10

MECH = [
 dict(
  name="NCCL_SOCKET_IFNAME / NCCL_IB_HCA selection picks an unroutable interface on a multi-homed host",
  hyp="H1: bootstrap succeeds over the management network but the data-plane transport is bound to the wrong device: NCCL auto-selects a docker0/bond/veth interface (or an IB HCA with no active port) that cannot reach peer nodes, so ring connection never completes. Falsifiable: NCCL_DEBUG=INFO 'NET/Socket : Using [0]<iface>' names an interface whose subnet is not shared by all nodes, or 'NET/IB : Using [0]<hca>' names an HCA whose ibstat state is not ACTIVE/LinkUp.",
  exp="Controlled experiment: (a) record the selected interface/HCA on every rank from NCCL_DEBUG=INFO; (b) relaunch pinning NCCL_SOCKET_IFNAME (and NCCL_IB_HCA) to the interface that is verifiably routable between all nodes, changing nothing else. Prediction under H1: the selected device differs from the routable one on exactly the stalling nodes, and the pinned relaunch reaches the first all-reduce.",
  meas="Selected iface/HCA per rank from NCCL logs; `ip -o addr` subnet table per node; ibstat port state and rate; MEASURED pairwise reachability on the chosen subnet; time-to-first-collective before and after pinning.",
  conf="A host may be routable on two interfaces with very different bandwidth, so a 'working' pin can still be a slow path; container network namespaces can make the same iface name mean different devices on different nodes.",
  risk=["Pinning an interface fleet-wide hard-codes a naming convention that breaks on heterogeneous nodes", "Forcing IB when a port is down makes the job fail immediately instead of falling back"],
  ev=["per-rank NCCL device-selection log lines", "ip addr / ibstat dumps for all nodes", "routing test results between every node pair on the candidate subnet", "inventory proving interface names are uniform across the fleet"],
  rb="Rollback gate: pin the interface for one trial only. If the pinned run does not reach the first collective, or nccl-tests bus bandwidth is below 80% of the MEASURED pre-incident baseline (ESTIMATE tolerance), unset the pin and return to auto-selection before further changes."),
 dict(
  name="Shared-memory (/dev/shm) exhaustion or restrictive container IPC blocks the SHM transport",
  hyp="H1: intra-node ranks cannot establish the SHM transport because /dev/shm is too small (container default 64 MB) or the container uses a private IPC namespace, so NCCL blocks creating shared buffers rather than erroring. Falsifiable: `df -h /dev/shm` shows near-zero free space during init, NCCL_DEBUG=INFO stops after a 'via SHM' line, and NCCL_SHM_DISABLE=1 changes the failure point.",
  exp="Controlled experiment: (a) sample /dev/shm usage and the container's ipc mode during the hang; (b) relaunch one node with --shm-size raised (e.g. 16 GB) and --ipc=host, leaving all other nodes unchanged. Prediction under H1: the modified node's local ranks pass the SHM phase while the control node still stalls at the same log line.",
  meas="/dev/shm size and free bytes per node during init; container ipc mode; NCCL last log line per rank; MEASURED intra-node all-reduce bandwidth with SHM enabled vs NCCL_SHM_DISABLE=1.",
  conf="Leftover /dev/shm segments from crashed runs consume space and self-heal on reboot, hiding the cause; --ipc=host also changes many other isolation properties at once.",
  risk=["--ipc=host removes IPC isolation between containers on the node", "Oversized /dev/shm counts against node memory and can trigger OOM for co-tenants"],
  ev=["/dev/shm utilization time series", "container runtime spec (ipc mode, shm-size)", "stale segment listing via ls -l /dev/shm", "node memory headroom before raising shm-size"],
  rb="Rollback gate: raise shm-size stepwise on one node; if node free memory drops below 10% (MEASURED from /proc/meminfo) or co-tenant OOM events appear, revert immediately. NCCL_SHM_DISABLE=1 is diagnostic only and must not ship."),
 dict(
  name="Firewall / security-group ACL permits the bootstrap port but blocks NCCL's ephemeral data ports",
  hyp="H1: rendezvous over MASTER_PORT succeeds because that single port is allowed, while NCCL's dynamically chosen data connections are dropped by a host firewall or cloud security group, so ranks hang after bootstrap. Falsifiable: `nc -z` on MASTER_PORT succeeds between nodes while a probe on an ephemeral port fails, and conntrack/iptables counters show drops in the ephemeral range during init.",
  exp="Controlled experiment: (a) run a paired port-reachability sweep (bootstrap port vs several ephemeral ports) between two stalling nodes; (b) relaunch with NCCL_PORT_RANGE (or the equivalent constrained range) restricted to an explicitly allowed window. Prediction under H1: only the bootstrap port is reachable, and constraining NCCL to the allowed range lets init complete.",
  meas="Per-port reachability matrix between node pairs; iptables/nftables drop counters delta during init; security-group rule export; time-to-first-collective with and without the constrained range.",
  conf="An overloaded conntrack table drops connections non-deterministically and mimics an ACL; stateful rules may allow the first flow and drop later ones, so a single successful probe is not sufficient.",
  risk=["Widening firewall rules to 'fix' the hang can expose the training data plane to untrusted networks", "Constraining NCCL's port range too narrowly causes bind failures at larger world size"],
  ev=["port sweep results per node pair", "firewall/security-group configuration export with change history", "conntrack table utilization during init", "security-owner approval for any rule change"],
  rb="Rollback gate: request the narrowest possible port-range exception rather than disabling the firewall. If a temporary rule is used for diagnosis, it must carry an expiry and be removed in the same session; verify removal with a repeat port sweep."),
 dict(
  name="Heterogeneous NCCL / driver / CUDA versions across nodes produce an incompatible handshake",
  hyp="H1: nodes were patched at different times, so ranks link against different NCCL builds (or incompatible driver/CUDA minor versions), and the version handshake stalls or negotiates an unsupported protocol. Falsifiable: the NCCL version banner differs across ranks in the INFO logs, or `nvidia-smi` driver versions and `python -c 'import torch;print(torch.cuda.nccl.version())'` disagree between the working and stalling node groups.",
  exp="Controlled experiment: (a) collect a version fingerprint (driver, CUDA runtime, NCCL, torch, container image digest) from every rank and diff them; (b) rerun restricted to a subset of nodes whose fingerprints are byte-identical. Prediction under H1: the fingerprints partition the fleet exactly along the working/stalling boundary, and the homogeneous subset initializes.",
  meas="Version fingerprint table per rank including container image digest; MEASURED pass/fail of the homogeneous-subset run; NCCL version banner per rank; package manifest diff between node groups.",
  conf="Version skew often coexists with a genuine network fault; a homogeneous subset may also happen to be a single rack and thus avoid an inter-rack fabric problem, so control for topology when selecting the subset.",
  risk=["Upgrading drivers on a live fleet requires node drains and can invalidate in-flight checkpoints' reproducibility", "Pinning everyone to the oldest version may reintroduce known NCCL bugs"],
  ev=["complete per-rank version fingerprint", "container image digests actually running (not the tag)", "subset run selected to span the same racks as the failing run", "change record for the last fleet patch"],
  rb="Rollback gate: roll the version alignment to one node first and verify with nccl-tests before fleet-wide rollout; if aligned nodes still stall, revert the upgrade and treat version skew as excluded rather than continuing to patch."),
 dict(
  name="GPU-to-NIC affinity / GPUDirect RDMA unavailable, so the transport silently falls back or blocks on a cross-socket path",
  hyp="H1: the GPU and the RDMA NIC are on different PCIe root complexes (or nvidia-peermem / dma-buf support is absent), so GDR cannot be used; NCCL either falls back to a staged path that is pathologically slow or blocks negotiating the transport. Falsifiable: NCCL_DEBUG_SUBSYS=NET logs show 'GDRDMA disabled' or a staged path, `lsmod | grep nvidia_peermem` is empty, and `nvidia-smi topo -m` shows the chosen HCA is not on the same PCIe switch as the GPU.",
  exp="Controlled experiment: (a) record the GPU/HCA affinity matrix and GDR availability per node; (b) rerun with NCCL_IB_HCA pinned to the affine HCA for each local rank and, as a control, with NCCL_NET_GDR_LEVEL forced off. Prediction under H1: forcing affine HCAs restores progress and raises MEASURED bus bandwidth, while the GDR-off control is functional but markedly slower.",
  meas="nvidia-smi topo -m GPU/NIC affinity; nvidia_peermem module presence; ib_write_bw MEASURED bandwidth per GPU-HCA pairing; NCCL bus bandwidth with and without GDR; PCIe link width/speed per device.",
  conf="A slow path is not the same as a hang; if the job never progresses at all, affinity may be a real but secondary finding. Kernel or firmware updates can silently drop peermem support.",
  risk=["Hard-pinning HCA per local rank breaks on nodes with a different PCIe layout", "Disabling GDR increases host memory bandwidth pressure and can starve the dataloader"],
  ev=["topo matrix and PCIe tree per node", "peermem/dma-buf support evidence", "per-pairing MEASURED bandwidth table", "confirmation the layout is identical across the fleet before pinning"],
  rb="Rollback gate: apply affinity pinning to one node and compare nccl-tests bus bandwidth against the MEASURED baseline; if it is not at least equal, revert the pin. Never leave NCCL_NET_GDR_LEVEL forced off in production without recording the throughput cost."),
 dict(
  name="Launcher/scheduler mis-assigns RANK, WORLD_SIZE or CUDA_VISIBLE_DEVICES so ranks collide or one never starts",
  hyp="H1: the environment the ranks actually receive is inconsistent with the intended layout: duplicated RANK values, a WORLD_SIZE larger than the number of started processes, or two local ranks mapped to the same GPU. The collective then waits for a participant that will never arrive. Falsifiable: a per-process dump of RANK/LOCAL_RANK/WORLD_SIZE/CUDA_VISIBLE_DEVICES shows duplicates or gaps, or the count of live trainer PIDs is less than WORLD_SIZE.",
  exp="Controlled experiment: before touching NCCL, have every process print its identity tuple to a shared file at startup and compare against the intended layout; then relaunch with an explicitly enumerated layout. Prediction under H1: the collected tuples show a gap or duplicate, and the explicit relaunch initializes.",
  meas="Sorted table of (host, PID, RANK, LOCAL_RANK, WORLD_SIZE, visible devices); count of live processes vs WORLD_SIZE; scheduler allocation record; per-GPU process count from nvidia-smi.",
  conf="A rank that started but is slow to reach the print looks identical to a rank that never started at short observation windows; scheduler retries can create extra processes that later exit.",
  risk=["Manually enumerating ranks bypasses the scheduler's placement and can oversubscribe GPUs", "Two processes on one GPU can OOM and corrupt a checkpoint in progress"],
  ev=["identity-tuple census file from the failing run", "scheduler allocation and node list", "nvidia-smi process listing per node", "diff between intended and observed layout"],
  rb="Rollback gate: an explicit manual layout is a diagnostic; return to the scheduler-managed launch once the mismatch is proven, and file the launcher defect. If manual launch oversubscribes any GPU, abort immediately."),
 dict(
  name="DNS / hostname resolution asymmetry: MASTER_ADDR resolves differently (or not at all) on some nodes",
  hyp="H1: some ranks cannot resolve MASTER_ADDR, or resolve it to a different address (split-horizon DNS, stale /etc/hosts, container resolver), so their TCP store connection retries silently until the job appears hung. Falsifiable: `getent hosts $MASTER_ADDR` returns different addresses on different nodes, or the store's listening socket shows fewer established connections than world_size.",
  exp="Controlled experiment: (a) resolve MASTER_ADDR from every node and compare; (b) relaunch using the master's literal IP address instead of a hostname, changing nothing else. Prediction under H1: resolution differs on exactly the non-connecting ranks and the IP-literal relaunch completes rendezvous.",
  meas="Resolution result per node; established connection count on the store port (`ss -tn state established sport = :29500 | wc -l`) vs world_size; DNS server config per node; retry/backoff log lines if any.",
  conf="A firewall drop looks identical from the client side; an IP literal also bypasses any load-balancer indirection, so success does not by itself prove DNS was the fault.",
  risk=["Hard-coding IPs breaks when the master is rescheduled to another node", "Editing /etc/hosts inside long-lived images creates drift that outlives the incident"],
  ev=["per-node resolution table", "store connection count timeline", "resolver configuration per node/container", "record of any /etc/hosts edit with an owner and removal plan"],
  rb="Rollback gate: the IP literal is a diagnostic configuration. Once DNS is fixed, restore the hostname form and re-verify rendezvous; remove any temporary /etc/hosts entries in the same session and confirm with a repeat resolution sweep."),
 dict(
  name="Fabric congestion or a degraded link makes init appear hung rather than failed (no timeout configured)",
  hyp="H1: the fabric is functional but severely degraded on one path: a link has trained down to a lower width/rate, or a co-tenant is saturating the same spine, so the first sizeable exchange takes far longer than the operator's patience. Falsifiable: NIC/switch counters show symbol errors or link-down events on a specific port, or `ibstatus`/`ethtool` reports a rate below the fleet standard on exactly the nodes involved.",
  exp="Controlled experiment: (a) collect link rate/width and error counters on every involved port and diff against the fleet standard; (b) run pairwise ib_write_bw across all node pairs to build a bandwidth matrix. Prediction under H1: one or a few pairs are dramatic outliers, and excluding the degraded node lets the job initialize.",
  meas="Link rate/width per port; symbol-error and link-down counter deltas over a fixed window; MEASURED pairwise bandwidth matrix (GB/s); fabric utilization from switch telemetry during the window.",
  conf="Transient congestion from another tenant disappears between measurements, so a single clean sweep does not exonerate the fabric; ib_write_bw uses a different traffic pattern than NCCL rings.",
  risk=["Excluding nodes shrinks world size and changes global batch size and convergence behaviour", "Cable reseating requires physical access and can disturb neighbouring ports"],
  ev=["port rate/width/error table for all involved ports", "pairwise bandwidth matrix with timestamps", "switch-side utilization for the same window", "co-tenant job schedule for the incident window"],
  rb="Rollback gate: quarantine a suspected degraded link only with fabric-owner agreement. If excluding the node does not restore init, return it to the pool rather than accumulating exclusions, and restore the original global batch size via gradient accumulation with the change recorded."),
 dict(
  name="Application-level deadlock: a barrier or all-reduce placed inside a rank-conditional code path",
  hyp="H1: the hang is in user code, not infrastructure: a collective (barrier, all_reduce, broadcast of a config, or a metric aggregation) sits inside an `if rank == 0` / `if is_last_batch` branch, so only a subset of ranks enters it and the rest wait forever. Falsifiable: py-spy dumps show ranks stopped at different source lines, with a proper subset inside a collective call and the remainder past it.",
  exp="Controlled experiment: take simultaneous py-spy dumps on all ranks and cluster them by stack; then rerun with the suspect conditional collective hoisted out of the branch (executed by all ranks). Prediction under H1: stacks split into exactly two clusters at a known line, and the hoisted version proceeds.",
  meas="Per-rank py-spy stack census with source line numbers; number of distinct stack clusters; rank membership of each cluster; pass/fail of the hoisted-collective control.",
  conf="Infrastructure faults also yield split stacks (some ranks further along) — the discriminator is whether the divergent line is a user-code conditional or a transport call; sampling skew can misattribute a rank that is merely slow.",
  risk=["Hoisting a collective changes semantics (e.g. logging or checkpoint frequency) and can alter training behaviour", "Adding collectives to a hot path costs throughput at every step"],
  ev=["simultaneous py-spy dumps from all ranks", "the source of the conditional branch", "diff of the hoisted-collective control", "confirmation that the hoisted call preserves intended semantics"],
  rb="Rollback gate: validate the hoisted version on a short run and compare loss curve and step time against the MEASURED baseline; if step time regresses by more than 5% (ESTIMATE) or loss diverges, revert and restructure so all ranks participate without extra synchronization."),
 dict(
  name="Elastic/fault-tolerant launcher rendezvous churn: a restarting agent keeps resetting the round",
  hyp="H1: an elastic launcher (torchrun with min<max, or a scheduler-side supervisor) has one agent that repeatedly crashes and rejoins, so the rendezvous round restarts before all members complete and the job never reaches a stable world. Falsifiable: the rendezvous backend shows an incrementing round/epoch counter with the same node leaving and rejoining, and one node's agent log contains repeated restart entries with a short interval.",
  exp="Controlled experiment: (a) watch the rendezvous state (round id, membership) for 3 minutes and record churn; (b) relaunch with min==max (no elasticity) so a failing member causes a hard error instead of a restart. Prediction under H1: membership churns around one identifiable node, and the rigid relaunch fails fast with that node named rather than hanging.",
  meas="Rendezvous round id and membership over time; restart count and interval per agent; exit codes of the churning worker; wall time to hard failure in the rigid control.",
  conf="Churn can be a symptom of an unrelated crash (OOM, Xid) rather than a rendezvous fault, so the agent's exit reason must be read before concluding; disabling elasticity changes recovery behaviour for real faults too.",
  risk=["min==max removes fault tolerance, so a single node failure kills long production runs", "Repeated restarts can leave orphaned GPU contexts that block the next attempt"],
  ev=["rendezvous membership timeline", "per-agent logs with exit codes and restart timestamps", "OOM/Xid evidence for the churning node", "orphan process census after each restart"],
  rb="Rollback gate: min==max is a diagnostic configuration. Restore the elastic range once the crashing member's root cause is fixed, and verify recovery works by deliberately failing one worker in a controlled test before returning to production."),
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
    seen = set()
    for p in sorted(glob.glob(os.path.join(RESDIR, "train-batch-*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for l in f:
                seen.add(hashlib.sha256(json.loads(l)["corrected_answer"].encode()).hexdigest())
    out = []
    for row, mech in zip(rows, MECH):
        msgs = row["messages"]
        su = next(m["content"] for m in msgs if m["role"] == "user")
        sa = next(m["content"] for m in msgs if m["role"] == "assistant")
        ans = build(mech)
        h = hashlib.sha256(ans.encode()).hexdigest()
        assert h not in seen, "duplicate corrected_answer: " + mech["name"]
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
