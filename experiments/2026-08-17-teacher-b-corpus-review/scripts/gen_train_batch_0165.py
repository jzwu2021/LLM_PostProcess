#!/usr/bin/env python3
"""teacher-B blind review generator for train-batch-0165 (corpus rows 1641-1650)."""
import json, hashlib, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0165.jsonl")
START, COUNT = 1640, 10

MECH = [
 dict(
  name="Rendezvous store: MASTER_ADDR resolves to loopback on a subset of nodes",
  hyp="H1: ncclCommInitRank never returns because the PyTorch TCPStore rendezvous never completes; on >=1 node MASTER_ADDR resolves to 127.0.0.1 (or a stale /etc/hosts entry), so those ranks connect to a local store and never join the global group. Falsifiable: if true, `ss -tnp state established '( sport = :29500 )'` on the master shows fewer than world_size-1 peer connections, and the master's store reports fewer keys than world_size.",
  exp="Controlled experiment: (a) on every node run `getent hosts $MASTER_ADDR` and `python -c \"import socket;print(socket.gethostbyname('$MASTER_ADDR'))\"`, record the tuple (hostname, resolved IP); (b) rerun with MASTER_ADDR pinned to the master's routable IPv4 literal instead of a name. Prediction under H1: the literal-IP run passes rendezvous; the name-based run hangs again on the same node set.",
  meas="Count of ESTABLISHED sockets to MASTER_PORT on master (expect world_size-1); torch.distributed init wall time per rank; TORCH_DISTRIBUTED_DEBUG=DETAIL logs; NCCL_DEBUG=INFO first 'bootstrap' line per rank.",
  conf="A slow but working rendezvous under high node count looks identical for the first ~30 s; DNS caching can make a failing node intermittently succeed.",
  risk=["Changing MASTER_ADDR resolution can break other jobs sharing /etc/hosts", "Restarting the job destroys in-flight state if no checkpoint was written"],
  ev=["getent hosts output from every rank", "ss/netstat socket census on the master", "TORCH_DISTRIBUTED_DEBUG=DETAIL init logs", "checkpoint timestamp before any restart"],
  rb="Rollback gate: if the literal-IP run does not complete rendezvous within 2x the historical median init time (ESTIMATE; derive the median from the last 10 successful runs of this job), revert MASTER_ADDR to the original value and move to H2 rather than accumulating changes."),
 dict(
  name="Interface selection: NCCL bootstrap binds a non-routable NIC (docker0/virbr0/lo)",
  hyp="H1: NCCL's automatic interface selection picked a non-routable virtual NIC, so the bootstrap ring cannot be closed. Falsifiable: NCCL_DEBUG=INFO lines of the form 'NET/Socket : Using [0]docker0:172.17.x.x' on at least one rank while others show the datacenter NIC.",
  exp="Controlled experiment: rerun unchanged except NCCL_SOCKET_IFNAME set to the known data NIC prefix (and NCCL_SOCKET_IFNAME excluded interfaces via '^docker,lo,virbr'). Prediction under H1: bootstrap completes; with the variable removed it hangs again, reproducibly.",
  meas="Per-rank 'Using [i]<ifname>' NCCL line; `ip -o addr` inventory per node; ping/nc reachability matrix on the chosen NIC subnet.",
  conf="Some clusters legitimately route over a bridge; a firewall drop (see other hypotheses) produces the same hang with correct interface names.",
  risk=["Pinning IFNAME cluster-wide can break heterogeneous nodes whose NIC names differ", "Excluding an interface may silently downgrade to a slower path"],
  ev=["NCCL_DEBUG=INFO NET/Socket lines from all ranks", "ip -o addr per node", "nc -vz reachability matrix on bootstrap port"],
  rb="Rollback gate: revert the IFNAME pin if any rank fails to find a matching interface (NCCL logs 'no usable interface') or if post-fix all-reduce bus bandwidth is below 80% of the recorded baseline (ESTIMATE threshold; baseline must be MEASURED with nccl-tests before the change)."),
 dict(
  name="Launcher env: duplicated/absent ranks — world_size mismatch",
  hyp="H1: the collective never forms because the union of RANK values is not exactly {0..world_size-1}: a launcher misconfiguration (nnodes/nproc_per_node mismatch, or a resurrected pod) produced a duplicate rank or a missing rank. Falsifiable: dumping (hostname, LOCAL_RANK, RANK, WORLD_SIZE) from every process yields a multiset whose RANK set != range(WORLD_SIZE).",
  exp="Controlled experiment: insert a pre-init barrier-free dump of the env tuple to a shared file, then reduce to a 2-node run with explicitly hand-set --node_rank. Prediction under H1: the dump shows the duplicate/missing rank, and the hand-set 2-node run succeeds.",
  meas="Env tuple census; scheduler allocation list (nodes actually assigned) vs nnodes; process count per node vs nproc_per_node.",
  conf="A crashed rank that exits after printing its env looks like a complete census; retry logic in the launcher can silently re-spawn a rank.",
  risk=["Editing launcher flags can change the effective global batch size and therefore training dynamics", "A missing rank may indicate an unhealthy node that will fail again mid-run"],
  ev=["Per-process env tuple dump", "scheduler allocation record", "`pgrep -c` per node", "launcher stdout showing spawn count"],
  rb="Rollback gate: if the corrected world configuration changes global batch size, revert and instead fix the node allocation; do not silently continue training with an altered effective batch size."),
 dict(
  name="Device visibility: two ranks mapped to the same GPU / CUDA_VISIBLE_DEVICES overlap",
  hyp="H1: two ranks call cudaSetDevice on the same physical GPU (mis-set CUDA_VISIBLE_DEVICES, or LOCAL_RANK not applied), and ncclCommInitRank deadlocks because the expected per-device communicators never all appear. Falsifiable: `nvidia-smi --query-compute-apps=pid,gpu_uuid` shows two job PIDs sharing one GPU UUID while another GPU has none.",
  exp="Controlled experiment: log (rank, torch.cuda.current_device(), GPU UUID) before init; then rerun with CUDA_VISIBLE_DEVICES unset and device chosen strictly as LOCAL_RANK. Prediction under H1: the first log shows the collision and the corrected run initializes.",
  meas="rank->GPU-UUID map; nvidia-smi compute-apps census; per-GPU memory footprint asymmetry before the hang.",
  conf="MPS or an intentional multi-process-per-GPU config also shows sharing without being a bug; a zombie process from a prior job can occupy a GPU.",
  risk=["Unsetting CUDA_VISIBLE_DEVICES may expose GPUs reserved for another tenant", "Killing the 'extra' process may kill an unrelated tenant job"],
  ev=["rank->UUID mapping file", "nvidia-smi --query-compute-apps output", "cgroup/device allowlist for the container"],
  rb="Rollback gate: if the node is shared, do not unset CUDA_VISIBLE_DEVICES in production; restore the original mapping and escalate to the scheduler owner."),
 dict(
  name="RoCE fabric: GID index / RoCEv1-vs-v2 mismatch stalls IB verbs connection setup",
  hyp="H1: the hang is in the IB/RoCE transport handshake, not in TCP bootstrap: ranks pick different GID indices (RoCEv1 vs RoCEv2, or an IPv6 link-local GID) so QP connection never reaches RTS. Falsifiable: NCCL logs reach 'NET/IB : Using [0]mlx5_0' on all ranks but never print the first 'Connected all rings', and `show_gids` reports differing v1/v2 index selection across nodes.",
  exp="Controlled experiment: run `ib_write_bw` between two hanging nodes with an explicitly matched -x GID index; then rerun NCCL with NCCL_IB_GID_INDEX pinned to the RoCEv2 index. Prediction under H1: ib_write_bw with matched GID succeeds while the default fails, and the pinned NCCL run gets past connection setup.",
  meas="show_gids table per node; ib_write_bw success/failure and MEASURED bandwidth (GB/s); NCCL log progression to 'Connected all rings'; PFC/ECN counters from `ethtool -S` for drops during the attempt.",
  conf="A lossless-fabric misconfiguration (missing PFC on one switch port) causes an identical stall with correct GIDs; MTU mismatch also stalls after successful GID selection.",
  risk=["Pinning a GID index cluster-wide breaks nodes whose index differs", "Falling back to NCCL_IB_DISABLE=1 masks the fault and silently drops throughput by a large factor"],
  ev=["show_gids output per node", "ib_write_bw logs with explicit GID", "switch PFC/ECN counter deltas", "NCCL_DEBUG=INFO NET/IB lines"],
  rb="Rollback gate: NCCL_IB_DISABLE=1 is a diagnostic only. If used, revert within the same session; do not run production training on the TCP fallback if measured all-reduce bus bandwidth is below 50% of the MEASURED IB baseline."),
 dict(
  name="Firewall/security group blocks the ephemeral bootstrap port range",
  hyp="H1: only MASTER_PORT was opened between nodes; NCCL's bootstrap and ring connections use ephemeral ports that are dropped, so the store rendezvous succeeds but the NCCL bootstrap ring hangs. Falsifiable: TCPStore-based `dist.barrier()` on the gloo backend completes while the NCCL communicator init does not, and packet capture shows SYNs without SYN-ACKs on non-MASTER_PORT ports.",
  exp="Controlled experiment: (a) run a gloo-only barrier across all ranks; (b) `tcpdump -n 'tcp[tcpflags] & tcp-syn != 0'` on a hanging pair during init. Prediction under H1: gloo barrier passes, tcpdump shows unanswered SYNs, and opening the full inter-node range (or setting NCCL_SOCKET_NTHREADS with a fixed port policy) unblocks it.",
  meas="gloo barrier pass/fail; SYN vs SYN-ACK counts per port; iptables/nftables counters (`nft list ruleset` with counters) incrementing on drop rules.",
  conf="A dropped path can also be caused by asymmetric routing; a host-level DROP and a switch ACL look identical from the endpoint.",
  risk=["Opening broad port ranges weakens the node security posture", "Changes to firewall rules may affect co-tenant workloads on the same subnet"],
  ev=["gloo barrier result", "tcpdump capture on both endpoints", "firewall ruleset with counters", "change-approval record for any ACL edit"],
  rb="Rollback gate: apply the ACL change to two nodes only first; if the pair-wise nccl-tests all-reduce does not complete within 60 s (ESTIMATE, based on a 2-node 8-GPU baseline), revert the ACL immediately and escalate to the network owner rather than widening the range cluster-wide."),
 dict(
  name="Container shared memory: /dev/shm too small or IPC namespace not shared",
  hyp="H1: intra-node transport setup hangs because NCCL's SHM path cannot allocate: the container has the default 64 MB /dev/shm, or ranks are in separate IPC namespaces so cudaIpc handles are not usable. Falsifiable: NCCL_DEBUG=INFO shows the last line around 'Channel .. via SHM' or 'shmOpen failed', and forcing NCCL_SHM_DISABLE=1 changes the failure point.",
  exp="Controlled experiment: (a) `df -h /dev/shm` and `ls -l /proc/<pid>/ns/ipc` for two ranks on one node; (b) rerun the single-node case with --shm-size=16g (or --ipc=host). Prediction under H1: single-node init succeeds after the shm/IPC fix while the original config still hangs.",
  meas="/dev/shm size and used bytes during init; IPC namespace inode equality across ranks; NCCL last-log-line before hang; single-node vs multi-node reproduction.",
  conf="NCCL_SHM_DISABLE=1 may 'fix' the hang by routing over P2P and hide the real limit; some runtimes silently ignore --shm-size.",
  risk=["--ipc=host removes container isolation between tenants", "Oversized /dev/shm consumes host RAM and can trigger the OOM killer on the trainer"],
  ev=["df -h /dev/shm sampled during the hang", "IPC namespace inodes per rank", "container runtime spec (docker inspect / pod securityContext)", "host free memory during the run"],
  rb="Rollback gate: prefer raising --shm-size over --ipc=host. If host free memory drops below 10% during init (MEASURED via /proc/meminfo), revert the shm size and reduce it stepwise."),
 dict(
  name="Version/config skew: mismatched NCCL or CUDA driver across nodes",
  hyp="H1: nodes run different NCCL builds (or a driver too old for the container's CUDA), so protocol/channel negotiation never converges and init blocks. Falsifiable: a per-node inventory shows differing NCCL version strings or driver versions; the hang disappears when the run is confined to a version-homogeneous node subset.",
  exp="Controlled experiment: collect `python -c 'import torch;print(torch.cuda.nccl.version())'`, `nvidia-smi --query-gpu=driver_version`, and container image digest from every node; then run only on the largest homogeneous subset. Prediction under H1: the homogeneous subset initializes and any mixed pairing reproduces the hang.",
  meas="version inventory table (node, NCCL, driver, image digest); pass/fail per node-pair; NCCL 'Bootstrap : Using' + version banner lines.",
  conf="A single bad node can correlate with a version difference without the version being causal; image digests can differ while the NCCL binary is identical.",
  risk=["Draining nodes to homogenize versions reduces cluster capacity", "Driver upgrades require a node reboot and can invalidate running jobs"],
  ev=["version inventory from all nodes", "pairwise nccl-tests matrix", "image digest per pod", "maintenance window approval for any driver change"],
  rb="Rollback gate: perform driver/NCCL upgrades on one node first; if that node fails a 2-node nccl-tests all-reduce, roll the node back to the previous image/driver and keep it cordoned rather than upgrading the fleet."),
 dict(
  name="Straggler rank: slow CUDA context init / ECC retirement / no init timeout set",
  hyp="H1: there is no true deadlock but an unbounded wait: one rank is extremely slow to reach ncclCommInitRank (cold page cache on a large checkpoint, GPU in a degraded state doing ECC page retirement, or MPS server contention), and because no init timeout is configured the healthy ranks wait forever. Falsifiable: py-spy dumps show N-1 ranks inside the collective and exactly one rank still in model load / cudaMalloc, and nvidia-smi on that node shows elevated retired-page or throttle counters.",
  exp="Controlled experiment: attach `py-spy dump --pid` to every rank at T+120 s, and set an explicit init timeout (e.g. timedelta(minutes=10)) so the run raises instead of hanging. Prediction under H1: exactly one rank has a distinct stack, and the timed-out run names that rank in the error.",
  meas="Per-rank py-spy stack at a fixed offset after launch; wall time from process start to init entry per rank; nvidia-smi ECC/retired-pages and clocks-throttle-reasons on the suspect node.",
  conf="A genuinely deadlocked run also shows one odd stack if that rank crashed; timeouts convert a hang into a crash without fixing the cause.",
  risk=["Setting a short timeout can abort otherwise-healthy large-scale runs during slow checkpoint loads", "Continuing on a GPU with ongoing ECC retirement risks silent data corruption"],
  ev=["py-spy dumps for all ranks", "nvidia-smi -q ECC and throttle sections", "checkpoint load duration per rank", "node health-check history"],
  rb="Rollback gate: if the suspect GPU reports uncorrectable ECC errors or pending page retirement, cordon the node and do not resume training on it; revert any timeout shortening if it aborts a run that previously completed init within that window."),
 dict(
  name="Double rendezvous / stale communicator ID from a previous job",
  hyp="H1: two rendezvous mechanisms are active simultaneously (MPI/PMIx plus torchrun's TCPStore), or a stale NCCL_COMM_ID / store key namespace from a previous job is reused, so ranks split across two half-groups and neither reaches world_size. Falsifiable: the store contains keys from an earlier run (timestamps predating this launch), or ranks report two distinct rendezvous endpoints in their logs.",
  exp="Controlled experiment: launch with a fresh, unique rendezvous id (torchrun --rdzv_id=$(uuidgen), --rdzv_backend=c10d) and with MPI-based init explicitly disabled; inspect the store/etcd namespace for pre-existing keys first. Prediction under H1: pre-existing keys are found, and the unique-id run initializes normally.",
  meas="Store/etcd key listing with creation timestamps; per-rank rendezvous endpoint from logs; count of ranks reaching the rendezvous barrier.",
  conf="A crashed previous job may leave keys that are harmless; a unique id also incidentally changes port assignment, which can mask a firewall issue.",
  risk=["Purging a shared store namespace can disrupt other jobs using the same backend", "Disabling MPI init may break launcher assumptions elsewhere in the pipeline"],
  ev=["store/etcd key dump with timestamps before purge", "per-rank rendezvous endpoint logs", "launcher command line as actually executed", "confirmation that no other job uses the namespace"],
  rb="Rollback gate: snapshot the store namespace before purging; if the unique-id run still hangs, restore the original launcher configuration and treat the port change as a confound to re-test under the firewall hypothesis."),
]

HEADER = ("Assumptions (state and verify before acting): multi-node, multi-GPU PyTorch + NCCL job; the hang is "
          "observed at process-group / communicator initialization, before the first training step; no rank has "
          "exited; you have shell access to all nodes and permission to set environment variables and relaunch. "
          "All numeric thresholds below are ESTIMATE unless a baseline is explicitly MEASURED first.\n\n"
          "Step 0 - freeze evidence before changing anything. Record per rank: hostname, PID, RANK, LOCAL_RANK, "
          "WORLD_SIZE, MASTER_ADDR, MASTER_PORT, CUDA_VISIBLE_DEVICES, NCCL_* variables, NCCL and driver versions, "
          "and the full NCCL_DEBUG=INFO log up to the last emitted line. Capture `py-spy dump` for every rank and "
          "`nvidia-smi` on every node. Rationale: the last NCCL log line localizes the hang to bootstrap, transport "
          "setup, or ring connection, and that single fact eliminates most of the hypothesis space. Changing "
          "environment variables before this capture destroys the only evidence you get for free.\n\n")

FOOTER = ("\n\nGeneral procedure and boundary conditions.\n"
          "1. Bisect the scale: single process -> single node all GPUs -> two nodes -> full world. The smallest "
          "configuration that reproduces the hang bounds the fault domain to intra-node (device/SHM/P2P) or "
          "inter-node (bootstrap/fabric/ACL).\n"
          "2. Always run a minimal collective as the probe, not the training job: nccl-tests `all_reduce_perf -b 8 "
          "-e 128M -f 2 -g <gpus>` or a 10-line torch script doing dist.init_process_group + a single all_reduce of "
          "one tensor. Probe latency is seconds, so a failed hypothesis costs little.\n"
          "3. Change exactly one variable per trial and keep a written log of (trial id, changed variable, outcome, "
          "duration). Compound changes make the result uninterpretable.\n"
          "4. Set an explicit init timeout so future occurrences fail loudly with a rank name instead of hanging; "
          "this is an observability fix, not a root-cause fix.\n"
          "5. Confounders that apply to every hypothesis: a genuinely slow checkpoint load, a co-tenant saturating "
          "the fabric, and retry logic in the launcher that silently re-spawns ranks. Control for these by recording "
          "wall-clock timestamps per rank and fabric counters during each trial.\n"
          "6. Global rollback gate: if two consecutive trials do not narrow the fault domain, revert every "
          "environment change to the last known-good configuration, re-verify with the minimal all-reduce probe, and "
          "escalate with the frozen evidence rather than continuing to mutate production configuration.\n"
          "7. Exit criterion: the minimal all-reduce probe passes on the full world size, and the training job "
          "reaches step 1 with measured all-reduce bus bandwidth within 10% (ESTIMATE tolerance) of the MEASURED "
          "pre-incident baseline. Without a recorded baseline, treat the fix as unverified.")


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
                "No rollback gate or evidence list in the source text",
            ],
            "evidence_required": mech["ev"] + [
                "MEASURED pre-incident all-reduce bandwidth baseline",
                "Frozen per-rank NCCL_DEBUG=INFO logs from before any configuration change",
            ],
            "confidence": 0.62,
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out), "records", rows[0]["id"], "->", rows[-1]["id"])


main()
