import json, os, hashlib

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
CORP = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT = os.path.join(EXP, "results", "train-batch-0162.jsonl")
START, N = 1610, 10

rows = [json.loads(l) for l in open(CORP, encoding="utf-8")]
sel = rows[START:START + N]

ASSUM = ("Assumptions: the hang is reproducible on the same node set; launcher, world size, container "
         "image digest and fabric path are held constant between control and treatment arms; exactly one "
         "variable changes per arm; measured facts are reported separately from estimates, and no "
         "vendor-specific behaviour is asserted without a log line, counter or command output backing it.")

TRIAGE = ("Ordered triage before any tunable is touched: (1) freeze the failing state - do not restart, "
          "capture per-rank stacks with py-spy dump / gdb -p first, because a restart destroys the only "
          "evidence that distinguishes 'blocked in the driver' from 'blocked in the collective'; (2) build "
          "a rank census table - hostname, LOCAL_RANK, global RANK, PID, CUDA device UUID, WORLD_SIZE, "
          "MASTER_ADDR:PORT, NCCL version and image digest; (3) re-run with NCCL_DEBUG=INFO "
          "NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV and TORCH_DISTRIBUTED_DEBUG=DETAIL and record the LAST "
          "stage reached per rank, since the ranks that never print a stage localise the fault; (4) "
          "reproduce with a minimal standalone all-reduce outside the training script; (5) shrink the "
          "world - 2 ranks on 1 node, then 1 rank on each of 2 nodes - to separate intra-node transports "
          "(SHM, P2P/NVLink) from inter-node transports (TCP sockets, IB/RoCE).")

CONF = ("Expected confounders. Scheduler- or image-injected environment variables that silently override "
        "CUDA_VISIBLE_DEVICES, NCCL_SOCKET_IFNAME, NCCL_IB_HCA or the timeout; a node drained, rebooted or "
        "re-imaged between arms; leftover debug env vars from an earlier incident still in the launch "
        "template; and host-to-host heterogeneity in driver, firmware, NUMA layout or kernel. Capture a "
        "full environment diff between control and treatment; without it the comparison is uncontrolled "
        "and any conclusion is anecdote rather than evidence.")

ROLLBACK_TAIL = ("General rollback criteria: if two consecutive controlled arms fail to move the stall to a "
                 "different stage, stop tuning, restore the last known-good launch template verbatim, and "
                 "hand off the evidence bundle instead of stacking further environment overrides. Never "
                 "raise the NCCL/process-group timeout as a fix - a longer timeout converts a fast, "
                 "diagnosable failure into a slow, expensive one and hides the real signal.")

M = [
 dict(
  mech="IPv6 / dual-stack resolution mismatch in the rendezvous bootstrap",
  why=("MASTER_ADDR resolving to an AAAA record on some hosts and an A record on others splits the world "
       "into two rendezvous groups; each group waits for members that are connecting on the other family, "
       "so no rank errors out and the job simply never reaches quorum."),
  hyp=("H1: the world is split by address family, not by a broken link. Falsifiable prediction: "
       "getent ahosts MASTER_ADDR returns a different first-family answer on at least one node, and "
       "forcing a literal IPv4 MASTER_ADDR lets the identical job reach quorum on the same node set."),
  exp=("Controlled experiment: keep world size, nodes, image and transport fixed; arm A uses the hostname "
       "as today, arm B substitutes the literal IPv4 address of the master interface. Quorum under arm B "
       "with a stall under arm A isolates name resolution. Do not simultaneously change "
       "NCCL_SOCKET_FAMILY and the address - that confounds two variables."),
  meas=["getent ahosts <MASTER_ADDR> output captured on every participating node, compared side by side",
        "ss -tnp on the master showing how many distinct peer addresses actually connected vs WORLD_SIZE",
        "/etc/nsswitch.conf, /etc/hosts and resolver config diffed across nodes",
        "quorum-reached wall time for arm A vs arm B, 3 repeats each with variance reported"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric's 'verify process-group rendezvous' step has no notion of address-family splitting, so a model imitating it will see a reachable master and wrongly clear the network.",
         "Hard-coding a literal IP into a launch template breaks on the next reschedule to a different master node; it is a diagnostic arm, not a durable configuration."],
  rollback=("Rollback gate: if the literal-IPv4 arm still stalls in 2 of 2 repeats, restore the hostname "
            "form immediately - leaving a pinned IP in the template creates a silent scheduling landmine "
            "with no demonstrated benefit."),
  qd=(3, 2, 2), conf=0.60),
 dict(
  mech="Host firewall / security group blocking the ephemeral ports NCCL opens beyond MASTER_PORT",
  why=("Only MASTER_PORT is usually allowed explicitly; NCCL's socket transport opens additional "
       "dynamically chosen ports between peers, so the TCPStore handshake succeeds and the subsequent "
       "peer-to-peer connection setup blocks with no rejection visible to the application."),
  hyp=("H2: the fault is a partially-open port policy, not a dead path. Falsifiable prediction: a TCP "
       "connect to MASTER_PORT succeeds between every node pair while a connect to a high ephemeral port "
       "on the same pair times out (SYN sent, no SYN-ACK), and firewall counters show drops in the stall window."),
  exp=("Controlled experiment: hold the job fixed and change only the port policy - arm A as today, arm B "
       "with NCCL_PORT_RANGE (or the equivalent pinned range) explicitly permitted end to end. Completion "
       "under arm B localises the fault to filtering. Verify with nc/ncat probes on the same ports before "
       "and after, so the conclusion does not rest on the job alone."),
  meas=["nc -zv node-to-node matrix on MASTER_PORT and on three sampled high ports, both directions",
        "iptables/nftables counters and cloud security-group flow logs filtered to the stall window",
        "tcpdump on one pair showing SYN retransmits with no SYN-ACK during the stall",
        "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET last line per rank, showing connect vs bootstrap stage"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric checks interface selection but never port reachability, so a model imitating it will conclude the interface is correct and stop while the policy still blocks the data path.",
         "Opening a wide ephemeral port range across the cluster to 'unblock' the job is a lasting security regression; scope any rule to the job's node set and record an expiry."],
  rollback=("Rollback gate: if the permitted-range arm does not clear the stall in 2 of 2 repeats, remove "
            "the firewall exception in the same change window - a standing open range with no proven "
            "benefit is pure attack surface."),
  qd=(3, 2, 2), conf=0.62),
 dict(
  mech="Topology detection stalling under virtualised / passed-through PCIe (no usable NCCL graph)",
  why=("In a VM or passthrough environment the PCIe tree exposed to the guest can omit the switch "
       "hierarchy, so NCCL's graph search explores a degenerate topology and either takes pathologically "
       "long or settles on a path that cannot carry the traffic - init appears frozen inside GRAPH."),
  hyp=("H3: the stall is in topology/graph search, not in transport connect. Falsifiable prediction: "
       "NCCL_DEBUG_SUBSYS=GRAPH is the last subsystem to emit output on the stalled ranks, and supplying "
       "an explicit NCCL_TOPO_FILE describing the real layout makes init complete on the same nodes."),
  exp=("Controlled experiment: identical job, arm A with autodetected topology, arm B with an explicit "
       "topology XML dumped from a known-good bare-metal peer of the same SKU. Completion under arm B "
       "isolates detection. Keep transport selection untouched in both arms so graph search is the only "
       "variable."),
  meas=["NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH full output, with the last emitted line per rank",
        "nvidia-smi topo -m inside the guest compared against the bare-metal reference for the same SKU",
        "lspci -tv in the guest, showing whether PCIe switches are visible or flattened",
        "init wall-clock time for arm A vs arm B, 3 repeats, with variance"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric's 'inspect topology' line does not distinguish a wrong topology from an undiscoverable one, so a model imitating it will accept whatever topo -m prints inside the guest as ground truth.",
         "A hand-written topology file that does not match the real hardware will produce silently wrong routing and misleading bandwidth numbers on every future run using that image."],
  rollback=("Rollback gate: if the explicit topology file does not both complete init and reproduce the "
            "reference busbw within 20%, delete the file from the image rather than shipping an unverified "
            "topology description to the whole fleet."),
  qd=(3, 2, 2), conf=0.55),
 dict(
  mech="Heterogeneous allocation: the scheduler granted nodes with differing GPU counts",
  why=("A launcher that assumes a uniform nproc_per_node across a heterogeneous allocation computes a "
       "WORLD_SIZE that some nodes cannot satisfy; the short node starts fewer ranks, quorum is never "
       "reached, and every started rank blocks indefinitely waiting for absent peers."),
  hyp=("H4: the world is arithmetically short because the allocation is not uniform. Falsifiable "
       "prediction: summing the visible GPU count over the allocated nodes is strictly less than "
       "WORLD_SIZE, and the rendezvous store's registered-member count equals that smaller sum and never grows."),
  exp=("Controlled experiment: without changing the fabric or the image, request an explicitly homogeneous "
       "allocation (same SKU, same GPUs-per-node) and rerun. Completion under the homogeneous arm confirms "
       "the mismatch. Alternatively set WORLD_SIZE to the measured sum on the same allocation - if it "
       "initialises, the arithmetic is causal."),
  meas=["scheduler allocation record (scontrol show job / equivalent) listing nodes and GPUs granted per node",
        "nvidia-smi -L on every allocated node, summed and compared against WORLD_SIZE explicitly",
        "registered-member count in the rendezvous store sampled over the stall, showing it plateaus below WORLD_SIZE",
        "per-node count of launched rank processes from ps, cross-checked with the rank census"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric treats world size as a given and only asks to 'record ranks', so a model imitating it will list the ranks that exist and never notice the ones that were never launched.",
         "Silently lowering WORLD_SIZE to make the job start changes the effective global batch size and invalidates comparison with prior runs unless the change is recorded in the run metadata."],
  rollback=("Rollback gate: if the homogeneous allocation still stalls, restore the original job spec "
            "before testing the next hypothesis so the baseline stays comparable, and do not leave a "
            "reduced WORLD_SIZE in the production template without re-recording the batch-size math."),
  qd=(3, 2, 2), conf=0.66),
 dict(
  mech="The stall is pre-NCCL: ranks blocked in a shared-storage read before the collective",
  why=("If every rank loads a checkpoint or dataset index from the same NFS/object mount before "
       "init_process_group, a stalled or throttled mount freezes all ranks in uninterruptible I/O; the "
       "symptom is indistinguishable from a collective hang unless stacks are inspected."),
  hyp=("H5: the ranks are blocked in filesystem I/O, not in NCCL. Falsifiable prediction: per-rank stack "
       "dumps show the top frame inside a read/open on the shared mount rather than inside a NCCL or "
       "c10d call, and process state is D (uninterruptible sleep) with no NCCL INIT lines emitted at all."),
  exp=("Controlled experiment: keep the cluster and job fixed and change only the data source - arm A "
       "reads from the shared mount, arm B reads a pre-staged copy on node-local NVMe. If arm B reaches "
       "the first collective while arm A does not, storage is causal and no fabric change is warranted."),
  meas=["py-spy dump (or gdb bt) for every rank, classified as: in NCCL, in c10d, or in filesystem I/O",
        "process state column from ps -eo pid,stat,wchan,comm for all ranks, counting D-state processes",
        "nfsstat -c / mount-server latency and mountstats retransmit counts during the stall window",
        "time-to-first-NCCL-log-line for arm A vs arm B, 2 repeats each"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric presumes the fault is inside collective initialization, so a model imitating it will run NCCL probes for hours on a job that never reached NCCL at all.",
         "Killing D-state ranks to 'reset' the job can leave stale NFS locks and partially written checkpoint files; unmount cleanly and validate checkpoint integrity before resuming."],
  rollback=("Rollback gate: if the node-local staging arm shows the same stall, revert to the shared mount "
            "and drop the storage hypothesis rather than rebuilding the data pipeline on a hunch; if it "
            "does clear, treat local staging as a temporary mitigation and file the mount issue with its owner."),
  qd=(3, 2, 2), conf=0.61),
 dict(
  mech="InfiniBand subnet manager down or ports not in ACTIVE state",
  why=("RDMA connection setup requires a live SM to have assigned LIDs and brought ports to ACTIVE; if the "
       "SM died or a port sits in INIT/DOWN, the socket bootstrap still succeeds over Ethernet while the "
       "IB queue-pair setup blocks, so the job hangs after rendezvous rather than at connect."),
  hyp=("H6: the fault is IB fabric state, not application logic. Falsifiable prediction: ibstat shows at "
       "least one participating port not in state ACTIVE (or with no assigned LID), and the identical job "
       "run with NCCL_IB_DISABLE=1 completes - slowly, over TCP - instead of hanging."),
  exp=("Controlled experiment: hold job, nodes and world size fixed and change only the transport, IB vs "
       "TCP sockets. Completion under TCP plus a non-ACTIVE port in ibstat localises the fault to the "
       "fabric. Confirm independently with ibping between the same pair before drawing the conclusion."),
  meas=["ibstat / ibstatus on every node: port state, physical state, LID, rate",
        "sminfo or the SM host's service status, with the SM's last restart timestamp",
        "ibping / ib_write_bw between the failing node pair, run outside the training job",
        "completion time of the identical job under NCCL_IB_DISABLE=1 as the TCP control arm"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric's network step stops at interface selection and has no concept of fabric-manager state, so a model imitating it will report the NIC as present and healthy while the port is not ACTIVE.",
         "Restarting a subnet manager re-sweeps the fabric and can briefly disrupt every RDMA job on it; this requires the fabric owner and a maintenance window, never an ad-hoc triage step."],
  rollback=("Rollback gate: NCCL_IB_DISABLE=1 is a diagnostic arm and a degraded-throughput mitigation "
            "only - remove it as soon as port state returns to ACTIVE, and if it is left in place for a "
            "production run, record the throughput penalty explicitly in the run metadata."),
  qd=(3, 2, 2), conf=0.64),
 dict(
  mech="Container image digest skew: nodes running different NCCL/CUDA builds",
  why=("A mutable tag re-pulled on some nodes but not others leaves the world running two different NCCL "
       "builds; protocol or feature negotiation between mismatched versions can block rather than fail "
       "cleanly, so init stalls with no version error surfaced to the user."),
  hyp=("H7: the ranks are not running identical software. Falsifiable prediction: the set of image "
       "digests collected across ranks has cardinality greater than one, and pinning every node to a "
       "single digest makes the same job initialise on the same hardware and fabric."),
  exp=("Controlled experiment: change only the image reference - arm A with the mutable tag as today, arm "
       "B with an explicit sha256 digest pinned across all nodes. Success under arm B isolates version "
       "skew. Do not upgrade NCCL and pin the digest in the same arm; that confounds skew with a version change."),
  meas=["image digest (sha256) reported by the container runtime on every node, tabulated and diffed",
        "per-rank NCCL version, CUDA runtime version and driver version from NCCL_DEBUG=VERSION output",
        "container image pull timestamps per node, to identify which nodes re-pulled a mutable tag",
        "init success/failure for the digest-pinned arm vs the mutable-tag arm, 2 repeats each"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric's 'record environment' step does not require image or library version identity, so a model imitating it will collect env vars and still miss a split-brain software stack.",
         "Re-pulling images mid-incident to force uniformity can pick up a newer upstream build and change two variables at once, destroying the controlled comparison."],
  rollback=("Rollback gate: if the digest-pinned arm still stalls, keep the digest pin (it removes a real "
            "variable) but revert any library upgrade applied alongside it, and record both digests in the "
            "incident notes before moving to the next hypothesis."),
  qd=(3, 2, 2), conf=0.65),
 dict(
  mech="PCIe ACS enabled on the root complex, breaking GPU-to-GPU P2P",
  why=("Access Control Services forces peer traffic up through the root complex instead of allowing direct "
       "P2P; NCCL may advertise a P2P path that then cannot be established or performs pathologically, so "
       "intra-node ring setup stalls or the first collective never returns."),
  hyp=("H8: intra-node peer access is administratively blocked, not physically absent. Falsifiable "
       "prediction: the CUDA p2pBandwidthLatencyTest reports peer access unavailable (or bandwidth at "
       "host-copy levels) between GPUs that nvidia-smi topo -m claims are directly connected, and ACS is "
       "enabled on the intervening bridges in lspci -vvv."),
  exp=("Controlled experiment: single node, same rank count, arm A as-is and arm B with NCCL_P2P_DISABLE=1. "
       "If arm B completes - slower - while arm A hangs, the P2P path is the fault. Disabling ACS on the "
       "bridges is the follow-up fix arm, done on a drained node, and must be measured separately."),
  meas=["lspci -vvv | grep -i 'ACSCtl' for every bridge between the GPUs, with SrcValid/TransBlk flags",
        "nvidia-smi topo -m peer matrix compared against p2pBandwidthLatencyTest measured bandwidth",
        "single-node all-reduce busbw with and without NCCL_P2P_DISABLE=1, 3 repeats",
        "NCCL_DEBUG=INFO lines showing which intra-node transport (P2P/SHM) was selected per pair"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric's single-node comparison is framed as a scale test, so a model imitating it will read a single-node hang as 'reproduces everywhere' rather than as evidence pointing at the intra-node P2P path.",
         "Disabling ACS weakens PCIe isolation between devices and is a security-relevant change on a multi-tenant host; it needs an owner sign-off and a drained node, not a live edit."],
  rollback=("Rollback gate: if disabling ACS does not raise measured intra-node busbw by at least 2x over "
            "the host-staged baseline, re-enable ACS before returning the node to the pool - an unproven "
            "isolation downgrade is not an acceptable residual state."),
  qd=(3, 2, 2), conf=0.58),
 dict(
  mech="Elastic-agent restart loop leaving duplicate members under one rendezvous run_id",
  why=("When a torchrun/elastic agent restarts a worker group without the store being cleaned, stale "
       "members from the previous generation remain registered; the new generation waits for peers that "
       "are already dead, so the round never reaches quorum and no rank errors out."),
  hyp=("H9: the store contains members from a previous generation. Falsifiable prediction: the rendezvous "
       "store lists more members than there are live rank PIDs on the cluster, some registered member "
       "addresses have no live process behind them, and a fresh unique run_id lets the same job start."),
  exp=("Controlled experiment: change only the rendezvous identity - arm A reuses the current run_id, arm "
       "B uses a freshly generated one against a cleared store, with nodes, image and world size fixed. "
       "Start under arm B isolates stale state. Keep the elastic min/max settings unchanged in both arms."),
  meas=["rendezvous store dump: registered members, their addresses and generation/round number",
        "live rank PID census across all nodes, cross-referenced against registered member addresses",
        "elastic agent logs showing restart events and their timestamps relative to the stall",
        "start success for a fresh run_id vs the reused run_id, 2 repeats each"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric's rendezvous check is a single line with no notion of generations or stale membership, so a model imitating it will see a reachable store with members present and call rendezvous healthy.",
         "Clearing a shared rendezvous store can evict a different, healthy job using the same backend; scope the deletion to this job's run_id key prefix and confirm ownership before deleting."],
  rollback=("Rollback gate: if the fresh run_id also stalls, stop touching the store - restore the original "
            "rendezvous configuration and move to the next hypothesis, because repeated store surgery adds "
            "uncontrolled variables and risks other tenants."),
  qd=(3, 2, 2), conf=0.63),
 dict(
  mech="Co-tenant GPU memory exhaustion blocking communicator buffer allocation",
  why=("NCCL allocates device-side buffers when the communicator is created; if a co-tenant or a leaked "
       "process from a previous run holds most of the GPU memory, that allocation blocks or retries "
       "instead of failing fast, so initialization appears to hang on the affected ranks only."),
  hyp=("H10: the stall is a device-memory shortage on a subset of GPUs, not a communication fault. "
       "Falsifiable prediction: nvidia-smi shows free memory below the communicator's buffer requirement "
       "on exactly the ranks that never emitted an INIT completion line, and those GPUs list a PID that "
       "does not belong to this job."),
  exp=("Controlled experiment: hold job and fabric fixed and change only device occupancy - arm A on the "
       "currently occupied nodes, arm B on nodes verified empty (zero non-job PIDs, full free memory). "
       "Init success on empty nodes isolates memory pressure. Do not simultaneously reduce buffer sizes; "
       "that would confound occupancy with configuration."),
  meas=["nvidia-smi --query-compute-apps=pid,used_memory --format=csv on every GPU, with owning user resolved",
        "nvidia-smi --query-gpu=memory.free,memory.total --format=csv sampled during the stall",
        "mapping of stalled ranks to GPU UUIDs, showing whether stalled ranks and low-free-memory GPUs coincide",
        "init success on a verified-empty node set vs the occupied set, 2 repeats each"],
  risks=["Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary about what a good answer would contain instead of performing the diagnosis.",
         "The rubric's GPU-visibility step only asks which devices are visible, not how much memory is free on them, so a model imitating it will confirm all GPUs are present and miss that they are full.",
         "Killing the processes holding GPU memory can terminate another tenant's multi-hour job; identify the owner and confirm the PIDs are orphans from a previous run before reclaiming anything."],
  rollback=("Rollback gate: if the verified-empty node set stalls identically, release those nodes and drop "
            "the memory hypothesis rather than continuing to reclaim GPUs; never leave a policy that "
            "auto-kills co-tenant PIDs in place as a workaround."),
  qd=(3, 2, 2), conf=0.67),
]

assert len(sel) == len(M) == N


def build(rec, m):
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
            "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV logs from every rank with the last stage identified",
            "TORCH_DISTRIBUTED_DEBUG=DETAIL output including any collective-mismatch warnings",
            "Full rank census: hostname, RANK, LOCAL_RANK, PID, CUDA device UUID, NCCL version, image digest",
        ] + m["meas"][:2] + [
            "Control-vs-treatment environment diff proving exactly one variable changed",
        ],
        "confidence": m["conf"],
    }


recs = [build(r, m) for r, m in zip(sel, M)]
h = {hashlib.sha256(r["corrected_answer"].encode()).hexdigest() for r in recs}
assert len(h) == N, "duplicate corrected_answer in batch"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(recs), [r["source_id"] for r in recs])
