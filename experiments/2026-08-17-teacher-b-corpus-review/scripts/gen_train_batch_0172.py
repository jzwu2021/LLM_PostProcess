#!/usr/bin/env python3
"""teacher-B blind review generator for train-batch-0172 (corpus rows 1711-1720)."""
import json, hashlib, os, glob

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
RESDIR = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
OUT = os.path.join(RESDIR, "train-batch-0172.jsonl")
START, COUNT = 1710, 10

MECH = [
 dict(
  name="torch.distributed store rendezvous deadlock: mismatched init_method, group ordering, or a stale rendezvous key",
  hyp="H1: the hang is upstream of NCCL entirely. Ranks disagree on the rendezvous identity (different MASTER_ADDR/MASTER_PORT pair, different c10d store key, or a leftover etcd/TCPStore key from a previous job), so init_process_group blocks in the store barrier and NCCL's communicator is never constructed. Falsifiable: py-spy dump on a stalled rank shows the stack inside the TCPStore/rendezvous wait rather than inside ncclCommInitRank, and NCCL_DEBUG=INFO has emitted no communicator lines at all.",
  exp="Controlled experiment: (a) py-spy dump every rank and classify the blocking frame as rendezvous-vs-collective; (b) relaunch with a freshly generated unique rendezvous id/port, changing nothing else. Prediction under H1: all stalled ranks block in the store wait, no NCCL init banner exists, and the fresh-id relaunch progresses past init_process_group.",
  meas="Per-rank py-spy stack classification; presence/absence of the NCCL version banner per rank; store port listener and established-connection count; MEASURED wall time from launch to init_process_group return on a known-good job.",
  conf="A rank that is merely slow to import torch looks identical to a stuck rank in a single dump, so take at least two dumps 30 s apart; a port already bound by an orphaned process from a prior job produces the same symptom via a different mechanism.",
  risk=["Killing an apparently orphaned store process can terminate a healthy co-tenant job sharing the node", "Randomizing the rendezvous port each launch defeats firewall allow-lists and can turn a hang into a hard connection failure"],
  ev=["two py-spy dumps per rank at least 30 s apart", "launcher-resolved MASTER_ADDR/MASTER_PORT as seen by each rank", "listing of processes holding the store port before launch", "record showing the previous job on these nodes exited cleanly"],
  rb="Rollback gate: change only the rendezvous id/port for one trial. If the fresh-id relaunch still blocks in the store wait, revert to the original launcher configuration and escalate to the network/ACL hypothesis rather than accumulating launcher edits."),
 dict(
  name="Collective-order divergence between ranks (conditional branch, uneven dataloader, or differing world-size assumption) causes a mismatched-op deadlock",
  hyp="H1: every rank initialized fine, but they do not issue the same sequence of collectives: one rank takes a conditional branch (e.g. skips an all-reduce on an empty shard, or a rank-0-only barrier) so ranks wait on different operations. Falsifiable: enabling TORCH_NCCL_DESYNC_DEBUG / the flight recorder shows different last-issued collective (op name, count, or datatype) on different ranks; the hang is reproducible at the same step index rather than at a random one.",
  exp="Controlled experiment: (a) capture the NCCL flight recorder / desync report and diff the last collective per rank; (b) rerun with the suspected conditional path forced to execute unconditionally on all ranks (e.g. pad the empty shard), leaving everything else fixed. Prediction under H1: the diff shows a specific rank issuing a different op, and forcing the uniform path removes the hang at that step.",
  meas="Per-rank last-collective record (op, size, dtype, sequence number) from the flight recorder; step index of the hang across repeated runs; per-rank sample counts per step; MEASURED variance in shard sizes.",
  conf="A genuinely slow rank (stragglering on I/O) also stalls the collective without any op mismatch; sequence numbers only align if all ranks use the same process group, so multiple subgroups can produce spurious 'mismatches'.",
  risk=["Padding shards changes the effective loss weighting and can silently alter training semantics", "Leaving desync debug enabled in production adds per-collective overhead and can mask the original timing"],
  ev=["flight-recorder dump from the failing run", "per-rank sample/shard count table", "reproducibility record: step index of the hang over at least three runs", "code path audit of every rank-conditional collective"],
  rb="Rollback gate: any padding or branch-uniformity change must be validated to leave the loss curve within the MEASURED run-to-run noise band (ESTIMATE: same-seed loss delta at step 100) before it ships; otherwise revert and fix the branch instead."),
 dict(
  name="No collective timeout configured, so a recoverable fault presents as an indefinite hang",
  hyp="H1: the underlying fault is transient or localized, but the process group was created with a very long (or effectively infinite) timeout and no watchdog, so instead of failing fast with a diagnostic the job blocks forever and destroys the evidence window. Falsifiable: the process-group timeout value in the launch config is far larger than the MEASURED p99 collective latency, and setting a short timeout converts the hang into an actionable exception naming the rank and op.",
  exp="Controlled experiment: relaunch with a timeout set to roughly 10x the MEASURED p99 collective latency and the NCCL watchdog/async-error-handling enabled, changing nothing else. Prediction under H1: the job now aborts within the timeout with a named rank/op instead of hanging indefinitely.",
  meas="Configured process-group timeout; MEASURED p50/p99 collective latency from a healthy baseline run; time-to-abort and the identity of the first aborting rank; frequency of abort across repeats.",
  conf="A timeout that is too aggressive will fire during legitimate long operations such as checkpoint save or the first CUDA-graph capture, producing false positives that look like the same fault.",
  risk=["Short timeouts can kill healthy long-tail jobs and waste hours of compute", "Aggressive async error handling may tear down the job before a checkpoint is flushed"],
  ev=["launch config showing the timeout in force", "baseline p99 collective latency distribution", "list of legitimately long operations and their MEASURED durations", "confirmation that checkpointing completes within the proposed timeout"],
  rb="Rollback gate: introduce the shortened timeout on a single trial run. If it fires during a known-good phase (checkpoint, graph capture, first step), raise it to above the MEASURED maximum of that phase or revert; never leave a timeout below the observed legitimate maximum."),
 dict(
  name="ECC / Xid fault or a wedged GPU context makes one rank's CUDA stream never complete",
  hyp="H1: one GPU is in a faulted state (double-bit ECC error, Xid 13/31/48/79, or a previously wedged context from a crashed process), so its kernels never retire and the collective it participates in never completes. Falsifiable: dmesg/nvidia-smi -q shows an Xid or retired-page/ECC event on exactly one device with a timestamp preceding the hang, and that device's utilization is pinned while others idle at the barrier.",
  exp="Controlled experiment: (a) correlate per-GPU Xid/ECC events and utilization with the identity of the rank that is not making progress; (b) rerun excluding the suspect GPU (or after a full device reset / node reboot), changing nothing else. Prediction under H1: the suspect device is the same one across repeats, and the run without it initializes and progresses.",
  meas="dmesg Xid entries with timestamps; nvidia-smi -q ECC aggregate and volatile counters plus retired/remapped page counts; per-GPU utilization and power during the hang; MEASURED pass/fail of the exclusion run.",
  conf="Persistence-mode transitions and other tenants on the node also generate Xids; a device reset clears the evidence, so capture nvidia-smi -q output before resetting anything.",
  risk=["Resetting or rebooting a node destroys in-memory state and any un-flushed checkpoint", "Silently excluding a GPU shrinks the world size and changes the effective batch size, invalidating comparisons"],
  ev=["pre-reset nvidia-smi -q full dump and dmesg excerpt", "mapping from rank to physical GPU UUID", "repeat-failure record showing the same UUID implicated", "hardware ticket if remapped pages exceed the vendor threshold"],
  rb="Rollback gate: exclusion is diagnostic. If the reduced-world run still hangs, restore the full node set and treat the GPU as exonerated. Any global-batch-size change made to accommodate a missing GPU must be reverted before comparing training metrics to prior runs."),
 dict(
  name="MPI / launcher bootstrap and NCCL bootstrap disagree, or a plugin (aws-ofi-nccl, SHARP) fails to load",
  hyp="H1: an external network plugin is selected but not loadable or misconfigured (missing libfabric provider, SHARP daemon down), so NCCL either retries plugin initialization indefinitely or negotiates a transport the peers do not have. Falsifiable: NCCL_DEBUG=INFO shows a 'NET/Plugin' load attempt with a failure or an unexpected provider, and forcing NCCL_NET=Socket changes the failure mode.",
  exp="Controlled experiment: (a) capture the plugin/provider lines from every rank and diff them; (b) relaunch with NCCL_NET=Socket as a functional control and, separately, with the plugin's provider explicitly pinned. Prediction under H1: the socket control initializes (slowly), proving the fault is in the plugin path rather than the fabric or the launcher.",
  meas="Plugin and provider selection lines per rank; fi_info provider list per node; SHARP daemon status if applicable; MEASURED bus bandwidth on the socket control vs the plugin path.",
  conf="The socket fallback is much slower, so 'it works' on the control does not mean the configuration is acceptable; a plugin present on some nodes and absent on others yields an asymmetric failure that mimics a network partition.",
  risk=["Shipping NCCL_NET=Socket as a workaround silently costs a large fraction of interconnect bandwidth", "Plugin version skew across nodes can corrupt collectives rather than just failing to initialize"],
  ev=["per-node plugin package versions and library paths", "fi_info output per node", "NCCL plugin selection log lines for all ranks", "MEASURED bandwidth comparison documenting the cost of the socket fallback"],
  rb="Rollback gate: NCCL_NET=Socket is a diagnostic control only. Restore the plugin path once fixed and confirm nccl-tests bus bandwidth returns to at least the MEASURED baseline; if it does not, roll back the plugin version rather than keeping the socket path."),
 dict(
  name="cgroup / MIG / device-visibility mismatch: the rank cannot see the GPU it is assigned",
  hyp="H1: the container's device cgroup, a MIG partition layout, or an exclusive-process compute mode means the GPU the rank tries to bind is not visible or is already owned by another process, so cudaSetDevice or the communicator's device binding blocks or errors late. Falsifiable: inside the container, the visible device count is smaller than LOCAL_WORLD_SIZE, or nvidia-smi shows compute mode EXCLUSIVE_PROCESS with a foreign PID holding the device.",
  exp="Controlled experiment: (a) from inside each container print the visible device list, MIG profile, and compute mode, and compare with the intended per-rank assignment; (b) relaunch after correcting the device mapping (or setting DEFAULT compute mode on one node as a control). Prediction under H1: the visible-device census is short on exactly the stalling node and the corrected mapping initializes.",
  meas="Per-container visible device UUIDs vs intended mapping; MIG instance layout per node; compute mode per device; foreign PIDs holding devices; MEASURED init success rate after correction.",
  conf="MIG instances change UUIDs when reconfigured, so a stale mapping file looks like a visibility bug; a co-tenant that exits mid-diagnosis makes the problem appear to fix itself.",
  risk=["Changing compute mode or reconfiguring MIG evicts co-tenant workloads on the node", "Rewriting device mappings by hand can place two ranks on one GPU and cause OOM"],
  ev=["in-container device census per rank", "MIG configuration snapshot with UUIDs", "compute-mode and process-ownership listing", "co-tenant inventory before any node-level change"],
  rb="Rollback gate: apply the mapping or compute-mode change to one node only, with the co-tenant owner notified. If the corrected node still stalls, restore the original MIG/compute-mode configuration before testing the next hypothesis."),
 dict(
  name="Persistent storage / checkpoint-loading barrier at startup is the real blocker, not the collective",
  hyp="H1: what looks like a collective hang is ranks waiting at a barrier while one rank performs a slow or stuck startup I/O operation — loading a large checkpoint or tokenizer from a saturated or degraded shared filesystem. Falsifiable: the slow rank's stack is in file I/O rather than NCCL, and the filesystem client shows outstanding operations or a stuck mount for that node.",
  exp="Controlled experiment: (a) stack-sample every rank and classify frames as I/O vs collective; (b) rerun with the checkpoint staged to node-local storage on the suspect node, changing nothing else. Prediction under H1: the stalled frame is in I/O, and local staging removes the delay.",
  meas="Per-rank stack classification; MEASURED checkpoint read throughput per node (MB/s); filesystem client outstanding-op counters and mount health; time from launch to first step with and without local staging.",
  conf="A slow first read may be cold-cache rather than a fault and disappears on the second attempt; local staging also changes memory pressure and page-cache behaviour, so it is not a clean single-variable change.",
  risk=["Staging large checkpoints to node-local disk can fill the disk and evict co-tenants", "Skipping checkpoint verification while chasing speed can silently load a truncated file"],
  ev=["per-rank stack samples with timestamps", "per-node read-throughput measurements against the same file", "filesystem health and outstanding-op metrics for the window", "checksum of the checkpoint proving completeness after staging"],
  rb="Rollback gate: verify the staged checkpoint's sha256 matches the source before training on it. If local disk free space drops below 15% (MEASURED from df) or the checksum mismatches, abort staging and revert to the shared path."),
 dict(
  name="Topology detection picks a pathological ring across a slow link (PCIe host bridge / cross-NUMA) so the first collective is effectively stalled",
  hyp="H1: NCCL's chosen ring or tree traverses a low-bandwidth hop — a cross-socket QPI/UPI path or a PCIe host bridge instead of NVLink — so the first sizeable all-reduce takes orders of magnitude longer than expected and is mistaken for a hang. Falsifiable: NCCL_DEBUG=INFO channel lines show a path through SYS/PHB where nvidia-smi topo -m reports NVLink is available, and the operation completes if given enough time.",
  exp="Controlled experiment: (a) record the channel/ring construction and compare against the topo matrix; (b) run nccl-tests all_reduce at the same message size and let it run to completion to distinguish 'slow' from 'stuck'; (c) as a control, force NCCL_P2P_DISABLE=1 and compare. Prediction under H1: the operation completes but at bandwidth far below the NVLink baseline, and the ring visibly avoids NVLink.",
  meas="NCCL channel/path lines per rank; nvidia-smi topo -m; MEASURED all_reduce bus bandwidth vs the NVLink baseline for the same size; completion time of the 'hung' operation when allowed to run.",
  conf="Process pinning by the launcher may be the real cause of the cross-NUMA path, not NCCL's detection; a slow path plus a genuine fault can coexist, so completion at low bandwidth does not exclude other hypotheses.",
  risk=["Disabling P2P as a workaround permanently caps intra-node bandwidth", "Rebinding processes to different NUMA nodes changes memory locality and can regress dataloader throughput"],
  ev=["topo matrix and NCCL channel construction from the same run", "MEASURED bandwidth table by message size against the baseline", "launcher CPU/NUMA binding configuration", "evidence the operation eventually completed rather than being stuck"],
  rb="Rollback gate: any affinity or P2P change is accepted only if MEASURED all_reduce bus bandwidth is at least equal to the pre-change baseline at the production message size; otherwise revert within the same session."),
 dict(
  name="MTU / jumbo-frame or PFC/ECN misconfiguration on RoCE makes large transfers stall while small ones succeed",
  hyp="H1: control-plane and small messages pass, but large RDMA transfers stall because MTU is inconsistent across the path, or PFC/ECN is enabled asymmetrically so pause frames deadlock a switch queue. Falsifiable: a size-sweep shows success below a threshold message size and stall above it, and switch counters show pause frames or ECN marks concentrated on the involved ports.",
  exp="Controlled experiment: run a message-size sweep (small to large) between the implicated node pair, then repeat after aligning MTU across all hops. Prediction under H1: there is a sharp size threshold where transfers stop completing, and MTU alignment moves or removes that threshold.",
  meas="Success/failure and MEASURED bandwidth per message size; MTU on every hop (host NIC, bond, switch port); PFC pause-frame and ECN-mark counters per port over the window; retransmission counters.",
  conf="A size threshold can also come from a memory-registration limit or a buffer-pool exhaustion rather than MTU; pause-frame counters accumulate from other tenants sharing the switch.",
  risk=["Changing MTU fleet-wide can black-hole traffic for any host missed by the change", "Disabling PFC to 'fix' a deadlock can destroy RoCE performance and cause massive retransmission"],
  ev=["end-to-end MTU inventory for every hop", "size-sweep results per node pair", "per-port PFC/ECN counter deltas scoped to the incident window", "network-team sign-off for any fabric-level change"],
  rb="Rollback gate: apply MTU changes to one node pair first and verify the size sweep completes end to end. If any hop cannot be aligned, revert the changed endpoints to the fleet MTU rather than leaving a mixed-MTU path in place."),
 dict(
  name="Host resource exhaustion (pinned-memory limit, file descriptors, or CPU oversubscription) blocks communicator setup",
  hyp="H1: the collective never initializes because a host-side limit is hit: RLIMIT_MEMLOCK too low for RDMA memory registration, file-descriptor exhaustion as world size grows, or CPU oversubscription starving the NCCL proxy threads. Falsifiable: `ulimit -l` inside the container is not unlimited while ib_write_bw fails to register memory, or open-fd count per process approaches the limit at the moment of the hang.",
  exp="Controlled experiment: (a) sample ulimit -l, open-fd counts, run-queue length and proxy-thread CPU time on the stalling ranks; (b) relaunch one node with memlock unlimited and the fd limit raised, changing nothing else. Prediction under H1: the resource ceiling is visibly hit on exactly the stalling ranks, and the raised-limit node passes the phase the control node stalls at.",
  meas="ulimit -l and -n inside each container; per-process open-fd count at hang time; run-queue length and proxy-thread CPU utilization; MEASURED registration success from a standalone ib_write_bw; fd count scaling vs world size.",
  conf="A raised limit can mask a genuine fd leak that will resurface at larger scale; CPU starvation may itself be caused by an unrelated co-tenant that leaves between measurements.",
  risk=["Unlimited memlock lets a buggy process pin all host memory and OOM the node", "Raising limits hides a leak that fails later at larger world size, at higher cost"],
  ev=["per-container ulimit snapshot", "fd-count time series per rank up to the hang", "host run-queue and per-thread CPU samples", "fd-count-vs-world-size scaling measurement to distinguish a ceiling from a leak"],
  rb="Rollback gate: raise limits on one node as a test. If the job progresses, still measure fd growth per step before rolling out — if fd count grows monotonically with steps (a leak), revert the limit change and fix the leak instead of raising ceilings."),
]


def build(m):
    return "\n".join([
        "Assumptions (state explicitly, correct me if the environment differs): a multi-node/multi-GPU PyTorch job using NCCL; the hang is at collective initialization or the first collective; no single rank has produced a Python traceback; and no configuration was changed between the last known-good run and this one. All numeric figures below are labelled ESTIMATE or MEASURED; nothing here is a platform-specific fact I have verified for your cluster.",
        "",
        "Primary falsifiable hypothesis under test: " + m["name"] + ".",
        m["hyp"],
        "",
        "Controlled experiment. " + m["exp"] + " Change exactly one variable per trial and keep a control node/rank untouched so the comparison is interpretable.",
        "",
        "Measurements to collect (all MEASURED unless noted): " + m["meas"],
        "",
        "Expected confounders: " + m["conf"] + " Additionally, a hang has no single signature: the same symptom is produced by network, launcher, device and host-resource faults, so record evidence that discriminates between them rather than evidence consistent with your favourite cause.",
        "",
        "Ordering discipline: capture evidence before mutating anything. Freeze per-rank NCCL_DEBUG=INFO logs, stack dumps and device state first; every subsequent change destroys some of that evidence. A hypothesis that cannot be tested without an irreversible action (node reset, MIG reconfiguration, fabric change) is tested last.",
        "",
        "Risks of the intervention: " + "; ".join(m["risk"]) + ".",
        "",
        "Evidence required before accepting this hypothesis: " + "; ".join(m["ev"]) + ".",
        "",
        m["rb"],
        "",
        "Exit criteria: the hypothesis is accepted only if the single-variable change reproducibly converts hang into progress across at least three trials (ESTIMATE: three is the minimum to distinguish a fix from a transient, given how often these faults are intermittent), and the mechanism is corroborated by an independent measurement rather than by the fix alone. Otherwise mark it excluded and move to the next hypothesis with the original configuration restored.",
    ])


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
                "Source assistant text is a grading rubric listing topics to mention, not an executable diagnosis; training on it teaches meta-commentary instead of reasoning",
                "Source text carries no rollback gate, no evidence list and no ESTIMATE/MEASURED discipline",
            ],
            "evidence_required": mech["ev"] + [
                "MEASURED baseline collective latency and bus bandwidth from the last known-good run",
                "Frozen per-rank NCCL_DEBUG=INFO logs and stack dumps captured before any configuration change",
            ],
            "confidence": 0.62,
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out), "records", rows[0]["id"], "->", rows[-1]["id"])


main()
