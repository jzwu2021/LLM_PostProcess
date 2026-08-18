import json, hashlib, os

CORPUS = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0154.jsonl"
START, N = 1530, 10  # 0-indexed line offset

ANS = {}

ANS["corpus-01689"] = """Mechanism: NCCL bootstrap happens over TCP sockets *before* any GPU transport is used. ncclCommInitRank first exchanges ncclUniqueId and per-rank bootstrap addresses over the interface NCCL auto-selects. If ranks on different nodes each pick a *different, non-routable* interface (docker0, virbr0, a second unconnected NIC), rank A publishes an address rank B can never connect to, and the bootstrap all-gather blocks forever with no error.

Falsifiable hypothesis H1: the hang is caused by interface auto-selection picking a non-routable NIC, not by GPU or fabric state. Prediction: NCCL_DEBUG=INFO shows different `NCCL INFO Bootstrap : Using <iface>:<ip>` lines across nodes, and those IPs are in disjoint/unroutable subnets.

Controlled experiment: change exactly one variable — set NCCL_SOCKET_IFNAME to the single verified data-plane interface on every node and rerun the identical job. If it initializes, H1 is confirmed; if it still hangs with all ranks now on one subnet, H1 is refuted and the fault lies upstream in rendezvous or downstream in transport.

Boundary conditions: valid only if the chosen interface exists with the same name on every node (heterogeneous NIC naming breaks this), and if the job is not inside a container namespace where the host NIC is invisible. Under those conditions the experiment is confounded and must be run with explicit per-node interface pinning instead.

Evidence required: NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP logs from *every* rank (not just rank 0); `ip -br addr` per node; a point-to-point `nc -zv <peer-ip> <port>` between the exact IPs NCCL printed; and confirmation that no rank died at import time (`sacct`/exit codes) since a dead rank is indistinguishable from a network partition at rank 0.

Confounders: a firewall dropping NCCL's ephemeral bootstrap ports produces the same symptom as bad interface selection; test connectivity on an ephemeral port, not just on MASTER_PORT.

Rollback gate: keep an explicit bounded init timeout (init_process_group(timeout=...)) so the failure is a stack trace rather than a silent wedge. Revert the NCCL_SOCKET_IFNAME pin if it does not fix init within one trial — do not accumulate env-var changes, since stacked NCCL overrides make the next hypothesis untestable."""

ANS["corpus-01690"] = """Mechanism: the process group is built from the *order* in which ranks call the collective, and every rank must call ncclCommInitRank / init_process_group with the same world size and the same group creation sequence. If one rank creates an extra subgroup (e.g. a TP group built conditionally on a flag that differs per node), the group IDs desynchronize and the ranks block on mismatched communicators — a hang, not an error, because NCCL has no way to detect the divergence.

Falsifiable hypothesis H1: the hang is a *group-construction divergence* across ranks, not a connectivity failure. Prediction: the count and order of `new_group()` / commInitRank calls logged per rank differ between at least two ranks, while raw TCP connectivity between all node pairs is healthy.

Controlled experiment: instrument every rank to log (group_index, ranks_list, world_size) at each group creation, then run the *same* job with parallelism degrees forced identical (TP=1, PP=1, DP=world) so only one group is created. If init succeeds with a single group and hangs whenever a second group is added, H1 is confirmed. Hold node set, launcher and NCCL env constant across both trials.

Boundary conditions: only valid if the framework does not lazily create groups inside the first forward pass; with lazy creation the divergence surfaces later and this experiment yields a false negative.

Evidence required: per-rank group-construction logs; the resolved parallelism config as *read by each process* (from /proc/<pid>/environ and the parsed config object, not the launch script); `py-spy dump --pid <pid>` on several hung ranks to show whether they are stopped at different call sites — divergent stacks are strong positive evidence.

Confounders: an uneven ranks-per-node allocation from the scheduler changes local group membership and mimics config divergence; verify the actual allocation before concluding.

Rollback gate: change one parallelism dimension per trial. If two consecutive trials do not move the hang location in the stack traces, abandon this hypothesis rather than tuning further, and revert to the last known-good parallelism config."""

ANS["corpus-01691"] = """Mechanism: on RoCE fabrics NCCL must pick a GID index that matches the peer's L3 addressing (RoCEv2/IPv4 vs RoCEv1/IPv6-mapped). GID selection happens during transport setup; if two nodes resolve incompatible GID indices, queue-pair connection requests are emitted but never completed, and NCCL sits in connect with no timeout — indistinguishable at the user level from a rendezvous hang.

Falsifiable hypothesis H1: init blocks in RDMA transport setup due to GID/index mismatch, and TCP-level rendezvous already completed successfully. Prediction: bootstrap lines appear in the logs on all ranks (rendezvous done) and the last log line is a NET/IB ring/channel line; `show_gids` reports different RoCE versions or a different index in use per node.

Controlled experiment: hold everything fixed and set NCCL_IB_DISABLE=1 to force the TCP socket path. If init completes, the fault is confined to the RDMA transport (H1 supported); if it still hangs, RDMA is exonerated and the fault is in bootstrap/config. Then re-enable IB and pin NCCL_IB_GID_INDEX to the verified RoCEv2 index on all nodes as the confirming trial.

Boundary conditions: NCCL_IB_DISABLE=1 changes achievable bandwidth by an order of magnitude, so it is a *diagnostic*, never a fix to leave in place. The experiment is invalid if the cluster mixes IB and RoCE nodes, where a single GID index cannot be correct everywhere.

Evidence required: `show_gids` / `ibv_devinfo -v` on every node; PFC and DSCP/ECN settings on the switch ports (a mismatched lossless config produces stalls that look like hangs); `ibv_rc_pingpong` between the two suspect nodes as an NCCL-independent transport control; NCCL_DEBUG_SUBSYS=NET logs from all ranks.

Confounders: link flap or an ARP/neighbour entry that resolves but blackholes traffic gives the same signature; check port counters (`ethtool -S`) for discards before and after the trial.

Rollback gate: any GID/PFC change is applied to one node pair first and reverted immediately if `ibv_rc_pingpong` does not improve; never roll a fabric-wide QoS change out on the strength of a single hung job."""

ANS["corpus-01692"] = """Mechanism: two ranks on the same node that map to the *same* physical GPU deadlock during init. NCCL builds intra-node transports assuming a bijection between ranks and devices; when LOCAL_RANK and CUDA_VISIBLE_DEVICES are both applied (double remapping), rank 0 and rank 1 can both land on device 0, one waiting for a peer that is itself, and commInitRank never returns.

Falsifiable hypothesis H1: the hang is caused by a non-bijective rank→device mapping on at least one node, not by inter-node networking. Prediction: `nvidia-smi` on the offending node shows fewer distinct GPUs with job contexts than local ranks, with two PIDs sharing one device — and the hang reproduces on a single node.

Controlled experiment: run the identical script with --nnodes=1 --nproc_per_node=<local_gpus>. Single-node removes the entire network variable. If it still hangs, H1 is strongly supported and the fault is local device binding; if single-node succeeds, device mapping is exonerated and attention moves to inter-node transport.

Boundary conditions: only decisive when the single-node run uses the same number of local ranks as the failing job; reducing nproc_per_node at the same time changes two variables and destroys the control.

Evidence required: `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv` on every node (count distinct UUIDs vs local ranks); the value of CUDA_VISIBLE_DEVICES and LOCAL_RANK read from /proc/<pid>/environ for each rank; the exact torch.cuda.set_device() call site; and `nvidia-smi topo -m` to confirm the visible device set is what the launcher intended.

Confounders: leftover zombie processes from a previous job holding GPU contexts look identical to double-binding; check process start times and kill stale PIDs before the trial. MPS or an exclusive-process compute mode adds a third look-alike failure.

Rollback gate: fix the mapping in exactly one place — either the launcher sets CUDA_VISIBLE_DEVICES or the code calls set_device(local_rank), never both. If one trial with the single-source mapping does not clear the hang, revert and re-open the hypothesis rather than layering more env overrides."""

ANS["corpus-01693"] = """Mechanism: intra-node NCCL transports fall back to a shared-memory path when P2P/NVLink is unavailable. That path allocates files under /dev/shm sized by NCCL_BUFFSIZE × channels × ranks. In a container with the default 64 MB /dev/shm, the allocation fails or blocks, and with P2P also disabled by IOMMU or ACS settings there is no remaining transport — init stalls rather than erroring cleanly.

Falsifiable hypothesis H1: init blocks on intra-node transport establishment because both P2P and the SHM fallback are unusable, not because of anything inter-node. Prediction: `p2pBandwidthLatencyTest` reports P2P=disabled between local GPUs, and /dev/shm is far smaller than channels × NCCL_BUFFSIZE × local_ranks.

Controlled experiment: hold the job fixed and rerun the container with --shm-size=16g (single variable). If init completes, the SHM ceiling was binding. As the second, independent trial, set NCCL_P2P_DISABLE=1 with the small shm restored: if that makes the hang *worse or identical*, the SHM path is confirmed as the only viable transport.

Boundary conditions: the experiment is only interpretable on a single node; run it with --nnodes=1 so inter-node transports cannot mask the result. It is invalid if the job also uses IPC-based dataloaders that independently consume /dev/shm.

Evidence required: `df -h /dev/shm` inside the container; NCCL_DEBUG_SUBSYS=INIT,SHM logs; `lspci -vvv | grep -i acs` and IOMMU state from dmesg (ACS enabled silently kills P2P); `nvidia-smi topo -m` to record the expected NVLink/PCIe path; and the container's `--ipc` mode.

Confounders: a host-level /dev/shm shared across containers can be exhausted by a *neighbouring* job, so the same config passes and fails nondeterministically — record free shm at failure time, not afterwards.

Rollback gate: raise shm and re-test once. Do not simultaneously set NCCL_P2P_DISABLE and NCCL_SHM_DISABLE as a "fix"; that leaves the job on the slowest possible transport and hides the real defect. Revert any ACS/IOMMU change immediately if a single-node all-reduce does not recover."""

ANS["corpus-01694"] = """Mechanism: rendezvous is a store-side barrier. torchrun's c10d backend has rank 0 bind MASTER_PORT and every other rank connect; init_process_group returns only when the store holds WORLD_SIZE entries. If the scheduler allocated fewer nodes than the launcher's --nnodes, or one rank crashed during import, the barrier can never be satisfied and all surviving ranks block silently and indefinitely.

Falsifiable hypothesis H1: fewer than WORLD_SIZE processes ever reached the store — a *participation shortfall*, not a transport fault. Prediction: the number of established TCP sessions on MASTER_PORT is strictly less than WORLD_SIZE, and at least one expected rank has no live process at all.

Controlled experiment: keep node set, image and env fixed and rerun with a bounded timeout (init_process_group(timeout=timedelta(minutes=2))) so the barrier converts the hang into a per-rank exception naming which ranks were missing. Then rerun with WORLD_SIZE reduced to exactly the ranks observed to connect. If that run initializes, H1 is confirmed and the defect is in allocation/launch, not in the fabric.

Boundary conditions: a bounded timeout is diagnostic only; the reduced-world run must keep the same nproc_per_node so local device binding is unchanged, otherwise two variables move at once.

Evidence required: `ss -tanp | grep <MASTER_PORT>` counted on all nodes at hang time; per-rank RANK/WORLD_SIZE/LOCAL_RANK read from /proc/<pid>/environ (never trust the submit script); `getent hosts $MASTER_ADDR` on every node to prove identical resolution; scheduler allocation (`scontrol show job`) compared against --nnodes; and the exit status of every rank.

Confounders: a zombie process from a prior job still holding MASTER_PORT causes new rank 0 to fail bind while others connect to the *old* store — a participation shortfall with a completely different root cause. Check port ownership PIDs, not just port state.

Rollback gate: one variable per trial; never leave the reduced world size in production. If two bounded-timeout trials name different missing ranks each time, the failure is nondeterministic (node health, not config) and the affected nodes should be drained before further experiments."""

ANS["corpus-01695"] = """Mechanism: NCCL's collectives are asynchronous with respect to the launching CPU thread, and by default a stuck collective has no host-side watchdog beyond the ProcessGroupNCCL timeout. A rank that entered init but is blocked in a CUDA call — e.g. waiting on a GPU whose context is wedged by a prior Xid error — never reports anything, so the *whole* job appears to hang at initialization even though only one device is faulty.

Falsifiable hypothesis H1: exactly one GPU/node is faulty and is holding the barrier, rather than a cluster-wide config error. Prediction: excluding that one node lets an otherwise identical job initialize, and dmesg on the suspect node contains an Xid/GPU-fallen-off-the-bus event.

Controlled experiment: bisect the node set. Run the identical job on the first half of the allocation, then the second half, holding image, launcher and env constant. A clean half plus a hanging half localizes the fault to a node; repeat within the failing half. If *every* subset hangs, H1 is refuted and the cause is global configuration.

Boundary conditions: bisection is only valid if the job can run at reduced world size without changing parallelism degrees; if TP size must equal a fixed GPU count, keep TP constant and shrink DP only, otherwise you are testing a different program.

Evidence required: dmesg -T | grep -i xid on all nodes; `nvidia-smi -q` ECC/retired-page and clock-throttle reasons; per-rank `py-spy dump` showing which ranks are inside a CUDA call vs inside the store barrier; TORCH_NCCL_ASYNC_ERROR_HANDLING=1 and TORCH_NCCL_BLOCKING_WAIT=1 to force the stuck rank to raise instead of wedging.

Confounders: thermal throttling or a slow filesystem mount on one node delays that rank past the timeout without any hardware fault; check whether the "faulty" node also fails a standalone single-GPU CUDA sample.

Rollback gate: drain a node only after it fails a standalone single-node test — bisection alone is correlational. Revert async-error-handling env vars after diagnosis; they change failure semantics and should not silently persist into production runs."""

ANS["corpus-01696"] = """Mechanism: mixed software versions across nodes break init deterministically. NCCL negotiates protocol and algorithm support during bootstrap; a rank running a different NCCL/CUDA/driver combination can advertise capabilities its peers do not implement, and the resulting handshake mismatch manifests as a block, not a version error, because the negotiation itself is a blocking all-gather.

Falsifiable hypothesis H1: at least one node runs a different NCCL/driver/container image than the others, and homogenizing the stack removes the hang. Prediction: the `NCCL version x.y.z+cudaA.B` banner differs between at least two ranks in NCCL_DEBUG=INFO output.

Controlled experiment: hold the node set and job fixed and rerun with every node pinned to a single image digest (not a mutable tag) and a single driver version verified by `nvidia-smi --query-gpu=driver_version`. If init succeeds, H1 is confirmed. As a negative control, deliberately run two nodes on the previously divergent versions and confirm the hang returns — a reproduced failure is stronger evidence than a single success.

Boundary conditions: valid only if all nodes can actually load the pinned image (a node with insufficient disk silently falls back to a cached older layer, which recreates the divergence you are trying to eliminate). Verify the running digest per rank, not the requested tag.

Evidence required: per-rank NCCL version banner, `nvidia-smi` driver version, container image digest from the runtime, and `ldd` on the process to confirm which libnccl.so was actually loaded (LD_LIBRARY_PATH can shadow the image's library).

Confounders: a mutable tag like :latest re-pulled mid-rollout means two nodes launched minutes apart legitimately differ; record pull timestamps. Also, a version match at the banner level does not exclude a mismatched libibverbs/rdma-core underneath.

Rollback gate: roll the pinned version to one node pair first and require a passing 2-node all-reduce before fleet-wide rollout. If the pinned version does not fix init in one trial, revert the pin immediately — leaving a half-rolled version makes every subsequent hypothesis untestable."""

ANS["corpus-01697"] = """Mechanism: the smallest sufficient probe decides whether the fault is in the framework or in the collective library. A minimal two-rank all-reduce exercises store rendezvous, communicator construction and one transport, but *none* of the model, dataloader or checkpoint code. Anything that hangs in the full job but passes in the probe is, by construction, not a NCCL/fabric problem.

Falsifiable hypothesis H1: the hang is in application-level setup (dataset shard scan, checkpoint load, tokenizer download) that runs before or interleaved with init_process_group, not in the collective stack. Prediction: a bare two-rank all-reduce on the same nodes, same image and same env completes in under a second, while the real job hangs with stack traces pointing outside NCCL.

Controlled experiment: run `torchrun --nnodes=2 --nproc_per_node=1` on a ~20-line script that only does init_process_group + all_reduce(tensor([1.0]).cuda()), holding node set, image and every NCCL env var identical to the failing job. Pass ⇒ H1 supported, escalate to the application layer. Hang ⇒ H1 refuted and the fault is in rendezvous/transport, which the probe now reproduces cheaply.

Boundary conditions: the probe must reuse the *exact* env (same MASTER_ADDR/PORT policy, same container, same NCCL_* vars). A probe run with a clean environment proves nothing about a job with twelve NCCL overrides.

Evidence required: probe exit status and wall time; per-rank py-spy dump from the real job showing the blocking frame (a frame in torch.distributed vs in a data or storage library is the decisive discriminator); NFS/S3 client metrics if the stack points at storage; and the elapsed time between the last log line and the hang.

Confounders: a job that hangs only at scale (e.g. 256 ranks) will pass a 2-rank probe for reasons unrelated to the application — scale-dependent effects such as store fan-in or port exhaustion require an intermediate-scale probe before concluding.

Rollback gate: escalate from 2 → 8 → full scale, one step per trial, and stop at the first scale that reproduces. Do not apply any "fix" until a probe reproduces the failure, because an unreproduced fix cannot be falsified."""

ANS["corpus-01698"] = """Mechanism: an environment that is correct in the submit script but wrong in the process is the most common silent cause. Schedulers, container entrypoints, module systems and shell rc files each rewrite the environment; NCCL and torch read only what the process actually has. A rank whose MASTER_ADDR was overwritten by a node-local hostname will connect to a store that only it can see and wait forever.

Falsifiable hypothesis H1: the effective per-process environment differs from the intended one on at least one rank, and this divergence — not the fabric — causes the block. Prediction: MASTER_ADDR / MASTER_PORT / WORLD_SIZE / NCCL_SOCKET_IFNAME read from /proc/<pid>/environ differ between at least two ranks, despite an identical submit script.

Controlled experiment: keep the job unchanged and add a pre-init dump of the resolved environment plus `hostname -f` from every rank into a shared log. Rerun once. If the dump shows divergence, remove the rewriting layer (e.g. unset the entrypoint override) as a single variable and rerun; init succeeding on that trial confirms H1. If the environment is provably identical across ranks and it still hangs, H1 is refuted and the environment is eliminated as a cause.

Boundary conditions: only valid if the dump happens inside the same process that calls init_process_group — a wrapper script's `env` output is a different process and may not reflect what Python sees.

Evidence required: /proc/<pid>/environ for every rank (NUL-separated, decoded), `hostname -f` and `getent hosts $MASTER_ADDR` per node, the scheduler's exported variables, and a diff of the environments pairwise across ranks.

Confounders: secrets and per-node paths legitimately differ between ranks; diff only the distributed-relevant keys or you will drown in false positives. A container entrypoint that sets NCCL vars conditionally on GPU count produces legitimate-looking divergence on heterogeneous nodes.

Rollback gate: set distributed env vars in exactly one place. Remove one rewriting layer per trial; if two trials produce no change in the hang signature, revert all env changes to the last known-good state before opening a new hypothesis, so the trial history stays interpretable."""

lines = open(CORPUS, encoding="utf-8").read().splitlines()[START:START+N]
out = []
seen = set()
for l in lines:
    d = json.loads(l)
    sid = d["id"]
    msgs = d["messages"]
    u = [m["content"] for m in msgs if m["role"] == "user"][0]
    a = [m["content"] for m in msgs if m["role"] == "assistant"][0]
    ca = ANS[sid]
    h = hashlib.sha256(ca.encode()).hexdigest()
    assert h not in seen, "duplicate corrected_answer " + sid
    seen.add(h)
    rec = {
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": ca,
        "quality_dimensions": {
            "technical_correctness": 3,
            "instruction_coverage": 2,
            "operational_safety": 3,
        },
        "risks": [
            "source_assistant is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis",
            "no concrete commands, thresholds or rollback gates in the source",
            "variant text is identical across many ids, risking memorization of a single template",
        ],
        "evidence_required": [
            "per-rank NCCL_DEBUG=INFO logs and /proc/<pid>/environ dumps",
            "node-level fabric/device state (nvidia-smi, ibv_devinfo, ss on MASTER_PORT, dmesg Xid)",
            "a reduced-scale control run holding all but one variable fixed",
        ],
        "confidence": 0.62,
    }
    out.append(json.dumps(rec, ensure_ascii=False))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("wrote", OUT, len(out))
