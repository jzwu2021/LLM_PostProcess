import json, hashlib, os

CORPUS = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0156.jsonl"
START, N = 1550, 10  # 0-indexed line offset

ANS = {}

ANS["corpus-01711"] = """Mechanism: NCCL bootstrap resolves MASTER_ADDR through the node's resolver stack (nsswitch -> /etc/hosts -> DNS). Rank 0 binds the store on the address its own resolver returns; peers connect to whatever *their* resolver returns for the same name. If one node has a stale /etc/hosts entry or a split-horizon DNS view, peers dial an address rank 0 never bound, the TCP SYN is silently dropped or lands on a dead host, and init_process_group blocks with no error because the store client retries indefinitely.

Falsifiable hypothesis H1: the hang is name-resolution divergence for MASTER_ADDR, not a fabric or GPU fault. Prediction: `getent hosts $MASTER_ADDR` returns different IPs on at least two nodes, and on the divergent node there is no established TCP session to rank 0's MASTER_PORT.

Controlled experiment: change exactly one variable -- replace MASTER_ADDR with rank 0's literal IPv4 address (the one rank 0 actually bound, taken from `ss -tlnp`) and rerun the identical job. Init succeeding confirms H1; still hanging refutes it and moves the fault to rendezvous participation or transport.

Boundary conditions: valid only if all nodes share one routable L3 path to that IP. Invalid where rank 0 is multi-homed and binds 0.0.0.0 -- then the literal-IP test can pass for a reason unrelated to DNS. Also invalid inside container network namespaces with NAT, where the bound address is not the reachable one.

Measurements and evidence required: `getent hosts $MASTER_ADDR` and `hostname -f` per node; `ss -tlnp` on rank 0 showing the bound address:port; `ss -tanp | grep $MASTER_PORT | wc -l` per node at hang time (expect < WORLD_SIZE-1 established); resolver config (`/etc/nsswitch.conf`, `/etc/resolv.conf`, `/etc/hosts`) diffed pairwise; timestamps of the last log line per rank.

Confounders: a firewall dropping SYN produces the same "connect never completes" signature with correct DNS -- distinguish by attempting `nc -zv <rank0-ip> $MASTER_PORT` from the suspect node. A rank that died at import also shows no session; check exit codes before blaming the network.

Rollback gate: literal IP is a diagnostic, not a fix -- it breaks on rescheduling. Revert to the name once the resolver defect is corrected, and require a passing 2-node bounded-timeout init (init_process_group(timeout=timedelta(seconds=120))) before restoring full scale. If one literal-IP trial does not clear the hang, revert and reopen the hypothesis rather than stacking env overrides."""

ANS["corpus-01712"] = """Mechanism: rendezvous uses one well-known port (MASTER_PORT), but NCCL's bootstrap and transport then open *ephemeral* ports in both directions. A security group or host firewall that allows only MASTER_PORT lets the store barrier complete and then wedges during communicator construction: the ranks have exchanged handles but cannot open the peer-to-peer bootstrap sockets, and NCCL retries connect without a timeout.

Falsifiable hypothesis H1: rendezvous completed and the block is in inter-rank ephemeral-port connectivity restricted by a firewall/security-group rule, not in the store. Prediction: every rank logs its bootstrap line (store barrier done) and the last log line on all ranks is a NET/Bootstrap connect line; `nc -zv <peer> <high-port>` fails between nodes while MASTER_PORT succeeds.

Controlled experiment: hold the job, image and node set fixed and set NCCL_SOCKET_NTHREADS/port policy aside -- instead open a *bounded* ephemeral range (e.g. 30000-30100) on both directions between the two suspect nodes only, and constrain the job to those two nodes. If init completes on the two-node run with the range open and hangs with it closed, H1 is confirmed. That is one variable (firewall rule state) toggled twice.

Boundary conditions: only interpretable if both directions are opened -- asymmetric rules give a false negative. Invalid if the cluster uses an overlay network where the enforcement point is not the host firewall.

Measurements and evidence required: `iptables -S` / cloud security-group rules on both nodes; conntrack or `nstat -az TcpExtListenDrops` deltas during the hang; NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET logs from every rank, not just rank 0; a plain `nc` listener/dialer pair on a high port as an NCCL-independent control; per-rank py-spy dump showing the blocking frame inside ncclCommInitRank.

Confounders: ephemeral port exhaustion (`ss -s` showing tens of thousands of TIME_WAIT) causes identical connect failures without any firewall rule; record port usage before concluding. MTU blackholes also fail only on large transfers, not on connect -- that would refute this hypothesis.

Rollback gate: never open a wide port range fleet-wide on the strength of one hung job. Apply the bounded range to one node pair, require a passing 2-node all-reduce, then expand one blast radius at a time; revert immediately if the pair test does not pass."""

ANS["corpus-01713"] = """Mechanism: GPUDirect RDMA requires the nvidia_peermem (or legacy nv_peer_mem) kernel module so the RDMA NIC can pin and address GPU BAR1 memory. If the module is absent on one node, NCCL's IB transport setup either falls back or, when NCCL_NET_GDR_LEVEL forces GDR, attempts a registration that never completes; the queue pairs are created but the connection handshake stalls, producing a hang at init rather than a clean capability error.

Falsifiable hypothesis H1: init blocks in RDMA memory registration because GPUDirect RDMA is unavailable on at least one node, while rendezvous and TCP bootstrap already succeeded. Prediction: `lsmod | grep -E 'nvidia_peermem|nv_peer_mem'` is empty on the suspect node, and NCCL logs on healthy nodes show `[GDR] GPU Direct RDMA Enabled` while the suspect node does not.

Controlled experiment: single variable -- set NCCL_NET_GDR_LEVEL=0 (disable GDR, keep IB) and rerun the identical job. If init completes, the fault is confined to the GDR path (H1 supported). Confirming trial: load nvidia_peermem on the suspect node, restore the original GDR level, and rerun; success there is the positive control.

Boundary conditions: disabling GDR routes traffic through host bounce buffers and will reduce achievable inter-node bandwidth substantially -- it is a diagnostic, never a production fix. The experiment is invalid if the node lacks a PCIe path between the NIC and GPU under the same root complex, in which case GDR was never usable and its absence is expected, not a defect.

Measurements and evidence required: `lsmod` and `dmesg -T | grep -i peermem` per node; `nvidia-smi topo -m` to record NIC-GPU affinity (PIX/PXB vs SYS); `ibv_devinfo -v` for active ports and MTU; NCCL_DEBUG_SUBSYS=NET,GRAPH logs from every rank; MEASURED all-reduce bus bandwidth in GB/s with GDR on vs off, so the cost of the diagnostic is quantified rather than assumed.

Confounders: a driver/rdma-core version mismatch prevents peermem from loading and looks identical to it simply being missing. Locked-memory limits (`ulimit -l` not unlimited in the container) also break RDMA registration with the same stall signature -- check both before concluding.

Rollback gate: load the module on one node and require a 2-node `ib_write_bw` plus a 2-node NCCL all-reduce to pass before touching the rest of the fleet. Revert NCCL_NET_GDR_LEVEL to its original value once diagnosis ends; a lingering GDR-disabled override silently halves throughput and hides regressions."""

ANS["corpus-01714"] = """Mechanism: RDMA queue pairs must pin host and GPU memory. Pinning is bounded by RLIMIT_MEMLOCK, which containers and some schedulers default to 64 KB. NCCL creates its IB transport buffers during communicator construction; when registration fails under the limit, the failure surfaces as a retry loop inside transport setup, so the job appears to hang at initialization even though rendezvous already succeeded.

Falsifiable hypothesis H1: init blocks because locked-memory limits prevent RDMA buffer registration, not because of fabric topology or GPU state. Prediction: `ulimit -l` inside the failing process is a finite small value (not unlimited), and NCCL logs stop immediately after the first NET/IB line with no ring/channel lines.

Controlled experiment: hold everything fixed and rerun the container with --ulimit memlock=-1 (single variable). Init completing confirms H1. Negative control: restore the small limit and confirm the hang returns -- a reproduced failure is stronger than a single success.

Boundary conditions: only valid if the limit is read from the *process* (cat /proc/<pid>/limits), not from an interactive shell on the host, which frequently differs. Invalid if the job already runs privileged with unlimited memlock, in which case another registration failure mode (missing peermem, wrong GID) must be tested instead.

Measurements and evidence required: /proc/<pid>/limits "Max locked memory" for every rank; the container runtime's ulimit config; `ibv_devinfo` max_mr_size and max_qp; NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET logs from all ranks; MEASURED time from process start to last log line; `ibv_rc_pingpong` between the two suspect nodes as an NCCL-independent registration control.

Confounders: cgroup memory limits that trigger the OOM killer on one rank produce a silent participation shortfall that mimics a registration stall -- check dmesg for oom-kill and verify every rank is still alive. Hugepage exhaustion is a third look-alike.

Rollback gate: raise memlock for one job at one node pair, require `ibv_rc_pingpong` and a 2-node all-reduce to pass, then widen. If the unlimited-memlock trial does not clear the hang in one attempt, revert the ulimit change and reopen the hypothesis -- do not accumulate container privilege escalations as a debugging habit, since each one enlarges the security blast radius."""

ANS["corpus-01715"] = """Mechanism: two rendezvous mechanisms can be active simultaneously. Under Slurm, srun exports PMI/PMIx variables while torchrun sets its own RANK/WORLD_SIZE/MASTER_ADDR. If the framework's init picks PMI on some ranks and c10d TCPStore on others, the two groups build disjoint barriers: neither ever reaches WORLD_SIZE, and both halves block indefinitely with no error, because each half is waiting for peers that joined the other rendezvous.

Falsifiable hypothesis H1: the hang is caused by two concurrent, incompatible rendezvous backends, not by connectivity. Prediction: /proc/<pid>/environ shows both PMI_* / SLURM_PROCID and RANK/WORLD_SIZE present, and the RANK values derived by the two paths disagree for at least one process.

Controlled experiment: single variable -- launch with exactly one mechanism. Trial A: `srun` alone with the framework's PMI init and all torchrun variables unset. Trial B: a single `srun -n1` per node that execs `torchrun` with PMI variables explicitly unset. If either trial initializes while the mixed launch hangs, H1 is confirmed; if both single-mechanism trials also hang, H1 is refuted and the fault is downstream.

Boundary conditions: the comparison is only valid if node set, nproc_per_node and image are held constant across trials. Invalid if the framework silently prefers one backend regardless of environment -- verify which init path executed by logging the resolved backend and store type, not by reading the launch script.

Measurements and evidence required: per-rank decoded /proc/<pid>/environ filtered to RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, SLURM_PROCID, SLURM_NTASKS, PMI_*; the resolved rank/world logged from inside Python before init; `scontrol show job <id>` allocation compared with --nnodes x --nproc_per_node; count of established sessions on MASTER_PORT; per-rank py-spy dump showing which ranks sit in the store barrier vs a PMI barrier -- divergent stacks are decisive positive evidence.

Confounders: a correct mixed launch is possible when the wrapper deterministically derives RANK from SLURM_PROCID; the mere presence of both variable sets is not proof. Duplicate or misordered hostnames in the hostfile create the same shortfall with a different root cause.

Rollback gate: pick one rendezvous mechanism and enforce it in exactly one place. Roll the change at 2 nodes with a bounded init timeout so failures raise rather than wedge, and revert to the last known-good launcher if one trial does not change the hang signature."""

ANS["corpus-01716"] = """Mechanism: NCCL topology detection reads the PCIe tree and GPU/NIC device nodes to build its graph. Inside a container whose cgroup device controller or `--gpus` filter exposes only a subset of devices -- or where /sys is partially masked -- NCCL can compute a topology that names peers it cannot actually reach. Communicator construction then waits on a transport that will never establish, so the failure presents as an initialization hang rather than a device-visibility error.

Falsifiable hypothesis H1: the hang originates in an inconsistent device/topology view between the container and the host, not in inter-node networking. Prediction: torch.cuda.device_count() inside the container differs from the local ranks the launcher spawns, or `nvidia-smi topo -m` inside the container shows fewer links/NICs than on the host, and the same job hangs with --nnodes=1.

Controlled experiment: reduce to one node (`--nnodes=1 --nproc_per_node=<visible GPUs>`), which removes every network variable. If it still hangs, H1 is strongly supported and the fault is local visibility/topology; if the single-node run succeeds, container topology is exonerated and attention moves inter-node. Then, as a second single-variable trial, rerun the container with full device visibility and unmasked /sys.

Boundary conditions: decisive only if the single-node trial uses the same number of local ranks as the failing job; shrinking nproc_per_node at the same time changes two variables. Invalid where the model's tensor-parallel degree requires more GPUs than are visible -- then the job is a different program and cannot be compared.

Measurements and evidence required: `nvidia-smi topo -m` and `lspci -tv` inside the container and on the host; torch.cuda.device_count() and CUDA_VISIBLE_DEVICES from /proc/<pid>/environ per rank; `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv` to prove a bijection between ranks and distinct GPU UUIDs; NCCL_DEBUG_SUBSYS=GRAPH,INIT logs from all ranks; container runtime device list.

Confounders: ACS or IOMMU enabled on the host silently disables P2P and produces a similar intra-node stall with a correct-looking device list; check `lspci -vvv | grep -i acs` and dmesg IOMMU lines. Stale processes holding GPU contexts are a third look-alike.

Rollback gate: change device exposure in exactly one place (runtime flag or CUDA_VISIBLE_DEVICES, never both). Require a single-node all-reduce to pass before returning to multi-node, and revert the container change if one trial does not move the blocking frame in the stack traces."""

ANS["corpus-01717"] = """Mechanism: RoCE depends on a consistent L2/L3 MTU across NICs and every switch hop. TCP rendezvous uses small packets and succeeds; RDMA transport negotiates a path MTU and then issues large RDMA writes. If one hop enforces a smaller MTU without fragmentation, the first large transfer is silently dropped, the queue pair retries up to its retry_cnt, and the job stalls at the transition from init to first collective -- observed by the user as an initialization hang.

Falsifiable hypothesis H1: small-packet connectivity is healthy and the block appears only for payloads above the smallest hop MTU, i.e. an MTU/PMTU blackhole, not a rendezvous or GPU fault. Prediction: `ping -M do -s 1472 <peer>` succeeds while `ping -M do -s 8972 <peer>` fails on jumbo-configured nodes, and `ibv_devinfo` active_mtu differs between the two suspect NICs.

Controlled experiment: hold everything fixed and pin the RDMA path MTU down to the smallest value all hops support (NCCL_IB_MTU set to the corresponding enum, e.g. 1024 B) and rerun. Completion confirms H1. Independent control: run `ib_write_bw -s 65536` between the two nodes with the original MTU; it should stall or report zero bandwidth, reproducing the failure without NCCL in the picture.

Boundary conditions: lowering the MTU reduces achievable bandwidth and raises per-message overhead, so it is a diagnostic, not a fix. Invalid if the fabric is InfiniBand with a uniform SM-assigned MTU, where per-hop divergence cannot occur; then investigate PFC/ECN instead.

Measurements and evidence required: `ip link show` MTU per interface on every node; `ibv_devinfo -v` active_mtu; switch port MTU from the network team (recorded as a fact, not inferred); `ethtool -S` discard and pause-frame counters sampled before and after a trial; MEASURED `ib_write_bw` GB/s at 4 KB vs 64 KB message sizes; NCCL_DEBUG_SUBSYS=NET logs from all ranks.

Confounders: missing or mismatched PFC/DSCP configuration causes congestion-driven stalls with the same "works small, hangs large" signature; pause-frame and ECN counters discriminate the two. A flapping link that resolves ARP but blackholes traffic is a third look-alike.

Rollback gate: apply the MTU pin to one node pair only, require `ib_write_bw` at 64 KB to pass, then expand. Never roll a fabric-wide MTU or QoS change on the strength of a single hung job; revert the pin immediately if the pair test does not improve, and escalate to the network owner with counter evidence rather than continuing to tune."""

ANS["corpus-01718"] = """Mechanism: an initialization hang and a dead rank are indistinguishable from rank 0's perspective, because the store barrier has no liveness signal. A rank killed by the cgroup OOM killer during model construction, or exiting on an import error, never joins the barrier; the survivors block forever. Nothing is logged at the collective layer because the collective layer never got the chance to time out.

Falsifiable hypothesis H1: at least one rank process is not alive (or never reached init), so the barrier is unsatisfiable -- a participation shortfall, not a transport fault. Prediction: the count of live job PIDs summed across nodes is strictly less than WORLD_SIZE, and dmesg on the affected node shows an oom-kill or the rank's stderr shows a traceback.

Controlled experiment: convert the silent wedge into data with a single change -- set init_process_group(timeout=timedelta(seconds=120)) and TORCH_DISTRIBUTED_DEBUG=DETAIL, then rerun the identical job. The timeout raises on each survivor and names the ranks that never checked in. Follow-up trial: rerun with WORLD_SIZE reduced to exactly the ranks observed to check in; if that initializes, H1 is confirmed and the defect is in launch/allocation or per-rank memory, not the fabric.

Boundary conditions: bounded timeout is diagnostic only, never a production fix; and the reduced-world trial must keep nproc_per_node unchanged so local device binding is not a second moving variable.

Measurements and evidence required: per-node `pgrep -af <job pattern> | wc -l` at hang time summed against WORLD_SIZE; exit status of every rank from the scheduler (`sacct -j <id> --format=JobID,State,ExitCode,MaxRSS`); dmesg -T grepped for oom-kill and Xid; per-rank stderr captured to distinct files (a single merged log hides which rank died); MEASURED peak host and device memory per rank; `ss -tanp | grep $MASTER_PORT | wc -l` on rank 0's node.

Confounders: a rank that is alive but blocked in slow storage (checkpoint or dataset scan on a cold NFS mount) also never checks in, and looks identical to a dead rank until you inspect its stack -- py-spy dump discriminates. A zombie process from a prior job holding MASTER_PORT is a third mechanism with the same symptom.

Rollback gate: never leave a reduced world size or a shortened timeout in production. Fix the actual cause (memory footprint, allocation, import), require a full-scale bounded-timeout init to pass once, then restore the original settings. If two bounded-timeout trials name different missing ranks each time, treat it as node health, drain the nodes, and stop config tuning."""

ANS["corpus-01719"] = """Mechanism: intra-node NCCL prefers direct GPU-to-GPU P2P over NVLink or PCIe. When ACS (Access Control Services) is enabled on upstream PCIe bridges, or the IOMMU forces translation, peer-to-peer DMA between GPUs is silently blocked. NCCL's P2P probe can report the path as usable while the actual transfer never completes, so communicator construction stalls on the first intra-node connection with no error message.

Falsifiable hypothesis H1: the block is intra-node P2P establishment defeated by ACS/IOMMU, not inter-node networking or rendezvous. Prediction: the job hangs identically with --nnodes=1, `lspci -vvv` shows ACSCtl with SrcValid+ on the bridges above the GPUs, and CUDA's p2pBandwidthLatencyTest reports 0 or fails for the affected GPU pairs.

Controlled experiment: two single-variable trials on one node. Trial A: set NCCL_P2P_DISABLE=1 and rerun -- if init completes, P2P is implicated and the SHM/host path works. Trial B: with P2P re-enabled, disable ACS on the relevant bridges (or boot with iommu=pt) and rerun -- success here is the confirming positive control, and it must be paired with a MEASURED p2pBandwidthLatencyTest before and after.

Boundary conditions: NCCL_P2P_DISABLE=1 routes intra-node traffic through host memory and can cut intra-node bandwidth by roughly an order of magnitude -- diagnostic only. ACS changes are host-wide and affect device isolation, so they are invalid to apply on multi-tenant nodes without an explicit security decision by the platform owner.

Measurements and evidence required: `nvidia-smi topo -m` showing expected NV#/PIX links; `lspci -vvv | grep -i acsctl` for every bridge above the GPUs; dmesg IOMMU lines and kernel cmdline; MEASURED p2pBandwidthLatencyTest matrix in GB/s before and after; NCCL_DEBUG_SUBSYS=INIT,P2P,SHM logs from every local rank; `df -h /dev/shm` since the fallback path needs space.

Confounders: a 64 MB default /dev/shm in the container makes the SHM fallback fail too, so Trial A can hang for a *different* reason and appear to refute H1 -- raise shm before running it. Two ranks bound to the same GPU also deadlock intra-node and must be excluded by checking distinct GPU UUIDs per PID.

Rollback gate: never disable ACS fleet-wide from a single hung job. Change one node, require a passing single-node all-reduce plus a P2P bandwidth measurement above a pre-agreed floor, and revert immediately if either check fails. Remove NCCL_P2P_DISABLE after diagnosis; leaving it set silently degrades every subsequent training run."""

ANS["corpus-01720"] = """Mechanism: NCCL's initialization cost is not constant in world size. The store fan-in at rank 0 is O(N) connections and the bootstrap all-gather is O(N) in handles; at large N the default connection backlog, file-descriptor limit, or store timeout can be exceeded. The result is a job that initializes correctly at 8 or 16 ranks and hangs only at full scale -- a scale-dependent failure that every small reproduction will falsely exonerate.

Falsifiable hypothesis H1: the hang is scale-dependent (store fan-in / fd / backlog exhaustion at rank 0), not a per-node configuration defect. Prediction: there exists a threshold world size below which init reliably completes and at or above which it reliably hangs, with the same nodes, image and environment; and at the failing scale rank 0 shows a plateau in established sessions well below WORLD_SIZE while `ss -s` or /proc/<pid>/limits indicates fd or backlog pressure.

Controlled experiment: scale bisection with one variable. Run the identical job at world sizes 8, 32, 128, then full, holding nproc_per_node, image, env and node pool constant, three repetitions per point to separate deterministic from flaky. If failures begin sharply at a threshold, H1 is supported; if the smallest scale also hangs, H1 is refuted and the cause is configuration, which is far cheaper to debug at 8 ranks.

Boundary conditions: valid only if reducing world size does not change parallelism degrees -- hold TP and PP fixed and shrink DP only, otherwise each scale point is a different program. Invalid if the node pool changes between points, since that reintroduces per-node health as a confounder.

Measurements and evidence required: MEASURED init wall time in seconds at each scale point (a superlinear curve is itself evidence); established-session count on rank 0's MASTER_PORT sampled over time; /proc/<pid>/limits Max open files for rank 0; `nstat -az TcpExtListenOverflows TcpExtListenDrops` deltas on rank 0's node; NCCL_DEBUG=INFO from a sampled subset of ranks (full-scale logging is itself a load); per-rank check-in timestamps.

Confounders: at large scale the probability that *any single node* is unhealthy rises with N, so a threshold can appear that is really "the bad node is only included above 128 ranks". Pin the node list across scale points and record it. Slow shared-storage image pull at scale is a second look-alike that delays check-in without any store defect.

Rollback gate: escalate 8 -> 32 -> 128 -> full, one step per trial, and stop at the first scale that reproduces. Raise fd/backlog limits or move to a distributed rendezvous backend only after the threshold is measured, roll it to one scale point at a time, and revert if init wall time does not improve. Do not apply a fix at full scale that was never falsified at the reproducing scale."""

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
            "no concrete commands, thresholds, units or rollback gates in the source",
            "variant text is identical across many ids, risking memorization of a single template",
        ],
        "evidence_required": [
            "per-rank NCCL_DEBUG=INFO/SUBSYS logs and decoded /proc/<pid>/environ from every rank",
            "node-level fabric and device state (nvidia-smi topo -m, ibv_devinfo, ss on MASTER_PORT, dmesg Xid/oom)",
            "a single-variable control run (reduced scale, single node, or toggled transport) with MEASURED wall time",
        ],
        "confidence": 0.62,
    }
    out.append(json.dumps(rec, ensure_ascii=False))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("wrote", OUT, len(out))
