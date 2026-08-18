import json, hashlib, os

CORPUS="research/ai-infra-expert/corpus/train.jsonl"
OUT="experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0157.jsonl"
START, N = 1560, 10

rows=[json.loads(l) for l in open(CORPUS,encoding="utf-8")]
sl=rows[START:START+N]

ANS = [
# 121
"""Mechanism: NCCL's bootstrap ring is built over the socket plugin before any GPU transport is chosen. Rank 0 publishes a bootstrap address derived from the *first* interface that passes NCCL's scoring, and every other rank must be able to open a TCP connection back to that exact address. On a node with multiple RDMA-capable and management NICs, an asymmetric NCCL_SOCKET_IFNAME (set on some nodes via the job script, unset on others so NCCL auto-selects) makes rank 0 advertise an address on a subnet that a subset of peers has no route to. Those peers' connect() sits in SYN-retransmit for the full kernel backoff and init_process_group blocks with no NCCL error, because the bootstrap layer has no timeout of its own until NCCL_COMM_INIT_TIMEOUT (or the c10d store timeout) fires.

Falsifiable hypothesis H1: the hang is bootstrap interface asymmetry, not a GPU, driver, or collective-algorithm fault. Prediction: dumping the effective NCCL_SOCKET_IFNAME and `ip -o addr` on every rank shows at least two distinct advertised subnets, and on the stalled ranks `ss -tan state syn-sent` shows a pending connection to rank 0's advertised bootstrap port. If instead all ranks advertise the same subnet and the SYN-SENT set is empty, H1 is falsified and the next hypothesis is the GPU-transport phase (H2: GDR/peermem) rather than bootstrap.

Controlled experiment: hold the job, model, and world size fixed and vary exactly one variable — the interface selection. Run A: launch with NCCL_SOCKET_IFNAME explicitly pinned to the same management interface name on all nodes plus NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,NET. Run B: launch unchanged. Predicted outcome under H1: Run A reaches the first all-reduce and prints a complete ring/tree topology line; Run B still hangs at the same rank set. Predicted outcome if H1 is false: both runs hang identically at the same phase.

Measurements to collect before touching anything: per-rank NCCL_DEBUG=INIT log up to the last line emitted; `ss -tanp` snapshots on a hung rank and on rank 0; `getent hosts $MASTER_ADDR` and `ip route get <rank0-ip>` from each node; `py-spy dump` or `gdb -p` backtraces to confirm the stack is inside bootstrapNetInit/ncclCommInitRank rather than in the model code; and the exact env of each rank (`tr '\\0' '\\n' < /proc/<pid>/environ`).

Confounders that must be ruled out explicitly: (a) a firewall that allows MASTER_PORT but blocks NCCL's ephemeral bootstrap ports — distinguishable because the c10d store rendezvous itself would succeed while NCCL bootstrap fails, so log the timestamp of 'rendezvous complete' vs the last NCCL line; (b) an OOM-killed or never-started rank making the barrier under-subscribed — check that the number of live processes equals world_size on every node before blaming networking; (c) DNS/hosts divergence for MASTER_ADDR, which produces a similar SYN-SENT signature but with *different* target IPs across nodes rather than a routing failure to one common IP.

Boundary conditions: this failure mode is scale- and placement-dependent. It will not reproduce on a single node (loopback/shared-memory bootstrap), and it may not reproduce on a two-node run if both nodes happen to be on the same rack switch with a default route that covers the advertised subnet. Reproduction must therefore use the same node set and the same launcher path as the failing job; a shrunk world size is a diagnostic, not a proof of absence.

Evidence that would close the ticket: a single-variable Run A that completes init, plus a captured pre-fix log showing divergent advertised interfaces, plus a routing-table diff between a healthy and a stalled node.

Rollback gate: apply the pinned NCCL_SOCKET_IFNAME only as a job-level environment change, never as a node image change, until Run A has passed twice at full scale. Revert immediately if the pinned interface reduces measured all-reduce bus bandwidth below the pre-incident baseline by more than 5 percent, since forcing bootstrap onto a management NIC must not silently move *data* traffic off the RDMA fabric — verify with a NCCL all-reduce bandwidth run and by confirming NET/IB (not NET/Socket) appears in the transport line before declaring the fix safe.""",
# 122
"""Mechanism: with a c10d TCPStore rendezvous, every rank must both write its own key and read all peers' keys before init_process_group returns. The store server lives inside rank 0's process. If rank 0 is placed on a node under memory pressure, or the launcher starts rank 0 last, peers that call the store before the listener binds get connection-refused and enter the store client's retry loop. That loop is bounded only by the store timeout (default 30 minutes in many launcher configs), so the job appears hung rather than failed. The distinguishing property is that the hang resolves *itself* if rank 0 eventually binds, and the wall-clock stall length clusters around the launcher's stagger, not around any NCCL timeout.

Falsifiable hypothesis H1: the stall is a rendezvous start-order/listener-availability race in the c10d store, not a fabric fault. Prediction: timestamps in the per-rank logs show peers attempting the store strictly before rank 0's 'store server listening' line, and `ss -tanl` on rank 0's node shows MASTER_PORT unbound during the stall window. If MASTER_PORT is bound from t0 and peers still stall, H1 is falsified and attention moves to the NCCL bootstrap phase after rendezvous.

Controlled experiment: fix everything except the store timeout and the start order. Run A: set the c10d store timeout to 120 seconds so the race converts from a hang into a fast, attributable exception naming the ranks that never joined. Run B: keep the long timeout but add a launcher barrier that blocks all ranks until rank 0 reports the listener bound. Under H1, Run A fails within ~2 minutes with an explicit missing-rank list and Run B succeeds; if H1 is false, Run A times out with *all* ranks reported present-but-blocked and Run B still hangs.

Measurements: per-rank monotonic timestamps for process start, rendezvous call, and rendezvous return; `ss -tanl` and `ss -tan state syn-sent` on rank 0's node sampled every second across the stall; the launcher's own scheduling log; `dmesg -T | grep -i oom` on rank 0's node; and a py-spy backtrace confirming the stack is in the store client rather than in ncclCommInitRank.

Confounders: (a) a genuinely dead rank (OOM, CUDA init failure) looks identical from the surviving ranks' side — separate them by counting live PIDs per node against the expected local world size; (b) a slow container image pull on one node delays that rank's start and mimics the race; (c) if MASTER_PORT is reused from a previous run still in TIME_WAIT or still held by a zombie process, the bind fails and rank 0 never listens — check for a stale process holding the port before concluding it is a pure ordering race.

Boundary conditions: shrinking world size often hides this because the stagger window shrinks with it, so a passing 2-node run is not evidence of absence. The failure is also launcher-specific: it appears with plain torchrun and static MASTER_ADDR, and typically disappears under an etcd/rendezvous-backend that has its own retry-with-backoff and explicit membership reporting.

Evidence required to close: a log pair showing peer-before-listener ordering, a reproduction under the shortened timeout that names the late rank, and a clean full-scale run after the launcher barrier.

Rollback gate: the shortened store timeout is a *diagnostic* setting. Keep it only if two consecutive full-scale runs start cleanly; if any legitimate slow-start path (image pull, checkpoint load on rank 0) exceeds it, restore the original timeout and fix the ordering instead. Never ship a shortened timeout as the fix on its own, because it converts a recoverable slow start into a hard job failure at scale.""",
# 123
"""Mechanism: NCCL selects a transport per peer pair during init. When it chooses the IB/RoCE path it must register pinned memory and open a queue pair per connection. On a fabric where the subnet manager has not yet assigned a LID/GID to a port, or where the port is in INIT rather than ACTIVE state, the queue-pair transition to RTR blocks or fails in a way NCCL retries. The job hangs after the topology line is printed but before the first collective completes, which is the signature that distinguishes a *transport* stall from a *bootstrap* stall: bootstrap failures hang before topology detection, transport failures hang after it.

Falsifiable hypothesis H1: at least one HCA port is not ACTIVE (or has no valid GID for the configured GID index) at job start, so QP establishment stalls. Prediction: `ibstat` / `ibv_devinfo` on the affected node shows state != ACTIVE or phys_state != LinkUp on the port NCCL selected, and the per-rank NCCL log's last line is a NET/IB connection setup entry naming that device. If every port is ACTIVE with a valid GID and the stall persists at the same place, H1 is falsified and the next candidate is memory registration limits (RLIMIT_MEMLOCK) or peermem.

Controlled experiment: single variable is the transport. Run A: force NCCL_IB_DISABLE=1 so all traffic falls back to the socket plugin. Under H1 the job initialises and runs (slowly); the throughput drop is expected and is itself the confirmation that the IB path was the blocked one. Run B: keep IB enabled but restrict NCCL_IB_HCA to a device whose port is confirmed ACTIVE. Under H1, Run B also initialises. If both A and B hang, the fault is not in HCA port state.

Measurements: `ibstat`, `ibv_devinfo -v`, and `show_gids` on every node; the port counters (`perfquery` or /sys/class/infiniband/*/ports/*/counters) sampled before and after the stall to see whether any packets moved at all; NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH per-rank logs; `ibv_rc_pingpong` between the two ranks whose connection is last in the log, as the minimal reproducer independent of PyTorch; and `dmesg -T` for HCA link-flap or firmware messages.

Confounders that must be excluded: (a) a link that is ACTIVE but negotiated at a degraded width/rate — that causes slowness, not a hang, so do not accept a width mismatch as the explanation for a hang without a corresponding blocked QP; (b) RoCE GID-index mismatch across nodes (v1 vs v2, or the wrong index selected by NCCL_IB_GID_INDEX), which produces a hang with all ports ACTIVE and is the most common false attribution here; (c) a PFC/ECN misconfiguration that stalls only under load, which will *not* reproduce with ibv_rc_pingpong and therefore must be tested with a sustained NCCL bandwidth run rather than a single-message test.

Boundary conditions: this diagnosis assumes the node set is fixed. A hang that migrates to a different node on every launch points at the scheduler or at a marginal cable/transceiver rather than a static port-state misconfiguration; in that case collect the port error counters over several launches and correlate with physical topology before changing any NCCL setting.

Evidence required: a pre-fix `ibstat` showing the bad port state on exactly the node whose rank is last in the NCCL log, a successful ibv_rc_pingpong after the port is restored, and a full-scale run whose all-reduce bus bandwidth matches the pre-incident baseline.

Rollback gate: NCCL_IB_DISABLE=1 is a diagnostic only and must never be left in a production job, because it silently moves all collective traffic to TCP and can cost an order of magnitude in bandwidth. Gate the real fix on: port ACTIVE on all nodes, bus bandwidth within 5 percent of baseline, and zero increment in symbol-error and link-downed counters across a 30-minute soak. Revert the fabric change if any of these fail.""",
# 124
"""Mechanism: PyTorch's init_process_group is a collective barrier — it returns only when every rank in world_size has checked in. A mismatch between the world_size the launcher believes and the number of processes actually started makes the barrier permanently under-subscribed. The classic source is a rank-count computed from nnodes * nproc_per_node in the launcher while one node's container is admitted with fewer visible GPUs (CUDA_VISIBLE_DEVICES trimmed by the scheduler, an MIG partition, or a GPU in a bad ECC state removed by the driver). The surviving ranks block forever in the rendezvous, and because no process crashed, there is no traceback to point at.

Falsifiable hypothesis H1: the number of live training processes is strictly less than world_size, i.e. this is a participation shortfall, not a network fault. Prediction: summing `pgrep -c -f train_entry` across all nodes yields a number below world_size, and `nvidia-smi -L` on at least one node lists fewer GPUs than nproc_per_node. If the process count equals world_size exactly and the stall persists, H1 is falsified and the fault is in the store or NCCL transport layer.

Controlled experiment: the single variable is world membership. Run A: shorten the rendezvous timeout to 120 seconds so the barrier converts into an exception that enumerates which ranks never joined — this turns an unattributable hang into data. Run B: relaunch with nproc_per_node derived at runtime from the actual device count on each node rather than from a static constant. Under H1, Run A names the missing ranks and Run B initialises cleanly. If Run A reports all ranks present and still blocked, H1 is false.

Measurements: per-node process count and PID list; `nvidia-smi -L`, `nvidia-smi -q -d ECC,PAGE_RETIREMENT` on every node; CUDA_VISIBLE_DEVICES as actually seen inside each container (`tr '\\0' '\\n' < /proc/<pid>/environ`); `dmesg -T | grep -iE 'oom|Xid'` for a rank killed before it could log; the scheduler's allocation record vs the launcher's assumed layout; and the last log line from each rank so a rank that died during CUDA context creation is not mistaken for one that never started.

Confounders: (a) a rank that started and then died silently (OOM killer, Xid 79 GPU fallen off the bus) leaves the same shortfall signature as a rank that never started — separate them with dmesg timestamps and the presence of a partial log; (b) a duplicated rank id from a launcher retry can make the process count look correct while the barrier still fails, so check rank uniqueness, not just count; (c) heterogeneous nodes where one host genuinely has fewer GPUs is a scheduling bug, not a runtime bug, and must be fixed in the placement policy rather than by shrinking world_size.

Boundary conditions: this reproduces only with the exact allocation that failed. Re-running on a fresh allocation frequently succeeds and proves nothing about the original failure; therefore capture the node list and the per-node device inventory from the failing allocation before releasing it, or the evidence is lost.

Evidence required to close: the short-timeout run's missing-rank list, the device inventory from the offending node, a dmesg or Xid record if a GPU was removed, and a clean full-scale run after the fix.

Rollback gate: do not paper over the shortfall by lowering world_size, because that silently changes global batch size and invalidates the training run's comparability. Gate any change on: process count == world_size on every node for two consecutive launches, device inventory matching the scheduler's allocation, and a first-step loss consistent with the pre-incident baseline. If the loss at step 1 deviates beyond the known run-to-run band, revert and treat the batch-size change as the prime suspect.""",
# 125
"""Mechanism: intra-node NCCL prefers CUDA IPC peer-to-peer between GPUs on the same PCIe root complex, and falls back to shared memory via /dev/shm when P2P is unavailable. In a container, /dev/shm defaults to 64 MB. NCCL's shared-memory transport allocates per-peer buffers sized by NCCL_BUFFSIZE; with eight local ranks the aggregate demand exceeds 64 MB. The allocation does not always fail cleanly — depending on the path, ranks can block waiting for a shared-memory segment that never becomes available, so init or the first collective hangs instead of raising ENOSPC. This is why the same image runs fine on bare metal and hangs under the orchestrator.

Falsifiable hypothesis H1: the stall is /dev/shm exhaustion on the shared-memory transport path, not a fabric or rendezvous fault. Prediction: `df -h /dev/shm` inside the container shows a 64 MB (or similarly small) tmpfs at or near 100 percent during the stall, the NCCL log's last transport line names SHM rather than P2P or IB, and strace on a hung rank shows a blocked or repeatedly failing shm operation under /dev/shm/nccl-*. If /dev/shm is large and mostly free, H1 is falsified and the P2P/IPC path becomes the suspect.

Controlled experiment: vary exactly one thing — the size of /dev/shm. Run A: relaunch the identical image and job with the container's shm sized to 8 GB. Run B: keep the small shm but set NCCL_SHM_DISABLE=1 so NCCL must use P2P or the network path instead. Under H1, Run A initialises and runs at normal speed, and Run B either initialises with degraded intra-node bandwidth or reports a clear transport error — either outcome distinguishes the shared-memory path from everything else. If both hang unchanged, H1 is false.

Measurements: `df -h /dev/shm` and `ls -l /dev/shm` sampled during the stall; the container runtime spec's shm size; NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,SHM,P2P logs showing the chosen transport per peer pair; `nvidia-smi topo -m` to record which pairs should have had P2P; `cat /proc/<pid>/maps | grep /dev/shm` on a hung rank; and dmesg for any tmpfs or cgroup memory pressure messages.

Confounders that produce a similar signature: (a) PCIe ACS enabled or IOMMU in a mode that blocks P2P, which pushes traffic onto SHM in the first place and is the *upstream* cause — check `lspci -vvv | grep -i acsctl` and the topo matrix, because fixing shm size without fixing ACS leaves intra-node bandwidth far below baseline; (b) a cgroup memory limit that counts tmpfs pages, so /dev/shm looks big but allocation still fails under the limit; (c) a leftover /dev/shm/nccl-* segment from a crashed prior run occupying space — clean it and re-measure before attributing to sizing.

Boundary conditions: this is a per-node, local-world-size-dependent failure. It will not reproduce with one or two local ranks and it does not depend on the number of nodes, so a passing small-scale run is not evidence of absence. Any reproduction must use the same local rank count and the same container runtime configuration.

Evidence required: a during-stall `df -h /dev/shm` at capacity, a NCCL log naming SHM as the last transport, a successful Run A, and a topo matrix plus an intra-node bandwidth measurement proving P2P was or was not available.

Rollback gate: enlarging /dev/shm is safe but must be paired with a bandwidth check — if intra-node all-reduce bandwidth after the fix is still materially below the NVLink/P2P baseline, the real defect is ACS/IOMMU and the shm change is masking it. Do not close the incident on a successful start alone; require a bus-bandwidth number within 5 percent of the known-good baseline, and revert the container spec change if node memory pressure (cgroup OOM events) appears during a 30-minute soak.""",
# 127
"""Mechanism: NCCL's IB/RoCE path registers memory regions with the HCA, which requires pinning pages. The amount a process may pin is bounded by RLIMIT_MEMLOCK. Container runtimes frequently apply a small default (64 KB or 64 MB) that does not inherit the host's 'unlimited'. When registration hits the ceiling, ibv_reg_mr returns ENOMEM; NCCL may retry or fall back in a way that leaves peers waiting on a connection that never reaches RTS, so init stalls after the topology line rather than raising a clean error on every rank.

Falsifiable hypothesis H1: the stall is an RLIMIT_MEMLOCK ceiling blocking RDMA memory registration, not a link-state or rendezvous fault. Prediction: `cat /proc/<pid>/limits | grep 'Max locked memory'` inside the container shows a finite, small value, and the NCCL log's last lines are NET/IB registration entries; ibv_reg_mr failures appear when the same registration size is attempted by a minimal reproducer under the same limit. If the limit is already unlimited on every rank, H1 is falsified and the next candidate is peermem/GDR or HCA port state.

Controlled experiment: single variable is the memlock limit. Run A: relaunch the identical job with the container's memlock set to unlimited and nothing else changed. Run B: keep the small limit but set NCCL_IB_DISABLE=1, removing the need for large pinned registrations. Under H1, both A and B start — A at full bandwidth, B at socket-transport bandwidth — and the contrast between their measured bus bandwidths is itself the confirmation. If A still hangs, the ceiling was not the binding constraint.

Measurements: per-rank /proc/<pid>/limits; the container runtime's ulimit spec; NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,NET logs; a standalone ibv_reg_mr probe sized to NCCL_BUFFSIZE times the connection count, run inside the same container; `ibstat` to prove ports are ACTIVE so link state is excluded; and `grep -i memlock /var/log/...` or dmesg for registration failures.

Confounders: (a) missing nvidia_peermem produces a similar post-topology stall on the GDR path — distinguish by testing NCCL_NET_GDR_LEVEL=0, which bypasses GPUDirect while keeping IB; if that alone unblocks the job, the cause is peermem, not memlock; (b) cgroup memory limits can make pinning fail even with unlimited RLIMIT_MEMLOCK, so check the cgroup's memory.max alongside the rlimit; (c) a partially applied fix, where the limit is raised for the shell but not for the process the launcher forks, gives a false negative on Run A — always read the limit from the actual training PID, not from an interactive shell.

Boundary conditions: the failure threshold depends on message size and connection count, so it is scale-dependent — a small world size may register under the ceiling and pass. Reproduction must use the same world size, the same NCCL_BUFFSIZE, and the same container runtime settings; a passing 2-node run does not clear the configuration.

Evidence required: the pre-fix limit read from the training PID, a failing standalone registration probe under that limit, a passing probe after the change, and a full-scale run whose bus bandwidth matches baseline.

Rollback gate: raising memlock to unlimited is a privilege-relevant change to the container spec — it must go through the same review as any other runtime capability change, and be applied to the job template rather than to the node image. Gate acceptance on: registration probe passes, full-scale init completes twice, bus bandwidth within 5 percent of baseline, and no increase in host memory pressure or OOM events during a 30-minute soak. Revert the spec change if any node shows pinned-memory growth that does not return to baseline after job exit, since that indicates a leak rather than a sizing fix.""",
# 128
"""Mechanism: on a RoCE fabric, the NCCL bootstrap and the c10d rendezvous use small TCP messages that traverse the same physical path as the RDMA traffic but fit inside the smallest MTU on the route. Once NCCL switches to large RDMA writes, packets larger than the smallest link MTU are dropped, and because RoCEv2 relies on the switch fabric rather than IP fragmentation, there is no ICMP-driven path-MTU discovery to fix it. The result is a job that completes rendezvous and topology detection, then hangs on the first sizeable collective — the small-packets-work, large-packets-vanish signature that separates an MTU blackhole from a link-down fault.

Falsifiable hypothesis H1: an MTU mismatch along the RoCE path silently drops large payloads, so the first large collective never completes. Prediction: small-message all-reduce (a few KB) completes and large-message all-reduce (tens of MB) hangs, with the switch/HCA drop counters incrementing only during the large run. If both small and large messages hang identically, H1 is falsified and the fault is earlier, in QP establishment.

Controlled experiment: single variable is message size. Run A: a NCCL all-reduce sweep from 8 B up to 1 GB on the same node pair, recording the exact size at which it stalls. Run B: repeat the sweep after clamping the effective MTU on both endpoints to the smallest value present on the path. Under H1, Run A shows a sharp cliff at a size corresponding to the smallest path MTU, and Run B completes the full sweep at reduced but stable bandwidth. If Run A shows no cliff, the hypothesis is wrong.

Measurements: `ip link show` and the HCA's active_mtu from `ibv_devinfo` on both endpoints; switch port MTU configuration for every hop on the path; per-port discard/drop counters sampled immediately before and after the large-message run; NCCL all-reduce sweep results with sizes and completion times; PFC pause-frame counters, since a congestion-control misconfiguration can also stall large transfers; and the NCCL log confirming NET/IB was the transport in use.

Confounders that must be separated: (a) PFC/ECN misconfiguration produces stalls that also appear only under load, but shows rising pause-frame counts rather than MTU-sized discards — collect both counter families or the attribution is unsound; (b) a marginal cable or transceiver produces symbol errors and intermittent, size-correlated failures that move with the physical port, so record the topology and repeat the sweep on a different port pair; (c) a GID-index mismatch can also break large transfers first, so confirm the GID configuration matches on both ends before touching MTU.

Boundary conditions: the failure is path-specific. It reproduces only on node pairs whose route includes the low-MTU hop, which is why the job may succeed on some allocations and hang on others with identical software. Any claim that the fabric is fixed must be tested across the full set of node pairs the job actually uses, not a single representative pair.

Evidence required to close: the message-size cliff from Run A, matching drop-counter increments on exactly the suspected hop, the MTU inventory showing the mismatch, and a post-fix full-scale run with bus bandwidth at baseline.

Rollback gate: MTU changes touch shared fabric configuration and can affect unrelated tenants. Stage them on one leaf switch pair first, verify with the sweep, and hold a documented revert command ready. Accept the fix only if: the size sweep completes to 1 GB with no stall, drop counters stay flat across a 30-minute soak, bus bandwidth is within 5 percent of baseline, and no other job on the same fabric reports a regression. Revert immediately on any counter growth or on any cross-tenant complaint, since a wrong global MTU is worse than the original single-job hang.""",
# 129
"""Mechanism: GPUDirect RDMA lets the HCA DMA directly into GPU BAR1 memory, which requires the nvidia_peermem (formerly nv_peer_mem) kernel module to expose GPU memory to the RDMA subsystem. If the module is absent or was not reloaded after a driver upgrade, NCCL may still select the GDR path based on topology distance and then block trying to register GPU buffers. The job therefore hangs *after* printing a topology that mentions GDR, which distinguishes it from a bootstrap failure (hangs before topology) and from a link-state failure (ports would not be ACTIVE).

Falsifiable hypothesis H1: nvidia_peermem is missing or stale on at least one node, so GPUDirect registration stalls. Prediction: `lsmod | grep -E 'nvidia_peermem|nv_peer_mem'` returns nothing on the node whose rank is last in the NCCL log, while the healthy nodes show it loaded, and dmesg lacks the peermem registration message that normally appears at driver load. If the module is loaded with a version matching the running driver on every node, H1 is falsified and the suspects become RLIMIT_MEMLOCK or HCA port state.

Controlled experiment: the single variable is whether GDR is used. Run A: relaunch unchanged except NCCL_NET_GDR_LEVEL=0, which keeps IB/RoCE transport but routes buffers through host memory instead of GPU BAR1. Under H1 the job initialises and trains, at measurably lower bandwidth — that bandwidth gap is the evidence that GDR was the blocked path, not incidental. Run B: load nvidia_peermem on the offending node and rerun with GDR enabled; under H1 the job initialises and bandwidth returns to baseline. If Run A still hangs, GDR is not the blocking path.

Measurements: `lsmod`, `modinfo nvidia_peermem`, and the loaded NVIDIA driver version on every node, checked for version agreement; dmesg around driver and module load; NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH logs showing whether GDR was selected per peer; `nvidia-smi topo -m` for GPU-to-HCA affinity; BAR1 usage from `nvidia-smi -q -d MEMORY`; and an all-reduce bandwidth number for both the GDR and non-GDR configurations so the difference is quantified rather than asserted.

Confounders: (a) RLIMIT_MEMLOCK exhaustion also blocks registration after topology detection — separate it by reading /proc/<pid>/limits from the training PID, because raising memlock and loading peermem at the same time destroys attribution; (b) a driver/module version skew where peermem is present but built against a different driver can be worse than absence, so compare modinfo's vermagic against the running driver; (c) topology-based GDR selection may differ per GPU-HCA pair, so a node can be half-broken — check every local rank, not one.

Boundary conditions: the failure is per-node and per-pair. It will not appear on single-node runs that never leave NVLink, and it may be masked if NCCL's distance heuristic declines GDR for the particular GPU-HCA pairing in a smaller allocation. Reproduce with the same node set and the same rank-to-GPU mapping.

Evidence required: the module inventory diff across nodes, the NCCL log showing GDR selection on the stalled pair, the Run A / Run B bandwidth pair, and a clean full-scale run.

Rollback gate: NCCL_NET_GDR_LEVEL=0 is a diagnostic and a temporary mitigation only — leaving it on permanently silently sacrifices bandwidth. Gate the real fix (loading and persisting peermem) on: module present and version-matched on all nodes, full-scale init twice, bus bandwidth within 5 percent of the pre-incident baseline, and no Xid errors during a 30-minute soak. Revert the module change and fall back to GDR-off if any Xid or host crash appears, since a kernel module fault at scale is a fleet-level risk, not a single-job one.""",
# 130
"""Mechanism: many clusters run two rendezvous mechanisms at once — the workload manager's PMI/PMIx bootstrap and PyTorch's own c10d TCPStore. If the launcher exports both a PMI environment and MASTER_ADDR/MASTER_PORT, some ranks can be initialised through one path and some through the other. Each subset forms a consistent view of a *smaller* world, so neither barrier ever reaches world_size and both halves wait indefinitely. No process dies, no error surfaces, and the logs from either half look internally coherent, which is what makes this failure mode so often misattributed to the network.

Falsifiable hypothesis H1: the ranks are split across two rendezvous backends, so the barrier is partitioned rather than blocked by a fabric fault. Prediction: the per-rank environment shows PMI/PMIX variables present on some ranks and absent or inconsistent on others, and the ranks that did check in to the TCPStore (readable from the store's keys, or inferable from rank 0's connection table) form a strict subset of world_size while the remaining ranks report a *different* rank/size pair. If every rank reports the same backend and the same world_size and the stall persists, H1 is falsified.

Controlled experiment: the single variable is the rendezvous backend. Run A: launch with the PMI/PMIx variables explicitly unset and c10d/TCPStore as the sole backend. Run B: launch with the workload manager's backend as the sole path and MASTER_ADDR/MASTER_PORT unset. Under H1, both A and B initialise cleanly while the mixed configuration hangs — the fact that either single backend works, and only the combination fails, is the confirming evidence. If both A and B hang, the partition hypothesis is wrong and the fault lies below the rendezvous layer.

Measurements: `tr '\\0' '\\n' < /proc/<pid>/environ` for every rank, diffed pairwise to expose environment asymmetry; each rank's reported (rank, local_rank, world_size) triple logged before init_process_group is called; rank 0's `ss -tan` connection count during the stall compared against world_size minus one; the launcher's generated job script; and py-spy backtraces from one rank in each suspected half to confirm both are blocked in a rendezvous, not one in rendezvous and one in NCCL.

Confounders: (a) a participation shortfall from a dead rank yields a similar under-subscribed barrier — distinguish by counting live PIDs, which will equal world_size in the partition case and fall short in the shortfall case; (b) duplicated rank ids from a retried launch also break the barrier while the process count looks right, so assert rank uniqueness explicitly; (c) heterogeneous container images across nodes can inject the PMI environment on only some hosts, making the split look random across allocations rather than deterministic.

Boundary conditions: the split requires at least two nodes and is sensitive to how the job was submitted; running the same code interactively on one node uses a single backend and always passes, which is why local reproduction is not evidence. Reproduce through the exact submission path that failed.

Evidence required to close: the per-rank environment diff showing the backend asymmetry, the two rank/size triples proving a partition, a passing Run A and Run B, and a full-scale run under the chosen single backend.

Rollback gate: standardising the rendezvous backend changes every job on the cluster, so roll it out to one queue first. Accept only if: all ranks report identical world_size, two consecutive full-scale runs initialise, and step-1 loss and step time match the pre-incident baseline. Revert the launcher template if any previously working job type fails to start, and keep the old template available for a documented fallback window rather than deleting it.""",
# 131
"""Mechanism: NCCL builds rings and trees from a topology graph derived from /sys (PCIe hierarchy, NVLink, NUMA, HCA placement). Inside a container with a masked or partially bind-mounted /sys, or with CUDA_VISIBLE_DEVICES reordered per rank, the graph NCCL computes does not match physical reality. Ranks then disagree about which peer owns which channel, and the ring never closes: every rank waits on a neighbour that believes it has a different neighbour. The job hangs during graph/channel setup, after device discovery but before the first collective, and the NCCL logs from different ranks show *inconsistent* ring strings — the diagnostic fingerprint that separates this from a pure connectivity fault, where the ring strings would agree and simply not complete.

Falsifiable hypothesis H1: ranks compute inconsistent topology graphs because of container /sys masking or a non-bijective rank-to-GPU mapping, not because a link is down. Prediction: collecting the ring/channel lines from NCCL_DEBUG_SUBSYS=GRAPH across all ranks yields at least two different ring topologies for the same communicator, and `nvidia-smi topo -m` inside the container differs from the same command on the host. If all ranks print an identical ring and the stall persists, H1 is falsified and the transport layer becomes the suspect.

Controlled experiment: single variable is topology visibility. Run A: relaunch with /sys exposed to the container as it is on the host (and with no per-rank CUDA_VISIBLE_DEVICES rewriting, letting local_rank index the full device list). Run B: keep the container as-is but pin NCCL to a single simple algorithm and channel count (for example forcing the ring algorithm with a reduced channel count) to see whether the graph search itself is where the disagreement arises. Under H1, Run A initialises with a single consistent ring across all ranks; Run B may start but with degraded bandwidth, which localises the fault to graph construction rather than to link availability.

Measurements: per-rank GRAPH-subsystem NCCL logs, collected and diffed; `nvidia-smi topo -m` inside every container and on every host; the per-rank CUDA_VISIBLE_DEVICES and the device UUIDs each rank actually opened (`nvidia-smi -L` plus the UUID logged from cudaGetDeviceProperties), which is the only reliable way to prove the mapping is bijective; the container spec's /sys mounts; and NUMA/PCIe path information for the HCAs.

Confounders: (a) MIG partitions change device enumeration and can make a bijective mapping look broken when it is merely unfamiliar — record MIG state explicitly; (b) two ranks mapped to the same physical GPU produce hangs *and* silent correctness loss, so this must be excluded by UUID, not by index; (c) ACS/IOMMU settings alter which pairs report P2P capability and therefore change the graph, meaning a topology disagreement can be a downstream symptom of a host BIOS setting rather than of container masking.

Boundary conditions: the fault is per-container and local-world-size dependent; it will not reproduce with one rank per node and it can vanish when the scheduler happens to allocate a uniform node type. Any reproduction must use the same container spec and the same local rank count, and the same MIG/ACS host configuration.

Evidence required: the diffed ring strings showing disagreement, the device-UUID map proving or disproving bijectivity, the in-container vs host topo matrices, and a post-fix run with one consistent ring and bus bandwidth at baseline.

Rollback gate: exposing /sys more broadly to containers is a security-relevant relaxation and must be reviewed as such rather than applied silently; prefer fixing the device mapping first if that alone restores a consistent graph. Accept the change only if: all ranks print an identical ring, device UUIDs are bijective across ranks, two consecutive full-scale runs initialise, and bus bandwidth is within 5 percent of baseline. Revert the mount change if any security scan or policy check flags it, and fall back to the mapping-only fix.""",
]

assert len(ANS)==N
recs=[]
seen=set()
for r,ans in zip(sl,ANS):
    m=r["messages"]
    u=[x for x in m if x["role"]=="user"][0]["content"]
    a=[x for x in m if x["role"]=="assistant"][0]["content"]
    h=hashlib.sha256(ans.encode()).hexdigest()
    assert h not in seen, ("dup answer", r["id"])
    seen.add(h)
    recs.append({
        "source_id": r["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": ans,
        "quality_dimensions": {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2},
        "risks": [
            "Source assistant turn is a grading rubric, not an answer; training on it directly teaches meta-commentary instead of diagnosis.",
            "Rubric wording is identical across the whole variant family, so unrewritten records invite template memorisation.",
            "No concrete mechanism, thresholds, or rollback gate in the source, so a model imitating it would produce unfalsifiable operational advice.",
        ],
        "evidence_required": [
            "Per-rank NCCL_DEBUG=INFO logs with INIT/NET/GRAPH subsystems, collected from every rank not just rank 0.",
            "Node-level fabric and device state: ibstat/ibv_devinfo, nvidia-smi topo -m, lsmod, /proc/<pid>/limits, df -h /dev/shm.",
            "A single-variable control run and an all-reduce bandwidth number compared against a recorded pre-incident baseline.",
        ],
        "confidence": 0.62,
    })

os.makedirs(os.path.dirname(OUT),exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f:
    for rec in recs:
        f.write(json.dumps(rec,ensure_ascii=False)+"\n")
print("WROTE",OUT,len(recs))
print("IDS",[r["source_id"] for r in recs])
