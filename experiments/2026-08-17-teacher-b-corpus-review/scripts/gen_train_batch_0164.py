import json, hashlib, os

ROOT = '/home/johnson/workspace/LLM_PostProcess'
CORPUS = os.path.join(ROOT, 'research/ai-infra-expert/corpus/train.jsonl')
OUT = os.path.join(ROOT, 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0164.jsonl')
START, N = 1630, 10

rows = [json.loads(l) for l in open(CORPUS)][START:START+N]

# Each variant gets a distinct dominant failure mechanism for NCCL/collective-init hang.
MECH = [
 ("rendezvous store deadlock: mismatched WORLD_SIZE vs launched ranks",
  "TCPStore/c10d rendezvous blocks until all `WORLD_SIZE` ranks check in. If the launcher starts fewer ranks (dead node, OOM-killed worker, SLURM ntasks mismatch), `init_process_group` hangs forever with no NCCL log line at all.",
  "H1: the hang occurs BEFORE any NCCL communicator is created; exactly k<WORLD_SIZE ranks reached the store.",
  "Falsified if all WORLD_SIZE ranks print a pre-init log line and NCCL_DEBUG=INFO emits `NCCL INFO Bootstrap` on every rank.",
  ["Instrument each rank to log (hostname, LOCAL_RANK, RANK, WORLD_SIZE, pid) to a shared file BEFORE init_process_group; count distinct RANK values.",
   "Set `TORCH_DISTRIBUTED_DEBUG=DETAIL` and a short `timeout=timedelta(seconds=120)` on init_process_group so it raises instead of hanging.",
   "On the store host: `ss -tanp | grep <MASTER_PORT>` and count ESTABLISHED connections; compare to WORLD_SIZE.",
   "Check scheduler view: `scontrol show job <id>` / `kubectl get pods -l job-name=<x>` for a missing or CrashLoopBackOff worker."]),

 ("NCCL_SOCKET_IFNAME selecting a non-routable interface (docker0/lo/virbr0)",
  "NCCL bootstrap picks interfaces by name-prefix heuristics. On hosts with docker0/br-*/virbr0, different ranks can select different, mutually unroutable subnets; bootstrap TCP connect then blocks in retry until the (default very long) timeout.",
  "H1: ranks selected different bootstrap interfaces / non-routable subnets.",
  "Falsified if `NCCL INFO Bootstrap : Using <ifname>:<ip>` shows the same routable fabric subnet on every rank yet the hang persists.",
  ["Collect `NCCL INFO Bootstrap : Using` line from all ranks (NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,ENV).",
   "Controlled experiment: rerun pinning `NCCL_SOCKET_IFNAME=<fabric-if>` (and `GLOO_SOCKET_IFNAME` for the gloo PG) on all ranks; hang should clear.",
   "Cross-node reachability matrix on the chosen IPs: `ping`/`nc -vz <ip> <port>` from every host to every host.",
   "`ip -o addr` + `ip route get <peer-ip>` on each node to prove routability."]),

 ("CUDA device visibility collision: two ranks bound to the same GPU",
  "If LOCAL_RANK->device mapping is wrong (missing `torch.cuda.set_device(local_rank)`, or CUDA_VISIBLE_DEVICES set identically for all local ranks), two ranks open communicators on the same device. NCCL then cannot complete the intra-node ring and blocks in `ncclCommInitRank`.",
  "H1: the multiset of (hostname, physical GPU UUID) used by ranks contains a duplicate.",
  "Falsified if every rank maps to a unique GPU UUID and the hang persists.",
  ["Each rank logs `torch.cuda.current_device()` and the GPU UUID from `nvidia-smi --query-gpu=uuid --format=csv`; assert uniqueness per host.",
   "`nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv` during the hang; look for 2 PIDs on one UUID.",
   "Controlled experiment: run with 1 rank/node (world = #nodes). If it initializes, the fault is intra-node mapping, not the fabric.",
   "Verify `CUDA_VISIBLE_DEVICES` and `LOCAL_RANK` per rank from /proc/<pid>/environ."]),

 ("P2P / NVLink path disabled or broken (IOMMU, ACS, MIG) causing intra-node ring stall",
  "NCCL probes P2P (NVLink or PCIe P2P) during init. With IOMMU on + PCIe ACS enabled, or MIG-partitioned GPUs, the probe can hang or the chosen transport can fail silently on first collective rather than erroring at init.",
  "H1: intra-node P2P transport is the blocking stage.",
  "Falsified if `NCCL_P2P_DISABLE=1` does not change the hang behaviour.",
  ["Baseline: `nvidia-smi topo -m` and `p2pBandwidthLatencyTest` from cuda-samples on the affected node.",
   "Controlled experiment: rerun with `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0`; then with `NCCL_P2P_LEVEL=SYS`. Record which config initializes.",
   "Check `dmesg | grep -i -e iommu -e 'AER'` and ACS state via `lspci -vvv | grep -i acsctl`.",
   "Confirm MIG state: `nvidia-smi -L` (MIG devices cannot use NVLink P2P between instances)."]),

 ("Rank-divergent collective call order / shape mismatch (a real deadlock, not a network fault)",
  "NCCL collectives are matched positionally per communicator. If one rank takes a different code branch (e.g. only rank 0 logs and calls an extra all_reduce, or batch shapes differ), the ranks block waiting on mismatched ops. This looks identical to a fabric hang.",
  "H1: at hang time, ranks are stopped at DIFFERENT collective call sites or with different tensor shapes.",
  "Falsified if all ranks show the identical opCount and the same source line.",
  ["`py-spy dump --pid <pid>` on every rank at hang time; compare Python stacks.",
   "`TORCH_NCCL_DESYNC_DEBUG=1` (older: `TORCH_DISTRIBUTED_DEBUG=DETAIL`) to get a mismatch report instead of a silent hang.",
   "Enable NCCL flight recorder `TORCH_NCCL_TRACE_BUFFER_SIZE=2000` and dump on timeout; compare opCount/sizes across ranks.",
   "Controlled experiment: run a fixed synthetic shape with no data-dependent branching; if it passes, the fault is in the model/data path."]),

 ("RDMA/RoCE fabric not usable: GID index, PFC/ECN, or MTU mismatch, NCCL silently retrying IB transport",
  "With `NCCL_IB_HCA` set, NCCL creates QPs over RoCEv2. A wrong GID index (v1 vs v2), missing lossless PFC on a switch hop, or MTU mismatch causes QP connect / first RDMA write to stall; init appears hung after `NCCL INFO NET/IB`.",
  "H1: the hang is inside the IB/RoCE transport, not bootstrap.",
  "Falsified if `NCCL_IB_DISABLE=1` (TCP socket fallback) still hangs at the same point.",
  ["Read the last NCCL log line per rank: bootstrap-stage vs `NET/IB ... via` stage discriminates the two.",
   "Controlled experiment: `NCCL_IB_DISABLE=1` run. If it completes (slower), the fault is RoCE config.",
   "`ibv_devinfo -v` (port state ACTIVE, MTU), `show_gids` for the RoCEv2 GID index; pin `NCCL_IB_GID_INDEX` explicitly.",
   "`ethtool -S <if> | grep -i -e pause -e discard` and switch counters for PFC pause storms / drops.",
   "`ib_write_bw` between the two hosts as an NCCL-independent fabric baseline."]),

 ("GPUDirect RDMA path unavailable: nvidia-peermem/dma-buf not loaded, host-bounce fallback stall",
  "GDR requires the peer-memory module (`nvidia_peermem`) or dma-buf support in the RDMA driver. Without it NCCL falls back to host staging; with `NCCL_NET_GDR_LEVEL` forced high it can attempt a path the driver rejects and block in registration.",
  "H1: GDR registration is the blocking stage.",
  "Falsified if `NCCL_NET_GDR_LEVEL=0` (disable GDR) leaves the hang unchanged.",
  ["`lsmod | grep -e nvidia_peermem -e nv_peer_mem` and `dmesg | grep -i peermem` on every node.",
   "NCCL_DEBUG_SUBSYS=NET log: look for `GPU Direct RDMA Enabled/Disabled` per rank.",
   "Controlled experiment: run with `NCCL_NET_GDR_LEVEL=0` vs `=SYS`; record init success and, if it runs, bus bandwidth from nccl-tests.",
   "Confirm HCA and GPU are under the same PCIe switch (`nvidia-smi topo -m` shows PIX/PXB, not SYS) before expecting GDR gain."]),

 ("Firewall / port-range blocking: MASTER_PORT reachable but NCCL bootstrap ephemeral ports dropped",
  "TCPStore uses MASTER_PORT only; NCCL's own bootstrap and socket transport use ephemeral ports. A firewall that allows MASTER_PORT but drops other inter-node TCP lets rendezvous succeed and then hangs at communicator init — a very common false 'NCCL is broken' report.",
  "H1: rendezvous completed on all ranks but NCCL bootstrap TCP connects are being dropped.",
  "Falsified if all ranks pass rendezvous AND arbitrary-port TCP between nodes succeeds, yet init still hangs.",
  ["Confirm stage: all ranks log 'process group initialized' but none logs a completed `ncclCommInitRank`.",
   "`nc -l <random-port>` on node A and `nc -vz A <random-port>` from node B to test non-MASTER_PORT reachability.",
   "Inspect `iptables -L -n` / `nft list ruleset` / cloud security groups for allow-rules narrower than the full inter-node range.",
   "Controlled experiment: open a known range and set `NCCL_SOCKET_FAMILY`/port policy per site; rerun."]),

 ("Hostname/DNS or MASTER_ADDR resolving differently per node (split rendezvous)",
  "If MASTER_ADDR is a short hostname resolved via per-node /etc/hosts, some ranks may resolve it to 127.0.0.1 (loopback entry) and form a separate store. Each partition waits for absent peers.",
  "H1: ranks resolved MASTER_ADDR to different IPs; ≥2 disjoint rendezvous groups exist.",
  "Falsified if `getent hosts $MASTER_ADDR` returns the identical routable IP on every node.",
  ["Run `getent hosts $MASTER_ADDR` and `hostname -I` on every node; assert one identical non-loopback IP.",
   "On the master, count store connections: `ss -tan state established '( sport = :$MASTER_PORT )' | wc -l` vs WORLD_SIZE.",
   "Controlled experiment: replace MASTER_ADDR with a literal IP and rerun.",
   "Grep /etc/hosts on all nodes for a 127.0.0.1 entry aliasing the master hostname."]),

 ("Timeout/backoff misconfiguration masking a slow-but-working init (not a true hang)",
  "Large worlds with many communicators (TP+PP+DP subgroups) can take minutes to initialize; with `NCCL_BLOCKING_WAIT`/long default PG timeout there is no output, so a slow init is misreported as a hang. Conversely a too-short timeout aborts a healthy run.",
  "H1: init is progressing but slow — elapsed time scales with #communicators/world size, and completes if given more wall clock.",
  "Falsified if a 30-minute run shows zero NCCL log progress and constant CPU/network counters.",
  ["Timestamped NCCL_DEBUG=INFO logs: check whether new `Channel`/`Connected` lines keep appearing over time (progress vs true stall).",
   "Sample `py-spy dump` twice 60s apart; a moving stack = progress, an identical stack = stall.",
   "Controlled experiment: sweep world size 2 -> 8 -> 32 and record init wall time; fit trend to predict expected time at full scale.",
   "Set an explicit `timeout=timedelta(minutes=30)` and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` so a genuine stall raises with a rank-attributed error."]),
]

def build(idx, row):
    mech_name, mech_desc, h1, falsify, steps = MECH[idx]
    m = row['messages']
    su = [x for x in m if x['role'] == 'user'][0]['content']
    sa = [x for x in m if x['role'] == 'assistant'][0]['content']
    steps_txt = "\n".join("%d. %s" % (i+1, s) for i, s in enumerate(steps))
    ca = f"""## Scope and assumptions

Symptom: a multi-GPU job stops making progress during collective initialization (`init_process_group` or the first `ncclCommInitRank`/first collective). Assumptions, to be confirmed before acting: PyTorch + NCCL backend; a launcher (torchrun/SLURM/K8s) sets RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT; no code change immediately preceded the failure; the job is non-production or a canary, so restarts are cheap.

## Stage discrimination first (do this before any tuning)

Nearly all "NCCL hang" reports collapse into three stages. Identify the stage before changing any env var:
- Stage A - rendezvous/store: no NCCL log lines at all on any rank.
- Stage B - NCCL bootstrap: `NCCL INFO Bootstrap : Using ...` appears but `ncclCommInitRank` never returns.
- Stage C - transport/first collective: `NET/IB` or `NET/Socket` lines appear, communicator rings are being built, then it stalls.

Run everything with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,NET`, `NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log`, and `TORCH_DISTRIBUTED_DEBUG=DETAIL`. Never diagnose a hang without per-rank logs.

## Dominant hypothesis for this variant: {mech_name}

Mechanism. {mech_desc}

H1 (falsifiable): {h1}
Falsification condition: {falsify}
H0 (null): the fabric and mapping are healthy and init is merely slow; then elapsed init time scales with world size and log lines keep appearing.

## Controlled experiment / measurement plan

{steps_txt}

Control the confound: change exactly one variable per run, keep the same node set, same container image, same world size, and repeat each configuration at least twice (init failures are often intermittent). Record for each run: config diff, stage reached, last NCCL log line per rank, wall time to failure.

## Independent baseline (fabric vs framework)

Before blaming application code, establish an NCCL-only baseline with `nccl-tests`:
`mpirun -np <world> all_reduce_perf -b 8 -e 1G -f 2 -g 1`.
If nccl-tests initializes and reports plausible busbw while the training job hangs, the fault is in the application/launcher, not the fabric. If nccl-tests also hangs at the same stage, the fault is environmental.

## Expected confounders

- A second, unrelated PG (gloo) using a different interface than the NCCL PG - `GLOO_SOCKET_IFNAME` must be set too.
- Stale zombie processes from a previous run still holding GPUs or MASTER_PORT (`fuser -k`, check `nvidia-smi`).
- Container networking (host vs bridge) changing interface names between nodes.
- A single sick node: always test the node set with one node removed before concluding a global config fault.
- ECC/Xid errors making one GPU unresponsive; check `nvidia-smi -q | grep -i xid` and `dmesg | grep Xid`.

## Numbers

No numbers are quoted here as MEASURED, because no telemetry from this cluster was provided. Any threshold below is an ESTIMATE with its derivation stated:
- ESTIMATE: healthy `init_process_group` for a single-node 8-GPU job completes in well under 60 s; derivation - it is dominated by one TCP rendezvous round plus intra-node transport setup, both sub-second operations, so a 60 s bound is ~2 orders of magnitude of headroom. Exceeding it justifies escalation to stage-A/B triage.
- ESTIMATE: multi-node init time grows roughly with the number of communicators created (TP x PP x DP subgroups), not with parameter count; derivation - each `new_group` performs its own bootstrap. Measure the 2/8/32-rank trend on this cluster to replace this estimate with MEASURED values.
Do not treat either number as a site SLO until measured locally.

## Evidence required before declaring root cause

- Per-rank NCCL logs showing the identical failing stage across ranks.
- One config change that reproducibly flips the outcome (hang -> success) across at least two paired runs.
- An independent, non-NCCL confirmation of the same fault (`ss`/`ip route`/`ibv_devinfo`/`nvidia-smi`/scheduler state).
- A negative control: the "fixed" config must still fail when the change is reverted.

## Rollback criteria

- Any env-var workaround (`NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_NET_GDR_LEVEL=0`) is a diagnostic, not a fix: it usually costs bandwidth. Roll it back once the real cause is found; if it must stay, record the measured throughput loss from nccl-tests before/after.
- Revert immediately if: post-change all_reduce busbw drops more than 10% vs the pre-incident baseline (ESTIMATE threshold - it is above run-to-run noise, which is typically a few percent; replace with the site's measured variance), or if any rank reports a new NCCL async error, or if step time regresses beyond the pre-incident band.
- Cluster-wide changes (driver, IOMMU/ACS, switch PFC) require a canary on one node pair, a defined bake window, and a documented one-command revert before rollout.

## Assessment of the reference answer

The reference answer lists the right checklist headings (ranks/environment, rendezvous, topology and GPU visibility, minimal all-reduce, interface selection and timeout, single-node vs reduced-world comparison) but stays at the level of section titles: it names no commands, no stage discrimination, no falsification condition, and no rollback threshold, so it is not directly usable during an incident. This rewrite keeps its checklist coverage and adds the executable mechanism, controls, evidence, and revert gates."""
    return {
        "source_id": row['id'],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": su,
        "source_assistant": sa,
        "corrected_answer": ca,
        "quality_dimensions": {
            "technical_correctness": 4,
            "instruction_coverage": 2,
            "operational_safety": 3,
        },
        "risks": [
            "Reference answer is a rubric checklist, not an answer; training on it teaches heading-listing rather than diagnosis.",
            "No falsification condition or negative control, so an operator could stop at a correlated env-var workaround and call it a root cause.",
            "Blanket env-var workarounds (NCCL_IB_DISABLE / NCCL_P2P_DISABLE / GDR off) silently cost interconnect bandwidth if left in place.",
            "Prompt supplies no cluster telemetry; any numeric threshold is necessarily an ESTIMATE and must not be treated as a site SLO.",
        ],
        "evidence_required": [
            "Per-rank NCCL_DEBUG=INFO logs (NCCL_DEBUG_FILE per host/pid) identifying the failing stage.",
            "Rank inventory: hostname, RANK, LOCAL_RANK, WORLD_SIZE, GPU UUID, pid for every launched process.",
            "Independent nccl-tests all_reduce_perf baseline on the same node set.",
            "A paired hang/success run differing in exactly one configuration variable, repeated at least twice.",
            "Pre- and post-change busbw measurements to quantify the cost of any retained workaround.",
        ],
        "confidence": 0.74,
    }

recs = [build(i, r) for i, r in enumerate(rows)]

# anti-template: corrected_answer must be pairwise distinct
h = [hashlib.sha256(r['corrected_answer'].encode()).hexdigest() for r in recs]
assert len(set(h)) == len(h), "duplicate corrected_answer detected"

with open(OUT, 'w') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs), "records; distinct hashes:", len(set(h)))
