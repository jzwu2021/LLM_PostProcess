import json, hashlib, os

CORPUS="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0158.jsonl"
START=1571; N=10

# Each variant gets a distinct dominant mechanism for the "hang during collective init" scenario.
MECH=[
 ("Rendezvous store unreachable (TCPStore master addr/port)",
  "MASTER_ADDR resolves to an interface no worker can reach, so init_process_group blocks in TCPStore connect until timeout.",
  "Hypothesis H1: the hang is a rendezvous-layer failure, not an NCCL transport failure. Falsifiable prediction: rank 0 shows a listening socket on MASTER_PORT while >=1 non-zero rank shows no ESTABLISHED connection to it; and a pure gloo process group over the same MASTER_ADDR/PORT hangs identically with zero NCCL logs emitted.",
  "Controlled experiment: keep world size and launcher identical, swap backend nccl->gloo. If gloo also hangs, the fault is rendezvous/network reachability; if gloo completes, rendezvous is healthy and the fault is in NCCL bootstrap/transport.",
  ["ss -ltnp on rank0 for MASTER_PORT","per-rank `ss -tnp | grep MASTER_PORT` ESTABLISHED count == world_size-1","getent hosts $MASTER_ADDR on every node","TORCH_DISTRIBUTED_DEBUG=DETAIL logs showing last completed rendezvous step"],
  "Rollback gate: if rendezvous connect success rate <100% of ranks within 60 s, abort the job, pin MASTER_ADDR to the verified data-plane IP, and rerun; do not raise NCCL timeouts to mask it."),

 ("Interface selection picked a non-routable NIC (NCCL_SOCKET_IFNAME)",
  "NCCL bootstrap auto-selects the first non-loopback interface, which may be a docker0/br- or management NIC with no cross-node route, so bootstrap ring construction stalls.",
  "Hypothesis H2: NCCL selected an interface that is up but not routable between nodes. Falsifiable prediction: NCCL_DEBUG=INFO lines 'NCCL INFO Bootstrap : Using <ifname>' name an interface whose subnet differs across nodes, and pinning NCCL_SOCKET_IFNAME to the verified NIC removes the hang.",
  "Controlled experiment: two runs differing only in NCCL_SOCKET_IFNAME (auto vs explicit data NIC), same ranks, same node set, 3 repetitions each; compare time-to-first-allreduce.",
  ["NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET bootstrap interface line per rank","ip -br addr and ip route get <peer-ip> on each node","ping/nc -z between nodes on the chosen NIC"],
  "Rollback gate: if explicit IFNAME does not reduce time-to-first-allreduce below 30 s in 3/3 runs, revert the env change and escalate to fabric/route inspection rather than accumulating env overrides."),

 ("GPU visibility / rank-to-device mapping collision",
  "Two ranks on a node bind the same CUDA device (LOCAL_RANK not honored or CUDA_VISIBLE_DEVICES rewritten by the scheduler), so the communicator never forms a consistent topology and init blocks.",
  "Hypothesis H3: device binding is not injective within a node. Falsifiable prediction: the (hostname, pid, CUDA device UUID) table has duplicate UUIDs; running with world size reduced to 1 rank per node completes init.",
  "Controlled experiment: log torch.cuda.current_device() and the device UUID per rank before init; then rerun with 1 rank/node. Duplicate UUIDs plus a successful 1-rank/node run confirms H3.",
  ["per-rank table of hostname, LOCAL_RANK, CUDA_VISIBLE_DEVICES, device UUID","nvidia-smi --query-gpu=uuid,index --format=csv per node","successful init log for the 1-rank/node control run"],
  "Rollback gate: if duplicates are absent, drop H3 immediately instead of rewriting the launcher; only change the binding code path when the UUID table shows a collision."),

 ("Topology/peer-access asymmetry (NVLink vs PCIe P2P blocked)",
  "P2P is advertised but blocked (IOMMU/ACS or a partially populated NVLink mesh), so NCCL's channel setup retries and appears hung.",
  "Hypothesis H4: intra-node P2P setup is the blocking stage. Falsifiable prediction: NCCL INFO stops after 'Channel' lines with no 'Connected all rings'; p2pBandwidthLatencyTest reports Disabled for the affected GPU pair; NCCL_P2P_DISABLE=1 makes init complete (slower).",
  "Controlled experiment: same job, single variable NCCL_P2P_DISABLE=0 vs 1, single node, 2 GPUs, repeated for each GPU pair to localize the bad link.",
  ["nvidia-smi topo -m","p2pBandwidthLatencyTest matrix","NCCL INFO last stage before stall","ACS/IOMMU state: lspci -vvv ACSCtl and dmesg | grep -i iommu"],
  "Rollback gate: NCCL_P2P_DISABLE=1 is a diagnostic, not a fix; if it is required to run, cap the job at the degraded-throughput SLO and drain the node for fabric repair rather than shipping it as config."),

 ("RDMA/RoCE path down while NCCL_NET is forced to IB",
  "NCCL_NET=IB or an ib plugin is selected but the RoCE lane has no GID/route (PFC or the L3 GID index is wrong), so transport connect blocks instead of falling back to sockets.",
  "Hypothesis H5: the failure is in the RDMA transport, not bootstrap. Falsifiable prediction: bootstrap lines appear normally and the stall occurs after 'NET/IB' selection; forcing NCCL_NET=Socket completes init.",
  "Controlled experiment: identical job with NCCL_NET=IB vs Socket. IB-hang + Socket-success isolates the RDMA lane; then bisect with ib_write_bw between the two node HCAs.",
  ["ibv_devinfo PORT_ACTIVE state","show_gids and the GID index NCCL selected","ib_write_bw point-to-point result between the two nodes","NCCL INFO NET/IB selection lines"],
  "Rollback gate: only run production on the Socket fallback if the measured collective bandwidth still meets the step-time SLO; otherwise fail the job and repair the RoCE lane (PFC/ECN, GID index) first."),

 ("Mismatched world: one rank crashed or never started",
  "The collective blocks because a rank exited before init_process_group; the remaining ranks wait for a peer that will never arrive.",
  "Hypothesis H6: actual joined ranks < world_size. Falsifiable prediction: the count of processes that logged 'entering init_process_group' is strictly less than WORLD_SIZE, and one node's log shows an earlier CUDA/OOM/import traceback.",
  "Controlled experiment: instrument each rank to write a heartbeat file on entry; run the job and count files. If count < world_size the hang is explained without touching NCCL settings.",
  ["per-rank heartbeat file count vs WORLD_SIZE","scheduler exit codes per task (sacct / kubectl get pods)","earliest traceback across all rank logs by timestamp","dmesg -T | grep -i xid on each node"],
  "Rollback gate: never raise NCCL_TIMEOUT before the rank census; if census shows a missing rank, fix the crashing rank and rerun rather than changing collective configuration."),

 ("Version / build skew between nodes (NCCL, CUDA driver, torch)",
  "Heterogeneous images mean ranks negotiate incompatible bootstrap or protocol assumptions and stall during handshake.",
  "Hypothesis H7: init failure is caused by version skew. Falsifiable prediction: the (torch, nccl, driver) triple is not identical across all nodes; a homogeneous re-deploy of one image completes init.",
  "Controlled experiment: collect the triple from every rank, then rerun the same world size pinned to one image digest. Skew present + homogeneous run succeeds confirms H7.",
  ["per-rank torch.__version__, torch.cuda.nccl.version(), nvidia-smi driver version","container image digest per node","successful init log for the pinned-digest run"],
  "Rollback gate: if the homogeneous run still hangs, discard H7 and revert the image pin; do not keep an unverified pin in the launch template."),

 ("Timeout/backoff masking a slow but progressing init",
  "Init is not deadlocked but extremely slow (e.g. large world size with serialized bootstrap over a shared control NIC), and looks like a hang below the timeout threshold.",
  "Hypothesis H8: init is progressing, not deadlocked. Falsifiable prediction: NCCL INFO line counts strictly increase over successive 60 s samples, and time-to-init scales roughly with world size across 2/4/8-node runs.",
  "Controlled experiment: scaling sweep at 2, 4, 8 nodes with all other variables fixed, recording time-to-first-allreduce; a monotonic, super-linear curve supports H8, a flat hang at all sizes refutes it.",
  ["timestamped NCCL INFO line-count samples at 0/60/120 s","time-to-first-allreduce per world size","control-NIC utilization during bootstrap"],
  "Rollback gate: raising the timeout is acceptable only if the scaling curve is measured and the projected init time is under the job's startup budget; otherwise revert and fix bootstrap bandwidth."),

 ("Firewall / security policy dropping ephemeral bootstrap ports",
  "MASTER_PORT is allowed but NCCL's ephemeral bootstrap and data ports are dropped by host firewall or network policy, so the ring never closes.",
  "Hypothesis H9: a subset of required ports is filtered. Falsifiable prediction: the TCPStore connection succeeds (rendezvous completes) but the NCCL ring setup stalls; nc -z on an ephemeral port in NCCL's range fails between the same node pair.",
  "Controlled experiment: with rendezvous confirmed healthy, pin NCCL_PORT_RANGE (or the equivalent bootstrap port control) to an explicitly allowed range and rerun; success under the allowed range confirms H9.",
  ["iptables -S / nftables ruleset and CNI NetworkPolicy dump","nc -z peer <ephemeral-port> result matrix","tcpdump SYN-without-SYNACK capture on the stalled pair","rendezvous-complete log line preceding the stall"],
  "Rollback gate: do not disable the firewall as the fix; if a scoped allow-rule cannot be obtained, abort the job and escalate — a blanket flush is an unacceptable production rollback."),

 ("Stale shared-memory / IPC namespace limits on the node",
  "Insufficient /dev/shm or a restricted IPC namespace blocks NCCL's intra-node shared-memory transport, stalling communicator creation before any network activity.",
  "Hypothesis H10: the block is intra-node SHM, not inter-node network. Falsifiable prediction: a single-node, 2-GPU run hangs identically with zero inter-node traffic, and NCCL_SHM_DISABLE=1 lets that same run complete.",
  "Controlled experiment: reduce to one node and two ranks (removes all network variables), then toggle NCCL_SHM_DISABLE. Hang that disappears only with SHM disabled isolates the SHM path.",
  ["df -h /dev/shm and the container shm-size setting","single-node 2-GPU control run outcome","NCCL INFO SHM transport lines","ipcs -m and ulimit -l per rank"],
  "Rollback gate: NCCL_SHM_DISABLE=1 is a diagnostic only; if it is needed to pass, raise /dev/shm to the documented size and re-verify, and roll the job back if intra-node bandwidth drops below the step-time SLO."),
]

BASE_ASSUMPTIONS=("Assumptions: the hang is reproducible; launcher, world size and node set are held constant across runs; "
 "no code change is made between control and treatment runs; measured facts are separated from estimates and no vendor-specific "
 "behaviour is asserted without a log line or command output to back it.")

TRIAGE=("Ordered triage (do this before touching any tunable): (1) record the full rank census - hostname, LOCAL_RANK, PID, "
 "CUDA device UUID, WORLD_SIZE, MASTER_ADDR/PORT, and the exact image digest for every rank; (2) confirm whether the rendezvous "
 "stage completed by reading TORCH_DISTRIBUTED_DEBUG=DETAIL and NCCL_DEBUG=INFO up to the last emitted line; (3) run a minimal "
 "standalone all-reduce (one small tensor, same world size) to reproduce without the training script; (4) shrink the world - "
 "single node, then 2 ranks - to separate intra-node from inter-node causes.")

def build(mech):
    name, mechanism, hyp, exp, evid, rollback = mech
    return (
      f"{BASE_ASSUMPTIONS}\n\n"
      f"Primary mechanism under test: {name}. {mechanism}\n\n"
      f"{TRIAGE}\n\n"
      f"Falsifiable hypothesis. {hyp}\n\n"
      f"Controlled experiment. {exp} Change exactly one variable per run and repeat 3x, because init hangs are "
      f"frequently intermittent and a single success is not evidence.\n\n"
      f"Expected confounders. Scheduler-injected environment variables that silently override CUDA_VISIBLE_DEVICES or "
      f"NCCL_SOCKET_IFNAME; a node that was drained or rebooted between runs; DNS/host-file differences between the login "
      f"node and compute nodes; and prior debug env vars left in the launch template from an earlier incident. Record the "
      f"full environment diff between control and treatment runs, otherwise the comparison is not controlled.\n\n"
      f"Measurements to collect: " + "; ".join(evid) + ".\n\n"
      f"{rollback} General rollback criteria: if two consecutive controlled runs fail to move the stall to a different "
      f"stage, stop tuning, restore the last known-good launch template verbatim, and hand off with the collected evidence "
      f"bundle rather than stacking further environment overrides."
    )

recs=[]
with open(CORPUS) as f:
    for i,l in enumerate(f,1):
        if i<START: continue
        if i>=START+N: break
        d=json.loads(l)
        msgs={m["role"]:m["content"] for m in d["messages"]}
        su=[m["content"] for m in d["messages"] if m["role"]=="user"][0]
        sa=[m["content"] for m in d["messages"] if m["role"]=="assistant"][0]
        mech=MECH[(i-START)%len(MECH)]
        ca=build(mech)
        recs.append({
          "source_id": d["id"],
          "teacher_lane":"teacher-B",
          "teacher_model":"claude-opus-5-current",
          "calibration_status":"provisional",
          "decision":"rewrite",
          "source_user":su,
          "source_assistant":sa,
          "corrected_answer":ca,
          "quality_dimensions":{"technical_correctness":3,"instruction_coverage":2,"operational_safety":2},
          "risks":[
            "Source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of diagnosis.",
            "Rubric omits any rollback threshold, so a model imitating it may recommend raising NCCL timeouts to mask a real fault.",
            "No explicit separation of intra-node (SHM/P2P) from inter-node (socket/RDMA) causes, risking mis-ordered triage."
          ],
          "evidence_required":[
            "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET logs from every rank, with the last emitted stage identified",
            "TORCH_DISTRIBUTED_DEBUG=DETAIL rendezvous completion status per rank",
            "Full rank census: hostname, LOCAL_RANK, PID, CUDA device UUID, image digest",
            "Minimal standalone all-reduce reproduction at the same and at reduced world size",
            "Control-vs-treatment environment diff proving exactly one variable changed"
          ],
          "confidence":0.62
        })

seen={}
for r in recs:
    h=hashlib.sha256(r["corrected_answer"].encode()).hexdigest()
    assert h not in seen, ("duplicate corrected_answer", r["source_id"], seen[h])
    seen[h]=r["source_id"]
assert len(recs)==N, len(recs)

os.makedirs(os.path.dirname(OUT),exist_ok=True)
with open(OUT,"w") as f:
    for r in recs:
        f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("WROTE",OUT,len(recs),"unique_answers",len(seen))
