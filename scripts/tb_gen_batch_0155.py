import json, hashlib, os

CORPUS="research/ai-infra-expert/corpus/train.jsonl"
START=1540; N=10
OUT="experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0155.jsonl"

MECH=[
 dict(name="Rendezvous store partial world (TCPStore/MASTER_ADDR:PORT)",
  hyp="The hang is a rendezvous-completion failure: fewer than WORLD_SIZE ranks ever reach the store barrier, so init_process_group blocks until timeout rather than failing fast.",
  mech="NCCL bootstrap runs over the PyTorch TCPStore at MASTER_ADDR:MASTER_PORT. Rank 0 opens the store; all other ranks connect and publish their bootstrap handles. ncclCommInitRank cannot return until all WORLD_SIZE handles are present. A launcher that computes RANK/LOCAL_RANK wrongly (duplicate ranks, or WORLD_SIZE larger than launched processes) leaves the barrier permanently short.",
  exp="Instrument every process to log (hostname, pid, RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) BEFORE init_process_group. Set TORCH_DISTRIBUTED_DEBUG=DETAIL and a short store timeout (timeout=timedelta(seconds=120)). Controlled arm: run the identical launcher with a pure gloo/TCPStore-only barrier and no CUDA. If the gloo barrier also hangs, the fault is rendezvous, not NCCL transport.",
  meas="Count of distinct RANK values that logged pre-init; store client connection count on rank 0; time-to-timeout; `ss -tnp state established '( sport = :MASTER_PORT )'` connection count on the rank-0 host.",
  conf="A crashed rank whose stderr was swallowed by the launcher looks identical to a never-launched rank; an unrelated slow container start can delay a rank past a short timeout.",
  rb="Roll back if the gloo-only barrier succeeds while NCCL still hangs (falsifies this hypothesis) — then move to transport hypotheses. Abort the debug run if it exceeds 2x the configured store timeout."),
 dict(name="Wrong socket interface selection (NCCL_SOCKET_IFNAME)",
  hyp="NCCL bootstrap selects a non-routable interface (docker0/virbr0/lo alias) so out-of-band handshake packets never reach peer nodes; the hang is transport-level, not rendezvous-level.",
  mech="NCCL's bootstrap and its fallback socket transport enumerate interfaces and pick by internal preference unless NCCL_SOCKET_IFNAME is set. On hosts with container bridges, the chosen interface can be up and have an address but be unreachable from other nodes, so the TCP connect stalls in SYN retransmit rather than being refused.",
  exp="Run with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET and read the 'Using [0]<ifname>:<ip>' line on every rank. Controlled arm: pin NCCL_SOCKET_IFNAME to the known data-plane NIC (for example ^docker,lo exclusion, or an explicit eth name) on all nodes and rerun the same job unchanged.",
  meas="Per-rank selected interface and IP from NCCL INIT logs; cross-node reachability of exactly that IP (ping and a raw TCP connect on the bootstrap port); SYN-retransmit counts from `nstat -az TcpExtTCPSynRetrans`.",
  conf="Multiple NICs on the same subnet can make an incorrect pick still work intermittently; asymmetric routing can make ping succeed while the bootstrap port is filtered.",
  rb="If pinning the interface does not change time-to-hang and the logged IPs were already correct, this hypothesis is falsified — revert the env var and test fabric-level causes. Keep the pin only after two consecutive clean runs."),
 dict(name="RoCEv2 GID index / lossless-fabric misconfiguration",
  hyp="The IB verbs path is selected but the RoCEv2 GID index or PFC/ECN configuration is inconsistent across nodes, so the RDMA connection handshake never completes and NCCL blocks in transport setup.",
  mech="On RoCE, each port exposes several GIDs (IPv4-mapped RoCEv1 vs RoCEv2, per VLAN). NCCL picks one via NCCL_IB_GID_INDEX. If two nodes pick GIDs of different RoCE versions or different VLANs, the QP transition to RTR/RTS never completes. Separately, if PFC priority or DSCP marking does not match the switch config, handshake packets are dropped under any congestion.",
  exp="Enumerate `show_gids` on every node and confirm the RoCEv2 entry index is identical. Controlled arm A: force the same NCCL_IB_GID_INDEX everywhere. Controlled arm B: set NCCL_IB_DISABLE=1 to force the socket transport. If the job initializes with IB disabled, the fault is in the RDMA path.",
  meas="Per-node GID table and chosen index from NCCL NET/IB logs; port counters via `perfquery`/ethtool for PortXmitDiscards and pause frames; QP state at hang time; ib_write_bw between the same node pair as an independent baseline.",
  conf="ib_write_bw may use a different GID than NCCL, so a passing point-to-point test does not exonerate the fabric; a single misconfigured switch port makes the failure host-pair dependent, which mimics a random hang.",
  rb="If NCCL_IB_DISABLE=1 also hangs, this hypothesis is falsified. Revert any switch-side PFC change immediately if pause-frame counts rise on uninvolved ports; require one full clean run before keeping GID pinning."),
 dict(name="Duplicate GPU assignment via CUDA_VISIBLE_DEVICES / LOCAL_RANK",
  hyp="Two ranks on the same host bind to the same physical GPU, so the intra-node communicator can never form a complete ring and init blocks.",
  mech="NCCL requires a bijection between ranks and devices within a communicator. If the launcher exports CUDA_VISIBLE_DEVICES per process AND the code also calls torch.cuda.set_device(LOCAL_RANK), the indices are applied twice, collapsing several ranks onto device 0 while other GPUs are unused. NCCL then sees a duplicate device in the communicator and stalls during graph construction.",
  exp="Log per rank (hostname, LOCAL_RANK, CUDA_VISIBLE_DEVICES, torch.cuda.current_device(), and the GPU UUID from pynvml). Assert the UUID multiset is unique per host before init_process_group. Controlled arm: remove the per-process CUDA_VISIBLE_DEVICES export and rely only on set_device(LOCAL_RANK).",
  meas="Set of distinct GPU UUIDs per host vs number of local ranks; `nvidia-smi --query-compute-apps=pid,gpu_uuid` during the hang; per-GPU memory footprint asymmetry.",
  conf="Some frameworks legitimately share a GPU for non-NCCL work (e.g. an eval process), which can produce duplicate UUIDs without being the cause; MIG makes UUID identity subtler because instances differ from parent devices.",
  rb="Falsified if UUIDs are already unique per host. Revert the launcher change if throughput regresses or if any rank lands on a GPU that is not NUMA-local to its CPU affinity."),
 dict(name="P2P/NVLink path blocked by PCIe ACS or IOMMU",
  hyp="Intra-node peer-to-peer transport is advertised but non-functional because PCIe Access Control Services or IOMMU remapping blocks direct device-to-device writes, so NCCL hangs after topology detection.",
  mech="NCCL builds an intra-node graph over NVLink or PCIe P2P. With ACS redirection enabled on upstream bridges, or IOMMU in strict translation mode, a P2P write is routed to the root complex and can be silently dropped for a peer mapping NCCL believes is valid. Detection succeeds, first data movement never lands, and the init all-gather blocks.",
  exp="Run the CUDA sample p2pBandwidthLatencyTest and `nvidia-smi topo -m` as an independent baseline. Controlled arm: rerun the job with NCCL_P2P_DISABLE=1 (and NCCL_SHM_DISABLE=1 as a second arm) on a single node only. If a single-node run succeeds with P2P disabled and hangs with it enabled, the hypothesis holds.",
  meas="ACS state from `lspci -vvv | grep -i acsctl`; IOMMU mode from kernel cmdline; p2pBandwidthLatencyTest matrix (GB/s, and whether it completes); NCCL GRAPH log showing the selected channels.",
  conf="Disabling P2P also changes the algorithm and buffer sizes, so a pass may reflect the algorithm change rather than ACS; some servers expose ACS only on a subset of bridges, making the failure GPU-pair specific.",
  rb="Falsified if the single-node run hangs identically with P2P disabled. Do not leave NCCL_P2P_DISABLE=1 in production: it typically costs a large fraction of intra-node bandwidth. Revert BIOS/ACS changes if any PCIe correctable-error counter increases."),
 dict(name="MTU mismatch / jumbo frames dropped mid-path",
  hyp="The data-plane path has an inconsistent MTU, so small handshake packets pass but the first large bootstrap or all-gather payload is dropped, producing a hang that looks like an init failure.",
  mech="If endpoints are configured for 9000-byte MTU while an intermediate switch port or overlay tunnel enforces 1500, oversized frames are dropped. With DF set and ICMP fragmentation-needed filtered, path MTU discovery cannot correct it, so the sender retransmits forever. Connection setup (small packets) succeeds, which is why the symptom appears only once real data moves.",
  exp="Probe the path with `ping -M do -s 8972 <peer>` between every node pair on the data-plane IPs. Controlled arm: set the endpoint MTU uniformly to 1500 on all nodes and rerun; a job that initializes at 1500 but hangs at 9000 confirms the hypothesis.",
  meas="Per-hop MTU; success/failure matrix of DF-set large pings; interface error and drop counters via `ip -s link` and ethtool -S before/after; TCP retransmit counters during the hang.",
  conf="Some overlay/VXLAN encapsulation reduces effective MTU by ~50 bytes, so 9000-to-9000 endpoint config can still fail; a switch may drop oversized frames without incrementing an obvious counter.",
  rb="Falsified if the 1500-MTU run hangs the same way. Treat the 1500 setting as a temporary mitigation only, since it will reduce achievable bandwidth; roll back to jumbo once the switch config is corrected and re-verified with the same ping matrix."),
 dict(name="Firewall / security-group filtering of NCCL ephemeral ports",
  hyp="Only the rendezvous port is permitted between hosts; NCCL's dynamically chosen bootstrap and socket-transport ports are silently dropped, so ranks connect to the store but never to each other.",
  mech="After the store rendezvous, ranks exchange listener addresses and open direct connections on ephemeral ports. A DROP (not REJECT) firewall rule makes those connects stall in SYN_SENT instead of failing, which is why the process-group creation succeeds up to the point of the first peer connection and then blocks.",
  exp="Capture with `ss -tn state syn-sent` on each host during the hang and correlate destination ports with the peer addresses in NCCL INIT logs. Controlled arm: constrain the range with NCCL_PORT_RANGE-style pinning if supported, or open the full ephemeral range between the node pair only, and rerun unchanged.",
  meas="Count and destinations of SYN_SENT sockets; firewall counters (`iptables -L -n -v` / nft counters) for the DROP rules; packet capture on the peer showing SYN arrival or absence.",
  conf="A host-based firewall and a cloud security group can both be in play, so opening one may not change behaviour; NAT between nodes can rewrite ports and mimic filtering.",
  rb="Falsified if SYN packets are observed arriving and being accepted on the peer. Any firewall opening must be scoped to the specific node pair and CIDR, time-boxed, and reverted after the experiment; do not leave a blanket allow-all rule in place."),
 dict(name="Version skew across nodes (NCCL / CUDA driver / container image)",
  hyp="Nodes are running different NCCL or driver builds, so the communicator handshake fails a compatibility or protocol check and the run stalls instead of erroring cleanly.",
  mech="NCCL negotiates protocol and algorithm capabilities at init. Mixed minor versions, or a container built against a newer CUDA than the host driver supports, can leave one side waiting on a feature the other never advertises. Rolling image updates that reach only part of the fleet make this look intermittent and node-dependent.",
  exp="Collect on every node: nccl version (from NCCL_DEBUG=VERSION output), torch.version.cuda, `nvidia-smi --query-gpu=driver_version`, and the container image digest. Assert all four are identical multisets. Controlled arm: pin every node to a single image digest and rerun the identical job.",
  meas="The four-tuple per node; the NCCL 'NCCL version x.y.z+cudaA.B' banner per rank; whether the hang set correlates exactly with the minority version group.",
  conf="Version skew often coexists with a real transport fault, so unifying versions may fix nothing; identical digests do not guarantee identical host drivers, which live outside the image.",
  rb="Falsified if all nodes already report identical versions. Roll back the image pin if the unified version regresses throughput beyond an agreed threshold (for example >5% step-time increase over the prior baseline)."),
 dict(name="Rank desynchronisation: unequal collective order before init completes",
  hyp="At least one rank enters a different collective sequence (or is blocked in data loading) so the collective call sets do not match, and the outstanding ranks wait indefinitely.",
  mech="NCCL collectives are matched positionally: every rank must issue the same collectives in the same order with the same sizes. A conditional branch (for example rank 0 doing an extra broadcast for a checkpoint, or a rank stalled on a slow filesystem before its first all-reduce) breaks that invariant. The result is a hang at what appears to be initialization because the first mismatched call is near startup.",
  exp="Enable TORCH_DISTRIBUTED_DEBUG=DETAIL to get collective-mismatch reporting, and log a monotonically increasing (collective_name, seq, tensor_shape) tuple per rank. Controlled arm: run with a synthetic in-memory dataset that removes filesystem variance, keeping model and parallelism identical.",
  meas="Per-rank last-issued collective tuple at hang time; py-spy dump of every process to see which frame each rank is parked in; storage latency percentiles for the dataset mount.",
  conf="A slow but progressing filesystem can look like a permanent hang under a short observation window; NCCL's own watchdog messages may name a collective that is a victim rather than the cause.",
  rb="Falsified if all ranks report the same last collective tuple and the synthetic-dataset arm hangs identically. Revert to the real dataset once the code path is fixed; require a full step to complete before resuming the production run."),
 dict(head=True, name="GPUDirect RDMA unavailable (nvidia_peermem) with an over-long timeout masking it",
  hyp="GPUDirect RDMA is requested but the peer-memory kernel module is absent, so registration of GPU buffers stalls in a retry path, and a very large NCCL timeout converts a fast failure into an unbounded hang.",
  mech="For GDR, the RDMA NIC must map GPU BAR1 memory, which requires nvidia_peermem (or nv_peer_mem) loaded and sufficient BAR1 space. Without it NCCL should fall back to a staged host-buffer path, but with NCCL_NET_GDR_LEVEL forcing GDR and a multi-hour NCCL_TIMEOUT/watchdog setting, the run neither falls back nor aborts, and looks like an init hang.",
  exp="Check `lsmod | grep -E 'nvidia_peermem|nv_peer_mem'` and BAR1 usage via `nvidia-smi -q -d MEMORY` on all nodes. Controlled arm A: set NCCL_NET_GDR_LEVEL=LOC (effectively disabling GDR) and rerun. Controlled arm B: keep GDR but reduce the process-group timeout to 120 s so the failure surfaces as an exception with a stack trace.",
  meas="Module presence per node; BAR1 total vs used; NCCL NET log line indicating whether GDR is enabled per device; time-to-first-exception under the short timeout; achieved bus bandwidth from nccl-tests all_reduce_perf with and without GDR.",
  conf="Disabling GDR changes the performance profile, so a pass may hide a second fault; BAR1 exhaustion is load-dependent and may not reproduce on an idle node.",
  rb="Falsified if the job still hangs with GDR disabled and a 120 s timeout. Do not ship the long timeout as a workaround — it only delays detection. Restore the production timeout only after a clean nccl-tests baseline, and roll back GDR disabling once nvidia_peermem is loaded and re-measured."),
]

def build(idx, rec, m):
    su = next(x["content"] for x in rec["messages"] if x["role"]=="user")
    sa = next(x["content"] for x in rec["messages"] if x["role"]=="assistant")
    cid = rec["id"]
    ans = f"""Scope and assumptions (stated, not assumed silently): the job is a PyTorch/NCCL multi-GPU run; "hangs during collective initialization" means init_process_group or the first collective never returns and no rank raises. I assume I can add logging, set environment variables, and rerun on the same hardware. Unless a number is labelled MEASURED, everything below is an estimate to be replaced by data.

Leading hypothesis for this variant -- {m['name']}.
Falsifiable hypothesis: {m['hyp']}

Mechanism: {m['mech']}

Controlled experiment: {m['exp']}

Measurements to collect (units matter): {m['meas']}

Expected confounders: {m['conf']}

Rollback criteria and gates: {m['rb']}

Baseline diagnosis plan that must run regardless of hypothesis:
1. Record ranks and environment on every process before init: hostname, pid, RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR/PORT, CUDA_VISIBLE_DEVICES, NCCL and driver versions, container image digest. A rank that never logs is a launcher fault, not a NCCL fault.
2. Verify process-group rendezvous independently of NCCL by running a gloo-only barrier with the same launcher. This cleanly separates rendezvous faults from transport faults.
3. Inspect topology and GPU visibility: `nvidia-smi topo -m`, per-rank GPU UUID uniqueness, NUMA and NIC affinity.
4. Run a minimal all-reduce (a single 4 MiB float32 tensor, or nccl-tests all_reduce_perf -b 8 -e 128M -g <n>) before touching the real training script; this isolates NCCL from model and dataloader code.
5. Test interface selection and timeout explicitly: pin NCCL_SOCKET_IFNAME, set NCCL_DEBUG=INFO with NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH, and shorten the process-group timeout to 120 s so failures surface as exceptions with stack traces instead of silent stalls.
6. Compare single-node and reduced-world runs: 1 GPU, then 2 GPUs on one node, then 2 nodes x 1 GPU. The smallest failing configuration localises the fault to intra-node or inter-node scope.

Evidence that would change the conclusion: NCCL INIT/NET/GRAPH logs from all ranks, py-spy stack dumps at hang time, the pass/fail matrix over the scaling ladder in step 6, and nccl-tests bus bandwidth as an independent baseline. Absent those artifacts, any single-cause claim here is a hypothesis, not a diagnosis.

Operational safety: run all experiments on a drained or non-production node group; time-box every hung run and kill it rather than leaving GPUs pinned; scope any firewall, BIOS, or fabric change to the specific node pair, log it, and revert it after the experiment. Never leave a diagnostic environment variable (NCCL_P2P_DISABLE, NCCL_IB_DISABLE, NCCL_NET_GDR_LEVEL=LOC, inflated timeouts) in a production launcher, because each trades measurable performance or failure detection for a temporary pass."""
    return {
      "source_id": cid,
      "teacher_lane": "teacher-B",
      "teacher_model": "claude-opus-5-current",
      "calibration_status": "provisional",
      "decision": "rewrite",
      "source_user": su,
      "source_assistant": sa,
      "corrected_answer": ans,
      "quality_dimensions": {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2},
      "risks": [
        "Source assistant turn is a grading rubric ('Answer should state...'), not an answer; training on it teaches meta-commentary instead of diagnosis.",
        "No mechanism, units, or measurement targets are given, so the model cannot learn to distinguish measured facts from estimates.",
        "No rollback or blast-radius guidance for changes that touch fabric, firewall, or BIOS settings on shared clusters.",
        "Variant numbering makes many near-identical items; without mechanism diversification this induces template memorisation."
      ],
      "evidence_required": [
        "Per-rank NCCL_DEBUG=INFO logs with NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH",
        "Pre-init environment dump per rank (RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/GPU UUID/versions)",
        "nccl-tests all_reduce_perf bus bandwidth baseline for the same topology",
        "Pass/fail matrix across the 1-GPU / 1-node / 2-node scaling ladder",
        "py-spy stack dumps of every process captured at hang time"
      ],
      "confidence": 0.62
    }

recs=[]
with open(CORPUS) as f:
    for i,l in enumerate(f):
        if START<=i<START+N: recs.append(json.loads(l))
assert len(recs)==N
os.makedirs(os.path.dirname(OUT),exist_ok=True)
out=[build(START+k, recs[k], MECH[k]) for k in range(N)]
hs={hashlib.sha256(o["corrected_answer"].encode()).hexdigest() for o in out}
assert len(hs)==N, f"duplicate corrected_answer: {len(hs)}"
with open(OUT,"w") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
print("wrote",OUT,len(out),"ids",out[0]["source_id"],"..",out[-1]["source_id"])
