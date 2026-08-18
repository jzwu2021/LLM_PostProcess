import json

CORPUS = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0150.jsonl"
START, N = 1490, 10

ANSWERS = [
# 1490 corpus-01647 Troubleshooting v47
"""Assumptions: single job, PyTorch + NCCL backend, containerised launch, no confirmed hardware fault yet. All numbers below are to be measured, not assumed.

Falsifiable hypothesis H1: the hang is a rendezvous/interface-selection failure, not a GPU or fabric fault. H1 predicts (a) every rank reaches init_process_group and none returns, (b) NCCL prints the bootstrap banner but no ring/tree construction line, (c) a 2-rank single-node run on the same image succeeds. If any of (a)-(c) is false, H1 is rejected.

Controlled experiment: keep image, driver, and code fixed; vary only world size and NCCL_SOCKET_IFNAME. Run matrix: (1 node, 2 ranks), (1 node, 8 ranks), (2 nodes, 2 ranks/node), each with the fabric interface pinned explicitly and then unset. A hang that appears only when NCCL_SOCKET_IFNAME is unset and world spans nodes confirms H1.

Measurements to record before touching anything: MASTER_ADDR/MASTER_PORT, RANK/LOCAL_RANK/WORLD_SIZE per process, CUDA_VISIBLE_DEVICES per process, `nvidia-smi topo -m`, `ibstatus`/`ibdev2netdev` if RDMA, NCCL_DEBUG=INFO plus NCCL_DEBUG_SUBSYS=INIT,NET logs from rank 0 and one remote rank, and py-spy dump of one stuck process to confirm the stack sits in ncclCommInitRank.

Mechanism and boundary conditions: NCCL bootstrap uses a TCP out-of-band channel chosen by interface heuristics; a docker0/virbr0 style interface that is present but unroutable between nodes causes the store handshake to complete on some ranks and stall on others. This is why the failure is world-size dependent and why single-node runs are clean. Duplicate LOCAL_RANK to device mapping produces a different signature: two ranks on the same device, visible in nvidia-smi.

Confounders: NCCL_ASYNC_ERROR_HANDLING masking a real timeout; a slow first-touch CUDA context making a healthy job look hung for 60-120 s; MPI/torchrun rendezvous store contention; firewall dropping the ephemeral port range.

Evidence that would settle it: matching bootstrap logs from two ranks showing agreement or disagreement on the chosen net device, and a minimal 4 MiB all-reduce test binary that reproduces or clears the hang independent of the training code.

Rollback criteria: if pinning the interface does not clear the hang within one full matrix pass, revert all env changes, drop back to the last known-good world size, and escalate to fabric/driver inspection rather than continuing to mutate NCCL variables.""",

# 1491 corpus-01648 Performance Analysis v48
"""Assumptions: the job is reported as hung, but 'hang' has not yet been distinguished from 'very slow initialisation'. Treat elapsed time as the primary measurement.

Falsifiable hypothesis H1: initialisation is not deadlocked but is dominated by a serialised cost that scales with world size (per-rank CUDA context creation plus bootstrap fan-in). H1 predicts init wall time grows roughly linearly with rank count and that the job eventually completes if left beyond the NCCL timeout. If init time is flat across world sizes and never completes, H1 is rejected and a true deadlock is indicated.

Controlled experiment: instrument with timestamps around init_process_group and the first collective. Sweep world size 2, 4, 8, 16 with everything else fixed; record init latency in seconds for each. Then repeat the 16-rank run with NCCL_TIMEOUT raised well above the observed 8-rank init time. Linear or superlinear growth plus completion under the raised timeout confirms H1.

Quantities to capture with units: init_process_group latency (s), first all-reduce latency (ms), busbw for a 1 GiB all-reduce (GB/s), per-process RSS at init (GiB), and CPU utilisation during init (%). Also record whether ranks share a filesystem-based rendezvous, since a cold NFS store adds seconds per rank.

Mechanism: bootstrap is an out-of-band all-gather of NCCL unique IDs, followed by topology detection and channel setup. Topology detection touches PCIe/NVLink and, on RDMA fabrics, enumerates HCAs; enumeration on a node with many devices or a stale IB subnet manager response is a known multi-second cost per rank. Slowness therefore concentrates in a phase that produces no output, which is what makes it look like a hang.

Confounders: page-cache-cold container image, driver persistence mode off (adds seconds per GPU on first context), ECC scrubbing after a reset, and other tenants saturating the same host.

Evidence required: per-phase timestamps from at least two ranks, nvidia-smi persistence mode state, and a repeat run with a warm image to separate one-time from recurring cost.

Rollback criteria: if raised timeouts merely defer the stall and busbw is unmeasurable, stop treating this as a performance problem, restore the original timeout, and reclassify as a correctness/deadlock defect.""",

# 1492 corpus-01649 System Design v49
"""Assumptions: we are designing the startup path so that this class of hang is observable and bounded, not merely debugging one instance.

Design goal: no collective initialisation may block indefinitely, and every failure must emit enough state to localise the faulty rank within one run.

Falsifiable hypothesis H1: adding a mandatory pre-flight barrier with a hard, short deadline before the real process group is built will convert silent hangs into fast, attributable failures without changing steady-state throughput. H1 predicts (a) mean time-to-diagnosis drops, (b) steady-state step time is unchanged within noise, (c) no new false failures at the chosen deadline. Any measured step-time regression above noise rejects H1.

Mechanism: a small TCP-store based liveness check runs first. Each rank publishes hostname, LOCAL_RANK, visible device UUIDs, chosen network interface and address. Rank 0 asserts world completeness and uniqueness of device UUIDs, then releases the barrier. Only then is ncclCommInitRank invoked. Because the check uses the same out-of-band path NCCL will use, an unroutable interface fails here with a named rank rather than inside NCCL.

Boundary conditions: the pre-flight deadline must exceed worst-case cold-start context creation. Set it from measured p99 init latency times a safety factor, and re-derive it whenever node count, image, or driver changes. On very large worlds the rank-0 fan-in becomes the bottleneck; switch to a tree aggregation above a measured threshold.

Operational safety: the check must be read-only with respect to cluster state, must not reset GPUs, and must not silently rewrite NCCL env vars in production; it may only report mismatches. Emit structured logs (one JSON line per rank) to the job log so post-mortem needs no live process.

Confounders: heterogeneous images across nodes, clock skew affecting deadline accounting, and schedulers that recycle a partially released allocation.

Evidence required before rollout: A/B step-time comparison over at least three runs per arm, p99 pre-flight latency, and a fault-injection test that blackholes one node's interface and shows the correct rank is named.

Rollback: gate behind a flag, default off; disable immediately if false-failure rate exceeds the agreed threshold or if step time regresses beyond noise.""",

# 1493 corpus-01650 Troubleshooting v50
"""Assumptions: hang reproduces, cluster otherwise healthy, no recent driver change confirmed or denied yet. Establish provenance before mutation.

Falsifiable hypothesis H1: a subset of ranks never entered the collective, so the rest block waiting for them. H1 predicts stack traces divide into two disjoint classes: some ranks inside ncclCommInitRank or an all-reduce wait, and at least one rank elsewhere (data loading, checkpoint restore, or already exited). If every rank shows the same collective frame, H1 is rejected and the fault is symmetric, pointing at transport rather than control flow.

Controlled experiment: attach py-spy (or gdb) to every process and classify stacks. Then rerun with a reduced world that excludes any node hosting an outlier rank. If the reduced run succeeds and re-including that node reproduces the hang, the fault is localised to that node.

Measurements: full stack of each rank, process start and last-log timestamps per rank, exit codes of any dead rank, dmesg on suspect nodes for Xid errors, and NCCL_DEBUG=INFO logs correlated by timestamp.

Mechanism and boundary conditions: collectives are synchronising; one absent participant stalls all others with no error until the watchdog timeout fires. A rank can be absent because it crashed before joining, because it is still restoring a large checkpoint, or because its device was claimed by a stale process. The asymmetric-stack signature separates these from fabric faults, which stall all ranks symmetrically.

Confounders: a watchdog that kills and restarts one rank makes the population look healthy on inspection; log buffering hides the true last line; container PID namespaces make attaching from the host misleading.

Evidence required: per-rank stack classification, dmesg Xid presence or absence, and a reduced-world reproduction that is positive on the suspect node and negative without it.

Rollback criteria: do not drain or reboot nodes on a single observation. Require two independent reproductions naming the same node before removing it from the pool, and restore it after a clean minimal all-reduce plus one full-scale run.""",

# 1494 corpus-01651 Performance Analysis v51
"""Assumptions: we can afford instrumented reruns; the goal is to quantify where initialisation time goes and whether the stall has a measurable rate.

Falsifiable hypothesis H1: the process is not making progress at all, i.e. the stall is a true deadlock rather than a throughput collapse. H1 predicts zero bytes transferred on the fabric counters during the stall window and zero GPU SM utilisation, sustained over a window several times longer than any plausible retry interval. Non-zero, slowly increasing counters reject H1 and reclassify the problem as pathological slowness, most often fallback to a socket path instead of the intended RDMA path.

Controlled experiment: during the stall, sample for a fixed window (e.g. 120 s at 1 Hz) nvidia-smi utilisation and memory, host NIC counters, and RDMA port counters if present. Then rerun with the transport forced to socket and separately to the RDMA path, comparing a 1 GiB all-reduce busbw in GB/s. A large gap plus counters that were creeping during the original stall confirms the fallback story.

Quantities with units: bytes/s per interface, packets/s, SM utilisation (%), init phase durations (s), and busbw (GB/s) at sizes 8 MiB, 128 MiB, 1 GiB.

Mechanism: when the intended transport is unusable, NCCL may select a slower path or retry discovery; either produces long, output-free periods. Distinguishing 'no progress' from 'slow progress' is the whole diagnostic, and it requires counters rather than logs.

Confounders: counter aggregation intervals coarser than the sampling window; shared NICs carrying other tenants' traffic; a monitoring agent itself generating the traffic you attribute to the job.

Evidence required: raw counter time series from at least two nodes, the transport actually selected as reported in NCCL init logs, and a baseline busbw measured on a known-good pair of nodes for comparison.

Rollback criteria: forcing a transport is a diagnostic action only. Revert to the default selection once the measurement is taken; do not ship a pinned transport into production without a documented busbw comparison and an owner.""",

# 1495 corpus-01653 Troubleshooting v53
"""Assumptions: intermittent hang, meaning some runs on the same allocation succeed. Intermittency is the strongest available clue and must drive the plan.

Falsifiable hypothesis H1: the hang is a race in rendezvous, triggered when ranks arrive outside a narrow window, not a static misconfiguration. H1 predicts (a) failure rate correlates with rank start-time spread, (b) artificially staggering rank starts changes the failure rate, (c) configuration is byte-identical between failing and passing runs. If configuration differs between runs, H1 is rejected in favour of a config-drift explanation.

Controlled experiment: run the same job 20 times, recording per-rank start timestamps and outcome. Then run 20 more with an injected stagger on half the ranks. Compare failure rates. A statistically distinguishable difference supports H1; identical rates across arms undermine it.

Measurements: rank start-time spread (ms), failure rate per arm with counts, rendezvous backend in use (TCP store vs file vs etcd), store server host load, and MASTER_PORT reuse across concurrent jobs on the same host.

Mechanism and boundary conditions: a shared MASTER_PORT or a stale file-based rendezvous artefact lets a new job's rank connect to a previous job's store. That mismatch is silent and world-size dependent, and it appears intermittent precisely because it needs overlapping lifetimes. Boundary: it cannot occur when the port is unique per job and the rendezvous path is per-job and cleaned on exit.

Confounders: scheduler retry logic hiding failures; log rotation losing the earliest evidence; nodes with different NTP state making timestamp correlation unreliable.

Evidence required: paired logs from a failing and a passing run with identical config hashes, the config hash itself, and store-server connection records showing which client connected.

Rollback criteria: any mitigation (unique ports, per-job rendezvous directories, cleanup hooks) must be validated by at least 20 consecutive clean runs before it is called a fix; if failures recur, revert and treat the earlier result as unexplained rather than fixed.""",

# 1496 corpus-01654 Performance Analysis v54
"""Assumptions: we want a budget for initialisation so that 'hung' is defined numerically rather than by operator patience.

Falsifiable hypothesis H1: a defensible init budget can be derived from measured components, and the observed stall exceeds that budget by a margin large enough to call it a fault. H1 predicts the sum of measured phase costs (CUDA context creation, topology detection, bootstrap all-gather, channel setup) accounts for well over 90% of healthy init time, leaving little unexplained. If a large fraction is unexplained even in healthy runs, the budget is not yet trustworthy and H1 is rejected.

Controlled experiment: on a known-good allocation, time each phase separately across world sizes 2, 4, 8, 16, five repetitions each. Report median and p95 in seconds per phase. Derive the budget as p95 of total times a stated safety factor. Then compare the failing run against it.

Quantities with units: per-phase latency (s), total init (s), variance across repetitions (s), and the resulting timeout setting (s). State explicitly that these are measured on this cluster and do not transfer to other hardware.

Mechanism: each phase has a different scaling law. Context creation is per-GPU and roughly constant; topology detection scales with devices per node; bootstrap all-gather scales with world size and with out-of-band link latency. Knowing which phase dominates tells you whether adding nodes or adding GPUs per node is the risk.

Confounders: first-run cache effects, persistence mode, concurrent jobs on the same nodes, and thermal or power state affecting nothing here but often blamed.

Evidence required: the raw per-phase timing table, the number of repetitions, and the exact software versions, since phase costs change across NCCL and driver releases and the budget must be re-derived after upgrades.

Rollback criteria: if the derived timeout causes any healthy run to be killed, the budget is wrong; restore the previous value immediately and re-measure rather than incrementally nudging the number.""",

# 1497 corpus-01655 System Design v55
"""Assumptions: designing the operational response, not only the technical fix. Multi-tenant cluster, jobs restart automatically.

Design goal: bound the blast radius of an init hang so that wasted GPU-hours are capped and the responsible component is identified without human inspection.

Falsifiable hypothesis H1: an automatic watchdog that kills a job whose init exceeds the measured budget, captures per-rank stacks, and quarantines the implicated node will reduce wasted GPU-hours without increasing job failure rate for healthy work. H1 predicts (a) wasted GPU-hours per week fall, (b) the healthy-job kill rate stays at or near zero, (c) quarantine decisions are reproducible on re-test. A rise in healthy-job kills rejects H1.

Mechanism: the launcher records init start per rank. A supervisor polls; on budget breach it triggers stack capture on all ranks, writes a structured incident record (job id, ranks, hostnames, chosen interfaces, stacks, dmesg tail), then terminates. A node is quarantined only if it appears in two independent incidents within a window, and quarantine is released only after an automated minimal all-reduce plus one full-scale job pass.

Boundary conditions: the budget must come from measurement on the same hardware and software versions, and must be invalidated on driver, NCCL, or image change. The two-incident rule prevents a single transient from draining capacity; the release gate prevents permanent capacity loss from a stale quarantine.

Operational safety: never auto-reboot or auto-reset GPUs; quarantine is scheduling-level only and reversible. Rate-limit kills so a cluster-wide fabric event cannot cascade into mass termination. Require a dry-run mode that logs the decision it would have made, run for at least a week before enforcement.

Evidence required: baseline wasted GPU-hours, dry-run decision log with false-positive count, and quarantine release test results.

Rollback: single feature flag returns the system to observe-only; any week where healthy-job kills exceed the agreed threshold triggers automatic reversion to dry-run.""",

# 1498 corpus-01656 Troubleshooting v56
"""Assumptions: the hang appeared after a change. Change correlation is the cheapest evidence available and should be exhausted before deep debugging.

Falsifiable hypothesis H1: the hang was introduced by a specific recent change (driver, NCCL version, container image, kernel, fabric firmware, or scheduler config). H1 predicts that pinning the stack to the last known-good versions on the same nodes clears the hang, and that reintroducing one component at a time reproduces it. If the known-good stack also hangs, H1 is rejected and the cause is environmental or load-dependent rather than a code/version change.

Controlled experiment: bisect over components, not over commits alone. Arm A: known-good image on current driver. Arm B: current image on known-good driver. Arm C: both known-good. Three runs per arm on the same node set. The arm that clears the hang names the component.

Measurements: exact versions of driver, CUDA runtime, NCCL, torch, kernel, and OFED/firmware on every node; image digests, not tags; and the scheduler's node allocation for each run so node identity is held constant.

Mechanism and boundary conditions: version skew across nodes is a frequent cause and is invisible if you check only one node. NCCL requires compatible transport support on both ends; a partial rollout leaves some pairs incompatible, which shows up only when the world spans the boundary. That predicts failure depends on which nodes are allocated, a directly testable claim.

Confounders: image tags that moved, nodes silently rebooted into a new driver, and caching layers serving a different image digest per node.

Evidence required: a version matrix across all allocated nodes with digests, and the three-arm result table.

Rollback criteria: if arm C (fully known-good) is clean, roll the cluster back to that stack as the immediate mitigation, freeze upgrades, and require the vendor-facing bug report to reference the version matrix. If arm C still hangs, revert the rollback to avoid carrying an unnecessary downgrade and reopen the investigation.""",

# 1499 corpus-01657 Performance Analysis v57
"""Assumptions: we need to decide whether the fabric itself is capable of the intended collective performance, independent of the training job.

Falsifiable hypothesis H1: the fabric and transport are healthy, and the hang is above the transport layer. H1 predicts a standalone collective benchmark (all_reduce_perf or equivalent) at the same world size and node set completes and reaches busbw within a stated fraction of the previously measured baseline. If the standalone benchmark also hangs or reaches only a small fraction of baseline busbw, H1 is rejected and the fault is at or below the transport.

Controlled experiment: run the benchmark on exactly the allocated nodes, sweeping message sizes from 8 MiB to 1 GiB, first intra-node, then pairwise inter-node, then full world. Escalating scope isolates whether the failure needs a specific pair or the full topology.

Quantities with units: algbw and busbw (GB/s) per size, latency (us) at small sizes, and the ratio to the recorded baseline (%). Record the baseline's date and software versions; a baseline from a different NCCL version is not comparable.

Mechanism: separating the benchmark from the training job removes checkpoint I/O, data loading, and framework-level synchronisation from the picture. If the benchmark is clean at full world size, transport-level explanations lose most of their support and attention should move to the framework's process-group construction order or to uneven rank arrival.

Confounders: the benchmark using different env defaults than the job; CPU affinity differences changing measured latency; other tenants sharing the fabric during the measurement window.

Evidence required: benchmark output for all three scopes, the env diff between benchmark and job, and a contemporaneous baseline from a known-good allocation.

Rollback criteria: if pairwise tests implicate specific links, do not reconfigure routing unilaterally; hand the pair list and counters to the fabric owner. Any temporary topology or env override applied for measurement must be removed before the next production run, and the run after removal must be verified clean.""",
]

QD = [
    (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2),
    (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2),
]
RISKS = [
    ["source answer is a rubric checklist, not an answer; training on it teaches meta-description",
     "no explicit falsifiable hypothesis despite the prompt asking for one",
     "mutating NCCL env vars without a revert path can leave a pinned config in production"],
    ["source lists measurement categories without units or expected magnitudes",
     "raising NCCL timeout can mask a real deadlock and waste GPU-hours",
     "no criterion separating slow init from true hang"],
    ["source gives no design, only rubric language",
     "pre-flight checks that rewrite env vars silently create configuration drift",
     "deadline set without measurement causes false failures on cold start"],
    ["source omits per-rank stack classification, the key discriminator",
     "draining or rebooting nodes on a single observation removes capacity without evidence",
     "watchdog restarts can hide the failing rank"],
    ["source has no counter-based progress test, so no-progress vs slow-progress is unresolved",
     "forcing a transport as a diagnostic can be shipped to production by accident",
     "baseline comparison across differing NCCL versions is invalid"],
    ["source ignores intermittency, which is the dominant clue here",
     "shared MASTER_PORT across concurrent jobs is a silent cross-talk hazard",
     "declaring a fix after one clean run is not statistically supported"],
    ["source provides no numeric budget, so 'hung' stays subjective",
     "a timeout derived from another cluster will kill healthy runs",
     "phase costs shift across driver/NCCL upgrades and stale budgets misfire"],
    ["source has no automation or blast-radius control",
     "auto-kill without rate limiting can cascade during a cluster-wide fabric event",
     "auto-reboot or GPU reset as remediation is unsafe and not reversible"],
    ["source omits change/version correlation, the cheapest first check",
     "version skew across nodes is invisible when only one node is inspected",
     "carrying an unnecessary rollback forward costs performance and support"],
    ["source does not separate fabric capability from framework behaviour",
     "temporary routing or env overrides left in place after measurement",
     "comparing busbw against a baseline from a different software stack"],
]
EVID = [
    ["NCCL_DEBUG=INFO INIT,NET logs from >=2 ranks", "nvidia-smi topo -m output", "per-rank env dump (RANK/LOCAL_RANK/WORLD_SIZE/CUDA_VISIBLE_DEVICES)", "result of minimal 4 MiB all-reduce test"],
    ["timestamped init_process_group latency (s) per world size", "busbw (GB/s) at 1 GiB", "nvidia-smi persistence mode state", "warm vs cold image repeat run"],
    ["A/B step-time over >=3 runs per arm", "p99 pre-flight latency (s)", "fault-injection test naming the correct rank", "structured per-rank JSON init log"],
    ["py-spy/gdb stacks for every rank", "dmesg Xid presence or absence per suspect node", "reduced-world reproduction result", "per-rank last-log timestamps"],
    ["NIC and RDMA port counter time series (bytes/s) during stall", "transport selected per NCCL init log", "busbw (GB/s) at 8 MiB/128 MiB/1 GiB", "known-good baseline from another node pair"],
    ["per-rank start-time spread (ms) and outcome for 20+ runs", "config hash for failing and passing runs", "rendezvous backend and MASTER_PORT uniqueness", "store-server connection records"],
    ["per-phase init timing table with median and p95 (s)", "repetition count and variance", "exact driver/NCCL/torch versions", "healthy-run kill rate after budget enforcement"],
    ["baseline wasted GPU-hours per week", "dry-run decision log with false-positive count", "quarantine release test results", "kill rate-limit configuration"],
    ["version matrix (driver, CUDA, NCCL, torch, kernel, OFED) across all allocated nodes", "image digests not tags", "three-arm bisect result table", "node allocation identity per run"],
    ["all_reduce_perf output intra-node, pairwise, full world", "env diff between benchmark and job", "contemporaneous baseline with matching versions", "per-pair busbw table"],
]
CONF = [0.72, 0.7, 0.68, 0.72, 0.69, 0.7, 0.68, 0.66, 0.71, 0.7]

rows = []
with open(CORPUS) as f:
    for i, line in enumerate(f):
        if i < START:
            continue
        if i >= START + N:
            break
        o = json.loads(line)
        msgs = {m["role"]: m["content"] for m in o["messages"]}
        k = i - START
        tc, ic, os_ = QD[k]
        rows.append({
            "source_id": o["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": msgs["user"],
            "source_assistant": msgs["assistant"],
            "corrected_answer": ANSWERS[k],
            "quality_dimensions": {
                "technical_correctness": tc,
                "instruction_coverage": ic,
                "operational_safety": os_,
            },
            "risks": RISKS[k],
            "evidence_required": EVID[k],
            "confidence": CONF[k],
        })

assert len(rows) == N, len(rows)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(rows))
print("ids", rows[0]["source_id"], "->", rows[-1]["source_id"])
