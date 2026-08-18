import io
P="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(P).read()
head="# Experiment: teacher-B corpus review (blind, independent second opinion)\n"
assert s.startswith(head)
entry = """
## Run: train-batch-0224.jsonl (provisional teacher-B BLIND review)

- Batch file: `results/train-batch-0224.jsonl`
- Corpus interval: positional slice [2230:2240) of `research/ai-infra-expert/corpus/train.jsonl`
- Source IDs: corpus-02457 .. corpus-02467 (10 items; corpus IDs are non-consecutive here - corpus-02458 is absent - so slicing is positional and never ID arithmetic)
- Progress: 2240 / 2500 train (stage target set by the user on 2026-08-18, down from the original 6000). Remaining: 260.
- Validation: 0 files, target 0. No validation batch was produced.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (`scripts/tb_verify_batch_0224.py`, derived from the 0223 verifier via sed rewrite of the batch filename)
- Repair actions: none required this run
- Final schema check: VERIFY_PASS, TOTAL 2240, aggregated sequence is a strict prefix of train.jsonl, no duplicate source_id, no validation-batch files
- Stance uniqueness: Stances 240-249 each appear exactly once in this batch and collide with no stance number used in any earlier batch
- Manifest: `MANIFEST.sha256` regenerated over every file in the experiment directory except itself; `sha256sum -c` all OK

Technical topics covered by this batch (Stances 240-249, all against the tensor-versus-pipeline-parallelism latency prompt family):
Crossing the node boundary making the fabric the deciding variable, with RoCE/InfiniBand small-message latency floors capping viable TP degree and lossless-domain misconfiguration (PFC pause storms, ECN) masquerading as a bad layout; GPUDirect RDMA and GPUDirect Storage determining whether transfers bounce through host memory, an additive plumbing constant routinely misattributed to the parallel axis and frequently different between benchmark and production hosts; KV-cache capacity rather than per-step latency being the binding constraint, with preemption and prefill recomputation landing in the TPOT tail; cost per request at the SLO boundary as the real decision unit, where the pipeline-bubble and wide-collective waste modes have opposite load dependence so latency ordering and efficiency ordering can cross; numerical non-invariance across layouts, since changing TP degree repartitions every GEMM reduction and changes kernel selection, making a layout change a behaviour change until output parity is demonstrated; traffic drift and model succession making a trace-optimal layout fragile given the asymmetric cost of re-sharding later; multi-tenancy and noisy neighbours hitting synchronisation-heavy TP hardest in the tail because an all-reduce runs at the pace of its slowest rank; mixture-of-experts architectures replacing the framing with a routing-dependent, load-imbalanced all-to-all whose capacity factor trades quality against tail latency; prefix and KV reuse in real traffic collapsing prefill so that synthetic zero-reuse benchmarks measure a workload the service does not have, with hit rate itself layout-dependent through eviction; and closing on a decision record scoped to the intersection of all conditioning variables, with invariant bands, monitors, expiry and a rehearsed rollback.

Every quantitative claim in this batch is explicitly labelled ESTIMATE with its derivation. No MEASURED values are reported, because no benchmark was executed for this review.

Status caveat: these outputs are PROVISIONAL teacher-B review material produced under blind review (teacher-A artifacts were not read at any point). They are not expert gold, they have not been human-adjudicated, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate later step outside this task.
"""
open(P,"w").write(head + entry + s[len(head):])
print("MD_UPDATED")
