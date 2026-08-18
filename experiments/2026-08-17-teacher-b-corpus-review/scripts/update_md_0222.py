import io
p="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(p).read()
hdr="# Experiment: teacher-B corpus review (blind, independent second opinion)\n\n"
assert s.startswith(hdr)
new = """## Run: train-batch-0222.jsonl (provisional teacher-B BLIND review)

- Batch file: `results/train-batch-0222.jsonl`
- Corpus interval: positional slice [2210:2220) of `research/ai-infra-expert/corpus/train.jsonl`
- Source IDs: corpus-02434 .. corpus-02446 (10 items; corpus IDs are non-consecutive here, slicing is positional and never ID arithmetic)
- Progress: 2220 / 2500 train (stage target set by the user on 2026-08-18, down from the original 6000). Remaining: 280.
- Validation: 0 files, target 0. No validation batch was produced.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (`scripts/tb_verify_batch_0222.py`, derived from the 0221 verifier via sed rewrite of the batch filename)
- Repair actions: none required this run
- Final schema check: VERIFY_PASS, TOTAL 2220, aggregated sequence is a strict prefix of train.jsonl, no duplicate source_id, no validation-batch files
- Stance uniqueness: Stances 220-229 each appear exactly once in this batch and collide with no stance number used in any earlier batch
- Manifest: `MANIFEST.sha256` regenerated over every file in the experiment directory except itself; `sha256sum -c` all OK

Technical topics covered by this batch (Stances 220-229, all against the tensor-versus-pipeline-parallelism latency prompt family):
Prefix caching and cross-request KV reuse shifting the prefill/decode mix so cold-cache benchmarks measure a system that never runs in production; intra-node topology non-uniformity across PCIe switches and NUMA boundaries making rank-to-device placement, not the parallel axis, the term the all-reduce is actually measuring; long-context requests moving the bottleneck from the weight-reading term to attention over the KV cache, which TP shards along the head dimension and PP does not, producing a context-length crossover; mixture-of-experts routing adding a data-dependent dispatch/combine all-to-all so expert parallelism is a third axis and routing imbalance dominates the layout gap; GPUDirect Storage and the checkpoint load path setting replica cold-start time, which becomes a latency term whenever capacity is elastic; host-side sampling and detokenisation forming a layout-independent floor that flattens device-time differences at small batch; RoCE PFC and ECN feedback loops making cross-node collective latency load-dependent so quiet-fabric numbers are optimistic bounds; non-associative floating-point accumulation and shard-shape-dependent kernel autotuning breaking numerical parity, requiring divergence bounding before latency is compared at all; horizontal replication as the competing lever that attacks queueing delay rather than service time and must be compared at equal GPU cost; and closing on a scoped decision record with invariant bands, monitors, expiry and a pre-tested rollback path.

Every quantitative claim in this batch is explicitly labelled ESTIMATE with its derivation. No MEASURED values are reported, because no benchmark was executed for this review.

Status caveat: these outputs are PROVISIONAL teacher-B review material produced under blind review (teacher-A artifacts were not read at any point). They are not expert gold, they have not been human-adjudicated, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate later step outside this task.

"""
open(p,"w").write(hdr+new+s[len(hdr):])
print("MD_UPDATED")
