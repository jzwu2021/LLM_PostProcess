import io
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
entry = """## Run: train-batch-0219.jsonl (provisional teacher-B BLIND review)

- Batch file: `results/train-batch-0219.jsonl`
- Corpus interval: positional slice [2180:2190) of `research/ai-infra-expert/corpus/train.jsonl`
- Source IDs: corpus-02403 .. corpus-02413 (10 items; corpus IDs are non-consecutive, slicing is positional)
- Progress: 2190 / 2500 train (stage target set by the user on 2026-08-18, down from the original 6000). Remaining: 310.
- Validation: 0 files, target 0. No validation batch was produced.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (`scripts/tb_verify_batch_0219.py`, derived from the 0218 verifier via sed offset rewrite)
- Repair actions: none required this run
- Final schema check: VERIFY_PASS, TOTAL 2190, aggregated sequence is a strict prefix of train.jsonl, no duplicate source_id, no validation-batch files
- Manifest: `MANIFEST.sha256` regenerated over every file in the experiment directory except itself; `sha256sum -c` all OK

Technical topics covered by this batch (Stances 190-199, all against the tensor-versus-pipeline-parallelism latency prompt family):
speculative decoding shifting decode into prefill-shaped verification passes and its effect on the layout gap; chunked prefill and scheduler admission policy as an unpinned confounder in prefill/decode interference; CUDA graph capture and host-side kernel launch overhead at small decode shapes; GPUDirect Storage / GPUDirect RDMA and the cold-start weight-load path with shard-dependent read patterns; NUMA, PCIe root-complex and NIC affinity determining whether nominal fabric bandwidth is reachable and whether GDR silently falls back to host staging; mixture-of-experts expert parallelism as a third axis with data-dependent all-to-all and token-dropping semantics; RoCE PFC/ECN congestion control as a co-determinant of cross-node tail latency; attention kernel dispatch changing with heads-per-rank so that a layout change becomes a kernel change; power/thermal throttling making synchronous-collective step time track the slowest device; and cost-per-served-token at the SLO-satisfying concurrency knee versus single-request latency.

Every quantitative claim in this batch is explicitly labelled ESTIMATE with its derivation. No MEASURED values are reported, because no benchmark was executed for this review.

Status caveat: these outputs are PROVISIONAL teacher-B review material produced under blind review (teacher-A artifacts were not read at any point). They are not expert gold, they have not been human-adjudicated, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate later step outside this task.

"""
s = io.open(P, encoding="utf-8").read()
lines = s.split("\n")
# insert after the leading title block: find first line starting with '## '
idx = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
new = "\n".join(lines[:idx]) + "\n" + entry + "\n".join(lines[idx:])
io.open(P, "w", encoding="utf-8").write(new)
print("MD_UPDATED", idx)
