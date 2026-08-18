import io
p="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(p).read()
hdr="# Experiment: teacher-B corpus review (blind, independent second opinion)\n\n"
assert s.startswith(hdr)
new = """## Run: train-batch-0220.jsonl (provisional teacher-B BLIND review)

- Batch file: `results/train-batch-0220.jsonl`
- Corpus interval: positional slice [2190:2200) of `research/ai-infra-expert/corpus/train.jsonl`
- Source IDs: corpus-02414 .. corpus-02423 (10 items; slicing is positional, never ID arithmetic)
- Progress: 2200 / 2500 train (stage target set by the user on 2026-08-18, down from the original 6000). Remaining: 300.
- Validation: 0 files, target 0. No validation batch was produced.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (`scripts/tb_verify_batch_0220.py`, derived from the 0219 verifier via sed offset rewrite)
- Repair actions: none required this run
- Final schema check: VERIFY_PASS, TOTAL 2200, aggregated sequence is a strict prefix of train.jsonl, no duplicate source_id, no validation-batch files
- Manifest: `MANIFEST.sha256` regenerated over every file in the experiment directory except itself; `sha256sum -c` all OK

Technical topics covered by this batch (Stances 200-209, all against the tensor-versus-pipeline-parallelism latency prompt family):
NCCL algorithm/protocol selection (Ring vs Tree vs NVLS, Simple vs LL vs LL128) as an unpinned confounder in small-message decode collectives; paged KV block fragmentation and preemption onset differing between TP and PP shardings; continuous-batching queue wait dominating TTFT below saturation and being misattributed to the layout; inter-node RDMA on RoCE/IB where TP pays 2*L fabric round trips per token versus PP's bounded per-token sends, with PFC/ECN counters as validity gates; prefill/decode disaggregation with phase-specific sharding and KV-transfer bytes as the new critical path (Mooncake store / NVIDIA Dynamo routing); grouped-query attention KV head count capping usable TP degree and producing a memory-saving knee; power and thermal steady-state clock divergence between the two duty-cycle shapes; production-shaped request-length distributions reinflating pipeline bubbles in the tail; weight-only quantization moving the bandwidth term that TP was dividing while leaving collective precision unchanged; and single-GPU fault fate-sharing with mean-time-to-restore dominated by weight reload.

Every quantitative claim in this batch is explicitly labelled ESTIMATE with its derivation. No MEASURED values are reported, because no benchmark was executed for this review.

Status caveat: these outputs are PROVISIONAL teacher-B review material produced under blind review (teacher-A artifacts were not read at any point). They are not expert gold, they have not been human-adjudicated, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate later step outside this task.

"""
open(p,"w").write(hdr+new+s[len(hdr):])
print("MD_UPDATED")
