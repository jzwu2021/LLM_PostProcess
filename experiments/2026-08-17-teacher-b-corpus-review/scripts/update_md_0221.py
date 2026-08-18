import io
p="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(p).read()
hdr="# Experiment: teacher-B corpus review (blind, independent second opinion)\n\n"
assert s.startswith(hdr)
new = """## Run: train-batch-0221.jsonl (provisional teacher-B BLIND review)

- Batch file: `results/train-batch-0221.jsonl`
- Corpus interval: positional slice [2200:2210) of `research/ai-infra-expert/corpus/train.jsonl`
- Source IDs: corpus-02424 .. corpus-02433 (10 items; slicing is positional, never ID arithmetic)
- Progress: 2210 / 2500 train (stage target set by the user on 2026-08-18, down from the original 6000). Remaining: 290.
- Validation: 0 files, target 0. No validation batch was produced.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (`scripts/tb_verify_batch_0221.py`, derived from the 0220 verifier via sed rewrite of the batch filename)
- Repair actions: none required this run
- Final schema check: VERIFY_PASS, TOTAL 2210, aggregated sequence is a strict prefix of train.jsonl, no duplicate source_id, no validation-batch files
- Stance uniqueness: Stances 210-219 each appear exactly once across the full results history (checked in-batch and against history)
- Manifest: `MANIFEST.sha256` regenerated over every file in the experiment directory except itself; `sha256sum -c` all OK

Technical topics covered by this batch (Stances 210-219, all against the tensor-versus-pipeline-parallelism latency prompt family):
CPU-side kernel launch overhead and CUDA graph capture dominating small-batch decode and asymmetrically favouring one layout; grouped-query attention KV head divisibility constraining legal TP degrees, with silent KV-head replication faking a 1/TP memory saving; chunked prefill and prompt-to-generation ratio blending two opposite regimes into one misleading latency average; thermal and power capping aliasing onto run order so sequential A/B runs bias the second arm, requiring counterbalancing and clock telemetry; speculative decoding changing collective payload shape and draft-model placement, requiring per-accepted-token metrics and acceptance-rate parity; quantization schemes differing in whether they shrink activation-sized collective payloads at all, gated on output quality parity; continuous-batching admission policy putting the two arms at different achieved operating points so the gap is scheduler-mediated; cross-node placement turning TP collectives into RoCE/IB fabric traffic with GPUDirect RDMA and PCIe affinity as first-order terms; prefill/decode disaggregation in Mooncake- and NVIDIA Dynamo-style architectures moving the cost into KV transfer whose crossover is a prompt-length sweep; and reframing the whole decision as GPU-seconds per served token subject to an SLO rather than a bare latency number.

Every quantitative claim in this batch is explicitly labelled ESTIMATE with its derivation. No MEASURED values are reported, because no benchmark was executed for this review.

Status caveat: these outputs are PROVISIONAL teacher-B review material produced under blind review (teacher-A artifacts were not read at any point). They are not expert gold, they have not been human-adjudicated, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate later step outside this task.

"""
open(p,"w").write(hdr+new+s[len(hdr):])
print("MD_UPDATED")
