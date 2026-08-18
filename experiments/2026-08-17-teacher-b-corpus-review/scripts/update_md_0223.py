import io
p="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(p).read()
hdr="# Experiment: teacher-B corpus review (blind, independent second opinion)\n\n"
assert s.startswith(hdr)
new = """## Run: train-batch-0223.jsonl (provisional teacher-B BLIND review)

- Batch file: `results/train-batch-0223.jsonl`
- Corpus interval: positional slice [2220:2230) of `research/ai-infra-expert/corpus/train.jsonl`
- Source IDs: corpus-02447 .. corpus-02456 (10 items; slicing is positional, never ID arithmetic)
- Progress: 2230 / 2500 train (stage target set by the user on 2026-08-18, down from the original 6000). Remaining: 270.
- Validation: 0 files, target 0. No validation batch was produced.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (`scripts/tb_verify_batch_0223.py`, derived from the 0222 verifier via sed rewrite of the batch filename)
- Repair actions: none required this run
- Final schema check: VERIFY_PASS, TOTAL 2230, aggregated sequence is a strict prefix of train.jsonl, no duplicate source_id, no validation-batch files
- Stance uniqueness: Stances 230-239 each appear exactly once in this batch and collide with no stance number used in any earlier batch
- Manifest: `MANIFEST.sha256` regenerated over every file in the experiment directory except itself; `sha256sum -c` all OK

Technical topics covered by this batch (Stances 230-239, all against the tensor-versus-pipeline-parallelism latency prompt family):
Mooncake-style prefill/decode disaggregation dissolving the single-layout question by letting each phase pick its own sharding while introducing a KV-transfer term that scales with prompt length and lands directly in TTFT; NCCL algorithm and protocol autotuning being threshold-based on message size and rank count, so a TP-degree sweep can compare implementations rather than a mechanism and produce a non-monotone curve; chassis power and thermal capping coupling GPUs so a wider TP group clocks down under sustained load and short benchmark runs measure pre-throttle behaviour; continuous batching changing batch composition every step so per-request TPOT depends on neighbours and PP's bubble-driven variance is hidden by aggregate percentiles; NVIDIA Dynamo-style runtime reconfiguration turning the layout into a policy whose switching cost, in-flight request handling and oscillation risk a static A/B cannot measure; failure blast radius and recovery time differing between the axes so a latency-only comparison buys a worse bad case; speculative decoding amortising TP's per-step collective cost across accepted tokens and thereby reversing rankings derived without it; weight quantisation shrinking the weight-reading term but not the activation-sized collective term, moving the optimal TP degree downward; benchmark client placement and streaming transport overhead sitting on the same wall-clock path and manufacturing differences invisible in server-side metrics; and closing on a decision record scoped to the intersection of all conditioning variables, with invariant bands, monitors, expiry and a rehearsed rollback.

Every quantitative claim in this batch is explicitly labelled ESTIMATE with its derivation. No MEASURED values are reported, because no benchmark was executed for this review.

Status caveat: these outputs are PROVISIONAL teacher-B review material produced under blind review (teacher-A artifacts were not read at any point). They are not expert gold, they have not been human-adjudicated, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate later step outside this task.

"""
open(p,"w").write(hdr+new+s[len(hdr):])
print("MD_UPDATED")
