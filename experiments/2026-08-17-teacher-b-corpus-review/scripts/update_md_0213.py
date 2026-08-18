import io
P="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(P).read()
HDR="# Experiment: teacher-B corpus review (blind, independent second opinion)\n\n"
assert s.startswith(HDR)
NEW = """## Run 0213 (train-batch-0213.jsonl)

- Batch file: results/train-batch-0213.jsonl
- Corpus range: positional rows 2121-2130 of research/ai-infra-expert/corpus/train.jsonl
- Source IDs: corpus-02337, corpus-02338, corpus-02339, corpus-02340, corpus-02341, corpus-02342, corpus-02343, corpus-02344, corpus-02345, corpus-02346 (slicing is positional, never ID arithmetic)
- Progress: 2130/2500 train (370 remaining). Validation target is 0; no validation-batch files exist or were created.
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: PASS on first run (scripts/tb_verify_batch_0213.py, derived from the 0212 verifier by sed substitution)
- Repairs performed: none to the batch output. The ad-hoc checker derived from 0212 carried the previous run's hard-coded offsets; the expected total, the prefix slice and the comparison slice were corrected to 2130, corpus[:2130] and corpus[2120:2130] in scripts/adhoc_verify_batch_0213.py before it was run. No corpus file and no previously committed batch was modified.
- Final schema check: VERIFY_PASS, TOTAL 2130; independent ad-hoc check scripts/adhoc_verify_batch_0213.py returned ADHOC_PASS, confirming per-line JSONL parse, 10 records, all 12 required fields, fixed-value fields correct, byte-exact source_user/source_assistant against the corpus, non-empty corrected_answer, confidence in [0,1], globally unique source_id, and the aggregate sequence a strict prefix of train.jsonl
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself (__pycache__ excluded); sha256sum -c reported all files OK

Technical topics covered by this batch: all ten items are near-homogeneous tensor-versus-pipeline parallelism scenario variants (137-146), so each answer carries a mutually exclusive analytical stance (Stance 130-139) disjoint from every stance used in prior batches, layered on the shared assumption/mechanism/boundary-condition frame. Stances cover speculative decoding raising arithmetic intensity and amortising tensor-parallel collectives over accepted tokens; weight-only quantisation shifting the bottleneck from HBM bandwidth toward invariant activation all-reduces; prefill and decode preferring opposite tensor degrees and the disaggregation trade that follows; continuous batching coupling layout to scheduler state so fixed-batch benchmarks measure an unoccupied regime; KV-cache capacity setting the preemption threshold that actually breaks the tail SLO; failure blast radius and NCCL communicator rebuild time as an availability term that grows with collective-domain size; multi-node crossings floored by RoCE/InfiniBand small-message latency with GPUDirect RDMA enablement required as evidence rather than intent; chunked prefill mixing phase composition and flattering tensor parallelism relative to a pure-decode benchmark; pre-registered SLO, minimum effect size and rejection rule as a precondition for the comparison concluding anything; and engine/collective-library/driver version pinning, since the arm gap is a difference of similar quantities and is version-fragile.

Status caveat: this output is PROVISIONAL teacher-B review material produced blind, without any visibility into the teacher-A calibration lane. It is not expert gold, has not been human-adjudicated, and is not evidence about any model's domain capability. Every quantitative claim in this batch is labelled ESTIMATE with its derivation stated; no value is MEASURED, because no benchmark run was executed for this review.

"""
open(P,"w").write(HDR+NEW+s[len(HDR):])
print("MD_UPDATED")
