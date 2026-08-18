P="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(P).read()
HDR="# Experiment: teacher-B corpus review (blind, independent second opinion)\n\n"
assert s.startswith(HDR)
NEW = """## Run 0214 (train-batch-0214.jsonl)

- Batch file: results/train-batch-0214.jsonl
- Corpus range: positional rows 2131-2140 of research/ai-infra-expert/corpus/train.jsonl
- Source IDs: corpus-02347, corpus-02349, corpus-02350, corpus-02351, corpus-02352, corpus-02353, corpus-02354, corpus-02355, corpus-02356, corpus-02357 (slicing is positional, never ID arithmetic; note the corpus ID sequence is non-consecutive here - corpus-02348 does not exist in the file)
- Progress: 2140/2500 train (360 remaining). Validation target is 0; no validation-batch files exist or were created.
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: PASS on first run (scripts/tb_verify_batch_0214.py, derived from the 0213 verifier by sed substitution)
- Repairs performed: none to the batch output. The ad-hoc checker derived from 0213 carried the previous run's three hard-coded offsets; the expected total, the prefix slice and the comparison slice were corrected to 2140, corpus[:2140] and corpus[2130:2140] in scripts/adhoc_verify_batch_0214.py before it was run. No corpus file and no previously committed batch was modified.
- Final schema check: VERIFY_PASS, TOTAL 2140; independent ad-hoc check scripts/adhoc_verify_batch_0214.py returned ADHOC_PASS, confirming per-line JSONL parse, 10 records, all 12 required fields, fixed-value fields correct, byte-exact source_user/source_assistant against the corpus, non-empty corrected_answer, confidence in [0,1], globally unique source_id, and the aggregate sequence a strict prefix of train.jsonl
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself (__pycache__ excluded); sha256sum -c reported all files OK

Technical topics covered by this batch: all ten items are near-homogeneous tensor-versus-pipeline parallelism scenario variants (147-157), so each answer carries a mutually exclusive analytical stance (Stance 140-149) disjoint from every stance used in prior batches, layered on the shared assumption/mechanism/boundary-condition frame. Stances cover NCCL algorithm and protocol selection dominating the observed collective penalty at latency-bound decode payload sizes; KV-cache capacity relief rather than step latency being the real mechanism behind tensor-parallel wins under concurrency; pipeline parallelism being defensible only as a tool for placing the cut on the cheapest interconnect boundary; the necessity of comparing layouts at matched achieved throughput under a latency bound rather than at matched offered rate; grouped-query and multi-query head-group counts imposing a hard ceiling past which sharding degenerates into replication; reconfiguration and failure-recovery cost differing by weight-reload volume between the axes and belonging in the availability budget; communication/computation overlap converting the tensor-parallel penalty from additive to partially hidden and being an implementation rather than architectural property; sequence-length distribution inverting the layout ranking across quantiles that an aggregate mean conceals; allocator fragmentation and workspace reservation undercutting the arithmetic capacity claim; and treating the layout as a versioned, canaryable configuration with a declared rollback gate because the correct answer drifts with traffic.

Status caveat: this output is PROVISIONAL teacher-B review material produced blind, without any visibility into the teacher-A calibration lane. It is not expert gold, has not been human-adjudicated, and is not evidence about any model's domain capability. Every quantitative claim in this batch is labelled ESTIMATE with its derivation stated; no value is MEASURED, because no benchmark run was executed for this review.

"""
open(P,"w").write(HDR+NEW+s[len(HDR):])
print("MD_UPDATED")
