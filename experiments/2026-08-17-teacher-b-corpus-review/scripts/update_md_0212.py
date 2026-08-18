BASE = '/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md'
s = open(BASE).read()
hdr = "# Experiment: teacher-B corpus review (blind, independent second opinion)\n\n"
entry = """## Run 0212 (train-batch-0212.jsonl)

- Batch file: results/train-batch-0212.jsonl
- Corpus range: positional rows 2111-2120 of research/ai-infra-expert/corpus/train.jsonl
- Source IDs: corpus-02324, corpus-02325, corpus-02327, corpus-02328, corpus-02329, corpus-02331, corpus-02332, corpus-02333, corpus-02334, corpus-02335 (slicing is positional, never ID arithmetic; the corpus IDs are non-consecutive here)
- Progress: 2120/2500 train (380 remaining). Validation target is 0; no validation-batch files exist or were created.
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: PASS on first run (scripts/tb_verify_batch_0212.py, derived from the 0211 verifier by sed substitution)
- Repairs performed: none to the batch output. The ad-hoc checker derived from 0211 carried three hard-coded offsets from the previous run (expected total 2110, prefix slice corpus[:2110], comparison slice corpus[2100:2110]); these were corrected in scripts/adhoc_verify_batch_0212.py to 2120, corpus[:2120] and corpus[2110:2120] respectively. No corpus file and no previously committed batch was modified.
- Final schema check: VERIFY_PASS, TOTAL 2120; independent ad-hoc check scripts/adhoc_verify_batch_0212.py returned ADHOC_PASS, confirming per-line JSONL parse, 10 records, all 12 required fields, fixed-value fields correct, byte-exact source_user/source_assistant against the corpus, non-empty corrected_answer, confidence in [0,1], globally unique source_id, and the aggregate sequence a strict prefix of train.jsonl
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself (__pycache__ excluded); sha256sum -c reported all 472 files OK

Technical topics covered by this batch: all ten items are near-homogeneous tensor-versus-pipeline parallelism scenario variants (124-135), so each answer carries a mutually exclusive analytical stance (Stance 120-129) disjoint from every stance used in prior batches, layered on the shared assumption/mechanism/boundary-condition frame. Stances cover host-side NUMA and PCIe root-complex placement as a hidden third topology; weight-loading, shard-format conversion and cold-restart cost as an availability term; proving output-contract and numerical equivalence across arms before comparing latency; change-management, approval and rollback-rehearsal gates that bind harder than performance; LoRA-adapter and multi-model tenancy reshaping the memory argument that motivated sharding; constrained/structured decoding adding a shard-independent per-step term that compresses the relative gap; client retry, timeout and hedging policy censoring the measured tail; analytic-model-first measurement with pre-registered predictions and sensitivity ranking; reporting the decision as a crossover surface over swept parameters rather than a scalar verdict; and mixture-of-experts routing imbalance making expert parallelism a third axis that invalidates any dense-model verdict.

Status caveat: this output is PROVISIONAL teacher-B review material produced blind, without any visibility into the teacher-A calibration lane. It is not expert gold, has not been human-adjudicated, and is not evidence about any model's domain capability. Every quantitative claim in this batch is labelled ESTIMATE with its derivation stated; no value is MEASURED, because no benchmark run was executed for this review.

"""
assert s.startswith(hdr)
open(BASE, 'w').write(hdr + entry + s[len(hdr):])
print('OK')
