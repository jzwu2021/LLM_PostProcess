import io
BASE = '/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md'
s = open(BASE).read()
hdr = "# Experiment: teacher-B corpus review (blind, independent second opinion)\n\n"
entry = """## Run 0211 (train-batch-0211.jsonl)

- Batch file: results/train-batch-0211.jsonl
- Corpus range: positional rows 2101-2110 of research/ai-infra-expert/corpus/train.jsonl
- Source IDs: corpus-02314, corpus-02315, corpus-02316, corpus-02317, corpus-02318, corpus-02319, corpus-02320, corpus-02321, corpus-02322, corpus-02323 (slicing is positional, never ID arithmetic)
- Progress: 2110/2500 train (390 remaining). Validation target is 0; no validation-batch files exist or were created.
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: PASS on first run (scripts/tb_verify_batch_0211.py, derived from the 0210 verifier by sed substitution)
- Repairs performed: none required
- Final schema check: VERIFY_PASS, TOTAL 2110; independent ad-hoc check scripts/adhoc_verify_batch_0211.py returned ADHOC_PASS, confirming per-line JSONL parse, 10 records, all 12 required fields, fixed-value fields correct, byte-exact source_user/source_assistant against the corpus, non-empty corrected_answer, confidence in [0,1], globally unique source_id, and the aggregate sequence a strict prefix of train.jsonl
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself (__pycache__ excluded); sha256sum -c reported all 466 files OK

Technical topics covered by this batch: all ten items are near-homogeneous tensor-versus-pipeline parallelism scenario variants (114-123), so each answer carries a mutually exclusive analytical stance (Stance 110-119) disjoint from every stance used in prior batches, layered on the shared assumption/mechanism/boundary-condition frame. Stances cover continuous-batching policy as the dominant confounder amortising per-step collectives; the opposite parallelism preferences of prefill versus decode and the danger of pooled metrics; prefill/decode disaggregation and KV-transfer cost on the TTFT critical path; prefix-cache hit rate silently redefining the measured baseline; quantization held fixed because numeric format moves both memory pressure and kernel path; multi-tenancy and interconnect contention degrading the higher-collective-count arm asymmetrically; statistical discipline where the layout gap is comparable to within-arm variance; devices-per-sustained-token as the true decision variable over raw latency; observability and instrumentation overhead symmetry as a precondition; and a closing epistemic-status stance.

Status caveat: this output is PROVISIONAL teacher-B review material produced blind, without any visibility into the teacher-A calibration lane. It is not expert gold, has not been human-adjudicated, and is not evidence about any model's domain capability. Every quantitative claim in this batch is labelled ESTIMATE with its derivation stated; no value is MEASURED, because no benchmark run was executed for this review.

"""
assert s.startswith(hdr)
open(BASE, 'w').write(hdr + entry + s[len(hdr):])
print('OK')
