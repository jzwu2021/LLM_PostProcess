import re
p="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(p).read()
entry = """## Run 0210 (train-batch-0210.jsonl)

- Batch file: results/train-batch-0210.jsonl
- Corpus range: positional rows 2091-2100 of research/ai-infra-expert/corpus/train.jsonl
- Source IDs: corpus-02303, corpus-02305, corpus-02306, corpus-02307, corpus-02308, corpus-02309, corpus-02310, corpus-02311, corpus-02312, corpus-02313 (IDs are non-consecutive; slicing is positional, never ID arithmetic)
- Progress: 2100/2500 train (400 remaining). Validation target is 0; no validation-batch files exist or were created.
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: PASS on first run (scripts/tb_verify_batch_0210.py, derived from the 0209 verifier by sed substitution)
- Repairs performed: none required
- Final schema check: VERIFY_PASS, TOTAL 2100, aggregate sequence confirmed a strict prefix of train.jsonl with globally unique source_id
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; sha256sum -c reported all files OK

Technical topics covered by this batch: all ten items are near-homogeneous tensor-versus-pipeline parallelism scenario variants (103-113), so each answer is assigned a mutually exclusive analytical stance (Stance 100-109) layered on a shared assumption/mechanism/boundary-condition frame. Stances cover NCCL transport discovery and silent fallback to PCIe or shared memory; context length as a hidden variable moving the memory-feasible degree via KV working set; attention and grouped-query key-value head divisibility limiting the usable tensor degree; pipeline schedule and in-flight micro-batch depth as the determinants of bubble fraction; per-stage partitioning imbalance and the max-stage-time throughput ceiling; speculative decoding's interaction with per-step collective amortisation and tail variance; engine/kernel/driver version pinning as a precondition for a portable ranking; power and thermal capping as a confounder that penalises the compute-denser arm; operational blast radius, failure-domain arithmetic and rehearsed rollback; and a closing epistemic-status stance.

Status caveat: this output is PROVISIONAL teacher-B review material produced blind, without any visibility into the teacher-A calibration lane. It is not expert gold, has not been human-adjudicated, and is not evidence about any model's domain capability. Every quantitative claim in this batch is labelled ESTIMATE with its derivation stated; no value is MEASURED, because no benchmark run was executed for this review.

"""
s = entry + s if not s.startswith("#") else re.sub(r"\A(# [^\n]*\n+)", r"\1"+entry.replace("\\","\\\\"), s, count=1)
open(p,"w").write(s)
print("OK")
