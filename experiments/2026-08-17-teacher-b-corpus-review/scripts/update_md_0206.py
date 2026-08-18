import os, subprocess, datetime
EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
MD = f"{EXP}/EXPERIMENT.md"
entry = """## Round 206 - train-batch-0206.jsonl

- Batch file: results/train-batch-0206.jsonl
- Corpus slice: train.jsonl positional lines 2051-2060 (0-indexed 2050-2059)
- Source IDs: corpus-02261 .. corpus-02270
- Progress: 2060/2500 train (440 remaining); validation target is 0 by user decision
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (tb_verify_batch_0206.py, derived from the 0205 verifier via sed)
- Repairs performed: none required
- Final schema check: VERIFY_PASS, aggregate TOTAL=2060, strict prefix of train.jsonl confirmed, no validation-batch files present
- Manifest: MANIFEST.sha256 regenerated over all files in the experiment directory; sha256sum -c passed for every entry

Technical topics covered by this batch: all ten items are the same degenerate TP-versus-PP family whose assistant turn is
a grading rubric rather than an answer, so all ten are rewrites carrying a shared mechanism frame plus ten mutually
disjoint analytical stances (60-69): attention-kernel dispatch and heads-per-rank fast paths; opposite prefill/decode
parallelism preferences and traffic length-ratio dependence; prefill/decode disaggregation in the Mooncake and NVIDIA
Dynamo sense with KV transfer cost as the deciding term; numerical non-associativity of reduced-precision reductions and
greedy-output equivalence as a precondition for timing; cross-node RoCEv2 PFC/ECN and GPUDirect RDMA attestation; long
context shifting the KV and collective-overhead arithmetic; operational runbook and game-day cost of each layout;
benchmark-client auditing and coordinated omission; pre-registered escalation and abandonment policy; and a closing
provenance bound.

Every quantitative claim in this batch is labelled ESTIMATE with its derivation attached; no value is MEASURED because no
benchmark was executed for this review. These records are PROVISIONAL teacher-B blind-review material. They are not
expert gold, they have not been adjudicated against any other lane, and they are not evidence about any model's domain
capability. Blind-review isolation held: no teacher-A artifact was read, opened or searched while producing this batch.

"""
cur = open(MD).read()
# insert after the top header block, before the first '## ' section
idx = cur.find("\n## ")
assert idx > 0
open(MD, "w").write(cur[:idx+1] + entry + cur[idx+1:])
print("md updated")
