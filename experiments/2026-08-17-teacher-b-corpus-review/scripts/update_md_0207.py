import re
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s = open(P).read()
entry = """## Round 207 - train-batch-0207.jsonl

- Batch file: results/train-batch-0207.jsonl
- Corpus slice: train.jsonl positional lines 2061-2070 (0-indexed 2060-2069)
- Source IDs: corpus-02271 .. corpus-02280
- Progress: 2070/2500 train (430 remaining); validation target is 0 by user decision
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (tb_verify_batch_0207.py, derived from the 0206 verifier via sed on batch number and slice offset)
- Repairs performed: none required
- Final schema check: VERIFY_PASS, aggregate TOTAL=2070, strict prefix of train.jsonl confirmed, no validation-batch files present
- Manifest: MANIFEST.sha256 regenerated over all files in the experiment directory (448 entries); sha256sum -c passed for every entry

Technical topics covered by this batch: all ten items are the same degenerate TP-versus-PP family (scenario variants 71-80) whose assistant turn is a grading rubric rather than an answer, so every item is a rewrite. Ten mutually disjoint analytical stances (60-series continuation, Stance 70-79) were used to avoid homogeneous output: quantisation/weight-dtype interaction with sharding granularity; scheduler batching policy (continuous batching, chunked prefill, max-batched-tokens) as the dominant latency term; warmup, graph capture and allocator fragmentation contaminating early windows; failure-domain sizing and blast radius as the upper bound on parallel degree; cost per delivered token at a fixed SLO rather than unloaded latency; statistical discipline on tail-percentile estimators and pre-registered sample size; multi-tenant noisy-neighbour and placement effects on shared fabric; engine/NCCL/driver version pinning and conclusion expiry; direct collective-layer instrumentation separating straggler dispersion from transfer time; and a closing provenance/authority-bound stance.

Every quantitative claim in this batch is explicitly tagged ESTIMATE with its derivation attached; no value is MEASURED, because no benchmark was executed for this review. These records are provisional teacher-B review material, NOT expert gold, and they are not evidence about any model's domain capability. Teacher-A artifacts were not read at any point during generation (blind review preserved).

"""
s = s.replace("## Round 206 - train-batch-0206.jsonl", entry + "## Round 206 - train-batch-0206.jsonl", 1)
open(P, "w").write(s)
print("ok")
