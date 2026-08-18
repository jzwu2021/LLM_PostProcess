import io
p = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s = open(p).read()
entry = """## Run 0203 - train-batch-0203.jsonl (provisional teacher-B blind review)

- Batch file: results/train-batch-0203.jsonl
- Corpus range: research/ai-infra-expert/corpus/train.jsonl positional lines 2021-2030 (0-indexed slice 2020:2030)
- Source IDs: corpus-02230 .. corpus-02239 (contiguous, corpus order preserved, none skipped or reordered)
- Progress: train 2030/2500 (stage target 2500, set by the user on 2026-08-18, replacing the original 6000 full-corpus target). Validation target is 0; no validation-batch file exists or was created.
- Remaining to stage target: 470
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: PASS on first run (scripts/tb_verify_batch_0203.py). No repair actions were needed this round.
- Repairs: none.
- Final schema check: VERIFY_PASS, aggregate TOTAL 2030, aggregated source_id sequence confirmed to be a strict prefix of train.jsonl with no duplicates; all 12 required fields present; source_user and source_assistant byte-identical to corpus; every corrected_answer non-empty, distinct, and distinct from the source assistant text; confidence in [0,1]; zero validation-batch files present.
- Manifest: MANIFEST.sha256 regenerated over all files in this experiment directory except MANIFEST.sha256 itself and __pycache__; sha256sum -c reports 431/431 OK with zero failures.

Technical topics covered by this batch: this is another homogeneous block of the tensor-parallel versus pipeline-parallel
family (scenario variants 30-39), so the batch was written with ten mutually disjoint analytical stances over a shared
assumption frame, to avoid ten near-identical answers. The stances are: decode arithmetic intensity and the
bandwidth-bound versus compute-bound distinction; a queueing-theoretic treatment where tail latency is driven by
utilization rather than by kernel time; decomposition of a vague "latency-sensitive" SLO into TTFT versus TPOT and the
dependence on output-length distribution; continuous-batching policy as a confounder that can invert the comparison;
weight and KV precision as the variable that moves the TP/PP crossover point; multi-node fabric health (RoCE lossless
configuration, PFC/ECN, GPUDirect RDMA engagement versus silent host-staged fallback) as a gate that precedes any
latency measurement; numerical non-determinism introduced by changing the reduction order when the shard axis changes,
and why token-level equality is the wrong regression gate; operational blast radius, gang scheduling and recovery time
as a function of parallel degree; prefill/decode disaggregation with KV transfer as the modern alternative to the
TP-versus-PP dichotomy; and pre-registration discipline to prevent overfitting the comparison to the tuning trace.

Measurement policy: every quantitative claim in this batch is explicitly labelled ESTIMATE and carries its derivation
from first principles. No value is labelled MEASURED, because no benchmark was executed for this review.

Status: these outputs are PROVISIONAL teacher-B review material produced under blind conditions. They are not expert
gold, they have not been human-verified, and they say nothing about any model's domain capability. No teacher-A artifact
was read, opened, or searched during the production of this batch.

"""
i = s.index("## Run ")
open(p, "w").write(s[:i] + entry + s[i:])
print("updated")
