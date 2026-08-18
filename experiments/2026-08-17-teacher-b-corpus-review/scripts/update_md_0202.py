P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s = open(P).read()
anchor = "## Run 2026-08-18 - train-batch-0201.jsonl"
entry = """## Run 2026-08-18 - train-batch-0202.jsonl

- Batch file: `results/train-batch-0202.jsonl`
- Corpus slice: `research/ai-infra-expert/corpus/train.jsonl` positional rows 2010-2019 (0-indexed), 10 records
- Source IDs: corpus-02220, corpus-02221, corpus-02222, corpus-02223, corpus-02224, corpus-02225, corpus-02226, corpus-02227, corpus-02228, corpus-02229 (positional slicing, not ID arithmetic)
- Progress: 2020/2500 train (stage target 2500, set by the user on 2026-08-18, superseding the original 6000-record full-corpus plan). No validation records are produced in this stage (validation target 0).
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: FAIL, 30 errors (`scripts/tb_verify_batch_0202.py`, derived from the batch-0201 verifier via sed). Two distinct generator defects: (1) the generator read `messages[0]`/`messages[1]` positionally, but these corpus rows carry a leading `system` message, so `source_user` and `source_assistant` were off by one and mismatched the corpus for all 10 records; (2) the generator omitted the required `Analytical stance under test:` prefix on `corrected_answer`.
- Repairs: fixed the generator to build a role-keyed map (`{x["role"]: x["content"] for x in d["messages"]}`) instead of positional indexing, and added the stance-marker prefix; regenerated the batch. No original corpus file and no previously committed batch was modified.
- Final schema check: PASS (TOTAL 2020, VERIFY_PASS). Checks: per-line JSON parse, batch count 10, all 12 required fields, teacher_lane/teacher_model/calibration_status/decision value constraints, exact character equality of source_user and source_assistant against the original corpus, non-empty corrected_answer, ESTIMATE labelling present, distinct answers and distinct openings, confidence in [0,1], global source_id uniqueness across all 202 batches, aggregated 2020-record train sequence a strict prefix of train.jsonl, and zero validation-batch files.
- Manifest: `MANIFEST.sha256` regenerated over every file in the experiment directory except itself (excluding `__pycache__`); `sha256sum -c` verified all entries.

### Technical topics covered by this batch

All ten source records are near-identical variants (scenario variants 20-29) of the same prompt: choosing between tensor and pipeline parallelism for a latency-sensitive multi-GPU service, with an explicit falsifiable hypothesis and controlled experiment. Because the batch is homogeneous, each corrected answer is written under a distinct, mutually exclusive analytical stance, disjoint from the stances used in batches 0200 and 0201. This batch's stances: roofline decomposition of decode latency into weight, KV and synchronisation terms; pipeline-bubble arithmetic as the structural disqualifier at low concurrency; fabric-first reasoning where the small-message all-reduce latency floor decides and TP must not cross node boundaries; memory-capacity framing via KV-bytes-per-token and what actually forces sharding; the hybrid TP-inside-node plus PP-across-nodes composition; prefill-versus-decode role separation with disaggregated serving; tail latency and failure-domain safety under fault injection; cost per served token as an SLO-constrained optimisation; silent configuration and placement traps that confound A/B results (NVLink domain placement, NCCL transport fallback, NUMA affinity, head-count divisibility, unbalanced PP stage splits); and an executable staged decision procedure with canary rollout.

Every answer carries a shared substrate stating the mechanism (TP splits every layer's GEMMs and adds two all-reduces per block to the critical path of every decoded token, so it is latency-additive but capacity-multiplying; PP splits by depth, sends only a small hidden state per boundary, but serialises a single request through stages and bubbles at low concurrency), the boundary conditions that flip the recommendation (interconnect class, whether the model fits at all, offered concurrency), the single-variable experiment design, the evidence artifacts required, and a pre-committed rollback gate. Every numeric statement is explicitly labelled ESTIMATE with its derivation inline; no value is labelled MEASURED because no benchmark was executed for this review.

- Status: PROVISIONAL. These are teacher-B second-opinion rewrites produced by the current conversation model under blind conditions; the teacher-A calibration directory was not read at any point during generation. They are not expert gold labels, have not been human-verified, and say nothing about any model's domain capability.

"""
assert anchor in s
s = s.replace(anchor, entry + anchor, 1)
open(P, "w").write(s)
print("OK")
