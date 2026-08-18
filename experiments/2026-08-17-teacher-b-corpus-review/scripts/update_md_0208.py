P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
BLOCK = """## Round 208 - train-batch-0208.jsonl

- Batch file: results/train-batch-0208.jsonl
- Corpus slice: train.jsonl positional lines 2071-2080 (0-indexed 2070-2079)
- Source IDs: corpus-02281 .. corpus-02291 (non-consecutive: corpus-02285 is absent from the corpus, so the slice is taken positionally, never by ID arithmetic)
- Progress: 2080/2500 train (420 remaining); validation target is 0 by user decision
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (tb_verify_batch_0208.py, derived from the 0207 verifier via sed on batch number and slice offset)
- Repairs performed: none required
- Final schema check: VERIFY_PASS, aggregate TOTAL=2080, strict prefix of train.jsonl confirmed, no validation-batch files present
- Manifest: MANIFEST.sha256 regenerated over all files in the experiment directory except itself (452 entries); sha256sum -c passed for every entry

Technical topics covered by this batch: all ten items belong to the same degenerate TP-versus-PP family (scenario variants 81-91) whose assistant turn is a grading rubric rather than an answer, so every item is a rewrite. Ten mutually disjoint analytical stances (Stance 80-89) were used to keep a near-homogeneous batch from producing homogeneous output: speculative decoding amortising per-step collective and bubble cost over accepted tokens; KV-cache sharding under grouped-query attention where TP degree does not divide the key/value head count, plus paged-allocator block granularity; pipeline stage-balance auditing where throughput is set by the slowest stage rather than the mean; request length distribution and tail shape loading the two arms asymmetrically; power and thermal throttling producing arm-dependent clock reductions; migration and mixed-layout rollout planning including checkpoint resharding and mixture-percentile behaviour; the framing correction that TP and PP are not exclusive and the real search space is the factorisation of the device budget including data-parallel replicas and expert parallelism; host-side per-token cost (detokenisation, stop matching, structured-output constraint masks) as a layout-invariant term that can exceed the gap being argued over; observability and on-call diagnosability of each layout's failure shape as a deployment gate; and a closing provenance/authority-bound stance.

Every quantitative claim in this batch is explicitly tagged ESTIMATE with its derivation attached; no value is MEASURED, because no benchmark was executed for this review. These records are provisional teacher-B review material, NOT expert gold, and they are not evidence about any model's domain capability. Teacher-A artifacts were not read at any point during generation (blind review preserved).

"""
c = open(P).read()
i = c.index("## Round 207")
open(P, "w").write(c[:i] + BLOCK + c[i:])
print("ok")
