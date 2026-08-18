import io

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
with io.open(EXP, encoding="utf-8") as f:
    txt = f.read()

marker = "## Run 2026-08-18 batch 0149"
assert marker in txt
entry = """## Run 2026-08-18 batch 0150

- Batch file: results/train-batch-0150.jsonl
- Corpus range: train.jsonl lines 1491-1500 (0-indexed 1490-1499)
- Source IDs: corpus-01647, corpus-01648, corpus-01649, corpus-01650, corpus-01651,
  corpus-01653, corpus-01654, corpus-01655, corpus-01656, corpus-01657
  (corpus order preserved verbatim; the gap at corpus-01652 exists in the source
  corpus itself and was not introduced here — no record was skipped or reordered)
- Progress: train 1500/5399, validation 0/601, total 1500/6000 (remaining 4500)
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema/ad-hoc check: PASS on first run
  (scripts/tb_adhoc_verify_0150.py, written fresh for this run and independent of the
  generator script scripts/tb_gen_batch_0150.py) — checks performed: per-line JSONL
  parse with physical-newline separation, exactly 10 records in the new batch, all 12
  required fields present, teacher_lane == "teacher-B", teacher_model ==
  "claude-opus-5-current", calibration_status == "provisional", decision in
  {keep, rewrite, reject}, quality_dimensions a dict of three integers in [1,5] with
  bool rejected, risks/evidence_required string arrays, corrected_answer non-empty,
  confidence a float in [0,1], global source_id uniqueness across all 1500 records in
  both lanes' result files, and the aggregate train sequence verified to be a strict
  prefix of train.jsonl by index-wise comparison of source_id, source_user and
  source_assistant against the corpus (character-identical). Validation lane not
  started, so its prefix check is vacuously satisfied at 0/601.
- Repairs applied: none required; the batch verified clean on the first attempt and no
  existing batch or corpus file was modified.
- Final schema check: PASS (same verifier rerun; MANIFEST regenerated after this
  EXPERIMENT.md edit so the digest covers the updated file).
- Manifest: MANIFEST.sha256 regenerated over all files in the experiment directory
  except itself; sha256sum -c PASS.
- Blind-review invariant: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened, grepped or listed at any point during this run. Only
  research/ai-infra-expert/corpus/train.jsonl source_user/source_assistant fields and
  this lane's own results were consulted, so teacher-A's corrected_answer text could not
  anchor these judgements. Agreement analysis remains a separate, later step.
- Technical topics covered: multi-GPU collective-initialization hangs, scenario variants
  47-57, all sharing one rubric-style source assistant turn. Each record was given a
  distinct, non-interchangeable diagnostic or design lane rather than a paraphrase:
  (47) rendezvous and out-of-band interface selection, with a world-size x
  NCCL_SOCKET_IFNAME run matrix and the docker0/virbr0 unroutable-interface mechanism;
  (48) reframing "hang" as possibly slow init, with a world-size sweep of
  init_process_group latency and a raised-timeout completion test;
  (49) designing a pre-flight barrier on the same out-of-band path NCCL will use, so
  failures name a rank instead of stalling silently, gated behind a flag with an A/B
  step-time regression check;
  (50) per-rank stack classification as the discriminator between an absent participant
  (asymmetric stacks) and a symmetric transport fault, plus a reduced-world localisation
  test and a two-reproduction rule before draining a node;
  (51) counter-based no-progress vs slow-progress test using NIC/RDMA byte counters and
  busbw at 8 MiB / 128 MiB / 1 GiB to detect silent fallback off the intended transport;
  (53) intermittency as the dominant clue, treating the failure as a rendezvous race,
  with a 20-run staggered-start experiment and the shared-MASTER_PORT / stale
  file-rendezvous cross-talk mechanism;
  (54) deriving a numeric initialization budget from measured per-phase costs (CUDA
  context creation, topology detection, bootstrap all-gather, channel setup) with their
  differing scaling laws, so "hung" stops being defined by operator patience;
  (55) operational design for blast-radius control — budget-triggered watchdog, stack
  capture, structured incident record, two-incident quarantine rule with an automated
  release gate, rate-limited kills, and a mandatory dry-run period;
  (56) change/version correlation as the cheapest first evidence, with a three-arm
  component bisect (image vs driver vs both) and the partial-rollout version-skew
  mechanism that makes failure depend on which nodes are allocated;
  (57) separating fabric capability from framework behaviour with a standalone
  all_reduce_perf sweep escalated intra-node -> pairwise -> full world, and the rule that
  a baseline from a different NCCL version is not a valid comparison.
  Every rewrite states assumptions, an explicitly falsifiable hypothesis with its
  rejection condition, a controlled experiment holding the right variables fixed,
  quantities with units, named confounders, the evidence required to settle the question,
  and an explicit rollback threshold. Diagnostic-only actions (forcing a transport,
  raising a timeout, pinning an interface) are marked as such and required to be reverted.
- Status caveat: these outputs are PROVISIONAL teacher-B review records produced by a
  general-purpose model under blind conditions. They are NOT expert gold labels, have not
  been validated against real hardware measurements, and do not constitute or demonstrate
  any model domain capability. They are inputs to a later human/expert adjudication and
  agreement analysis step only.

"""
txt = txt.replace(marker, entry + marker, 1)
with io.open(EXP, "w", encoding="utf-8") as f:
    f.write(txt)
print("updated", len(txt))
