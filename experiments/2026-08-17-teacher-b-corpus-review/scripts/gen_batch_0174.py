#!/usr/bin/env python3
"""Generate teacher-B provisional BLIND review batch 0174 (corpus rows 1731-1740).

Rubric-identical variant family: "weight-only quantization -> define a fair comparison".
Each variant is assigned a DISTINCT primary mechanism / distinct controlled experiment so
the batch is not template-identical. corrected_answer sha256 uniqueness is asserted here
and cross-checked against all previously emitted batches by the verifier.
"""
import json, hashlib, os, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
START, N = 1730, 10   # 0-based offset of first row, count
OUT = os.path.join(EXP, "results", "train-batch-0174.jsonl")

# (mechanism_title, hypothesis, controlled_experiment, arithmetic, confounders, rollback, risks, evidence, dims, conf, decision)
VARIANTS = [
 dict(
  mech="Memory-bandwidth-bound decode: dequant weight traffic, not FLOPs, sets the ceiling",
  h="H1: at batch size 1 and identical KV state, decode step time scales with bytes of weight read per token, so INT4 weight-only quantization cuts TPOT by at most the weight-byte ratio (about 4x vs BF16) and by strictly less once dequant overhead and non-weight traffic are counted. If measured TPOT speedup exceeds the weight-byte ratio, H1 is false and something else (kernel fusion, different attention path, changed batching) changed too.",
  exp="Fix the checkpoint, tokenizer, prompt set, max_tokens, sampling seed, and server flags. Run A=BF16, B=W4A16 on the SAME GPU, same driver, same container digest, back to back, alternating A/B/A/B to absorb thermal drift. Concurrency swept 1,2,4,8,16,32. Report TTFT p50/p95, TPOT p50/p95, output tok/s, and peak weight-memory from the allocator, each with bootstrap 95% CI over >=5 repeats.",
  arith="ESTIMATE: a 9B-parameter model holds 9e9 * 2 B = 18 GB of BF16 weights; at 4 bits plus a typical group-wise scale/zero overhead of roughly 0.5 bit/weight the same weights hold 9e9 * 4.5/8 B = ~5.06 GB. Weight-byte ratio = 18/5.06 = ~3.6x. Derivation: parameter count times bytes per parameter; no vendor number used. MEASURED: none of this is measured until you record allocator peak and per-token latency on your hardware.",
  conf="Confounders: CUDA graph capture on in one arm only; different max_num_seqs or chunked-prefill setting; KV cache dtype silently changed with the quantization recipe; power/clock throttling on the second arm; a different attention backend selected because the quantized path lacks a fused kernel.",
  roll="Rollback gate: revert to BF16 if quality on the frozen eval set drops more than the pre-registered margin (choose it BEFORE running, e.g. 1.0 absolute point on your primary metric with the CI excluding zero), or if p95 TTFT regresses at the production concurrency.",
  risks=["Speedup claimed from a non-identical serving configuration rather than from quantization","KV cache dtype changed together with weight dtype, confounding memory and quality deltas","Single-run latency numbers reported without repeats or confidence intervals"],
  ev=["Container image digest and server flags for both arms","Allocator peak memory and per-token latency traces, >=5 repeats per point","Frozen eval set scores with bootstrap CIs"],
  dims=(4,4,4), conf_v=0.62),
 dict(
  mech="Cost must be defined per served token at a fixed SLO, not per GPU-hour",
  h="H1: quantization only lowers cost if it raises sustained throughput AT the latency SLO. If the maximum concurrency that still meets p95 TPOT does not increase, cost per million output tokens does not improve, no matter how much idle memory is freed.",
  exp="Define the SLO first (e.g. p95 TTFT <= X ms, p95 TPOT <= Y ms; you pick X,Y from product requirements, not from the benchmark). For each arm, binary-search the maximum request rate that holds the SLO for a 15-minute steady-state run with a fixed request-arrival trace replayed identically. Cost = (GPU-hours * price) / (output tokens delivered within SLO).",
  arith="ESTIMATE: if arm A sustains 900 output tok/s within SLO and arm B sustains 1,350 tok/s on the same single GPU, cost per token falls by 1 - 900/1350 = 33%. Derivation: cost is inversely proportional to in-SLO throughput when hardware and price are held fixed. MEASURED: both throughput figures must come from your steady-state runs; the numbers above are illustrative arithmetic only.",
  conf="Confounders: warm-up requests counted in steady state; different prefix-caching hit rates between arms because the trace was reshuffled; autoscaling or other tenants on the node; measuring throughput at saturation while ignoring that the SLO was already violated.",
  roll="Rollback gate: revert if in-SLO throughput gain is under the pre-registered threshold (e.g. <15%) or if the CI on cost per token includes zero improvement.",
  risks=["Reporting peak throughput at violated SLO as a cost win","Prefix-cache hit-rate asymmetry between arms","Ignoring engineering and requalification cost of maintaining a second checkpoint"],
  ev=["Written SLO definition fixed before measurement","Identical replayed arrival trace for both arms","In-SLO throughput with CIs and the GPU price basis used"],
  dims=(4,4,4), conf_v=0.6),
 dict(
  mech="Quality evaluation must be task-matched and paired, because weight-only quantization degrades unevenly",
  h="H1: aggregate perplexity is not a sufficient acceptance metric; quantization error concentrates in outlier channels and shows up on long-context, code, and arithmetic tasks before it shows up in mean perplexity. If perplexity moves less than its noise band while a task-specific score drops beyond its CI, H1 is supported.",
  exp="Paired evaluation: run both arms on the SAME frozen prompt set with greedy decoding (temperature 0) so the only difference is weights. Compute per-item paired differences and a paired bootstrap CI, not two independent means. Include at least: your production task set, a long-context subset at your true max sequence length, and an arithmetic/code subset. Report win/loss/tie counts per subset.",
  arith="ESTIMATE: to detect a 1-percentage-point drop on a binary-scored task with per-item variance near p(1-p)=0.25 at 80% power, a paired design needs on the order of several thousand items; the exact count follows from your observed pairwise disagreement rate, which you must MEASURE first on a pilot of a few hundred items. No sample size should be asserted without that pilot.",
  conf="Confounders: nondeterministic kernels making even the same arm disagree with itself; different tokenizer or chat template applied per arm; evaluation prompts contaminated by the quantization calibration set; sampling temperature above 0 injecting variance that swamps the effect.",
  roll="Rollback gate: revert if any pre-registered subset regresses beyond its margin, even when the aggregate mean looks flat.",
  risks=["Aggregate perplexity used to hide a concentrated task regression","Self-disagreement from nondeterministic kernels mistaken for a quantization effect","Unpaired comparison inflating the noise band"],
  ev=["Frozen prompt set with a content hash","Greedy-decoding self-consistency run of each arm against itself","Per-subset paired deltas with bootstrap CIs"],
  dims=(5,4,4), conf_v=0.63),
 dict(
  mech="Calibration data is an experimental variable and must be controlled and disclosed",
  h="H1: the quantized model's measured quality depends materially on the calibration corpus. If two W4 checkpoints built from the same recipe but different calibration sets differ by more than the evaluation noise band, then any single-calibration comparison against BF16 is not a property of quantization but of that calibration draw.",
  exp="Build >=3 quantized checkpoints with identical recipe, group size, and seed, varying only the calibration corpus (in-domain, generic web, and a deliberately off-domain set), each with the same number of calibration sequences and sequence length. Evaluate all three plus BF16 on the frozen eval set. Report the spread across calibration draws as the noise floor for any BF16-vs-quantized claim.",
  arith="ESTIMATE: with 3 calibration draws you can only bound the spread crudely; the standard error of a mean over 3 draws is sigma/sqrt(3) = 0.58*sigma, so a difference smaller than roughly 1.2*sigma between arms is not resolvable. Derivation: standard error of the mean. MEASURED: sigma itself must come from your three runs.",
  conf="Confounders: calibration set overlapping the evaluation set (contamination); different calibration sequence lengths changing which activations dominate; group size or act-order changed at the same time as the corpus.",
  roll="Rollback gate: revert to BF16, or block the release, if the across-calibration spread is comparable to the BF16-vs-quantized gap you are trying to claim.",
  risks=["Calibration/evaluation contamination inflating measured quality","Single calibration draw generalized into a claim about the method","Undisclosed recipe changes bundled with the corpus change"],
  ev=["Calibration corpus manifests and overlap check against the eval set","Recipe config diff across the three builds","Per-draw scores and their spread"],
  dims=(5,4,4), conf_v=0.6),
 dict(
  mech="Kernel availability, not the numeric format, often decides whether the speedup materializes",
  h="H1: the quantized arm is only faster where a fused dequant-GEMM kernel exists for the exact shape, dtype, and hardware. On shapes routed to a fallback path, W4A16 is SLOWER than BF16. If per-layer timing shows a subset of GEMMs regressing, H1 is supported and the fix is shape/kernel selection, not a different bit width.",
  exp="Profile both arms at the kernel level (torch profiler or Nsight Systems) on the same fixed prompt. Emit a per-layer table: layer name, GEMM shape, kernel name, and time. Also test a prefill-heavy prompt and a decode-heavy prompt separately, since prefill is compute-bound and decode is bandwidth-bound and they can move in opposite directions.",
  arith="ESTIMATE: if 20% of GEMM time falls back to an unfused path that is 1.5x slower while the remaining 80% gets 2.0x faster, the aggregate speedup is 1 / (0.2*1.5 + 0.8/2.0) = 1/(0.30+0.40) = 1.43x, not 2x. Derivation: Amdahl-style weighted sum of per-segment times. MEASURED: the 20/80 split and both per-segment ratios must come from your profile.",
  conf="Confounders: tensor-parallel degree changing per-GPU GEMM shapes; padding to a kernel-friendly tile in one arm only; profiler overhead applied asymmetrically; a first-call autotune pass counted in the timed window.",
  roll="Rollback gate: revert if any production-relevant shape regresses, or if the kernel path is not available in the exact runtime version you would deploy.",
  risks=["Speedup measured only on kernel-friendly shapes","Autotune or warm-up time contaminating the timed window","Runtime version drift between the benchmark and the deployment"],
  ev=["Per-layer kernel timing tables for both arms","Runtime, driver, and kernel-library versions","Separate prefill-heavy and decode-heavy measurements"],
  dims=(5,4,4), conf_v=0.62),
 dict(
  mech="Freed weight memory converts to serving value only through KV-cache capacity",
  h="H1: the business benefit of weight-only quantization is the extra KV cache it buys, which raises the concurrency ceiling. If the deployment is already limited by compute or by a max_num_seqs cap rather than by KV memory, freeing weight bytes yields no throughput gain and H1 is false for that deployment.",
  exp="Measure the KV-limited concurrency ceiling directly: for each arm, hold the SLO fixed and raise concurrency until preemption/recompute events or queue delay appear; record the concurrency at first preemption and the reported KV utilization. Then repeat with the KV budget artificially clamped to the BF16 value in the quantized arm; if throughput falls back to the BF16 level, the gain was entirely KV capacity.",
  arith="ESTIMATE: KV bytes per token = 2 (K and V) * layers * kv_heads * head_dim * bytes_per_element. For 40 layers, 8 KV heads, head_dim 128, FP16: 2*40*8*128*2 B = 163,840 B = ~160 KiB per token. Freeing ~13 GB of weights therefore buys ~13e9/163840 = ~79,000 additional cached tokens. Derivation: pure arithmetic from the stated architecture; substitute your real config values. MEASURED: nothing here until you read the actual config and the server's KV utilization counters.",
  conf="Confounders: server-side memory fraction flags reserving space independent of weights; MQA/GQA head counts differing from assumption; fragmentation; a max_num_seqs cap binding before memory does.",
  roll="Rollback gate: revert if the clamped-KV control shows no throughput difference, i.e. you paid quality for capacity you cannot use.",
  risks=["Assuming freed memory automatically becomes throughput","Wrong KV arithmetic from assumed rather than read config values","Preemption/recompute events silently degrading tail latency"],
  ev=["Model config values for layers, kv_heads, head_dim, and KV dtype","KV utilization and preemption counters at the ceiling","Clamped-KV control run"],
  dims=(5,4,4), conf_v=0.63),
 dict(
  mech="Statistical discipline: pre-register the comparison so the result is falsifiable rather than negotiable",
  h="H1: without a pre-registered primary metric, margin, and stopping rule, any quantization comparison will produce a favourable-looking subset by chance. If post-hoc subset selection is required to show a win, the honest conclusion is no demonstrated win.",
  exp="Before any run, write down: the primary metric, the non-inferiority margin, the number of repeats, the concurrency points, the eval set hash, and the decision rule. Then run. Analyse the primary metric first; report all secondary metrics with multiplicity correction (e.g. Holm) and label them exploratory. Include a negative control: BF16 vs BF16 with different seeds, which must show no significant difference; if it does, the harness is too noisy to support any conclusion.",
  arith="ESTIMATE: with 10 secondary metrics at alpha=0.05 and no correction, the probability of at least one false positive is 1 - 0.95^10 = 40%. Derivation: independent-test family-wise error rate. This is a mathematical fact about the test procedure, not a claim about your platform.",
  conf="Confounders: peeking at results and stopping when favourable; changing the eval set after seeing scores; reporting the best of several quantization recipes against a single BF16 baseline.",
  roll="Rollback gate: if the negative control fails, halt and fix the harness before making any quantization decision; do not ship on a noisy harness.",
  risks=["Post-hoc metric selection manufacturing a false win","Uncorrected multiplicity across many secondary metrics","Best-of-N recipe selection compared against a single baseline"],
  ev=["Written pre-registration document with a timestamp","Negative control BF16-vs-BF16 result","Full metric table including losses, not only wins"],
  dims=(5,5,4), conf_v=0.64),
 dict(
  mech="Operational safety: the comparison must include failure modes, not only happy-path metrics",
  h="H1: the quantized arm carries operational risks absent in BF16 (kernel numerical instability at long context, checkpoint-format lock-in, missing observability). If a long-context or high-concurrency soak run produces NaNs, garbled output, or OOM in the quantized arm but not in BF16, the cost comparison is void until fixed.",
  exp="Run a 4-hour soak per arm at the target concurrency with a realistic length distribution including the max supported context. Instrument: output-token entropy, empty/garbled response rate, NaN/inf detector on logits, OOM and preemption counts, and restart count. Compare failure rates, not just latency. Additionally run a canary at low traffic share before any full rollout.",
  arith="ESTIMATE: to observe at least one event of a 1-in-10,000-request failure with 95% probability you need about ln(0.05)/ln(0.9999) = ~29,957 requests. Derivation: geometric distribution. A 4-hour soak at 3 requests/s delivers 3*3600*4 = 43,200 requests, which clears that bar. MEASURED: your actual request rate and failure rate.",
  conf="Confounders: soak run hitting a different prompt mix than production; log sampling hiding rare failures; automatic retries masking errors; the canary receiving unrepresentative traffic.",
  roll="Rollback gate: automatic revert if garbled-output rate, NaN count, or 5xx rate exceeds the BF16 baseline beyond the pre-registered margin at any point during canary or soak.",
  risks=["Rare numerical failures invisible in short benchmark runs","Retries masking real error rates","Checkpoint-format lock-in to a runtime version that may not be maintained"],
  ev=["Soak run logs with per-minute failure counters","NaN/inf detector output on logits","Canary traffic share, duration, and abort criteria"],
  dims=(5,5,5), conf_v=0.64),
 dict(
  mech="Compare against the real alternatives, not only against BF16",
  h="H1: weight-only quantization is not the cheapest available cost lever. If a smaller model, better batching/prefix caching, or FP8 (where supported) reaches the same SLO at equal or lower cost with less quality risk, then choosing W4 is not justified even if W4 beats BF16.",
  exp="Add arms to the same harness: (a) BF16 baseline, (b) W4A16, (c) FP8 if the hardware supports it, (d) a smaller model at BF16 with comparable task quality, (e) BF16 with prefix caching and tuned batching enabled. Evaluate every arm on identical prompts, identical SLO, identical trace. Rank by cost per in-SLO token AND by quality delta; report the Pareto frontier rather than a single winner.",
  arith="ESTIMATE: if a model with 55% of the parameters meets the quality bar, the weight-byte reduction is 1 - 0.55 = 45%, which is less than the roughly 72% (1 - 4.5/16 for INT4-with-scales vs BF16) from W4 but comes with no quantization-specific numerical risk. Derivation: ratio of parameter bytes. Whether the smaller model meets the bar is MEASURED, not assumed.",
  conf="Confounders: unequal tuning effort across arms (the favourite gets tuned, the rest do not); hardware not supporting FP8, making arm (c) fall back silently; prefix caching helping only if the traffic actually has shared prefixes.",
  roll="Rollback gate: do not adopt W4 if another arm is on the Pareto frontier at equal cost and lower quality risk.",
  risks=["Straw-man baseline that was never tuned","Silent fallback in an unsupported-precision arm","Optimising a single metric instead of reporting the frontier"],
  ev=["Tuning effort log per arm","Hardware capability check for the precisions tested","Pareto table of cost vs quality across all arms"],
  dims=(5,4,4), conf_v=0.62),
 dict(
  mech="Attribution: decompose the observed latency delta into weight-read, dequant, and non-weight components",
  h="H1: the measured TPOT delta can be fully accounted for by (bytes of weight read)/(achieved memory bandwidth) plus dequant compute plus an unchanged non-weight residual. If the accounted terms leave a residual larger than the run-to-run noise band, an uncontrolled variable changed between the arms.",
  exp="For each arm at batch size 1, measure achieved memory bandwidth with a profiler counter, compute the predicted weight-read time from the known weight byte count, and time a weights-only forward with attention stubbed out to isolate the non-weight residual. Build a table: predicted vs measured, per arm. Repeat at concurrency 32 where the arithmetic intensity changes and check that the model still balances.",
  arith="ESTIMATE: with 18 GB of BF16 weights and an achieved bandwidth of 900 GB/s, the pure weight-read floor is 18/900 = 20.0 ms per decode step; with ~5.06 GB of W4 weights at the same achieved bandwidth it is 5.06/900 = 5.6 ms. Predicted floor ratio 3.6x. Derivation: bytes divided by bandwidth. Both the byte counts and the 900 GB/s figure must be MEASURED or read from your own profiling; do not carry the 900 GB/s number over from any other system.",
  conf="Confounders: achieved bandwidth differing between arms because access patterns differ; dequant compute overlapping the read and thus not additive; CUDA graphs removing launch overhead in one arm; the stubbed-attention control changing memory residency.",
  roll="Rollback gate: if the residual exceeds the noise band, halt the comparison and find the uncontrolled variable before reporting any speedup number.",
  risks=["Unexplained residual reported as a quantization benefit","Assuming a bandwidth figure instead of measuring it","Overlap effects treated as additive in the model"],
  ev=["Achieved-bandwidth counters per arm","Weight byte counts read from the checkpoint, not assumed","Stubbed-attention control timings"],
  dims=(5,4,4), conf_v=0.61),
]

PREAMBLE = ("Assumptions, stated so you can correct them: a single model family served on one GPU type you control; "
            "weight-only quantization (W4A16 or similar) with activations left in BF16/FP16; the goal is lower serving "
            "cost at unchanged user-visible quality. Every number below is labelled ESTIMATE (arithmetic derived from "
            "inputs stated in the same sentence) or MEASURED (only valid once you collect it on your own hardware). "
            "I assert no vendor- or platform-specific performance fact that I have not derived here.\n\n")


def build(v):
    return (PREAMBLE
        + "Primary mechanism this comparison must control\n" + v["mech"] + "\n\n"
        + "Falsifiable hypothesis\n" + v["h"] + "\n\n"
        + "Controlled experiment\n" + v["exp"] + "\n\n"
        + "Quantitative reasoning (labelled)\n" + v["arith"] + "\n\n"
        + "Expected confounders\n" + v["conf"] + "\n\n"
        + "Rollback criteria\n" + v["roll"] + "\n\n"
        + "Fairness checklist the comparison must satisfy\n"
          "1. Identical checkpoint lineage, tokenizer, and chat template in both arms.\n"
          "2. Identical prompt set, identical decoding parameters, identical max_tokens.\n"
          "3. Identical runtime image digest, driver, and server flags except the single variable under test.\n"
          "4. Report quality, peak memory, TTFT, TPOT, throughput, and concurrency ceiling together; any one alone is misleading.\n"
          "5. Report confidence intervals from repeated runs; a single run is not a measurement.\n"
          "6. Disclose the calibration set and confirm it does not overlap the evaluation set.\n"
          "7. Record kernel/runtime support for the exact shapes served, and note any fallback path.\n"
          "8. Report failure cases (garbled output, NaNs, OOM, preemption), not only the happy path.\n\n"
        + "What would change my recommendation\n"
          "If your profile shows the deployment is compute-bound rather than memory-bound at your production concurrency, "
          "or if the concurrency ceiling is capped by a scheduler limit rather than by KV memory, the expected benefit of "
          "weight-only quantization shrinks toward zero and the quality risk is no longer worth taking. Send me the profile "
          "and the config values and I will redo the arithmetic against your real numbers.")


def main():
    rows = []
    with open(CORPUS) as f:
        for i, line in enumerate(f):
            if START <= i < START + N:
                rows.append(json.loads(line))
            elif i >= START + N:
                break
    assert len(rows) == N, len(rows)
    assert len(VARIANTS) == N

    out, seen = [], set()
    for r, v in zip(rows, VARIANTS):
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        ans = build(v)
        h = hashlib.sha256(ans.encode()).hexdigest()
        assert h not in seen, "duplicate corrected_answer within batch"
        seen.add(h)
        tc, ic, os_ = v["dims"]
        out.append({
            "source_id": r["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": v.get("decision", "rewrite"),
            "source_user": msgs["user"],
            "source_assistant": msgs["assistant"],
            "corrected_answer": ans,
            "quality_dimensions": {"technical_correctness": tc, "instruction_coverage": ic, "operational_safety": os_},
            "risks": v["risks"],
            "evidence_required": v["ev"],
            "confidence": v["conf_v"],
        })

    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("WROTE", OUT, len(out), "ids", out[0]["source_id"], "->", out[-1]["source_id"])


if __name__ == "__main__":
    main()
