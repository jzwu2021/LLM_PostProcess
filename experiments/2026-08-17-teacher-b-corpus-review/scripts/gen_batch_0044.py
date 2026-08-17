#!/usr/bin/env python3
"""Generate teacher-B provisional BLIND review batch 0044 (train rows 431-440)."""
import json, os, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0044.jsonl")
START, END = 431, 440  # 1-indexed inclusive

ASSUMPTIONS_CORE = """Assumptions that must be on the record before any speculative-decoding performance claim.

1. Hardware and topology. GPU model, HBM capacity and measured (not spec-sheet) achievable bandwidth, whether draft and target are co-resident on the same device or on separate devices, and the tensor-parallel degree of each. A speedup measured with TP=1 on one A30 does not transfer to TP=4, because verification introduces an extra all-reduce-bound step whose cost does not shrink with the draft model.

2. Workload distribution. Prompt-length and output-length distributions, request arrival rate, and the concurrency level at which the number is measured. Speculative decoding is a bandwidth-to-tokens converter: it wins in the memory-bound, low-batch decode regime and degrades once the target model is already compute-bound.

3. Sampling configuration. Temperature, top-p, and whether the verifier uses the exact rejection-sampling rule (output distribution provably identical to the target model alone) or a relaxed/lossy acceptance criterion. A "speedup" obtained with relaxed acceptance is a quality change, not a pure systems win, and must be reported alongside a task-level quality metric.

4. Draft/target pairing and measured acceptance rate. Report the empirical mean accepted length E[a] on the actual traffic, not on a held-out benchmark, plus the draft-to-target per-step cost ratio c_draft/c_target and the speculation length gamma.

5. Baseline definition. The comparison must be against the same server, same batching policy, same quantization, same KV layout, with speculation disabled -- not against a differently configured system.

Concrete mechanism. In memory-bound decode, one target forward costs about t_step = model_bytes_read / effective_HBM_bandwidth and is nearly independent of how many token positions are scored, because the weights stream once. The draft proposes gamma tokens; the target verifies all gamma+1 positions in one forward; the accepted prefix is emitted. Expected wall-clock speedup is approximately E[a] / (1 + gamma * c_draft/c_target).

Boundary condition. The technique stops paying once the target's verification forward becomes compute-bound. Concretely, when the effective batch already saturates the GEMM units, the gamma extra positions per sequence consume FLOPs that would otherwise serve other requests, and rejected drafts are pure waste; aggregate tokens/sec can fall even while single-stream TPOT improves.

Falsifiable hypothesis. "At concurrency <= C and gamma = g, speculation reduces p90 TPOT by >= X% with no measurable change in task accuracy."

Evidence required. Per-request TPOT/TTFT distributions (p50/p90/p99, not means) with and without speculation at several concurrency levels; measured acceptance-rate histogram; GPU utilization and achieved memory bandwidth from a profiler; KV-cache occupancy and admitted-concurrency delta caused by the resident draft weights; a task-quality metric to prove the acceptance rule is lossless.

Rollback gate. Disable speculation (a server flag, no redeploy) if p99 TPOT regresses, if aggregate tokens/sec drops at production concurrency, or if admitted concurrency falls because draft weights displaced KV cache."""

TRAIN_INFER = """How speculative decoding differs between training and inference.

Direct answer: speculative decoding is an inference-time-only execution strategy. It changes nothing about the loss, the gradients, or the parameters of either model. Training is involved only indirectly, when a draft model is trained or distilled to raise the acceptance rate.

Concrete mechanism. At inference in the memory-bound decode regime, one target forward costs roughly t_step = model_bytes_read / effective_HBM_bandwidth and is nearly independent of the number of token positions scored, because the weights stream once. Speculative decoding exploits that flat region: the draft autoregressively proposes gamma tokens, the target scores all gamma+1 positions in a single batched forward, and a rejection-sampling test accepts a prefix. With the standard acceptance rule the emitted distribution is provably identical to sampling from the target model alone, so the technique is a pure latency optimization with no quality delta to defend.

Why training does not need it. Training already scores every position of a sequence in one parallel forward via teacher forcing -- there is no sequential token-by-token dependency to hide, and the step is compute-bound, not bandwidth-bound. There is no idle bandwidth for speculation to convert into tokens, so applying it to a training step is meaningless. Anything resembling speculation during training is really draft-model distillation: minimizing KL between draft and target next-token distributions on on-policy target outputs, which raises E[a] later at inference.

Boundary condition. The training/inference boundary must not be crossed on tokenizers. Draft and target must share a vocabulary and tokenizer, otherwise the acceptance test is undefined and requires lossy re-tokenization. If the draft is fine-tuned separately after the target changes -- new quantization, new fine-tune, new system prompt -- the acceptance rate drifts and the measured speedup silently decays. Treat draft-target compatibility as a versioned pair, not two independent artifacts.

Second boundary condition. Any RL or preference-optimization stage that samples from the policy may use speculative decoding in its generation phase, but only with the exact lossless acceptance rule; a relaxed acceptance criterion biases the sampling distribution and therefore biases the gradient estimator.

Falsifiable hypotheses. (1) "Enabling speculation in the RL rollout phase leaves the reward curve statistically indistinguishable from the non-speculative baseline." (2) "Distilling the draft on target on-policy outputs raises E[a] from a to a' on production traffic."

Evidence required. Measured acceptance-rate histogram before and after any target-side change; a training-step profile showing the step is compute-bound (so speculation is inapplicable); paired quality/reward curves for any generation-phase use; explicit draft-target version pinning in the deployment manifest.

Rollback gate. Revert to non-speculative generation if the acceptance rate drops below the level at which speedup = E[a]/(1 + gamma*c_draft/c_target) exceeds 1.0, or if any reward/quality curve diverges from the baseline beyond its noise band."""

MISLEADING = """One misleading intuition about speculative decoding, and the correction.

Misleading intuition: "Speculative decoding makes the model faster, so it raises throughput -- adding it is strictly free, and a higher acceptance rate always means a bigger win."

Correction, part one: latency and throughput move in opposite directions under load. Speculative decoding does not make the target model compute faster; it exploits the fact that a memory-bound decode step wastes FLOPs. One target forward costs about t_step = model_bytes_read / effective_HBM_bandwidth and is nearly independent of how many token positions are scored, because the weights stream once. The draft proposes gamma tokens, the target verifies gamma+1 positions in a single forward, and a prefix is accepted. That converts idle bandwidth into emitted tokens at low batch. But at high concurrency the target is already compute-bound: the gamma extra positions per sequence consume real FLOPs, rejected drafts are pure waste, and aggregate tokens/sec can fall even while single-stream inter-token latency improves. "Faster" is true only for a specific batch regime and must always be stated with the concurrency level.

Correction, part two: acceptance rate alone does not determine the win. The expected speedup is approximately E[a] / (1 + gamma * c_draft/c_target). A draft that is large enough to reach a high acceptance rate can be so expensive that the denominator eats the gain. Both terms must be measured on production traffic.

Correction, part three: it is not free in memory. The draft's weights are resident alongside the target and reduce the KV-cache budget, which lowers maximum admitted concurrency; verification also needs KV slots for the speculated positions plus a rollback path for the rejected suffix. A latency win can therefore show up as a capacity loss.

Boundary condition. With the standard rejection-sampling acceptance rule, the emitted distribution is provably identical to sampling from the target alone, so there is no quality cost. With any relaxed or "lossy" acceptance criterion that claim is void and a task-level quality metric must be reported alongside the speedup.

Falsifiable hypothesis. "At production concurrency C, enabling speculation with gamma = g changes aggregate tokens/sec by no more than -2% while reducing p90 TPOT by >= X%."

Evidence required. Tokens/sec and TPOT p50/p90/p99 curves swept across concurrency, not a single-stream number; measured E[a] and c_draft/c_target; KV-cache occupancy and admitted-concurrency delta from resident draft weights; a task-quality metric whenever acceptance is not the exact lossless rule.

Rollback gate. Turn speculation off via server flag if aggregate tokens/sec regresses at production concurrency, if p99 TPOT widens, or if admitted concurrency falls below the capacity SLO."""

ANSWERS = {
    "assumptions": ASSUMPTIONS_CORE,
    "train_infer": TRAIN_INFER,
    "misleading": MISLEADING,
}

# per-source-id topic mapping (derived from the user prompt text itself, not from any other lane)
KIND = {
    "corpus-00478": "assumptions", "corpus-00479": "assumptions", "corpus-00480": "assumptions",
    "corpus-00481": "train_infer", "corpus-00482": "train_infer", "corpus-00483": "train_infer",
    "corpus-00484": "train_infer", "corpus-00485": "train_infer",
    "corpus-00487": "misleading", "corpus-00489": "misleading",
}

RISKS = [
    "Source answer is a two-clause definition; it states neither the assumptions the prompt asks for nor any boundary condition, so it fails instruction coverage.",
    "Source answer implies speedup is a property of the technique rather than of a measured (draft, target, workload, concurrency) tuple, which invites unsupported production performance claims.",
    "Source answer omits the memory cost of resident draft weights and speculated KV slots, so a reader may enable speculation and silently lose admitted concurrency.",
    "Source answer does not distinguish the lossless rejection-sampling acceptance rule from relaxed acceptance, so a quality regression could be reported as a pure systems win.",
]

EVIDENCE = [
    "TPOT/TTFT p50/p90/p99 distributions with and without speculation, swept across concurrency levels on the same server build.",
    "Measured acceptance-rate histogram E[a] and draft/target per-step cost ratio c_draft/c_target on production traffic.",
    "Profiler trace showing achieved HBM bandwidth and GPU compute utilization, to establish whether decode is memory-bound or compute-bound.",
    "KV-cache occupancy and maximum admitted concurrency before and after loading the draft model.",
    "Task-level quality metric whenever the acceptance rule is not the exact lossless rejection-sampling rule.",
]

def main():
    rows = []
    with open(CORPUS, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i < START:
                continue
            if i > END:
                break
            rows.append(json.loads(line))
    assert len(rows) == 10, len(rows)

    out = []
    for d in rows:
        sid = d["id"]
        msgs = {m["role"]: m["content"] for m in d["messages"]}
        user = msgs["user"]
        assistant = msgs["assistant"]
        kind = KIND[sid]
        rec = {
            "source_id": sid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": user,
            "source_assistant": assistant,
            "corrected_answer": ANSWERS[kind],
            "quality_dimensions": {
                "technical_correctness": 4,
                "instruction_coverage": 1,
                "operational_safety": 2,
            },
            "risks": RISKS,
            "evidence_required": EVIDENCE,
            "confidence": 0.82,
        }
        out.append(rec)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out))

if __name__ == "__main__":
    main()
