"""Contamination and overlap audit: corpus_v04 training data against benchmark.jsonl.

Contamination is not one thing. This separates four questions that get conflated:

  1. Item overlap   - does a training item reproduce a benchmark item?
  2. Answer leakage - do training answers carry the benchmark's expected content?
  3. Topic coverage - does training cover the same subject areas? (expected, not a fault)
  4. Process        - did benchmark items inform how the training data was authored?

Only (1) and (2) are measurable from the artifacts. (3) is reported because it
bounds what the benchmark can still measure. (4) is a provenance fact that no
string comparison can establish and is recorded separately in EXPERIMENT.md.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAIN = ROOT / "research/ai-infra-expert/corpus_v04/train.jsonl"
BENCH = ROOT / "research/ai-infra-expert/benchmark.jsonl"
V03 = ROOT / "research/ai-infra-expert/corpus/train.jsonl"

NGRAM = 5
STOP = set("""the a an and or of to in for on with that this is are be as by at from it its
into than then when where which while not no any all each per via using use used than""".split())

# Which v0.4 mechanisms address each benchmark topic. Authored by hand from the
# mechanism list; this is a judgement about subject matter, not a text measurement,
# and it is recorded here so a reviewer can disagree with a specific line.
DECLARED_COVERAGE = {
    "GPU memory hierarchy": ["decode_bandwidth_bound"],
    "HBM versus DDR": ["decode_bandwidth_bound"],
    "PCIe versus NVLink": ["tp_allreduce_cost", "collective_latency_floor"],
    "NVSwitch": ["tp_allreduce_cost"],
    "RDMA": ["rdma_property_bundle", "gdr_silent_staging"],
    "RoCE and InfiniBand": ["pfc_configuration_consistency", "ecmp_flow_collision"],
    "CUDA streams": [],
    "NCCL collectives": ["rendezvous_vs_datapath", "collective_tail_statistic", "straggler_amplification"],
    "Transformer attention": ["kv_capacity_ceiling"],
    "KV cache": ["kv_capacity_ceiling", "block_fragmentation", "max_len_overreservation"],
    "prefill": ["prefill_decode_interference", "sequence_parallel_prefill"],
    "tensor parallelism": ["tp_allreduce_cost", "tp_degree_memory_tradeoff", "shard_divisibility"],
    "continuous batching": ["prefill_decode_interference", "batched_cost_attribution"],
    "data parallelism": ["replica_vs_degree"],
    "quantization": ["weight_quant_regime", "kv_quant_quality_cost", "activation_outliers"],
    "speculative decoding": [],
    "MIG partitioning": ["multitenant_interference"],
    "CUDA Graphs": ["collective_latency_floor", "cold_start_measurement_bias"],
    "GQA": ["kv_capacity_ceiling"],
    "MoE": ["expert_parallel_alltoall"],
    "communication": ["tp_allreduce_cost", "collective_incast", "collective_tail_statistic"],
    "serve a 70B model on 8 GPUs": ["tp_degree_memory_tradeoff", "kv_capacity_ceiling"],
    "design multi-node inference over RoCE": ["pfc_configuration_consistency", "collective_incast"],
    "design long-context serving": ["max_len_overreservation", "sequence_parallel_prefill"],
    "design an MoE inference cluster": ["expert_parallel_alltoall"],
    "design a safe model rollout": ["rollout_stage_sizing", "canary_detection_power", "incomplete_rollback"],
    "optimize an agent inference service": ["agent_context_budget", "agent_reprefill_amplification",
                                            "agent_latency_budget"],
    "plan GPU capacity": ["kv_capacity_ceiling", "utilisation_target_conflict", "shape_fragmentation"],
    "debug distributed training startup": ["rendezvous_vs_datapath", "effective_batch_hidden"],
    "compare quantization deployment choices": ["weight_quant_regime", "dequant_batch_amortisation"],
    "build a benchmark harness": ["missing_baseline_band", "generation_cap_truncation",
                                  "cold_start_measurement_bias"],
    "compute KV-cache bytes": ["kv_capacity_ceiling"],
    "validate a tensor-parallel world size": ["shard_divisibility"],
    "estimate transfer time": ["cache_transfer_budget"],
    "detect duplicate request IDs": [],
    "parse a structured tool call": ["tool_call_trust_boundary", "constrained_decoding_scope"],
    "calculate paged KV blocks": ["block_fragmentation"],
    "classify prefill/decode workload": ["prefill_decode_interference"],
    "check NCCL environment completeness": ["rendezvous_vs_datapath"],
    "implement retry with bounded backoff": ["nested_retry_amplification", "tool_idempotency"],
    "aggregate latency percentiles": ["percentile_aggregation"],
}


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


def grams(text: str, n: int = NGRAM) -> set:
    t = toks(text)
    return {tuple(t[i:i + n]) for i in range(max(len(t) - n + 1, 0))}


def content_terms(text: str) -> set:
    return {w for w in toks(text) if len(w) > 2 and w not in STOP}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def best_match(target_grams, index, docs):
    """Highest-jaccard document sharing at least one gram with target."""
    cand = collections.Counter()
    for g in target_grams:
        for i in index.get(g, ()):
            cand[i] += 1
    best, best_i = 0.0, None
    for i in cand:
        j = jaccard(target_grams, docs[i])
        if j > best:
            best, best_i = j, i
    return best, best_i


def build_index(doc_grams):
    idx = collections.defaultdict(list)
    for i, gs in enumerate(doc_grams):
        for g in gs:
            idx[g].append(i)
    return idx


def main():
    train = [json.loads(l) for l in TRAIN.open() if l.strip()]
    bench = [json.loads(l) for l in BENCH.open() if l.strip()]

    tq = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in train]
    ta = [next(m["content"] for m in r["messages"] if m["role"] == "assistant") for r in train]
    bq = [r["question"] for r in bench]
    ba = [r["reference_answer"] for r in bench]

    print("=" * 72)
    print("0. BENCHMARK'S OWN COMPOSITION")
    print("=" * 72)
    print(f"benchmark records            : {len(bench)}")
    print(f"distinct questions           : {len(set(bq))}")
    print(f"distinct reference answers   : {len(set(ba))}")
    print(f"distinct topics              : {len(set(r['topic'] for r in bench))}")
    dup_ref = collections.Counter(ba)
    print(f"reference answers reused     : {sum(1 for v in dup_ref.values() if v > 1)} "
          f"covering {sum(v for v in dup_ref.values() if v > 1)} records")

    print()
    print("=" * 72)
    print("1. ITEM OVERLAP  (training item reproducing a benchmark item)")
    print("=" * 72)
    exact_q = set(q.strip() for q in tq) & set(q.strip() for q in bq)
    print(f"exact question matches       : {len(exact_q)}")

    bq_grams = [grams(q) for q in bq]
    bq_idx = build_index(bq_grams)
    q_scores = []
    for i, q in enumerate(tq):
        j, bi = best_match(grams(q), bq_idx, bq_grams)
        q_scores.append((j, i, bi))
    q_scores.sort(reverse=True)
    over = [s for s in q_scores if s[0] >= 0.30]
    print(f"questions with jaccard >=0.30: {len(over)} of {len(tq)}")
    print(f"max question jaccard         : {q_scores[0][0]:.3f}")
    for j, i, bi in q_scores[:3]:
        if bi is None:
            continue
        print(f"  {j:.3f}  train {train[i]['id']} ({train[i]['mechanism']})")
        print(f"         bench {bench[bi]['id']} [{bench[bi]['topic']}]")

    # a zero 5-gram score only means no shared word sequences; term overlap is the
    # weaker signal that shows whether the two are talking about the same thing
    bq_terms = [content_terms(q) for q in bq]
    term_scores = []
    for i, q in enumerate(tq):
        tt = content_terms(q)
        best, bi = 0.0, None
        for k, bt in enumerate(bq_terms):
            s = jaccard(tt, bt)
            if s > best:
                best, bi = s, k
        term_scores.append((best, i, bi))
    term_scores.sort(reverse=True)
    print(f"max question TERM overlap    : {term_scores[0][0]:.3f}")
    print(f"questions with term >=0.40   : {sum(1 for s in term_scores if s[0] >= 0.40)}")
    for j, i, bi in term_scores[:3]:
        print(f"  {j:.3f}  train {train[i]['id']} ({train[i]['mechanism']})")
        print(f"         bench {bench[bi]['id']} [{bench[bi]['topic']}]")

    print()
    print("=" * 72)
    print("2. ANSWER LEAKAGE  (training answer carrying benchmark expected content)")
    print("=" * 72)
    ba_grams = [grams(a) for a in ba]
    ba_idx = build_index(ba_grams)
    a_scores = []
    for i, a in enumerate(ta):
        j, bi = best_match(grams(a), ba_idx, ba_grams)
        a_scores.append((j, i, bi))
    a_scores.sort(reverse=True)
    print(f"answers with jaccard >=0.20  : {sum(1 for s in a_scores if s[0] >= 0.20)} of {len(ta)}")
    print(f"max answer jaccard           : {a_scores[0][0]:.3f}")
    for j, i, bi in a_scores[:3]:
        if bi is None:
            continue
        print(f"  {j:.3f}  train {train[i]['id']} ({train[i]['mechanism']})")
        print(f"         bench {bench[bi]['id']} [{bench[bi]['topic']}]")

    # key-point verifier items are the ones where verbatim phrase reuse would matter most
    kp = [r for r in bench if r.get("verifier") == "contains_key_points"]
    train_blob = "\n".join(ta).lower()
    leaked = []
    for r in kp:
        phrases = [p.strip() for p in re.split(r"[;.]", r["reference_answer"]) if len(p.strip()) > 40]
        for p in phrases:
            if p.lower() in train_blob:
                leaked.append((r["id"], p[:80]))
    print(f"key-point items              : {len(kp)}")
    print(f"verbatim key phrases in train: {len(leaked)}")
    for x in leaked[:5]:
        print("   ", x)

    print()
    print("=" * 72)
    print("3. TOPIC COVERAGE  (expected overlap; bounds what the benchmark can measure)")
    print("=" * 72)
    b_terms = collections.Counter()
    for r in bench:
        b_terms.update(content_terms(r["topic"]))
    t_terms = set()
    for r in train:
        t_terms |= content_terms(" ".join(r["concepts"]) + " " + r["mechanism"] + " " + r["category"])

    # Concept tags are too sparse to answer "is this subject present"; search the
    # full training text instead, requiring every content word of the topic.
    train_terms_per_record = [content_terms(q + " " + a + " " + " ".join(r["concepts"]))
                              for q, a, r in zip(tq, ta, train)]

    lexical = {}
    for topic in sorted(set(r["topic"] for r in bench)):
        tt = content_terms(topic)
        lexical[topic] = sum(1 for terms in train_terms_per_record if tt <= terms) if tt else 0
    print(f"benchmark topics                    : {len(lexical)}")
    print(f"  lexically present in training text: {sum(1 for v in lexical.values() if v)}")
    print("  NOTE: this is a weak proxy. It requires every content word of the topic")
    print("  phrase to co-occur in one record, so it marks 'design a safe model")
    print("  rollout' absent even though rollout mechanisms are present. Use the")
    print("  declared mapping below for the coverage judgement.")
    print()
    print("  Declared mapping (author judgement, not a measurement):")
    for topic, mechs in sorted(DECLARED_COVERAGE.items()):
        n = lexical.get(topic, 0)
        mark = "covered" if mechs else "NOT COVERED"
        print(f"    [{mark:11s}] {topic:42s} lexical={n:4d}  {', '.join(mechs) if mechs else '-'}")
    declared_covered = sum(1 for v in DECLARED_COVERAGE.values() if v)

    print()
    print("=" * 72)
    print("4. CROSS-CHECK AGAINST THE v0.3 CORPUS")
    print("=" * 72)
    if V03.exists():
        v03 = [json.loads(l) for l in V03.open() if l.strip()]
        v03q = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in v03]
        v03_norm = set(re.sub(r"scenario variant \d+|case variant \d+|case [ivxlcdm]+", "", q.lower()).strip()
                       for q in v03q)
        exact_v03 = sum(1 for q in tq if q.lower().strip() in v03_norm)
        print(f"v0.3 rows                    : {len(v03)}")
        print(f"v0.4 questions also in v0.3  : {exact_v03}")
        v03_idx = build_index([grams(q) for q in v03q[:2000]])
        v03_grams = [grams(q) for q in v03q[:2000]]
        top = 0.0
        for q in tq[::7]:
            j, _ = best_match(grams(q), v03_idx, v03_grams)
            top = max(top, j)
        print(f"max v0.4-to-v0.3 q jaccard   : {top:.3f} (sampled)")
    else:
        print("v0.3 corpus not present")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    item_clean = len(exact_q) == 0 and q_scores[0][0] < 0.50
    answer_clean = a_scores[0][0] < 0.30 and not leaked
    print(f"item overlap    : {'CLEAN' if item_clean else 'CONTAMINATED'} "
          f"(exact {len(exact_q)}, max jaccard {q_scores[0][0]:.3f})")
    print(f"answer leakage  : {'CLEAN' if answer_clean else 'CONTAMINATED'} "
          f"(max jaccard {a_scores[0][0]:.3f}, verbatim phrases {len(leaked)})")
    print(f"topic coverage  : {declared_covered}/{len(DECLARED_COVERAGE)} benchmark topics have "
          f"a corresponding training mechanism (declared)")
    print()
    print("Topic coverage is not contamination, but it does mean this benchmark")
    print("cannot measure generalisation to unseen subject areas. It can only")
    print("measure whether training on these mechanisms transfers to differently")
    print("worded questions about the same mechanisms.")
    print()
    print("Not measurable from artifacts: whether benchmark items informed authoring.")
    print("See EXPERIMENT.md for the provenance statement, which is the only")
    print("evidence available on that question.")
    return 0 if (item_clean and answer_clean) else 1


if __name__ == "__main__":
    sys.exit(main())
