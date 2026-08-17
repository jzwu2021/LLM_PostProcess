import json
CORPUS='research/ai-infra-expert/corpus/train.jsonl'
OUT='experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0030.jsonl'
START=290; N=10
rows=[json.loads(l) for l in open(CORPUS) if l.strip()][START:START+N]

A={}
common_risks=["Seed answer is a one-sentence definition and does not answer the asked task",
              "No units, no measurement methodology, no rollback threshold",
              "Risk of over-generalizing MoE gains across unrelated serving regimes"]
common_ev=["Per-token active-parameter count and total-parameter count for the compared dense/MoE pair",
           "Router top-k, expert count, expert-parallel degree, and capacity factor",
           "Measured TTFT/TPOT p50/p95/p99, throughput (tok/s), and GPU HBM high-water mark at fixed request rate",
           "All-to-all time share from NCCL/profiler traces (nsys or torch profiler)",
           "Quality parity check on a held-out task set to confirm the comparison is iso-quality"]

MEAS = ("Measurement plan (falsifiable, iso-quality):\n"
 "1. Assumptions to state up front: same tokenizer, same max context, same quantization/dtype, same batching policy (continuous batching), same KV-cache layout, same hardware topology. Declare which numbers are measured vs estimated.\n"
 "2. Baseline: pick a dense model whose *quality* matches the MoE candidate on a held-out eval, not one that merely matches total parameters. Otherwise any throughput win is a quality trade, not an MoE win.\n"
 "3. Mechanism under test: MoE cuts active FLOPs per token (active = shared + top-k experts), but adds (a) router compute, (b) an all-to-all (dispatch + combine) per MoE layer under expert parallelism, and (c) a memory-capacity cost, because all experts must be resident even though only top-k are used per token.\n"
 "4. Sweep: request rate from 10% to 120% of expected peak; input/output length mixes representative of production (e.g. 2k/256 and 8k/1k). Record TTFT and TPOT p50/p95/p99, throughput at a fixed SLO, tokens/s/GPU, HBM high-water mark, and the all-to-all share of step time.\n"
 "5. Boundary condition (the key one): at small batch / low concurrency the workload is memory-bandwidth- and latency-bound, not FLOP-bound, so MoE's reduced active FLOPs buys little while the all-to-all and weight-residency cost is paid in full — MoE typically loses there. The win appears only at batch sizes large enough that expert GEMMs are compute-bound and expert load is well balanced.\n"
 "6. Second boundary: routing imbalance. With capacity factor C, tokens beyond an expert's capacity are dropped or rerouted; measure the per-expert token histogram. If the max/mean expert load ratio is high, the slowest expert sets step time and the FLOP saving does not materialize.\n"
 "7. Falsifiable hypothesis: 'At iso-quality and at >= target concurrency, MoE gives >= X% higher tokens/s/GPU at equal p95 TTFT and p95 TPOT.' Reject if not met on two independent runs.\n"
 "8. Rollback gate: if p95 TTFT regresses beyond SLO, or HBM headroom drops below the level needed for peak KV cache, or expert-load max/mean exceeds the agreed bound, revert to the dense baseline. Keep the dense deployment routable behind the same endpoint until the gate passes.\n"
 "Caveat: do not quote vendor or paper speedups as if they were your measurements; they use different batch shapes and interconnects.")

ASSUM = ("Assumptions that must be stated before any MoE performance claim:\n"
 "1. What is held constant: quality. State the held-out eval and the score parity band. A throughput claim against a weaker dense baseline is not an MoE claim.\n"
 "2. Model configuration: total params, active params per token, number of experts, top-k, shared-expert presence, number of MoE vs dense layers, capacity factor, and the token-drop policy.\n"
 "3. Parallelism and topology: tensor-parallel, expert-parallel, pipeline degrees; how many GPUs per node; intra-node link (e.g. NVLink vs PCIe) and inter-node fabric (e.g. IB/RoCE) with its per-GPU bandwidth. All-to-all cost is dominated by the slowest hop, so an expert-parallel group that spans nodes behaves very differently from one that fits inside a node.\n"
 "4. Serving regime: batch size / concurrency, input and output length distribution, whether prefill and decode are colocated or disaggregated, scheduler policy, and whether the number quoted is at a fixed SLO or at saturation.\n"
 "5. Numerics: dtype/quantization for weights, activations and KV cache; whether the router runs in higher precision.\n"
 "6. Measurement hygiene: warmup discarded, steady state reached, run duration, number of repeats, percentile (not just mean), and client-side vs server-side timing.\n"
 "7. Which numbers are measured and which are estimated. Analytical active-FLOP savings are an upper bound, not a result.\n"
 "Mechanism: the claimed saving comes from executing only top-k experts per token; the offsetting costs are router overhead, dispatch/combine all-to-all, expert load imbalance, and full-weight HBM residency.\n"
 "Boundary condition: if expert-parallel groups cross the node boundary, all-to-all runs at inter-node fabric bandwidth and can dominate step time, erasing the FLOP saving; measure the all-to-all share before claiming anything.\n"
 "Falsifiable form and rollback: state the claim as 'iso-quality, at concurrency C and length mix L, tokens/s/GPU improves by >= X% with p95 TTFT/TPOT within SLO'; revert to dense if two independent runs fail it, or if HBM headroom for peak KV cache is lost.")

TRAINF = ("How MoE differs between training and inference:\n"
 "Shared mechanism: a router scores tokens and sends each to top-k experts; only those experts' FFN weights are used per token, so active FLOPs per token are far below total parameters. Under expert parallelism this becomes a dispatch all-to-all before the expert FFN and a combine all-to-all after it.\n"
 "Training-specific:\n"
 "- Auxiliary objectives exist: load-balancing / router-z losses (or loss-free bias correction) push the token distribution toward uniform across experts. These have no inference-time counterpart.\n"
 "- Capacity factor and token dropping are tuned for throughput; dropped tokens are a training-dynamics concern.\n"
 "- Backward pass adds a second pair of all-to-alls, and optimizer state plus gradients for *all* experts must be held, so memory is dominated by total (not active) parameters times optimizer-state multiplier.\n"
 "- Batches are large and token-dense, so expert GEMMs are comfortably compute-bound and imbalance amortizes over many tokens.\n"
 "Inference-specific:\n"
 "- Router is frozen; no balancing loss can fix skew, so real traffic can persistently overload a few experts. Prefill (many tokens at once) behaves like training and balances well; decode (few tokens per step) is where skew and latency hurt.\n"
 "- Decode is memory-bandwidth- and latency-bound: with one token per sequence per step, only a handful of experts are touched per step, yet all expert weights must be resident in HBM. This is the core asymmetry — inference pays total-parameter memory cost for active-parameter compute.\n"
 "- Token dropping is normally unacceptable at inference, so capacity must be effectively unbounded or overflow rerouted, which changes the cost model.\n"
 "- All-to-all latency is now on the critical path of every decode step, and small messages make it latency- rather than bandwidth-bound.\n"
 "Boundary condition: if the expert-parallel group spans nodes, per-step all-to-all latency (not bandwidth) can dominate decode; keeping the EP group inside one NVLink domain is usually the deciding factor.\n"
 "Evidence to collect: per-expert token histogram under real traffic, all-to-all time share in decode vs prefill, HBM high-water mark, and p95 TPOT.\n"
 "Falsifiable hypothesis and rollback: 'moving the EP group inside one node reduces decode all-to-all share below Y% and p95 TPOT below SLO'; if not reproduced twice, revert to the dense or lower-EP configuration.")

for r in rows:
    m={x['role']:x['content'] for x in r['messages']}
    u=m['user']
    if u.startswith('Give a measurement plan'): ans=MEAS
    elif u.startswith('What assumptions'): ans=ASSUM
    else: ans=TRAINF
    A[r['id']]=ans

out=[]
for r in rows:
    m=r['messages']
    u=[x['content'] for x in m if x['role']=='user'][0]
    a=[x['content'] for x in m if x['role']=='assistant'][0]
    out.append({
      "source_id": r['id'],
      "teacher_lane": "teacher-B",
      "teacher_model": "claude-opus-5-current",
      "calibration_status": "provisional",
      "decision": "rewrite",
      "source_user": u,
      "source_assistant": a,
      "corrected_answer": A[r['id']],
      "quality_dimensions": {"technical_correctness": 4, "instruction_coverage": 1, "operational_safety": 2},
      "risks": common_risks,
      "evidence_required": common_ev,
      "confidence": 0.78,
    })
with open(OUT,'w') as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
print("wrote",len(out),OUT)
