import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0255.jsonl"
START, END = 2540, 2550

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
 ("STANCE 121 - Report routing entropy alongside the load histogram, because a skewed distribution and a low-entropy collapsed router are different defects with different fixes.",
  """Mechanism. The router produces a distribution over experts per token. Skew in the aggregate token counts can arise either because the input distribution genuinely concentrates on a few experts while the router remains confident and diverse per token, or because the router has collapsed and assigns nearly every token to the same small set regardless of input. The histogram looks similar in both cases, but the first is a data property and the second is a learned-parameter defect that capacity and placement cannot repair.

Falsifiable hypothesis. H1: per-token routing entropy is materially below the entropy implied by the aggregate load histogram, indicating collapse rather than input concentration. Falsified if per-token entropy is high while aggregate counts are skewed, which localises the cause in the input distribution and makes placement and capacity the correct levers.

Metrics. Mean and quantiles of per-token routing entropy, aggregate per-expert token counts, the entropy of the aggregate histogram for contrast, gate probability of the top-1 expert, fraction of tokens whose top-1 and top-2 gate probabilities are within a small margin, token-drop rate, all-to-all time per layer, and p99 step time. Entropies and counts are MEASURED from router logits captured in a fixed replay; any inferred cause is an ESTIMATE until a controlled arm is run.

Controlled experiment. Capture router logits for a fixed replayed token stream at a pinned checkpoint, computing per-token entropy and aggregate counts from the same capture so the two cannot diverge through sampling. Then repeat the capture on a deliberately different input mix; collapse predicts the load pattern is stable across mixes, while input concentration predicts it moves with the mix.

Confounders. Entropy is layer-dependent, so pooling across layers hides collapse confined to a few layers and each layer must be reported separately. Capturing logits adds memory traffic and can perturb step time, so behavioural claims require the profiler-inertness check. Numerical precision of the gate softmax affects entropy at the low end.

Rollback criteria. Do not spend on placement or capacity tuning if the measurement indicates collapse, since neither addresses a learned router; escalate to a training-side change with the layer indices attached. Any logit-capture instrumentation must be revertible by a single flag and must be disabled before performance measurements are taken."""),
 ("STANCE 122 - Separate what can be changed at serving time from what is frozen in the weights: the router was shaped by a training-time balancing objective and cannot be rebalanced by configuration.",
  """Mechanism. Load balance in a sparse MoE is largely determined during training by the auxiliary balancing loss and its coefficient. At serving time the reachable levers are capacity factor, placement, parallelism degree and batch composition. None of these alter which expert the router prefers for a given token; they only change what happens after the assignment. Proposals that promise to rebalance load at serving time are therefore either changing the drop pattern or changing which device is congested, and those must not be described as fixing routing.

Falsifiable hypothesis. H1: sweeping every available serving-side lever leaves the per-expert token assignment distribution unchanged within the noise band, confirming that assignment is frozen in the weights. Falsified if any serving lever measurably shifts the assignment distribution, which would identify a configuration-dependent routing path worth investigating in its own right.

Metrics. Per-expert assignment distribution under each serving-lever setting, token-drop and padding rates, all-to-all time, per-device busy time, p99 step time, and eval-set quality on a fixed set. Assignment distributions are MEASURED before capacity truncation is applied, since measuring after truncation confuses assignment with admission.

Controlled experiment. Hold the checkpoint and token stream fixed and vary capacity factor, expert placement and parallelism degree one at a time, recording the pre-truncation assignment distribution at each setting. Any observed change indicates the assignment path depends on configuration, for example through batch-composition effects on a batched router implementation, which is itself a finding.

Confounders. Measuring assignment after capacity truncation makes drops look like routing changes. Batch composition varies with the scheduler and shifts which tokens are present, changing the aggregate distribution without changing per-token behaviour. Numerics differences across kernels flip borderline tokens.

Rollback criteria. Reject any serving-side proposal presented as a routing fix and reclassify it as a drop-pattern or placement change with its own metrics. If a training-side rebalance is undertaken, the prior checkpoint remains the revert target and must stay loadable, since a rebalanced router is a new model requiring full quality re-evaluation rather than a configuration rollback."""),
 ("STANCE 123 - Count per-slot rather than per-token when top-k is greater than one, because a token occupying two experts contributes twice to load and to all-to-all volume.",
  """Mechanism. With top-k routing each token is dispatched to k experts, so the quantity that fills expert capacity and crosses the interconnect is the number of token-expert slots, not the number of tokens. Reporting per-token counts understates offered load by roughly a factor of k and, more importantly, hides that a change in k alters both capacity pressure and all-to-all bytes simultaneously. Comparisons across k that use token counts are therefore not comparing like with like.

Falsifiable hypothesis. H1: expressed per slot, the offered load per expert at the deployed k is close to k times the per-token figure, and the capacity overflow threshold is reached at a lower nominal capacity factor than the per-token accounting suggests. Falsified if slot and token accounting coincide, which would indicate k is effectively one for most tokens and the distinction is immaterial here.

Metrics. Token-expert slots per expert per microbatch, tokens per expert for contrast, slots per token realised, all-to-all bytes per layer, token-drop rate computed on slots, per-device busy time, p99 step time, and fixed eval-set quality. Slot counts and all-to-all bytes are MEASURED from dispatch instrumentation; any projected effect of changing k is an ESTIMATE until that arm is run.

Controlled experiment. Instrument the dispatch to emit slot counts directly rather than deriving them, then compare k as deployed against an adjacent value on a fixed token stream, holding capacity factor, placement and parallelism constant. Report quality and all-to-all volume together, since reducing k reduces communication but changes the function being computed.

Confounders. Some implementations renormalise gate weights after top-k selection, so changing k changes the combination weights as well as the routing breadth. Dropped slots reduce measured all-to-all volume, making a congested configuration look cheap. Capacity is often specified per expert per token, which silently rescales when k changes.

Rollback criteria. Revert a change in k if fixed eval-set quality declines, regardless of the communication saving, since k is a model-architecture parameter and not a tuning knob. Any capacity factor tuned under the previous k must be re-derived rather than carried over, and the prior k and capacity pair must be recorded as a single revertible configuration unit."""),
 ("STANCE 124 - Consider expert-choice routing before tuning token-choice capacity, because it makes balance structural at the cost of dropping the guarantee that every token is served.",
  """Mechanism. Token-choice routing lets each token select experts and then enforces balance by truncating at capacity, so imbalance appears as dropped tokens. Expert-choice routing inverts the selection: each expert takes a fixed number of the tokens that scored highest for it, which makes per-expert load exactly uniform by construction. The imbalance does not disappear; it changes form, because now some tokens are selected by no expert and receive only the residual path.

Falsifiable hypothesis. H1: switching to expert-choice eliminates capacity overflow entirely while producing a non-zero rate of unserved tokens, and fixed eval-set quality changes by less than the pre-declared tolerance. Falsified if quality degrades beyond that tolerance, which would mean the uniform-load benefit is paid for in model behaviour and token-choice with tuned capacity is preferable.

Metrics. Per-expert load dispersion under each scheme, capacity overflow rate, unserved-token rate, all-to-all time and bytes, p99 step time, per-device busy time, and fixed eval-set quality with per-slice breakdown. Load dispersion and overflow are MEASURED; the quality comparison must use the same eval set and decode configuration under both schemes or it is an ESTIMATE of unknown validity.

Controlled experiment. Replay one fixed token stream through both routing schemes at matched effective capacity, so the comparison is at equal compute rather than equal nominal parameters. Report the distribution of unserved tokens across sequence positions and token types, since uniform aggregate rates can hide that unserved tokens concentrate on a particular class of input.

Confounders. Expert-choice requires global visibility of scores within a batch, so its behaviour depends on batch composition and it degrades toward token-choice at very small batch sizes such as single-sequence decode. Matched effective capacity is nontrivial to define and must be stated explicitly. Training-time and inference-time routing schemes must match, so this is not a serving-only switch unless the model was trained accordingly.

Rollback criteria. Do not deploy expert-choice at inference for a model trained under token-choice; treat that as a model change requiring full re-evaluation rather than a configuration change. Revert if unserved tokens concentrate on any identifiable input class, since a uniformly small rate that is concentrated is a targeted quality regression."""),
 ("STANCE 125 - Measure expert utilisation over a long window as a cumulative distribution, because dead experts are invisible in any single batch and represent stranded capacity.",
  """Mechanism. Per-batch histograms answer whether load was balanced at that moment. They cannot answer whether an expert is never used, because a single batch is a small sample over a large expert count. A permanently underused expert consumes memory, participates in every all-to-all and contributes nothing, which is a different problem from transient skew: it is stranded capacity and a signal that model capacity is being wasted.

Falsifiable hypothesis. H1: over a long window, a non-trivial fraction of experts receive a share of tokens far below uniform, and that set is stable across disjoint windows rather than rotating. Falsified if the low-utilisation set is not stable across windows, which indicates transient skew rather than dead experts and redirects the fix to capacity and scheduling.

Metrics. Cumulative distribution of per-expert token share over a long window, per-layer count of experts below a pre-declared share threshold, set overlap of the low-utilisation experts between disjoint windows, memory occupied by low-utilisation experts, all-to-all bytes attributable to them, and fixed eval-set quality. Utilisation shares are MEASURED by accumulating dispatch counters, which is far cheaper than logging per-token routing.

Controlled experiment. Accumulate per-expert counters over two disjoint long windows drawn from comparable traffic and compare both the distributions and the identity of the low-utilisation set. Where the set is stable, ablate those experts on a fixed eval set to quantify their contribution before drawing any conclusion about pruning or merging.

Confounders. Traffic mix shifts between windows and can move the low-utilisation set for reasons unrelated to the router, so the windows must be matched on mix. Counter resets during deployment silently truncate a window. A low-utilisation expert may still be essential for a rare but important input class, which aggregate quality metrics will not reveal.

Rollback criteria. Do not prune or merge experts on utilisation evidence alone; require an ablation showing no degradation on the rare-class slices before any structural change, and retain the prior weights as the revert target. Because pruning changes the artifact, revert is a model rollback rather than a configuration change and must be planned as such."""),
 ("STANCE 126 - Scope every balance measurement to a traffic mix, because routing is input-dependent and a placement tuned on one mix will be wrong when the mix shifts.",
  """Mechanism. Which experts are hot is a function of the input distribution: language, domain, prompt template and task type all shift the routing pattern. Placement and capacity tuned against one mix encode that mix into the serving configuration. When traffic changes, whether diurnally, by campaign, or by onboarding a new tenant, the tuned configuration becomes actively wrong and the tail degrades without any change to the model or the fleet.

Falsifiable hypothesis. H1: per-expert load distributions differ materially between two contrasting traffic mixes drawn from production, so a placement optimised for one is not optimal for the other. Falsified if the distributions agree within the noise band across mixes, which would make a single static placement adequate and remove mix as a scoping axis.

Metrics. Per-expert load distribution per mix with the mix definition recorded, divergence between mixes, p99 step time under each mix with placement held fixed, per-device busy time, all-to-all time, token-drop rate, and fixed eval-set quality. Distributions are MEASURED per mix; any claim that a placement is optimal is an ESTIMATE scoped to the mixes on which it was measured.

Controlled experiment. Select two production windows with deliberately contrasting composition, replay each through the same checkpoint and placement, and compare load distributions and tail step times. Then optimise placement for one mix and evaluate it on both, reporting the cross-mix regression explicitly rather than only the in-mix gain.

Confounders. Windows differ in load level as well as in composition, so the comparison must control offered rate. Sequence length differs by mix and changes the ratio of prefill to decode tokens, which have different routing statistics. A mix defined by tenant identity conflates content differences with usage-pattern differences.

Rollback criteria. Revert to the prior placement if the cross-mix evaluation shows a regression beyond the pre-declared tolerance on any production mix, even when the target mix improves. Placement must be recorded as a versioned artifact bound to the mixes on which it was validated, and a mix shift outside that set is a trigger to re-validate rather than to assume transfer."""),
 ("STANCE 127 - State the expert-parallelism degree with every measurement, because it determines all-to-all size and therefore how much a given imbalance costs.",
  """Mechanism. Expert parallelism shards experts across devices, and the all-to-all exchanges tokens between every pair of ranks in the expert-parallel group. Increasing the degree spreads experts more widely, which reduces per-device memory and can improve balance granularity, but it enlarges the collective and increases sensitivity to the slowest participant. The same expert-level imbalance therefore has a different cost at different degrees, and a result quoted without the degree is unscoped.

Falsifiable hypothesis. H1: holding the checkpoint, token stream and placement policy fixed, p99 step time as a function of expert-parallel degree is non-monotonic, with a minimum at an intermediate degree rather than at the extremes. Falsified if step time is monotonic across the swept range, which would make the degree a straightforward capacity trade rather than a tuning decision.

Metrics. p99 and median step time, all-to-all time and bytes per rank, rank wait time at the collective, per-device memory footprint, per-device busy time, achieved interconnect bandwidth, token-drop rate, and fixed eval-set quality, each reported per expert-parallel degree. Bandwidth and wait times are MEASURED from profiler and fabric counters; any extrapolation beyond the swept degrees is an ESTIMATE.

Controlled experiment. Sweep the degree over the feasible range on a fixed replayed token stream, holding batch composition, capacity factor and placement policy constant, and record both compute and communication terms separately so the trade is visible rather than summarised in step time alone. Repeat at two offered load levels, since the optimum shifts with batching.

Confounders. Changing the degree changes per-device memory, which can alter the maximum batch size and therefore batch composition, introducing a second difference. Topology is not uniform, so a degree that fits within a node behaves very differently from one that spans nodes. Some frameworks change kernel selection with the degree.

Rollback criteria. Revert to the recorded prior degree if p99 step time or eval-set quality regresses, and treat degree changes as requiring a full restart and reshard rather than a live reconfiguration. Because the degree constrains the placement space, any placement tuned at one degree must be re-derived after a degree change rather than carried across."""),
 ("STANCE 128 - Separate rank wait time from transfer time in the all-to-all, because waiting for a straggler and moving bytes have different causes and different remedies.",
  """Mechanism. A collective's wall time at a rank is the sum of time spent blocked waiting for peers to arrive and time spent actually moving data. Imbalance in expert load produces the first; insufficient bandwidth or excessive volume produces the second. Both appear as a large all-to-all time in a coarse profile, and the two point to opposite interventions: rebalancing placement versus reducing bytes or improving the fabric. Without the split, the team will optimise the wrong term.

Falsifiable hypothesis. H1: at the deployed configuration the majority of all-to-all wall time at the median rank is wait rather than transfer, identifying imbalance rather than bandwidth as the binding constraint. Falsified if transfer dominates, which redirects the work to volume reduction and fabric capacity and makes placement changes low-value.

Metrics. Per-rank wait time and transfer time at each all-to-all, their ratio, all-to-all bytes per rank, achieved versus theoretical link bandwidth, per-device busy time before the collective, arrival-time spread across ranks, p99 step time, and token-drop rate. Wait and transfer are MEASURED by instrumenting a barrier immediately before the collective so arrival spread is observable directly rather than inferred.

Controlled experiment. Insert an explicit synchronisation point before the all-to-all in a diagnostic build and record per-rank arrival times, then compute wait as the gap to the last arrival and transfer as the remainder. Confirm the added barrier is inert by comparing step time with and without it, because the barrier itself changes overlap behaviour and must not be left enabled during performance measurement.

Confounders. The added barrier serialises what may otherwise be overlapped, so absolute step times under diagnosis are not comparable to production. Fabric contention from co-tenants inflates transfer time independently of this workload. Clock skew across ranks corrupts arrival-spread measurement unless timestamps are taken on a synchronised source.

Rollback criteria. Remove the diagnostic barrier before any capacity or latency claim is made, and re-measure without it. Do not fund fabric upgrades or placement changes until the split is measured, and record which term dominated as the justification for whichever intervention is chosen, so a later re-measurement can falsify it."""),
 ("STANCE 129 - Check whether the all-to-all is overlapped with compute before treating imbalance as a latency problem, because a fully overlapped collective may hide it entirely.",
  """Mechanism. Modern implementations attempt to overlap expert communication with computation on other layers or microbatches. When overlap succeeds, moderate imbalance is absorbed into slack and does not appear in step time at all. When it fails, the collective is exposed and every millisecond of imbalance is on the critical path. Whether a given imbalance matters is therefore a property of the schedule, not of the histogram, and the schedule is directly observable.

Falsifiable hypothesis. H1: at the deployed configuration the all-to-all is substantially exposed rather than overlapped, so reducing imbalance would reduce step time roughly one for one. Falsified if the collective is already well overlapped, in which case reducing imbalance yields little step-time improvement and the effort should move to the exposed critical path instead.

Metrics. Exposed communication time as a fraction of step time, overlap efficiency, stream and kernel timeline occupancy, per-device compute idle time during the collective, all-to-all time, p99 step time, and token-drop rate. Exposure is MEASURED from a kernel timeline trace rather than inferred by subtracting compute from step time, since that subtraction assumes perfect serialisation.

Controlled experiment. Capture a kernel-level timeline for a fixed replayed stream and measure exposed versus overlapped communication directly. Then artificially perturb imbalance, for example by a deliberately skewed placement, and confirm that step time responds only in proportion to the exposed fraction; a response larger than that indicates the overlap model is wrong.

Confounders. Tracing perturbs scheduling and can itself break overlap, so the inertness of the tracer must be checked before its output is trusted. Overlap depends on microbatch count and on whether the framework uses separate streams, both of which change with configuration. Power and clock throttling alters compute duration and thus the available overlap window.

Rollback criteria. Do not fund placement or routing work on the basis of all-to-all wall time alone when the collective is overlapped; record the exposed fraction as the gating measurement. Any change made to increase overlap, such as raising microbatch count, must be reverted if it increases memory footprint beyond headroom or degrades p99 through longer pipelines."""),
 ("STANCE 130 - Prioritise intra-node placement over inter-node balance, because the fabric is hierarchical and a token crossing the node boundary costs far more than one that does not.",
  """Mechanism. Interconnect is not uniform. Within a node, devices communicate over a high-bandwidth link; between nodes they traverse a network with an order of magnitude less bandwidth and higher latency. An all-to-all whose traffic is dominated by inter-node hops will be limited by the slower tier regardless of how evenly tokens are spread across experts. Placement that co-locates frequently co-activated experts within a node therefore reduces the binding term even when it makes the expert-level histogram look less uniform.

Falsifiable hypothesis. H1: inter-node all-to-all bytes dominate the collective's transfer time, and a placement that reduces inter-node traffic lowers p99 step time even though it increases per-expert load dispersion. Falsified if step time follows load dispersion rather than inter-node volume, which would mean the fabric is not the binding constraint and uniform spreading is correct.

Metrics. All-to-all bytes split by intra-node and inter-node paths, transfer time per tier, achieved bandwidth per tier, per-expert load dispersion, per-device busy time, rank wait time, p99 step time, and fixed eval-set quality. The byte split and per-tier bandwidth are MEASURED from fabric and NVLink counters; any predicted gain from a proposed placement is an ESTIMATE until the placement is run.

Controlled experiment. Measure the co-activation structure of experts on a fixed token stream, then construct a placement that co-locates high co-activation pairs within nodes and compare it against the current placement, holding the router, capacity factor, parallelism degree and batch composition fixed. Report both the dispersion increase and the inter-node byte reduction so the trade is explicit.

Confounders. Co-activation is mix-dependent, so a placement derived from one traffic window inherits that window's structure. Node-local bandwidth is shared among the devices in the node, so concentrating traffic intra-node can saturate the local link. Topology discovery can be wrong after maintenance, making the assumed hierarchy incorrect.

Rollback criteria. Revert to the recorded prior placement if p99 step time regresses or if intra-node link utilisation saturates, and re-derive co-activation after any traffic-mix shift rather than assuming the structure persists. Placement must be a versioned artifact with the topology snapshot it was derived against attached, so a topology change invalidates it explicitly."""),
]

DECISIONS = ["rewrite"] * 10

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

CONF = [0.72,0.74,0.73,0.7,0.71,0.73,0.72,0.74,0.71,0.7]

RISKS = [
 ["Source answer asks for the routing distribution without distinguishing input concentration from a collapsed router.",
  "Entropy pooled across layers hides collapse confined to a few layers.",
  "Capturing router logits adds memory traffic and can perturb the step times being measured."],
 ["Source answer lists routing and capacity policies together without separating frozen weights from serving configuration.",
  "Measuring assignment after capacity truncation makes drops look like routing changes.",
  "A rebalanced router is a new model requiring full quality re-evaluation, not a configuration rollback."],
 ["Source answer counts tokens where the quantity that fills capacity and crosses the fabric is token-expert slots.",
  "Gate renormalisation after top-k means changing k also changes combination weights, not just breadth.",
  "Dropped slots reduce measured all-to-all volume, making a congested configuration appear cheap."],
 ["Source answer treats routing policy comparison without noting that expert-choice moves the failure from drops to unserved tokens.",
  "Expert-choice depends on batch composition and degrades toward token-choice at decode-time batch sizes.",
  "A uniformly small unserved-token rate can still concentrate on one input class and become a targeted regression."],
 ["Source answer measures routing distribution per batch, where permanently unused experts are invisible.",
  "Traffic-mix shifts between windows can move the low-utilisation set for reasons unrelated to the router.",
  "A rarely used expert may still be essential for an important minority input class."],
 ["Source answer requires balance measurements without scoping them to the traffic mix that produced them.",
  "Windows differ in offered load as well as composition, confounding mix with rate.",
  "Sequence length differs by mix and changes the prefill-to-decode token ratio, which have different routing statistics."],
 ["Source answer treats all-to-all time as a single quantity without recording the expert-parallel degree that sets its size.",
  "Changing the degree changes per-device memory and therefore the feasible batch size, introducing a second difference.",
  "A degree that fits within a node behaves very differently from one that spans nodes."],
 ["Source answer measures all-to-all time as one number, conflating waiting for stragglers with moving bytes.",
  "An inserted synchronisation barrier serialises otherwise overlapped work, so diagnostic step times are not production step times.",
  "Clock skew across ranks corrupts arrival-spread measurement unless timestamps share a synchronised source."],
 ["Source answer assumes all-to-all time is on the critical path without checking whether it is overlapped with compute.",
  "Tracing perturbs scheduling and can itself break the overlap being measured.",
  "Overlap depends on microbatch count and stream structure, both of which change with configuration."],
 ["Source answer treats placement as a balance problem without accounting for the hierarchical fabric.",
  "Concentrating traffic intra-node can saturate the shared node-local link.",
  "Expert co-activation structure is mix-dependent, so a placement inherits the window it was derived from."],
]

EVID = [
 ["Per-token routing entropy and aggregate per-expert counts computed from a single router-logit capture at a pinned checkpoint.",
  "Repeat capture on a deliberately different input mix, reported per layer, with a profiler-inertness check before any behavioural claim."],
 ["Pre-truncation per-expert assignment distribution recorded at each serving-lever setting on a fixed token stream and checkpoint.",
  "One-factor-at-a-time sweep of capacity factor, placement and parallelism degree with the assignment distribution compared against the noise band."],
 ["Token-expert slot counts emitted directly by dispatch instrumentation rather than derived from token counts.",
  "All-to-all bytes and fixed eval-set quality reported together for the deployed k and an adjacent k on the same token stream."],
 ["Per-expert load dispersion, capacity overflow rate and unserved-token rate under both routing schemes at matched effective capacity.",
  "Distribution of unserved tokens across sequence positions and token types, with eval-set quality measured under identical decode configuration."],
 ["Cumulative per-expert token-share distribution accumulated from dispatch counters over two disjoint long windows matched on traffic mix.",
  "Set overlap of low-utilisation experts between windows, plus an ablation of those experts on rare-class eval slices before any structural change."],
 ["Per-expert load distributions for two contrasting production mixes replayed through the same checkpoint and placement at controlled offered rate.",
  "Cross-mix evaluation of a mix-optimised placement reporting the regression on the non-target mix alongside the in-mix gain."],
 ["p99 and median step time, all-to-all bytes, rank wait time and per-device memory recorded at each swept expert-parallel degree.",
  "Compute and communication terms reported separately at two offered load levels with batch composition, capacity factor and placement held constant."],
 ["Per-rank arrival times captured at an explicit pre-collective synchronisation point in a diagnostic build.",
  "Wait-versus-transfer split with achieved link bandwidth, plus a step-time comparison with and without the barrier to establish its cost."],
 ["Kernel-level timeline trace giving exposed versus overlapped communication as a fraction of step time.",
  "Step-time response to a deliberately perturbed placement checked against the exposed fraction, with tracer inertness verified first."],
 ["All-to-all bytes and transfer time split by intra-node and inter-node path from NVLink and fabric counters.",
  "Expert co-activation structure measured on a fixed token stream, with the candidate placement compared at fixed router, capacity, degree and batch composition."],
]

rows = []
for i, s in enumerate(src):
    m = {x["role"]: x["content"] for x in s["messages"]}
    head, body = STANCES[i]
    tc, ic, os_ = QD[i]
    rows.append({
        "source_id": s["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": DECISIONS[i],
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": head + "\n\n" + body,
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": RISKS[i],
        "evidence_required": EVID[i],
        "confidence": CONF[i],
    })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(rows))
