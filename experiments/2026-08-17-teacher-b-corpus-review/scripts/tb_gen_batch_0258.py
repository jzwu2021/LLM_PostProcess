import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0258.jsonl"
START, END = 2570, 2580

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
 ("STANCE 151 - Plan the rollback before the placement change, because remapping experts to devices requires a reshard and a restart rather than a configuration flip.",
  """Mechanism. Expert placement determines which weights are resident on which device. Changing it means moving weights, rebuilding the communication groups and restarting the serving process, so the change is a deployment with drain, cold start and warm-up rather than a live parameter update. The rollback has exactly the same cost, which means an unplanned revert during an incident is slow precisely when speed matters most.

Falsifiable hypothesis. H1: the measured time to revert a placement change, including drain, reshard, restart and warm-up to steady-state latency, exceeds the incident response objective for this service. Falsified if the measured revert completes within that objective, which would make placement changes operationally routine.

Metrics. Measured drain time, reshard and weight-load time, restart time, warm-up time to steady-state p99, total revert time, capacity lost during the operation, error-budget consumed by the operation, and the standard balance set of per-device load, token-drop rate and p99 step time. All revert timings are MEASURED by rehearsal; a revert time quoted from component estimates is an ESTIMATE and must be labelled.

Controlled experiment. Rehearse the full revert in staging at production scale and again on one production replica during a low-traffic window, timing each phase separately so the dominant term is identified rather than assumed. Keep the prior placement artifact and its topology snapshot available for the entire validation period rather than deleting it after the forward rollout succeeds.

Confounders. Staging clusters are often smaller and load weights from warmer caches, understating both reshard and warm-up. Weight-load time depends on storage contention, which varies with concurrent deployments. Warm-up duration depends on traffic mix at the moment of restart.

Rollback criteria. Do not begin a placement rollout whose rehearsed revert exceeds the response objective; reduce the blast radius until it does, for example by staging one replica at a time. The prior placement must remain a deployable versioned artifact, and the rollout is blocked if that artifact cannot be produced on demand."""),
 ("STANCE 152 - Treat any change in expert-parallel size as invalidating the tuned placement, because the expert-to-device map is a function of the group size.",
  """Mechanism. Placement assigns experts to ranks within an expert-parallel group. Autoscaling, node failure or a capacity change that alters the group size renumbers those ranks and changes which experts share a device. A placement tuned for one size therefore becomes an arbitrary mapping at another, and the co-location structure that reduced inter-node traffic is silently destroyed. The system continues to run, so nothing signals that the tuning is no longer in effect.

Falsifiable hypothesis. H1: after a change in expert-parallel size, the inter-node all-to-all byte share returns toward its untuned value, demonstrating that the tuned co-location did not survive the resize. Falsified if the byte share is preserved across sizes, which would mean the placement policy generalises and needs no re-derivation.

Metrics. Inter-node and intra-node all-to-all byte share before and after a resize, per-device load dispersion, p99 step time, expert-to-device map recorded per size, time since last placement derivation, and the size at which the current placement was derived. The byte shares are MEASURED from fabric counters at each size; any assumption that a placement transfers is an ESTIMATE requiring this check.

Controlled experiment. Deliberately resize the expert-parallel group in a staging deployment while holding checkpoint, traffic and capacity fixed, and measure the byte share and dispersion before and after. Then repeat with a placement re-derived at the new size to quantify how much of the loss is recoverable by re-derivation.

Confounders. A resize usually coincides with a restart, so warm-up effects overlap the measurement and must be excluded. Traffic mix at the time of an autoscaling event is by definition unusual, since load changed. Some frameworks re-derive placement automatically on startup, which would mask the effect and must be checked before attributing anything.

Rollback criteria. Block automatic scaling of the expert-parallel dimension until placement re-derivation is part of the startup path, or pin the size and scale on another axis. Record the derivation size with the placement artifact so a mismatch between the running size and the derivation size is detectable rather than silent."""),
 ("STANCE 153 - Check whether the hot experts are also the worst quantised, because per-expert quantisation error and per-expert load are independent and their interaction is where quality is lost.",
  """Mechanism. Post-training quantisation computes scales per tensor or per channel, and each expert receives its own. Experts differ in weight distribution, so quantisation error differs across them. Load differs across experts for unrelated reasons. If a heavily used expert happens to be one with large quantisation error, a disproportionate share of tokens is processed through the worst-approximated weights, and aggregate quality degrades more than a uniform error analysis would predict.

Falsifiable hypothesis. H1: per-expert quantisation error is correlated with per-expert token share, so the load-weighted error materially exceeds the unweighted mean error. Falsified if the two are uncorrelated and the load-weighted error matches the unweighted mean, which would justify treating quantisation error as uniform across experts.

Metrics. Per-expert quantisation error measured against the unquantised weights, per-expert token share, load-weighted versus unweighted mean error, fixed eval-set quality for the quantised and unquantised models, per-expert activation range and outlier incidence, and token-drop rate. Per-expert errors and shares are MEASURED; the predicted quality impact of their correlation is an ESTIMATE until a mixed-precision arm is run.

Controlled experiment. Compute per-expert error directly by comparing quantised and unquantised weights, join it to the measured token share from a replayed window, and test the correlation. Then run a mixed-precision arm keeping only the highest load-weighted-error experts at higher precision, holding everything else fixed, and measure the quality and memory trade.

Confounders. Quantisation error in weight space is only a proxy for output error, which also depends on activation distributions and must be checked with an activation-aware measure. Calibration data determines the scales, so an unrepresentative calibration set creates error that is an artifact of calibration rather than of the expert. Mixed precision may not be supported by the deployed kernels.

Rollback criteria. Revert to uniform precision if the mixed-precision arm fails to improve fixed eval-set quality beyond the noise band, since heterogeneous precision complicates every subsequent kernel and memory decision. Because a precision change alters the served artifact, its rollback is a model rollback and the prior artifact must remain deployable."""),
 ("STANCE 154 - Consider which tenants are co-scheduled, because tenants have different routing profiles and mixing them can flatten or sharpen the load.",
  """Mechanism. Routing is driven by content, and tenants differ systematically in content. Two tenants whose traffic activates disjoint expert sets will, when co-batched, produce a flatter aggregate distribution than either alone; two tenants sharing a hot expert will concentrate load further. Co-scheduling is therefore an active control on balance that operates without touching the model, and it is currently being exercised implicitly by whatever the scheduler happens to do.

Falsifiable hypothesis. H1: per-expert load dispersion in mixed-tenant microbatches differs materially from the dispersion predicted by combining single-tenant profiles independently, showing that co-scheduling composition matters beyond simple addition. Falsified if mixed dispersion matches the independent prediction, which would make tenant mixing neutral and remove it from the intervention set.

Metrics. Per-tenant expert profile over a long window, pairwise profile overlap between tenants, dispersion in single-tenant versus mixed microbatches, p99 step time by microbatch composition, per-tenant end-to-end latency and fairness, token-drop rate, and admitted concurrency per tenant. Profiles and dispersion are MEASURED per tenant; any co-scheduling policy recommendation is an ESTIMATE until evaluated end to end.

Controlled experiment. Compute per-tenant profiles, then replay arrival streams under a scheduler that groups complementary tenants and under the current scheduler, holding capacity, placement and parallelism fixed. Report per-tenant latency alongside aggregate dispersion, since a policy that improves balance by delaying one tenant's requests is a fairness regression rather than an optimisation.

Confounders. Tenant profiles drift as their applications change, so a grouping derived once decays. Small tenants provide too few tokens to estimate a stable profile. Grouping by tenant can leak information about co-tenancy through timing, which is a privacy consideration independent of performance.

Rollback criteria. Revert any co-scheduling policy that increases p99 latency or queue wait for any tenant beyond its pre-declared guardrail, regardless of the aggregate balance improvement. The grouping policy must be a single revertible configuration and must be re-derived on a schedule rather than assumed stable."""),
 ("STANCE 155 - Choose the objective before tuning capacity, because throughput and tail latency move in opposite directions along the capacity axis.",
  """Mechanism. Raising the capacity factor reduces drops and improves quality but processes more padded slots, lowering useful throughput and lengthening steps. Lowering it does the reverse. There is no setting that optimises quality, throughput and tail latency simultaneously, so tuning proceeds by an implicit objective that is rarely stated. Different teams then reach different settings from the same data and the disagreement is mistaken for a technical dispute.

Falsifiable hypothesis. H1: over the feasible capacity range the settings that minimise p99 step time and that maximise useful tokens per GPU-second are different, so no single setting is jointly optimal. Falsified if the two optima coincide, which would remove the trade and make the choice unambiguous.

Metrics. Useful tokens per GPU-second, p99 step time, token-drop rate and fixed eval-set quality across a capacity sweep, with the declared objective and its acceptance threshold recorded before the sweep. All four series are MEASURED on the same replayed stream; the selected operating point is a decision recorded with its objective rather than an ESTIMATE presented as an optimum.

Controlled experiment. Sweep capacity across the feasible range on a fixed stream, holding placement, parallelism and batch policy constant, and plot all four series on the same axis so the trade is visible. Pre-declare the objective and the constraint set, for example maximise throughput subject to a quality floor and a p99 ceiling, before examining the curves.

Confounders. Quality effects of drops may be below eval-set resolution over part of the range, leaving the quality constraint unenforceable exactly where it matters. Throughput measured with padding included, rather than useful throughput, inverts the apparent trade. The feasible range is bounded by memory, so part of the curve may be unreachable.

Rollback criteria. Do not change the operating point without recording the objective under which it was chosen, so a later reviewer can falsify the choice rather than relitigate it. Revert to the prior capacity if the quality floor or the p99 ceiling is breached at production load, since both were declared as constraints rather than as trade terms."""),
 ("STANCE 156 - Re-derive placement and capacity after every checkpoint change, because a fine-tune moves the router and therefore moves the load.",
  """Mechanism. Placement and capacity are tuned against a measured routing distribution. Any further training, including a small fine-tune or a preference pass, updates the router weights and shifts which experts are hot. The serving configuration is not updated by that process, so a model upgrade silently pairs new routing with an old, now mismatched, physical layout. Nothing fails, and the regression appears only as a slow degradation in tail latency after a model release.

Falsifiable hypothesis. H1: the per-expert load distribution differs materially between two consecutive checkpoints on the same replayed token stream, so the placement tuned for the earlier one is no longer aligned. Falsified if the distributions agree within the noise band, which would allow placement to be carried across checkpoints in this lineage.

Metrics. Per-expert load distribution per checkpoint on a fixed stream, divergence between consecutive checkpoints, inter-node byte share and p99 step time under the old placement with the new checkpoint, the same under a re-derived placement, and the checkpoint identifier recorded with every placement artifact. Distributions are MEASURED on identical input so the checkpoint is the only varying factor.

Controlled experiment. Replay one fixed token stream through both checkpoints, compare distributions, then evaluate the new checkpoint under both the inherited and the re-derived placement to quantify the cost of not re-deriving. This isolates the configuration effect from the model-quality effect of the upgrade itself.

Confounders. A model upgrade usually coincides with other changes such as a new serving build or tokenizer, so the comparison must hold those fixed. Traffic mix at release time differs from the tuning window. Router changes may be concentrated in a few layers, so a pooled divergence understates a severe per-layer shift.

Rollback criteria. Block a checkpoint promotion until placement and capacity have been re-derived and evaluated, or explicitly accept the inherited configuration with the measured cost recorded. Bind the placement artifact to the checkpoint hash so a mismatch is detectable at startup rather than discovered through latency drift."""),
 ("STANCE 157 - Stratify routing statistics by context position, because expert selection is content-dependent and long sequences do not route like short ones.",
  """Mechanism. The router acts on hidden states that depend on position and on accumulated context. Early tokens of a prompt, tokens deep in a long document, and tokens generated during decode occupy different regions of representation space and therefore select different experts. A deployment whose traffic shifts toward longer contexts will see its load distribution move even with an unchanged router and an unchanged tenant mix, and a capacity setting tuned on short traffic will be wrong for long.

Falsifiable hypothesis. H1: per-expert load distributions differ materially between early-position and late-position tokens within the same sequences, so context length changes the aggregate distribution independently of tenant or task. Falsified if distributions are position-invariant, which would let context length be ignored as a scoping axis.

Metrics. Per-expert load distribution by position bucket, divergence between early and late buckets, sequence-length distribution of production traffic, dispersion and drop rate for long-context versus short-context microbatches, p99 step time by sequence-length stratum, and fixed eval-set quality on long-context slices. Position-bucketed distributions are MEASURED from a single capture so buckets share the same input population.

Controlled experiment. Capture routing for sequences spanning the production length range and compute distributions per position bucket within the same sequences, which controls for content by construction. Then compare capacity settings tuned on short-context traffic against settings tuned on long-context traffic, evaluating each on both strata.

Confounders. Long sequences are not a random sample of content, since documents and code differ from chat, so position and content are correlated and the within-sequence comparison is the only clean control. Chunked prefill splits long sequences across microbatches, changing the observed composition. Attention sinks and other position-specific mechanisms can dominate early positions.

Rollback criteria. Revert to the prior capacity setting if a long-context-tuned configuration regresses short-context p99 or quality, and record which length strata each setting was validated on. Treat a sustained shift in the production sequence-length distribution as a trigger to re-derive capacity rather than as a routine traffic change."""),
 ("STANCE 158 - Set capacity against the smallest expected microbatch, because relative skew is largest exactly when the token count is smallest.",
  """Mechanism. Capacity is expressed as a multiple of the average tokens per expert in a microbatch. When a microbatch is large the realised counts concentrate near that average and a modest factor suffices. When it is small, sampling variation dominates and one expert can easily receive several times the mean, overflowing a capacity that would be ample at scale. Tuning against average or peak-load microbatches therefore guarantees drops during the low-load periods that operators consider safe.

Falsifiable hypothesis. H1: the capacity factor required to hold the drop rate below its threshold is materially higher for the smallest microbatches observed in production than for the largest, so a single factor tuned at high load under-provisions at low load. Falsified if the required factor is flat across microbatch sizes, which would allow a single setting tuned anywhere.

Metrics. Required capacity factor as a function of microbatch token count to hold a fixed drop threshold, realised microbatch size distribution over a full daily cycle, drop rate by microbatch size bucket, useful-token fraction by bucket, p99 step time by bucket, and memory headroom at the largest required factor. The required-factor curve is MEASURED by sweeping capacity within each size bucket rather than extrapolated from the aggregate.

Controlled experiment. Bucket microbatches by token count from a full-cycle replay, sweep capacity within each bucket, and derive the required factor per bucket. Then check whether the serving stack can vary capacity dynamically with microbatch size; if it cannot, the binding setting is the one required by the smallest bucket and that constraint must be stated explicitly.

Confounders. Small microbatches occur during low load, when latency headroom is largest, so the operational cost of drops there may be lower and the threshold arguably different. Batch size and composition are correlated, since low load also changes which tenants are active. Padding waste at high capacity is worst precisely in small microbatches.

Rollback criteria. Do not tune capacity on a peak-hour window alone; require the full-cycle sweep before changing the setting. If the smallest-bucket requirement is unaffordable in memory or padding terms, record the accepted drop rate at low load as an explicit decision with its measured quality effect rather than leaving it as an unnoticed consequence."""),
 ("STANCE 159 - Pre-register the deciding metric and require the offline counterfactual before any production change, because this system offers too many defensible metrics to choose after the fact.",
  """Mechanism. A single capacity or placement change moves drop rate, useful-token fraction, dispersion, all-to-all time, step-time quantiles, end-to-end latency and eval-set quality, and these do not move together. With that many candidate outcomes, a change can almost always be described as an improvement by selecting the metric afterwards. Naming the deciding metric and its threshold in advance, and requiring the cheap offline prediction first, converts the decision into something that can fail.

Falsifiable hypothesis. H1: the pre-registered deciding metric moves past its declared threshold in the production arm, and the offline counterfactual predicted that movement within tolerance. Falsified if the deciding metric fails its threshold, or if the offline prediction and the realised value disagree beyond tolerance, either of which blocks the change.

Metrics. The single pre-registered deciding metric with its threshold, the full secondary set covering drop rate, useful-token fraction, dispersion, all-to-all time, step-time quantiles, end-to-end p99 and eval-set quality, the offline prediction for each, and the prediction error. Predictions are ESTIMATE and the realised values are MEASURED, and both must appear in the record so the pair can be checked later.

Controlled experiment. Register the deciding metric, threshold and analysis plan before running anything, generate the offline counterfactual from recorded routing scores, and only then run the production arm. Report every secondary metric including those that moved unfavourably, since selective reporting of the secondary set reintroduces the bias the registration was meant to remove.

Confounders. A deciding metric chosen without understanding the trade may be the wrong one, so the registration must state the objective it serves rather than the metric alone. Offline prediction quality varies by policy type and is weakest exactly for policies that change scheduling. Threshold values set without a measured quality anchor are arbitrary.

Rollback criteria. Revert the change if the deciding metric fails, irrespective of favourable secondary movement, and record the failure rather than re-registering a different metric on the same data. If offline and realised values disagree beyond tolerance, revert and repair the offline model before attempting further policy changes, since its predictions gate the cheap path."""),
 ("STANCE 160 - Report per-layer drop rates jointly, because a token dropped at one layer alters the representation entering the next and the effects compound rather than add.",
  """Mechanism. Each MoE layer routes independently, and a dropped token bypasses its expert, typically passing through a residual path unchanged. The representation entering the next layer therefore differs from what the model was trained to expect, which changes that layer's routing decision and its own drop probability. Reporting a single fleet-wide drop rate averages over a sequential process in which errors propagate, and it cannot distinguish one severe layer from uniform mild dropping.

Falsifiable hypothesis. H1: tokens dropped at an early MoE layer have a higher probability of being dropped again at later layers than the marginal rate implies, indicating positive dependence rather than independence across layers. Falsified if per-token drop events are independent across layers, which would make the aggregate rate a sufficient summary.

Metrics. Per-layer drop rate, joint distribution of drop counts per token across layers, conditional drop probability at layer L+1 given a drop at layer L, share of tokens dropped in more than one layer, fixed eval-set quality as a function of multi-layer drop share, and p99 step time. The joint distribution is MEASURED by tagging drop events per token per layer rather than aggregating counts per layer independently.

Controlled experiment. Instrument dispatch to record drop events keyed by token identity and layer, then compute the conditional probabilities directly on a replayed window. Compare quality on the subset of sequences containing multi-layer drops against length-matched sequences without them, to test whether compounding is where the quality effect concentrates.

Confounders. Token identity must survive across layers for the join, which some implementations do not preserve and which must be added carefully so as not to perturb the hot path. Sequences with multi-layer drops may be atypical in content, so length matching alone is insufficient and content stratification is needed. Residual paths differ across architectures, changing how much a drop actually perturbs the next layer.

Rollback criteria. Do not set a capacity target from an aggregate drop rate once dependence is demonstrated; re-derive it against the multi-layer drop share. If per-token drop tagging perturbs step time beyond the noise band, disable it and fall back to per-layer aggregates, recording that the dependence structure is then unmeasured rather than assumed absent."""),
]

DECISIONS = ["rewrite"] * 10

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

CONF = [0.74,0.73,0.7,0.71,0.73,0.74,0.71,0.72,0.73,0.7]

RISKS = [
 ["Source answer proposes placement changes without accounting for reshard and restart in the rollback path.",
  "Staging clusters are smaller and load weights from warmer caches, understating reshard and warm-up time.",
  "An unplanned revert during an incident is slowest exactly when response time matters most."],
 ["Source answer treats placement as durable without binding it to the expert-parallel size it was derived at.",
  "A resize coincides with a restart, so warm-up effects overlap the measurement.",
  "Frameworks that re-derive placement automatically on startup would mask the effect and change the attribution."],
 ["Source answer does not consider that per-expert quantisation error and per-expert load are independent and interact.",
  "Weight-space quantisation error is only a proxy for output error and needs an activation-aware measure.",
  "An unrepresentative calibration set creates error that is an artifact of calibration rather than of the expert."],
 ["Source answer treats batch composition without recognising that tenant mixing is an active control on balance.",
  "A co-scheduling policy that improves balance by delaying one tenant is a fairness regression rather than an optimisation.",
  "Tenant routing profiles drift as their applications change, so a grouping derived once decays."],
 ["Source answer compares capacity policies without declaring the objective the comparison is meant to serve.",
  "Throughput measured with padding included rather than useful throughput inverts the apparent trade.",
  "Quality effects of drops may fall below eval-set resolution exactly where the quality constraint should bind."],
 ["Source answer treats the serving configuration as independent of the checkpoint that routes the tokens.",
  "A model upgrade usually coincides with a new serving build or tokenizer, confounding the comparison.",
  "Router shifts concentrated in a few layers are understated by a pooled divergence figure."],
 ["Source answer measures routing distribution without stratifying by context position or sequence length.",
  "Long sequences are not a random sample of content, so position and content are correlated.",
  "Chunked prefill splits long sequences across microbatches and changes the observed composition."],
 ["Source answer tunes capacity against typical load, where relative skew is smallest.",
  "Small microbatches occur at low load, so batch size and tenant composition are correlated.",
  "Padding waste at a high capacity factor is worst precisely in the small microbatches that require it."],
 ["Source answer permits a change to be justified by whichever of many metrics happens to move favourably.",
  "Offline prediction quality is weakest exactly for policies that change scheduling.",
  "A threshold set without a measured quality anchor is arbitrary and cannot fail meaningfully."],
 ["Source answer reports token dropping as a single rate, hiding whether drops compound across layers.",
  "Token identity must survive across layers for the join, which some implementations do not preserve.",
  "Sequences containing multi-layer drops may be atypical in content, so length matching alone is insufficient."],
]

EVID = [
 ["Rehearsed full revert timed by phase covering drain, reshard, weight load, restart and warm-up to steady-state p99.",
  "Prior placement retained as a deployable versioned artifact with its topology snapshot for the entire validation period."],
 ["Inter-node and intra-node all-to-all byte share measured before and after a deliberate expert-parallel resize at fixed checkpoint and traffic.",
  "Placement re-derived at the new size and compared against the inherited one, with the derivation size recorded on the artifact."],
 ["Per-expert quantisation error computed against unquantised weights and joined to measured per-expert token share.",
  "Mixed-precision arm keeping the highest load-weighted-error experts at higher precision, with fixed eval-set quality and memory recorded."],
 ["Per-tenant expert profiles over a long window with pairwise overlap and dispersion in single-tenant versus mixed microbatches.",
  "Per-tenant end-to-end latency and queue wait reported alongside aggregate dispersion under each scheduling policy."],
 ["Useful tokens per GPU-second, p99 step time, drop rate and fixed eval-set quality across a capacity sweep on one stream.",
  "Objective, constraint set and acceptance thresholds recorded before the sweep, with the selected operating point logged against them."],
 ["Per-expert load distributions for two consecutive checkpoints on one fixed token stream with serving build held constant.",
  "New checkpoint evaluated under both inherited and re-derived placement, with the placement artifact bound to the checkpoint hash."],
 ["Per-expert load distributions by position bucket computed within the same sequences to control content by construction.",
  "Capacity settings tuned on short-context and long-context traffic each evaluated on both strata, with the sequence-length distribution recorded."],
 ["Required capacity factor per microbatch-size bucket derived by sweeping capacity within each bucket over a full daily cycle.",
  "Realised microbatch size distribution with drop rate, useful-token fraction and memory headroom at the largest required factor."],
 ["Pre-registered deciding metric, threshold and analysis plan recorded before any arm is run.",
  "Offline counterfactual prediction and realised value reported together for every primary and secondary metric, including unfavourable ones."],
 ["Per-token per-layer drop tagging giving the joint drop distribution and the conditional probability at layer L+1 given a drop at layer L.",
  "Fixed eval-set quality compared between sequences with multi-layer drops and content-stratified length-matched sequences without them."],
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
