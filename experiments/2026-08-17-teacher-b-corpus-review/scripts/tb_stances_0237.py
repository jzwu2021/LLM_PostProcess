from tb_stances_0236 import COMMON_HEAD, COMMON_TAIL, BASE_RISKS, BASE_EV  # noqa: F401

S = [
 dict(v=102, f="Cost accounting: what a redundant call actually costs end to end.",
  body="""Before choosing an intervention, the cost of the defect must be expressed in the same units as the budget it competes for. A redundant calculator call consumes: (a) one decode of the tool-call token sequence, (b) one tool round-trip, (c) one extra prefill of the appended result on the current turn, and (d) an amortized prefill surcharge on every subsequent turn in the same trajectory, since the result stays in the context window.

Term (d) dominates and is the term teams routinely omit. A redundant call at turn 2 of an eight-turn trajectory is re-prefilled six more times unless prefix caching absorbs it — and prefix caching only absorbs it if the redundant call sits in a shared prefix, which it does not, because it is task-specific. State the accounting model explicitly, then instrument it, rather than quoting a per-call number.""",
  h="H102: the amortized re-prefill term (d) exceeds the sum of terms (a)-(c) for trajectories longer than roughly four turns (ESTIMATE; derivation: terms (a)-(c) are paid once, term (d) is paid once per remaining turn, so the crossover is near the turn index at which remaining turns exceed the fixed cost ratio; refuted if measured token-hours attributed to re-prefill of redundant results are below the one-time cost on the production trajectory-length distribution).",
  exp="Instrument the serving stack to emit prompt_tokens per turn with a flag marking tokens originating from calculator results, replay one hash-pinned task set, and attribute token-hours to each of the four terms separately. Compare against a counterfactual replay in which flagged redundant results are elided from the context.",
  gate="Rollback gate: no intervention is funded until the four-term attribution is published and the redundancy cost exceeds a pre-registered share of serving token-hours. If the share is below threshold, the defect is logged as cosmetic and closed.",
  risks=["quoting a per-call cost without the amortized re-prefill term understates the true cost and misprioritises the fix",
         "eliding results from a counterfactual replay changes model behavior downstream, so the counterfactual is an approximation and must be labelled as one"],
  ev=["per-turn prompt_token counts with provenance flags from the serving stack, not reconstructed from client logs",
      "the production distribution of trajectory turn counts, since the crossover point depends on it"],
  qd=[2,1,2], conf=0.62),

 dict(v=103, f="Prefix-cache interaction: why cache hit rate can move in the wrong direction.",
  body="""Any intervention that edits the system prompt or tool descriptions to discourage redundant calls changes the shared prefix. On a stack with prefix caching, that invalidates every cached prefix simultaneously at rollout. The immediate effect is a cache-hit-rate collapse and a prefill spike that looks exactly like a regression caused by the intervention itself.

The correct reading requires separating the transient from the steady state. Deploy the prompt change, then hold traffic steady and let the cache repopulate before reading any latency number. A p95 measured inside the repopulation window is an artifact of the rollout mechanism, not a property of the intervention.""",
  h="H103: p95 time-to-first-token rises immediately after a system-prompt edit and returns to within the pre-change noise band once cache occupancy recovers, so any latency regression read inside the repopulation window is an artifact (ESTIMATE; derivation: a prefix edit invalidates cached blocks keyed on that prefix, forcing full prefill until re-warm; refuted if p95 stays elevated after cache hit rate returns to its pre-change level).",
  exp="Roll the prompt change to one replica group. Record prefix-cache hit rate and p95 TTFT at one-minute resolution from before the change through full cache recovery. Declare the steady-state read only after hit rate has been within its pre-change band for a sustained window, and report both the transient and steady-state numbers.",
  gate="Rollback gate: revert only if p95 TTFT remains outside the pre-change noise band after cache hit rate has recovered. A latency alarm inside the repopulation window is explicitly not a rollback trigger and must be pre-registered as such.",
  risks=["auto-rollback automation keyed on p95 will fire during normal cache re-warm and mask a working intervention",
         "staged rollout across replicas with different cache states makes aggregate latency uninterpretable unless replicas are reported separately"],
  ev=["prefix-cache hit rate and p95 TTFT time series at one-minute resolution spanning the full re-warm",
      "the replica-group assignment map, so aggregate metrics can be decomposed"],
  qd=[2,1,2], conf=0.64),

 dict(v=104, f="Tool-schema surface as the primary lever, tested before any model change.",
  body="""The tool description is part of the prompt and is the cheapest reversible surface available. If the calculator's description reads as an unconditional capability advertisement, the policy has no textual signal about when not to call it. Adding an explicit precondition and a worked non-call example changes the conditional distribution without touching weights.

Order of attempts matters and should be pre-registered: (1) tool-description precondition, (2) system-prompt policy statement with a non-call exemplar, (3) decoding or structured-output constraints, (4) preference optimization on trajectories. Steps 1-3 are reversible within one deploy; step 4 is not, and must not be started until steps 1-3 are measured and exhausted.""",
  h="H104: a tool-description precondition plus one non-call exemplar reduces adjudicated redundant calls by a materially larger margin than any decoding-parameter change on the same task set (ESTIMATE; derivation: tool descriptions and exemplars enter the conditional context directly and shift the tool-call token distribution, whereas temperature and top-p rescale an already-peaked distribution; refuted if the prompt arm's reduction falls inside the decoding arm's confidence interval).",
  exp="Four arms on one hash-pinned replay with identical model version and serving stack: control, tool-description precondition, system-prompt exemplar, decoding change. Report adjudicated redundant-call rate and task_success_rate per arm with bootstrap confidence intervals, and pre-register the comparison before running.",
  gate="Rollback gate: no arm is promoted if task_success_rate drops below the control lower confidence bound, regardless of how large the redundancy reduction is. Preference optimization is not started while any reversible arm remains untested.",
  risks=["a non-call exemplar can generalise into refusing legitimate calls, converting a cost defect into a correctness defect",
         "tool-description edits are silently versioned in many stacks, so an unrecorded edit can contaminate later comparisons"],
  ev=["version-pinned tool schema text per arm, stored as an artifact rather than referenced by name",
      "per-arm task_success_rate with bootstrap confidence intervals, not point estimates"],
  qd=[2,1,2], conf=0.66),

 dict(v=105, f="Stratification: redundancy is a per-template property, not a global rate.",
  body="""A single global redundancy rate hides the structure that determines which intervention is correct. Stratify by prompt template, task family, trajectory length and turn index before designing anything. Two very different worlds produce the same global rate: uniform low-level redundancy across all templates, which implies a policy-level fix, versus concentration in a small number of templates, which implies a targeted prompt edit.

The distinction is decidable from the trace data already available and costs one query. Running a model-level intervention without making it is how teams spend weeks of training compute on a defect that was three lines of prompt text.""",
  h="H105: adjudicated redundant calls are concentrated such that a minority of prompt templates account for a majority of them, making a targeted template edit dominate any global intervention on cost-effectiveness (ESTIMATE; derivation: templates differ in whether they instruct the agent to verify arithmetic, which directly conditions the tool-call decision; refuted if the per-template redundancy distribution is approximately uniform).",
  exp="Group all flagged calls from one hash-pinned replay by template identifier, compute the concentration curve, and publish it before choosing an intervention class. Then edit only the top-contributing templates and re-replay, holding everything else pinned.",
  gate="Rollback gate: a global or model-level intervention is not approved while the concentration curve shows the top templates dominating and a targeted edit has not been attempted. Targeted edits revert if task_success_rate on the affected templates drops.",
  risks=["template identifiers are often missing from traces, and reconstructing them by string matching introduces misattribution",
         "fixing the top templates can shift traffic patterns so that the concentration curve is no longer valid for the next iteration"],
  ev=["per-template flagged-call counts and the concentration curve from a single pinned replay",
      "template identifier provenance: emitted by the caller, not inferred post hoc"],
  qd=[2,1,2], conf=0.65),

 dict(v=106, f="Multi-tenant serving effects: batching makes per-call latency a poor signal.",
  body="""Under continuous batching, the latency of any single request is a function of concurrent load, not only of its own token count. Removing redundant calls shortens sequences, which increases the number of sequences that fit in a batch, which changes queueing for everyone. The observable effect on p95 is therefore not the sum of saved per-call latencies and can even be negative in the short run as freed capacity is immediately consumed by admitted queued work.

Measure at the fleet level in throughput and KV-cache occupancy terms, and treat per-request latency deltas as a secondary, load-confounded signal.""",
  h="H106: removing redundant calls increases sustained request throughput at fixed p95 while leaving mean per-request latency approximately unchanged, because freed KV-cache capacity is absorbed by higher batch occupancy rather than by faster individual requests (ESTIMATE; derivation: continuous batching schedulers admit work up to a memory or latency bound, so freed capacity converts into admitted concurrency; refuted if throughput at fixed p95 does not move while mean latency drops).",
  exp="Run control and intervention arms as separate replica groups under an identical replayed load generator at matched offered load. Report sustained throughput at a fixed p95 target, KV-cache occupancy, batch size distribution and preemption or recompute counts per arm.",
  gate="Rollback gate: revert if preemption or recompute counts rise, since that indicates the scheduler is now operating against a memory bound and the apparent throughput gain is being paid for in tail instability.",
  risks=["comparing arms at different offered loads produces a throughput difference that is entirely an artifact of load, not of the intervention",
         "shared upstream dependencies between replica groups can couple the arms and break independence"],
  ev=["KV-cache occupancy, batch size distribution and preemption counts per replica group under matched offered load",
      "the load generator's offered-rate trace, to prove the arms were matched"],
  qd=[2,1,2], conf=0.63),

 dict(v=107, f="Held-out generalisation: proving the fix is not replay overfitting.",
  body="""Any intervention tuned while looking at a replay set will fit that set. The redundancy rate on the tuning set after several iterations of prompt editing is a training metric and must not be reported as the result.

Partition tasks into a tuning set and a sealed held-out set before the first edit, hash-pin both, and evaluate the held-out set exactly once at the end. If the held-out set is opened mid-programme, it is burned and a new one must be cut from unseen traffic. State the number of tuning iterations alongside the result, because it bounds how much optimism to expect.""",
  h="H107: redundancy reduction on the sealed held-out set is materially smaller than on the tuning set, with the gap widening in the number of tuning iterations (ESTIMATE; derivation: iterative prompt editing against a visible metric is optimisation against that sample, which is the standard mechanism of selection optimism; refuted if held-out and tuning reductions agree within their confidence intervals).",
  exp="Cut and hash-pin two disjoint task sets from the same traffic window before any edit. Iterate only on the tuning set, logging every iteration. Evaluate the held-out set once, report both numbers side by side with the iteration count, and do not iterate further on the held-out result.",
  gate="Rollback gate: the intervention is not promoted on a tuning-set result alone. If held-out reduction is inside the control noise band, the intervention is recorded as unproven regardless of tuning-set performance.",
  risks=["held-out sets cut from the same window as the tuning set share transient traffic characteristics and overstate generalisation to future traffic",
         "informal peeking at held-out results between iterations silently converts it into a second tuning set"],
  ev=["hash pins and cut timestamps for both task sets, recorded before the first edit",
      "a complete log of tuning iterations, so the optimism bound is stated rather than assumed"],
  qd=[2,1,2], conf=0.66),

 dict(v=108, f="Failure-mode inversion: monitoring for under-calling after the fix.",
  body="""Every suppression intervention has a symmetric failure mode: the agent stops calling the calculator when it should. This failure is quieter than the one being fixed, because it manifests as wrong answers rather than as extra spend, and wrong answers on arithmetic-bearing tasks are not always visible in aggregate success metrics if those metrics are lenient.

Instrument both directions from the start. Define an under-call metric — tasks whose ground truth requires computation but whose trajectory contains no calculator call — and alert on it with equal severity to the redundancy metric. A dashboard that only shows call-count going down is an instrument that cannot detect the fix failing.""",
  h="H108: at suppression strengths sufficient to materially reduce redundant calls, the under-call rate rises above its control band before task_success_rate shows a detectable drop, making under-call the earlier and more sensitive indicator (ESTIMATE; derivation: success metrics aggregate over many tasks and dilute a subpopulation regression, whereas under-call is measured directly on the affected subpopulation; refuted if success rate degrades first or if under-call stays flat while success drops).",
  exp="Sweep suppression strength across at least four levels on one hash-pinned replay. Plot redundant-call rate, under-call rate and task_success_rate against strength on a common axis, and identify which crosses its control band first.",
  gate="Rollback gate: revert immediately if under-call rate exits its control band, independent of the redundancy improvement and independent of whether task_success_rate has yet moved.",
  risks=["under-call is undefined without task-level ground truth about whether computation was required, and labelling that ground truth is itself a project",
         "lenient success graders mask arithmetic errors, so the success metric may never move even when the fix is harming correctness"],
  ev=["per-task labels for whether computation is required, produced independently of the intervention",
      "the suppression-strength sweep with all three metrics on a common axis and their control bands drawn"],
  qd=[2,1,2], conf=0.67),

 dict(v=109, f="Attribution discipline: separating model change from runtime change.",
  body="""Interventions in this space fall into two categories that must never be reported together: runtime and system changes such as prompts, tool schemas, decoding parameters, caching and scheduling; and model changes such as fine-tuning or preference optimization. A metric that moves after a prompt edit says nothing about the model's capability, and a metric that moves after a training run says nothing until the runtime is held fixed across the comparison.

The reporting rule is mechanical: every result carries a label naming which category changed, and a comparison spanning both categories is invalid and must be re-run. This is the single discipline that most often separates a real capability claim from a serving-configuration artifact.""",
  h="H109: a substantial share of redundancy reduction attributed to model-level training in loosely controlled comparisons is in fact produced by concurrent runtime changes, and vanishes when the runtime is pinned across arms (ESTIMATE; derivation: training rollouts in practice ship alongside prompt, template and serving-version updates, so the arms differ in more than one variable; refuted if pinning the runtime leaves the training arm's advantage inside its original confidence interval).",
  exp="Re-run the training arm against control with model version as the only differing variable: identical system prompt, tool schema, decoding parameters, serving version and hash-pinned task set. Report the delta before and after pinning, and publish both.",
  gate="Rollback gate: no capability claim is published from a comparison in which more than one variable moved. Such a comparison is re-run with the runtime pinned or the claim is withdrawn.",
  risks=["serving-version drift between arms is invisible unless the version is captured as an artifact per run",
         "reporting a combined delta as a model capability gain is the specific error this discipline exists to prevent"],
  ev=["per-arm manifest capturing model version, system prompt hash, tool schema hash, decoding parameters and serving version",
      "the paired before-and-after-pinning deltas with confidence intervals"],
  qd=[2,1,2], conf=0.68),

 dict(v=110, f="Preference-optimization arm: preconditions, data construction and reversibility.",
  body="""If and only if the reversible arms are exhausted and the defect still exceeds its cost threshold, a preference-optimization arm may be opened. Its preconditions are strict: an adjudicated redundancy detector with known precision, a sealed held-out set, a defined under-call metric, and a checkpoint restore path proven by an actual restore, not by the presence of files on disk.

Pair construction is where this arm usually fails. Preferred and rejected trajectories must differ only in the redundant call, holding the final answer correct in both; otherwise the objective learns answer style rather than call discipline. Construct pairs by ablating the redundant call from a correct trajectory and re-running from that point, then discard any pair whose final answer changed.""",
  h="H110: preference pairs constructed without holding final-answer correctness fixed shift answer style measurably while leaving adjudicated redundancy near its control band, i.e. the objective optimises the wrong difference (ESTIMATE; derivation: the objective learns whatever systematically separates preferred from rejected sequences, and answer-text differences are a stronger and more consistent signal than a single omitted call; refuted if a style-uncontrolled pair set yields redundancy reduction indistinguishable from the controlled pair set).",
  exp="Build two pair sets from the same trajectories, one with final answers held identical and one without, train matched runs from the same base checkpoint with identical hyperparameters and seed, and evaluate both on the sealed held-out set for redundancy, under-call and answer-length distribution.",
  gate="Rollback gate: the training arm is abandoned and the base checkpoint restored if held-out under-call exits its band, if answer-length distribution shifts beyond a pre-registered bound, or if the restore path has not been exercised end to end beforehand.",
  risks=["ablating a call and re-running changes downstream context, so ablated trajectories are not drawn from the deployed policy's distribution and carry a distribution shift",
         "a training arm without a rehearsed restore path converts a reversible experiment into an outage"],
  ev=["the pair-construction script and the count of pairs discarded for answer change, reported as a fraction",
      "a restore-drill log showing the base checkpoint was actually reloaded and served, with output equivalence checked"],
  qd=[2,1,2], conf=0.64),

 dict(v=111, f="Programme close-out: what may and may not be claimed at the end.",
  body="""The deliverable of this programme is a defensible statement, and the shape of that statement is fixed in advance. Permitted: a redundancy reduction on a sealed held-out set, at a stated suppression strength, with the runtime pinned, with under-call inside its band, with confidence intervals, and with the intervention category named. Not permitted: any claim that the agent 'reasons better', 'uses tools more intelligently', or that domain capability improved.

Write the close-out template before running the experiments and fill it in mechanically. Pre-committing to the claim shape is what prevents a serving-configuration result from being narrated as a capability result once the numbers are in hand.""",
  h="H111: the final defensible claim will be narrower than the claim stated in the programme's original proposal, specifically by being scoped to one intervention category and one held-out task set (ESTIMATE; derivation: proposals are written before the confounders are enumerated, and each confounder identified during the programme removes scope from the claim; refuted if the pre-registered close-out template can be filled with the original proposal's claim unmodified).",
  exp="Write and hash-pin the close-out template before the first arm runs. At close-out, fill it strictly from artifacts, and diff the filled template against the original proposal text. Publish the diff alongside the result.",
  gate="Rollback gate: the programme does not close while any field in the template is unfillable from artifacts. An unfillable field is reported as an open gap, never as an inference.",
  risks=["retrospective narrative expansion, in which a narrow measured result is described in capability language in the summary while the artifacts remain narrow",
         "close-out templates written after results are known are reverse-engineered to fit those results and provide no discipline"],
  ev=["the hash-pinned close-out template with its pre-run timestamp",
      "the diff between the filled template and the original proposal claim"],
  qd=[2,1,2], conf=0.67),
]
