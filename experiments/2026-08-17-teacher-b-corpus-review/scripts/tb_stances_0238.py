COMMON_HEAD = (
    "Assumptions (state before measuring): the agent has a deterministic tool-call channel with "
    "per-call logging (call id, arguments, latency, result, and whether the result changed the final "
    "answer); 'answer already known' is operationalised as the answer being derivable from the current "
    "context window without arithmetic beyond what the model reliably does token-internally; and the "
    "serving stack (vLLM/SGLang-style continuous batching behind a Dynamo-style router) is unchanged "
    "across arms so tool-call count is the only manipulated variable."
)

COMMON_TAIL = (
    "Rollback gate (shared): revert the intervention if final-answer correctness on the held-out agentic "
    "eval drops by more than 1.0 absolute point, or if p99 end-to-end trajectory latency regresses more "
    "than 10% versus the control arm, measured over at least 2 consecutive canary hours. All numeric "
    "targets below are ESTIMATE unless explicitly tagged MEASURED; ESTIMATE values are derived from the "
    "stated arithmetic on assumed baselines and must be replaced with MEASURED values from the canary "
    "before any promotion decision. This review output is provisional teacher-B commentary, not expert "
    "gold, and says nothing about any model's domain capability."
)

BASE_RISKS = [
    "Suppressing tool calls can silently convert a verifiable computation into an unverified model guess.",
    "Redundancy metrics computed only on successful trajectories bias the estimate downward (survivorship).",
    "Reward shaping against tool use can generalise into under-calling on genuinely hard arithmetic.",
]

BASE_EV = [
    "Per-call trace log with call arguments, latency, and a counterfactual flag for whether removing the call changes the final answer.",
    "Paired control/treatment run on the same prompt set with fixed decoding seed and identical serving config.",
    "Held-out arithmetic-heavy slice to detect under-calling regressions.",
]

S = [
    dict(v=112, f="defining 'unnecessary call' counterfactually rather than heuristically",
         body=("Mechanism: an unnecessary call is not 'a call whose result already appears in context'; that heuristic "
               "over-fires on re-verification. Define it counterfactually: replay the trajectory with the call's result "
               "replaced by a no-op and the tool description unchanged; if the final answer is unchanged under 3 seeds, "
               "the call is redundant. Metrics: redundant-call rate (redundant calls / total calls), calls-per-solved-task, "
               "tool success rate, final correctness, trajectory token length, mean and p99 tool latency, and post-tool-error "
               "recovery rate. Boundary condition: counterfactual replay is only valid when the tool is side-effect free; "
               "a calculator qualifies, a write-capable tool does not."),
         h="H1: at least 30% (ESTIMATE, derived from assuming ~1 in 3 calls is a re-verification of an in-context value) of calculator calls are counterfactually redundant; a no-tool-needed stop action reduces calls-per-task by >=20% (ESTIMATE) with final correctness within 1.0 point.",
         exp=("two arms on a frozen 1k-task agentic set, control = current policy, treatment = policy plus an explicit "
              "'answer_directly' action with the same logit treatment as tool actions. Fixed seeds, identical KV-cache and "
              "batching settings, 3 repetitions per arm to bound seed variance."),
         gate="Promotion gate: treatment ships only if redundant-call rate drops >=15 points absolute AND correctness delta >= -1.0 point at 95% CI.",
         qd=(3, 3, 3), conf=0.62,
         risks=["Counterfactual replay is invalid for any tool with side effects; misapplying it corrupts the label set."],
         ev=["Side-effect audit of every tool in the action space before replay-based labelling."]),

    dict(v=114, f="separating decision error from execution error in the trajectory",
         body=("Mechanism: redundant calling has two distinct causes — (a) the policy cannot tell that the value is already "
               "resolved (decision error), and (b) the policy distrusts its own earlier tool result and re-runs it (execution "
               "distrust). These need different interventions and must be measured separately. Instrument by tagging each call "
               "with whether an identical argument tuple was called earlier in the same trajectory: exact-repeat calls indicate "
               "distrust, novel-argument redundant calls indicate decision error. Metrics: exact-repeat call rate, novel-redundant "
               "rate, tool success rate, correctness, trajectory length, latency, recovery rate."),
         h="H2: exact-repeat calls account for >=40% (ESTIMATE, assuming distrust dominates in loops) of redundant calls; injecting the prior call's result verbatim into the system scratchpad cuts exact-repeat rate by >=50% (ESTIMATE) without changing correctness.",
         exp=("three arms: control, scratchpad-echo, and scratchpad-echo plus a hard dedup guard that returns the cached result "
              "for identical argument tuples. The dedup arm isolates whether the gain is from policy change or from caching."),
         gate="Promotion gate: ship the dedup guard only if it does not increase stale-result incidents (cached value used after an intervening state change) above zero on the audit slice.",
         qd=(3, 3, 3), conf=0.60,
         risks=["A dedup cache can return stale results if any tool in the loop is stateful."],
         ev=["Argument-tuple hash log per trajectory to compute exact-repeat rate."]),

    dict(v=116, f="cost accounting so the intervention is justified in latency and tokens, not just call counts",
         body=("Mechanism: call count is a proxy; the operational quantity is added end-to-end latency and added prefill/decode "
               "tokens. Each tool round trip costs one extra prefill of the growing context plus the tool's own latency. With a "
               "4k-token context and a 24GB-class GPU serving a 9B model, a redundant round trip costs roughly tool_latency + "
               "one prefill of the appended result; this is an ESTIMATE and must be replaced with MEASURED per-arm p50/p99 from "
               "the serving layer. Metrics: added tokens per redundant call, added seconds per redundant call, GPU-seconds per "
               "solved task, plus the correctness and recovery metrics."),
         h="H3: redundant calls account for >=15% (ESTIMATE) of total trajectory GPU-seconds; removing them yields a >=10% (ESTIMATE) reduction in GPU-seconds per solved task at unchanged correctness.",
         exp=("A/B at fixed request rate with server-side metrics scraped per arm (time-to-first-token, inter-token latency, "
              "batch occupancy) so the saving is attributed to fewer round trips rather than to lower load."),
         gate="Promotion gate: require MEASURED GPU-seconds-per-solved-task improvement >=5% with correctness delta >= -1.0; ESTIMATE values alone are insufficient.",
         qd=(3, 3, 3), conf=0.58,
         risks=["GPU-second savings can be an artefact of differing batch occupancy between arms rather than fewer calls."],
         ev=["Per-arm serving metrics (TTFT, ITL, batch occupancy, queue depth) captured at matched request rates."]),

    dict(v=117, f="confounders that make a naive before/after comparison unfalsifiable",
         body=("Mechanism: the three dominant confounders are (1) task-mix drift between the observation window and the treatment "
               "window, (2) tool-latency drift changing the policy's effective timeout behaviour, and (3) prompt/template changes "
               "shifting the tool-call token distribution. Control all three by freezing the task set, pinning tool latency with a "
               "fixed-delay stub in at least one arm, and byte-freezing the chat template and tool schema. Metrics as in the base set, "
               "stratified by task difficulty decile so a mix shift is visible rather than absorbed."),
         h="H4: with task set, tool latency and template frozen, the between-arm difference in calls-per-task is attributable to the intervention with residual confounding <2 points (ESTIMATE); if unfrozen, the same comparison shows >5 points of drift-driven difference.",
         exp=("a 2x2: {frozen, unfrozen} x {control, treatment}. The frozen/control cell is the negative control and must show no "
              "significant difference across repetitions; if it does, the measurement rig is not trustworthy and the experiment stops."),
         gate="Stop gate: if the negative-control cell shows a significant difference, halt and fix instrumentation before interpreting any treatment effect.",
         qd=(3, 4, 3), conf=0.63,
         risks=["Freezing tool latency with a stub removes real tail behaviour and can hide timeout-driven retry loops."],
         ev=["Difficulty-decile stratified metrics to expose task-mix drift.", "Negative-control cell results across >=3 repetitions."]),

    dict(v=118, f="preference/reward signal design and its failure modes",
         body=("Mechanism: the cleanest signal is a pairwise preference over trajectories that reach the same correct answer, "
               "preferring the one with fewer tool calls — this makes call reduction lexicographically secondary to correctness "
               "instead of trading against it. A scalar reward of the form correctness - lambda*calls is fragile: any lambda large "
               "enough to change behaviour also buys correctness losses. Metrics: win-rate of low-call trajectories among "
               "correctness-tied pairs, calls-per-task, correctness, recovery rate, plus a KL-to-reference term to detect policy drift."),
         h="H5: correctness-tied pairwise preference reduces calls-per-task by >=15% (ESTIMATE) with correctness delta within 0.5 points, whereas a scalar lambda tuned to the same call reduction costs >=1.5 points (ESTIMATE) of correctness.",
         exp=("train both objectives from the identical SFT checkpoint with identical data volume and compute; compare on the same "
              "held-out set. Report KL to the reference policy for both so behaviour change is not confused with degeneration."),
         gate="Rollback gate: revert if KL-to-reference exceeds the pre-registered budget, even if headline metrics improve.",
         qd=(4, 4, 3), conf=0.66,
         risks=["Correctness-tied pairing shrinks the usable training set and can bias toward easy tasks where ties are common."],
         ev=["Count of correctness-tied pairs by difficulty decile to check the pairing is not concentrated on easy tasks.", "KL-to-reference trace during training."]),

    dict(v=119, f="the stop / no-tool evaluation as a standalone capability probe",
         body=("Mechanism: build an eval whose items are constructed so the answer is fully determined by the prompt and no tool is "
               "needed; the only correct trajectory is a direct answer. Score two things separately: stop accuracy (did it refrain) "
               "and answer accuracy (was the direct answer right). A policy can score well on stop accuracy by refusing everywhere, "
               "so pair the probe with a tool-required slice and report the joint confusion matrix. Metrics: stop accuracy, "
               "tool-required recall, joint routing accuracy, correctness, calls-per-task, recovery."),
         h="H6: the current policy's stop accuracy on the no-tool slice is <=70% (ESTIMATE) and the intervention lifts it to >=85% (ESTIMATE) while tool-required recall stays >=95%.",
         exp=("evaluate the same checkpoint on the paired no-tool and tool-required slices before and after the intervention; report "
              "the 2x2 routing confusion matrix rather than a single scalar."),
         gate="Promotion gate: no shipping on stop accuracy alone; tool-required recall must not fall below its pre-intervention value minus 2 points.",
         qd=(4, 4, 3), conf=0.65,
         risks=["Optimising the stop action can produce a policy that under-calls on genuinely tool-required tasks."],
         ev=["Paired no-tool / tool-required slices with the routing confusion matrix reported for each arm."]),

    dict(v=120, f="online guardrail and canary rollout mechanics",
         body=("Mechanism: ship the intervention behind a per-request flag routed at the gateway so control and treatment share the "
               "same model replicas and the same KV cache pressure, eliminating hardware confounding. Start at 5% of traffic, hold "
               "for at least one full diurnal cycle, then step to 25% and 50%. Metrics streamed per arm: calls-per-task, correctness "
               "proxy (user-visible retry rate or downstream validator pass rate), p50/p99 latency, tool error rate, recovery rate."),
         h="H7: at 5% canary the treatment shows a calls-per-task reduction of >=10% (ESTIMATE) with validator pass rate within 1.0 point; if the effect is not visible at 5% within one diurnal cycle, the effect size is below operational relevance.",
         exp=("gateway-level randomisation with sticky assignment per session so multi-turn trajectories are not split across arms; "
              "arm assignment logged with every trace."),
         gate="Automatic rollback triggers: validator pass rate down >1.0 point, p99 latency up >10%, or tool error rate up >2x, any sustained for 2 consecutive canary hours.",
         qd=(3, 4, 4), conf=0.64,
         risks=["Non-sticky arm assignment splits a single multi-turn trajectory across arms and destroys attribution."],
         ev=["Per-request arm assignment in the trace log with session stickiness verified.", "Streaming validator pass rate as the online correctness proxy."]),

    dict(v=121, f="statistical power and how many tasks the experiment actually needs",
         body=("Mechanism: the primary endpoint is a paired difference in calls-per-task on the same items, which is far more "
               "powerful than an unpaired comparison. Required n scales with the per-item variance of call counts; with a "
               "call-count standard deviation of roughly 1.5 calls and a target detectable difference of 0.3 calls, a paired "
               "design needs on the order of 200 items at 80% power (ESTIMATE, from n ~ (1.96+0.84)^2 * sd^2 / delta^2 with "
               "sd=1.5, delta=0.3, halved for pairing). The correctness endpoint is the binding constraint: detecting a 1-point "
               "difference in a ~90% accuracy rate needs thousands of items, so correctness is treated as a non-inferiority "
               "guardrail with a pre-registered margin, not as a powered primary endpoint."),
         h="H8: with 1k paired items the call-count endpoint is adequately powered while the correctness endpoint can only support a non-inferiority claim at a 1.0-point margin, not an equivalence claim at 0.2 points.",
         exp=("pre-register n, the primary endpoint, the non-inferiority margin and the analysis plan before running; report "
              "confidence intervals rather than point estimates for both endpoints."),
         gate="Analysis gate: any post-hoc change to the endpoint or margin invalidates the run; rerun with the revised pre-registration.",
         qd=(4, 4, 3), conf=0.61,
         risks=["Under-powered correctness comparisons get read as 'no harm' when they are simply uninformative."],
         ev=["Pre-registered analysis plan with n, endpoint, margin and stopping rule fixed before data collection.", "Per-item paired call counts to compute the paired variance empirically."]),

    dict(v=122, f="failure taxonomy and recovery behaviour after a tool error",
         body=("Mechanism: any intervention that reduces calling must be checked against the recovery path, because the same "
               "mechanism that suppresses a redundant call can suppress the retry after a genuine tool failure. Instrument a "
               "taxonomy: transient tool error (timeout, 5xx), argument error (malformed input), semantic error (tool returns a "
               "wrong-but-well-formed value). Metrics: recovery rate per error class, retries-per-error, correctness conditional "
               "on an error having occurred, and time-to-recovery. Boundary condition: semantic errors are undetectable without an "
               "independent oracle, so that class is measured only on synthetic fault injection."),
         h="H9: recovery rate conditional on injected transient errors stays within 2 points of control after the intervention; if it falls more, the intervention is suppressing legitimate retries rather than redundant calls.",
         exp=("fault injection at fixed rates (5% transient, 2% argument) applied identically to both arms, with the injection "
              "schedule seeded so both arms see the same failures on the same items."),
         gate="Rollback gate: revert if conditional recovery rate drops >2 points, regardless of the headline call-count improvement.",
         qd=(4, 4, 4), conf=0.67,
         risks=["Call-suppression interventions can degrade legitimate retry behaviour, which is invisible in aggregate call counts."],
         ev=["Seeded fault-injection schedule shared across arms.", "Recovery rate broken out by error class, not aggregated."]),

    dict(v=123, f="an end-to-end decision protocol tying the pieces into a go/no-go",
         body=("Mechanism: sequence the work so cheap falsification happens first. Step 1: counterfactual labelling on 200 logged "
               "trajectories to estimate the redundant-call rate — if it is below 10%, stop, the problem is not worth an "
               "intervention. Step 2: offline paired A/B of the stop action on 1k items with the negative-control cell. Step 3: "
               "preference training only if step 2 shows a real effect. Step 4: 5% gateway canary with automatic rollback. Each step "
               "has an explicit kill criterion so the programme can fail cheaply. Metrics carried through every step: redundant-call "
               "rate, calls-per-task, correctness, tool success, recovery rate, p99 latency."),
         h="H10: the redundant-call rate estimated on 200 trajectories predicts the canary's realised call reduction within a factor of 2 (ESTIMATE); if it does not, offline labelling is not a valid screen and later cohorts must be sized from online data.",
         exp=("run all four steps with the step-1 estimate recorded and sealed before the canary, then compare the sealed prediction "
              "against the MEASURED canary delta."),
         gate="Kill criteria: stop at step 1 if redundant rate <10%; stop at step 2 if the paired call-count CI includes zero; stop at step 4 on any automatic rollback trigger.",
         qd=(4, 4, 4), conf=0.68,
         risks=["Sequential gating with reused data inflates false-positive rates unless each step uses a disjoint slice."],
         ev=["Disjoint data slices per step, documented in the pre-registration.", "Sealed step-1 prediction recorded before the canary starts."]),
]
