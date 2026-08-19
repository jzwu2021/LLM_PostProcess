import json, hashlib, os, glob

SRC = 'research/ai-infra-expert/corpus/train.jsonl'
START, N = 2390, 10  # zero-based positional start
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0240.jsonl'

rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if START <= i < START + N:
            rows.append(json.loads(line))
        elif i >= START + N:
            break
assert len(rows) == N, len(rows)

# Ten distinct causal mechanisms for "agent re-calls the calculator when the answer
# is already known", each with its own metric family and its own intervention.
MECH = [
    dict(
        title="Tool-result grounding loss: the observation is evicted from the context window before the final answer is composed",
        hyp="H1: redundant calculator calls are a context-eviction artifact — the prior tool observation has already been dropped or summarised out of the prompt by the time the model plans the next step, so from the model's point of view the value is genuinely unknown and re-calling is rational.",
        exp="Controlled experiment: hold the policy, decoding params and task set fixed and vary only the retention rule for tool observations (verbatim retention of the last K results vs the current summariser). One variable changed. Predicted outcome if H1 holds: redundant-call rate falls monotonically as K grows and the effect is concentrated on trajectories whose token length exceeds the summariser trigger.",
        meas="Per step: prompt token count, whether the earlier tool observation string is byte-present in the rendered prompt, redundant-call indicator, final-answer exact match. Aggregate: redundant calls per solved task, redundant-call rate stratified by trajectory token length, tokens spent on tool observations.",
        conf="Longer retention also raises total prompt length, which changes latency and can change attention behaviour on unrelated content — a confound with the intervention itself. Task difficulty correlates with trajectory length, so the stratification can pick up difficulty rather than eviction.",
        fix="Intervention: pin tool results in a structured scratchpad slot that the summariser is forbidden to compress, and render it as an explicit key/value table (`calc(expr) = value`) rather than free prose, so a lookup is cheaper than a re-call.",
        rb="Rollback gate: if the redundant-call rate is flat across K and the observation string is present in the prompt in >95 percent of redundant calls, H1 is falsified — revert the retention change (it costs tokens for nothing) and move to the reward/decoding hypotheses.",
        risks=["Pinning observations grows the prompt and can push long trajectories into truncation of *task-relevant* content, trading one failure for a worse one.",
               "Byte-presence of the observation is a proxy for the model attending to it; a pass on that check does not prove the model used it."],
        ev=["Per-step trace with rendered prompt hash, observation-presence flag and redundant-call flag",
            "Redundant-call rate vs retention K with confidence intervals over >=3 seeds",
            "Token-length histogram of trajectories, split by whether the summariser fired",
            "Final-task accuracy under each K, to prove the fix does not trade correctness for call count"],
        dims=(4, 3, 4), conf_v=0.55,
    ),
    dict(
        title="Reward specification gap: training signal rewards tool use but never penalises a redundant call",
        hyp="H2: the behaviour is learned, not confused — the SFT/RL objective gave credit for emitting a well-formed tool call and none for abstaining, so the policy's expected return is maximised by calling whenever a call is syntactically available.",
        exp="Controlled experiment: same base checkpoint, same data, vary only the reward/loss term — add an explicit per-call cost c to the return and sweep c over at least three values including 0 (control). If H2 holds, redundant calls fall monotonically in c while final accuracy stays flat until c becomes large enough to suppress *necessary* calls, producing a visible knee.",
        meas="Redundant calls per task, necessary-call recall (fraction of tasks that genuinely need arithmetic where the tool was still invoked), tool success rate, final correctness, trajectory length in steps and tokens.",
        conf="A per-call cost also shortens trajectories in general, so a correctness drop may come from suppressed *legitimate* calls rather than from the cost being wrong; reward hacking can move the behaviour into an equally wasteful non-tool form such as long redundant chain-of-thought.",
        fix="Intervention: shape the reward as correctness minus c times (calls issued minus calls whose result changed the answer), so only calls with no information gain are penalised, and add explicit no-tool positives to the training mix.",
        rb="Rollback gate: revert the reward change if necessary-call recall drops more than 2 points absolute, or if final correctness on the held-out set drops at all beyond the seed noise band measured over >=3 seeds.",
        risks=["Penalising calls can teach the model to guess arithmetic it should verify — a safety regression that looks like an efficiency win in the headline metric.",
               "Reward changes retrain the policy; the resulting checkpoint must be re-qualified on the full eval suite, not just on the call-count metric."],
        ev=["Reward-sweep table: c vs redundant calls, necessary-call recall, final accuracy, >=3 seeds each",
            "Definition and audit sample of the 'information gain' label used in the shaped reward",
            "Held-out eval results for the retrained checkpoint against the pre-change baseline",
            "Trajectory-length distribution to detect the chain-of-thought substitution failure mode"],
        dims=(5, 4, 4), conf_v=0.6,
    ),
    dict(
        title="No stop condition: the prompt/scaffold offers no explicit 'answer directly' action",
        hyp="H3: this is a scaffold defect, not a model defect — the action space as rendered contains only tool calls, with finishing expressed implicitly, so the highest-probability well-formed continuation is another call.",
        exp="Controlled experiment: keep the identical checkpoint and identical tasks and vary only the system prompt/action schema by adding a first-class `final_answer` action with the same syntactic weight as `calculator`. No retraining. If H3 holds, redundant calls drop immediately at fixed weights — an effect no weight-level hypothesis can explain.",
        meas="Redundant-call rate before/after, fraction of trajectories terminating via the explicit finish action, malformed-action rate, steps-to-termination, final accuracy.",
        conf="A new action token shifts the whole output distribution and can degrade unrelated formatting; prompt edits are not free and can interact with few-shot exemplars still showing the old schema.",
        fix="Intervention: make finishing an explicit, exemplified action, and add one or two few-shot exemplars in which the correct behaviour is to answer from an existing tool result without calling again.",
        rb="Rollback gate: revert the schema change if malformed-action rate rises above the pre-change rate plus 1 point, or if final accuracy regresses; the prompt is version-controlled so revert is a single config edit and must be verified by re-running the same eval seed.",
        risks=["Prompt-level fixes are checkpoint-specific and silently regress when the base model is upgraded; they need a regression test pinned in CI.",
               "Adding exemplars consumes context and can crowd out task content on long inputs."],
        ev=["Exact diff of the system prompt / action schema between arms",
            "Per-arm eval on the same fixed seed set with termination-reason breakdown",
            "Malformed-action counts per arm",
            "Confirmation that few-shot exemplars were updated consistently with the new schema"],
        dims=(4, 4, 5), conf_v=0.62,
    ),
    dict(
        title="Cache-key mismatch: identical expressions miss a memo the scaffold believes exists",
        hyp="H4: a deduplication layer already exists but is ineffective because its cache key is the raw expression string, and the model re-emits semantically identical but textually different expressions (`2*3` vs `2 * 3` vs `3*2`), producing cache misses that surface as duplicate executions.",
        exp="Controlled experiment: replay the *recorded* trajectories offline against two key functions — raw string vs a normalised/canonicalised key (whitespace-stripped, parsed to an AST, commutative operands sorted). Nothing about the policy changes, so any difference is purely the key function.",
        meas="Cache hit rate under each key function, count of distinct keys mapping to the same computed value, executed calls per task, tool latency saved, and any collisions where different values share a normalised key.",
        conf="Aggressive canonicalisation can collide expressions that are not actually equal (floating-point associativity, integer vs float division), turning a redundancy fix into a correctness bug; offline replay cannot capture behaviour changes the cache would induce online.",
        fix="Intervention: canonicalise on a parsed AST with a conservative rule set, cache value plus provenance, and render cache hits back to the model as an ordinary tool observation so the trajectory stays consistent.",
        rb="Rollback gate: disable canonicalisation (fall back to exact-string keys) immediately if any normalised-key collision produces two different computed values in replay, regardless of the hit-rate gain.",
        risks=["A wrong cache hit silently returns a stale or incorrect number — worse than the redundant call it removes.",
               "Caching hides real tool failures: an error cached once is replayed as if authoritative."],
        ev=["Offline replay log with per-call raw key, normalised key, and computed value",
            "Collision report: normalised keys with >1 distinct value (must be empty to ship)",
            "Cache hit rate and executed-call delta per key function",
            "Unit tests covering float/int division and non-commutative operators"],
        dims=(4, 4, 4), conf_v=0.5,
    ),
    dict(
        title="Sampling entropy: redundant calls are a decoding artifact, not a policy belief",
        hyp="H5: the policy already assigns most mass to finishing, but temperature/top-p leaves enough tail mass on the call token that at multi-step horizons the per-trajectory probability of at least one spurious call is high — a compounding-sampling effect rather than a learned preference.",
        exp="Controlled experiment: fix everything and sweep only decoding (greedy, T=0.3, T=0.7, T=1.0) on the same task set and same seeds. If H5 holds, redundant-call rate falls sharply toward greedy and the per-step probability assigned to the call token at the redundancy point is already below 0.5 in the failing trajectories.",
        meas="Per-step logprob of the first token of the call action at each decision point, redundant-call rate by temperature, variance across seeds, final accuracy by temperature, trajectory length.",
        conf="Greedy decoding also changes reasoning quality and can reduce diversity-dependent accuracy, so a redundancy win may be paid for elsewhere; logprobs from a quantised or batched serving path can differ from the training-time distribution.",
        fix="Intervention: lower temperature for action selection only (structured decoding over the action head) while leaving free-text reasoning at the higher temperature, so redundancy is suppressed without flattening reasoning.",
        rb="Rollback gate: if per-step call-token probability at redundancy points is >0.5, H5 is falsified — the model believes the call is right, and lowering temperature would only mask a policy defect; revert decoding and go to H2.",
        risks=["Split-temperature decoding complicates the serving path and can desynchronise with the training-time distribution, producing silent drift after a model update.",
               "Tuning decoding to a benchmark overfits to that benchmark's horizon length."],
        ev=["Per-decision-point logprob dump for the action token in failing trajectories",
            "Temperature sweep table with >=3 seeds and confidence intervals",
            "Confirmation that eval-time serving stack (quantisation, batching) matches the one measured",
            "Accuracy alongside redundancy for every temperature arm"],
        dims=(4, 4, 4), conf_v=0.48,
    ),
    dict(
        title="Self-verification loop: the agent re-calls to double-check because it has no confidence signal",
        hyp="H6: the repeated call is deliberate verification — the agent has no calibrated way to express 'I already know this with sufficient confidence', so its cheapest available verification action is re-running the tool, and the loop repeats whenever the two results are rendered in different formats.",
        exp="Controlled experiment: inject a verification affordance (a state slot holding `value, source, verified=true`) and compare against the control with identical prompts otherwise. If H6 holds, redundant calls collapse specifically on trajectories where the two calls had identical arguments and the second occurred after a formatting difference in the rendered result.",
        meas="Fraction of redundant calls with byte-identical arguments to a prior call, time/steps between the pair, whether rendered formats differed (trailing zeros, scientific notation), and post-intervention redundancy on that subpopulation.",
        conf="Some verification is legitimate — for non-deterministic or time-varying tools, re-calling is correct behaviour, so a blanket suppression is wrong; the formatting-difference signal is correlational and can be a spurious marker of harder tasks.",
        fix="Intervention: render tool results in a single canonical numeric format with an explicit `verified` flag, and separate deterministic tools (safe to memoise) from non-deterministic tools (must be re-callable) in the tool registry.",
        rb="Rollback gate: revert if suppressed re-calls include any non-deterministic tool, or if tasks that legitimately require re-reading changing state regress in accuracy.",
        risks=["Treating a non-deterministic tool as memoisable produces stale answers that are hard to detect downstream.",
               "A `verified` flag the model can emit itself is not evidence of correctness; it must be set by the scaffold, not the policy."],
        ev=["Paired-call table: argument equality, rendering diff, step distance",
            "Tool registry annotated with determinism per tool, reviewed by an owner",
            "Accuracy on a held-out set containing genuinely time-varying tasks",
            "Before/after redundancy restricted to the identical-argument subpopulation"],
        dims=(4, 4, 4), conf_v=0.52,
    ),
    dict(
        title="Training-data imitation: demonstrations contain the redundancy the policy is reproducing",
        hyp="H7: the model is faithfully imitating its SFT data — the demonstration set was generated by an earlier scaffold that re-called tools, so redundancy is in the targets and no amount of prompt or decoding change will remove it.",
        exp="Controlled experiment: audit the SFT set for redundant calls, then fine-tune two checkpoints from the identical base on (a) the raw set and (b) the same set with redundant calls removed and the trajectory re-stitched — identical hyperparameters, identical seeds, one variable changed. If H7 holds, arm (b) shows a large redundancy drop with unchanged accuracy.",
        meas="Redundancy rate in the training targets themselves, redundancy in each fine-tuned policy's rollouts, held-out accuracy, and the correlation between per-topic redundancy in data and in rollouts.",
        conf="Removing calls changes trajectory length distribution and token budget, which itself affects learning; the cleaned set is smaller, so a capability difference may reflect data volume rather than data quality.",
        fix="Intervention: filter/re-stitch redundant calls out of the demonstration set, and add explicitly labelled negatives (a redundant call followed by a correction) so the model learns the boundary rather than just never seeing it.",
        rb="Rollback gate: if the data audit finds redundancy in <5 percent of demonstrations, H7 cannot explain a double-digit rollout rate — abandon the data intervention and do not retrain.",
        risks=["Re-stitching trajectories can produce observations that reference calls no longer present, teaching the model to hallucinate tool results.",
               "Retraining consumes the full training budget; running it on a falsified hypothesis is the expensive failure mode here."],
        ev=["Redundancy audit of the SFT set with the exact detector definition and a hand-checked sample",
            "Diff statistics of the re-stitched set (examples changed, tokens removed, dangling references found)",
            "Paired fine-tune results, identical hyperparameters and >=3 seeds",
            "Held-out accuracy for both arms against the pre-fine-tune baseline"],
        dims=(5, 4, 4), conf_v=0.57,
    ),
    dict(
        title="State-rendering defect: the result is in the transcript but not in the model-visible state",
        hyp="H8: an engineering bug — the scaffold stores the tool result in its own state object for logging but the template that renders the next prompt drops it (wrong field name, silent exception, role filtered out), so redundancy is a plumbing failure with a deterministic reproduction.",
        exp="Controlled experiment: dump the exact rendered prompt at the redundancy step and diff it against the scaffold's internal state for the same step. This is a deterministic check with no statistics required — either the value is present in the rendered prompt or it is not.",
        meas="Byte-level presence of the result value in the rendered prompt, count of steps where internal state and rendered prompt disagree, exception counts in the renderer, and the redundancy rate restricted to steps with a disagreement.",
        conf="A renderer can drop the value only for certain result types (long output truncation, non-ASCII, nested JSON), so a sample that happens to use short integers will show no bug; log-only reproduction may not match the production template version.",
        fix="Intervention: add an invariant assertion in the scaffold — every tool result recorded in state must appear in the next rendered prompt or the step fails loudly — plus a regression test with a long, nested and non-ASCII result.",
        rb="Rollback gate: if internal state and rendered prompt agree at every redundancy step, H8 is falsified; remove the assertion from the hot path if it costs measurable latency and move to the model-level hypotheses.",
        risks=["A hard assertion in the serving path can turn a cosmetic bug into an outage; ship it behind a flag that logs before it enforces.",
               "Byte-presence checks are brittle against legitimate reformatting and can produce false alarms that erode trust in the alarm."],
        ev=["Paired dump of scaffold state and rendered prompt for every redundancy step",
            "Renderer exception/warning counts over the eval run",
            "Regression test covering long, nested and non-ASCII tool results",
            "Redundancy rate on the disagreement subpopulation vs the agreement subpopulation"],
        dims=(5, 4, 4), conf_v=0.63,
    ),
    dict(
        title="Retry/timeout layer double-executes: the duplicate is infrastructure, not the policy",
        hyp="H9: the model emitted one call and the transport executed two — a client-side timeout fired while the tool server was still running, the retry succeeded, and both executions are recorded, so a policy-level metric attributes an infrastructure duplicate to the agent.",
        exp="Controlled experiment: instrument the tool client with a per-call idempotency key and compare emitted calls (policy side) against executed calls (server side) for the same run. No policy change. If H9 holds, emitted counts match the non-redundant expectation and the gap is entirely on the server side, correlated with calls whose latency exceeded the client timeout.",
        meas="Emitted-call count vs executed-call count per trajectory, per-call latency distribution against the client timeout threshold, retry counter, and duplicate executions grouped by idempotency key.",
        conf="Emitted and executed counts are collected by different systems whose clocks and sampling may differ; a load-dependent effect will vanish in a quiet reproduction run and reappear under production concurrency.",
        fix="Intervention: make tool execution idempotent by key with server-side dedup within a TTL, raise the client timeout above the measured p99.9 tool latency, and report only *emitted* calls in the agent-behaviour metric so the two failure classes stay separable.",
        rb="Rollback gate: if emitted and executed counts are equal for every trajectory, H9 is falsified and no transport change should ship; keep the emitted/executed split in telemetry regardless, since it is cheap and prevents this misattribution recurring.",
        risks=["Raising the client timeout increases tail latency and can cascade into upstream queue growth under load.",
               "Server-side dedup with too long a TTL suppresses legitimate repeat calls for non-deterministic tools."],
        ev=["Paired emitted-vs-executed call counts per trajectory with idempotency keys",
            "Tool latency distribution with p50/p99/p99.9 and the configured client timeout marked",
            "Retry counters from the tool client over the same window",
            "Load level (concurrency, QPS) during both the failing run and the reproduction"],
        dims=(5, 4, 5), conf_v=0.6,
    ),
    dict(
        title="Metric definition error: 'redundant' is measured wrongly and the intervention would chase noise",
        hyp="H10: before any intervention, the measurement itself is suspect — the redundancy detector counts any repeated tool name as redundant, so legitimate calls with different arguments, or calls on genuinely changed state, inflate the rate and would make a real fix look ineffective.",
        exp="Controlled experiment on the *metric*, not the model: hand-label a stratified sample of at least 200 call pairs as redundant / not redundant, then score the automatic detector against those labels. Change nothing in the system. If precision is below roughly 0.9, every downstream A/B on this metric is uninterpretable.",
        meas="Detector precision, recall and F1 against human labels with confidence intervals; inter-annotator agreement (Cohen's kappa) on a doubly-labelled subset; rate breakdown by the specific rule that fired.",
        conf="Human labels themselves are noisy on borderline verification cases; a stratified sample over-weights rare strata unless the estimate is re-weighted back to the population.",
        fix="Intervention: define redundancy operationally as 'a call whose canonicalised arguments match a prior successful call in the same trajectory, on a tool registered as deterministic, with no intervening state mutation', and freeze that definition with a labelled regression set before running any behavioural experiment.",
        rb="Rollback gate: do not ship or evaluate any behavioural intervention while detector precision is below 0.9 on the frozen labelled set; if a shipped change is later found to have been evaluated on a sub-0.9 detector, treat its reported gain as unverified and re-measure before keeping it.",
        risks=["Optimising a mis-specified metric can make real behaviour worse while the dashboard improves — the dominant failure mode of agent-efficiency work.",
               "Freezing a definition too early bakes in the current tool registry; the definition must be revisited whenever a non-deterministic tool is added."],
        ev=["Stratified sample of >=200 hand-labelled call pairs with the labelling guideline",
            "Detector precision/recall/F1 with confidence intervals against those labels",
            "Cohen's kappa on the doubly-labelled subset",
            "Written, version-controlled operational definition of redundancy plus its regression set"],
        dims=(5, 5, 5), conf_v=0.66,
    ),
]


def build(m):
    return (
        "Scope and assumptions. The observed behaviour is an agent issuing a calculator tool call whose result is already "
        "determinable from the trajectory. Assumptions stated up front: the calculator is deterministic and side-effect free; "
        "the trajectory is single-agent; and the baseline redundancy rate quoted below is MEASURED from the recorded eval run, "
        "not assumed. Any number produced by the design below must be labelled ESTIMATE or MEASURED at the point it is reported. "
        "Mechanism under examination: " + m["title"] + ".\n\n"
        "Falsifiable hypothesis. " + m["hyp"] + "\n\n"
        "Controlled experiment. " + m["exp"] + " The control arm is the unmodified current system evaluated on the identical "
        "task set and identical seeds in the same window, so that model version, tool version and load are held fixed.\n\n"
        "Metrics and instrumentation. " + m["meas"] + " Report every rate with a confidence interval over at least three seeds; "
        "a single-seed delta on agent-behaviour metrics is not evidence. Any capacity or latency figure carried into a decision "
        "must be MEASURED on the same hardware and serving configuration as the arm it describes; projections from a different "
        "batch size or quantisation are ESTIMATE and must be re-measured before they gate anything.\n\n"
        "Expected confounders. " + m["conf"] + " Additionally: eval-set composition drift between arms, and any change to the "
        "tool server that lands mid-experiment, both invalidate the comparison and must be checked from deployment records "
        "rather than assumed absent.\n\n"
        "Intervention. " + m["fix"] + " Deploy it behind a flag on a canary slice first, keeping the control arm live for "
        "simultaneous comparison rather than comparing against a historical baseline measured under different load.\n\n"
        "Rollback criteria. " + m["rb"] + " Independent of the hypothesis: revert immediately if end-task accuracy on the frozen "
        "held-out set drops outside the seed noise band, if necessary-tool-call recall falls, or if tool-error rate rises; and "
        "require any change that cannot be undone by a single flag flip or config edit to have a written revert procedure agreed "
        "before it is attempted."
    )


recs = []
for row, m in zip(rows, MECH):
    msgs = row['messages']
    u = [x for x in msgs if x['role'] == 'user'][0]['content']
    a = [x for x in msgs if x['role'] == 'assistant'][0]['content']
    tc, ic, os_ = m["dims"]
    recs.append({
        "source_id": row["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": build(m),
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": m["risks"],
        "evidence_required": m["ev"],
        "confidence": m["conf_v"],
    })

h = [hashlib.sha256(r["corrected_answer"].encode()).hexdigest() for r in recs]
assert len(set(h)) == len(recs), "duplicate corrected_answer within batch"
seen = set()
for f in glob.glob('experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-*.jsonl'):
    for l in open(f):
        seen.add(hashlib.sha256(json.loads(l)["corrected_answer"].encode()).hexdigest())
assert not (set(h) & seen), "corrected_answer collides with an existing batch"
for r in recs:
    assert r["corrected_answer"] != r["source_assistant"]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs), "ids", recs[0]["source_id"], "->", recs[-1]["source_id"])
