"""Agent, tool-calling and structured-output mechanisms (topic: agents)."""
from __future__ import annotations

from core import Mechanism, Quant, Setting, fmt_int, gib, register


def q_context_growth(s: Setting) -> Quant:
    turn = 800
    turns = max(s.ctx // turn, 1)
    per_tok = s.kv_bytes_per_token
    return Quant(
        label="how many tool-calling turns fit before the context window is exhausted",
        steps=[
            f"Assume roughly {turn} tokens per turn once the call and its result are included",
            f"Context window is {fmt_int(s.ctx)} tokens",
            f"Turns before exhaustion = floor({fmt_int(s.ctx)} / {turn}) = {turns}",
            f"Cache held at exhaustion = {fmt_int(per_tok)} * {fmt_int(s.ctx)} = {gib(per_tok * s.ctx)}",
        ],
        value=f"about {turns} turns, holding {gib(per_tok * s.ctx)} of cache at the limit",
        interpretation=(
            "An agent loop consumes context monotonically, so the turn budget is fixed by the window "
            "rather than by the task. A loop with no turn cap will reach the limit and fail on a "
            "boundary the task never anticipated."),
    )


def q_reprefill(s: Setting) -> Quant:
    turn = 800
    turns = max(s.ctx // turn, 1)
    total = sum(turn * i for i in range(1, turns + 1))
    return Quant(
        label="the prompt tokens reprocessed across an agent loop without prefix caching",
        steps=[
            f"Turn i re-sends the whole history: roughly {turn} * i tokens",
            f"Summing over {turns} turns gives {fmt_int(total)} prompt tokens processed",
            f"A single-shot request of the same final length would process {fmt_int(turn * turns)}",
            f"Amplification = {total / max(turn * turns, 1):.1f}x",
        ],
        value=f"{fmt_int(total)} prompt tokens across the loop, {total / max(turn * turns, 1):.1f}x amplification",
        interpretation=(
            "Agent loops are prefill-heavy by construction. Prefix caching is not an optimisation "
            "here; without it the loop's cost grows quadratically with turn count."),
    )


def q_schema_validation(s: Setting) -> Quant:
    return Quant(
        label="the validation surface between a generated call and an executed action",
        steps=[
            "The payload must parse as an object before any field is read",
            "The tool name must be a string and must be in the allowed set",
            "The arguments must be an object and must match the tool's declared schema",
            "Only after all of that may the call reach anything that acts",
        ],
        value="four checks, all of which must pass before any field reaches an executor",
        interpretation=(
            "Model output is untrusted input. Validating lazily as fields are accessed means an "
            "unvalidated value has already reached the code that acts on it."),
    )


def q_retry_amplification(s: Setting) -> Quant:
    depth = 3
    attempts = 3
    return Quant(
        label="how nested retries multiply into total attempts",
        steps=[
            f"Suppose {depth} layers each retry up to {attempts} times",
            f"Total attempts = {attempts}^{depth} = {attempts ** depth}",
            f"At {s.concurrency} concurrent requests that is up to "
            f"{attempts ** depth * s.concurrency} in-flight attempts",
            "Each attempt consumes a full generation, not a cheap network call",
        ],
        value=f"{attempts ** depth} attempts per request from {depth} nested layers",
        interpretation=(
            "Retry layers compose multiplicatively. In an agent stack where each attempt is a model "
            "generation, that turns a transient dependency failure into a load multiplier."),
    )


def q_tool_latency(s: Setting) -> Quant:
    tool_ms = 400
    turns = 5
    gen_ms = s.slo_ms // 4
    return Quant(
        label="how the latency budget divides between generation and tool execution",
        steps=[
            f"End-to-end objective is {s.slo_ms} ms",
            f"A {turns}-turn loop generates {turns} times at roughly {gen_ms} ms each",
            f"Tool execution adds about {tool_ms} ms per turn: {turns * tool_ms} ms",
            f"Total {turns * gen_ms + turns * tool_ms} ms against a {s.slo_ms} ms budget",
        ],
        value=f"{turns * gen_ms + turns * tool_ms} ms for a {turns}-turn loop against {s.slo_ms} ms",
        interpretation=(
            "The loop multiplies both terms by turn count, so an objective set for a single "
            "generation cannot survive a multi-turn agent without being restated per turn."),
    )


def q_grammar_constraint(s: Setting) -> Quant:
    return Quant(
        label="what constrained decoding does and does not guarantee",
        steps=[
            "A grammar constrains which tokens may be sampled at each position",
            "That guarantees the output parses against the grammar",
            "It does not guarantee the field values are correct, permitted or safe",
            "It also does not guarantee the tool exists or that its arguments are meaningful",
        ],
        value="syntactic validity guaranteed, semantic validity not addressed",
        interpretation=(
            "Constrained decoding removes parse failures and can create the impression that "
            "validation is unnecessary. The remaining checks are exactly the ones that matter for "
            "safety."),
    )


def q_idempotency(s: Setting) -> Quant:
    return Quant(
        label="why an agent retry needs idempotency that a plain request does not",
        steps=[
            "A retried tool call may execute an action that already succeeded",
            "The model cannot observe whether the first attempt took effect",
            "Without an idempotency key the second execution is a second real action",
            f"At {s.concurrency} concurrent loops, duplicate actions accumulate quickly",
        ],
        value="every retryable tool call needs a caller-supplied idempotency key",
        interpretation=(
            "Retry safety is a property of the tool, not of the retry policy. A policy that retries "
            "non-idempotent tools is unsafe regardless of how carefully its backoff is tuned."),
    )


def q_loop_detection(s: Setting) -> Quant:
    turn = 800
    turns = max(s.ctx // turn, 1)
    return Quant(
        label="the cost of an undetected repeating loop",
        steps=[
            f"A loop that repeats the same call runs until the {fmt_int(s.ctx)}-token window fills",
            f"That is roughly {turns} turns, each a full generation",
            f"Each turn also re-prefills the growing history",
            "The loop then terminates on context exhaustion rather than on task completion",
        ],
        value=f"up to {turns} wasted generations before the window forces termination",
        interpretation=(
            "Context exhaustion is not a loop detector; it is the absence of one. A repeated "
            "state must be detected explicitly or the window becomes the only bound."),
    )


def q_parallel_tools(s: Setting) -> Quant:
    n = 4
    tool_ms = 400
    return Quant(
        label="what parallel tool execution can and cannot recover",
        steps=[
            f"{n} independent tool calls at about {tool_ms} ms each",
            f"Serial: {n * tool_ms} ms",
            f"Parallel: about {tool_ms} ms, bounded by the slowest",
            "The generation turns between them remain serial and are unaffected",
        ],
        value=f"{n * tool_ms} ms serial against about {tool_ms} ms parallel, generation unchanged",
        interpretation=(
            "Parallelism helps only the tool-execution term. If generation dominates the loop, "
            "parallel tools change a small part of the total and the effort is misdirected."),
    )


def q_output_truncation(s: Setting) -> Quant:
    return Quant(
        label="what happens when a tool result exceeds the space left in the window",
        steps=[
            f"The window is {fmt_int(s.ctx)} tokens and the history already occupies most of it",
            "A large tool result must be truncated to fit",
            "Truncation usually removes the end, which is often where the answer sits",
            "The model then reasons over a fragment without knowing it is a fragment",
        ],
        value=f"a result larger than the remaining share of {fmt_int(s.ctx)} tokens is silently cut",
        interpretation=(
            "Truncating a tool result is a data-loss event that the model cannot detect. It must be "
            "signalled explicitly in the result, or summarised deliberately rather than cut."),
    )


register(
    Mechanism(
        key="agent_context_budget", topic="agents",
        title="an agent loop consumes context monotonically, so the turn count is bounded by the window",
        concepts=("agents", "context_window", "tool_calling"),
        symptom="Agent runs fail partway through with a context-length error after behaving correctly for several turns.",
        chain="Each turn appends the model's call and the tool's result to the history, so the prompt grows on every iteration and the loop terminates on window exhaustion rather than on task completion.",
        metric="Prompt token count per turn, plotted across the loop rather than sampled at the end.",
        signature="Prompt length grows monotonically and approximately linearly with turn count, and the failure occurs at the window limit rather than at a task boundary.",
        confounders=(
            "A single oversized tool result, which exhausts the window in one step rather than gradually.",
            "System prompt growth from dynamically injected context, which raises the baseline without any loop effect.",
            "Model output length varying by turn, which changes the slope without changing the mechanism.",
        ),
        fixes=(
            "Set an explicit turn cap derived from the window and the observed per-turn growth.",
            "Summarise or drop older turns once the history passes a stated share of the window.",
            "Move bulk tool output into external storage and pass a reference rather than the content.",
        ),
        rollback="Restore the previous history policy if task success rate falls after summarisation, since discarding context trades window space for capability.",
        options=("setting an explicit turn cap", "summarising older turns once a share of the window is used"),
        tradeoff="whether the task can be completed within the turns the window allows",
        flip="tasks legitimately need more turns than the window permits, at which point summarisation rather than a cap is required and its quality cost must be measured",
        falsifier="prompt length is flat across turns, which means history is already being managed",
        wrong_claim="The context window is large, so a few tool-calling turns will not come close to filling it.",
        wrong_why="Growth is cumulative and includes every tool result, so the window is consumed by history rather than by any single message, and the limit arrives sooner than per-message sizes suggest.",
        threshold="Cap turns so predicted history stays within a stated share of the window, leaving room for the largest expected tool result.",
        cost="A loop that fails at the window limit has already paid for every generation it performed and returns nothing.",
        scaling="Per-turn growth is roughly constant while the window is fixed, so the turn budget shrinks as tool results grow richer.",
        quant=q_context_growth,
    ),
    Mechanism(
        key="agent_reprefill_amplification", topic="agents",
        title="an agent loop reprocesses its whole history every turn, so cost grows quadratically",
        concepts=("prefill", "prefix_caching", "agents"),
        symptom="A multi-turn agent costs far more per task than the token counts of its individual messages suggest.",
        chain="Each turn sends the accumulated history as the prompt, so a turn near the end reprocesses everything that came before, and total prompt tokens across the loop grow with the square of the turn count rather than linearly.",
        metric="Total prompt tokens processed across the loop, compared against the final history length.",
        signature="The ratio of processed prompt tokens to final history length grows with turn count rather than staying near one.",
        confounders=(
            "Prefix caching already eliminating most of the reprocessing, which removes the effect.",
            "History summarisation shortening the prompt between turns.",
            "Tool results dominating the token count, which changes the constant but not the growth.",
        ),
        fixes=(
            "Enable prefix caching and verify the hit rate on real agent traffic rather than assuming it.",
            "Route all turns of one conversation to the instance holding its prefix.",
            "Keep the prefix stable by placing volatile content at the end of the prompt rather than the beginning.",
        ),
        rollback="Disable prefix routing if it concentrates load badly enough that queueing exceeds the prefill it saves.",
        options=("routing a conversation's turns to the instance holding its prefix", "keeping the prefix stable by ordering the prompt"),
        tradeoff="whether saved prefill exceeds the queueing that constrained routing adds",
        flip="conversation turns arrive too far apart for the cache to survive between them, at which point routing buys nothing and prompt ordering is the only remaining lever",
        falsifier="processed prompt tokens are close to the final history length",
        wrong_claim="Each turn only adds a few hundred tokens, so a ten-turn loop is a small amount of work.",
        wrong_why="Every turn reprocesses all previous turns, so the loop's total prompt work grows with the square of the turn count rather than with the tokens added.",
        threshold="Require a measured prefix cache hit rate on agent traffic before budgeting agent cost from message sizes.",
        cost="Reprocessed prefix tokens are billed as prefill compute on every turn and produce no new information.",
        scaling="Total cost grows with the square of turn count, so the difference between a five-turn and a ten-turn loop is fourfold rather than twofold.",
        quant=q_reprefill,
    ),
    Mechanism(
        key="tool_call_trust_boundary", topic="agents",
        title="a generated tool call is untrusted input and must be fully validated before dispatch",
        concepts=("validation", "security", "tool_calling"),
        symptom="A malformed or unexpected tool call reached an executor and produced an action nobody intended.",
        chain="Model output is generated text, so a call payload is untrusted input, and any field read before structural validation completes is an unvalidated value already flowing toward code that acts.",
        metric="Count of dispatches per validation outcome, including which rule rejected each refused call.",
        signature="Rejections are attributable to a specific rule, and no code path returns a partially populated call object.",
        confounders=(
            "A grammar-constrained decoder making malformed payloads rare, which hides the gap rather than closing it.",
            "Retries masking the rejection rate so the underlying frequency is not visible.",
            "Upstream schema enforcement in the provider, which may change between versions.",
        ),
        fixes=(
            "Complete structural validation before returning anything a caller could act on.",
            "Reject unknown top-level fields and duplicate keys explicitly rather than ignoring them.",
            "Check the tool name against an allow list rather than dispatching on whatever string arrives.",
        ),
        rollback="Fail closed and refuse the call if any validation rule cannot be evaluated, rather than falling through to dispatch.",
        options=("validating structure fully before returning a call object", "checking the tool name against an allow list"),
        tradeoff="whether the risk is malformed structure or a well-formed call to something that should not be reachable",
        flip="the payloads are always well formed because of constrained decoding, at which point the allow list rather than the parser is doing all the useful work",
        falsifier="no path returns a call object on any error, and every field is validated before it is readable",
        wrong_claim="The decoder is grammar-constrained, so the output is guaranteed valid and does not need checking.",
        wrong_why="A grammar guarantees the output parses, not that the tool exists, that the caller may invoke it, or that the arguments are within permitted ranges.",
        threshold="Require every field a dispatcher reads to have passed structural and allow-list validation first.",
        cost="An unintended action taken on a malformed call is not recoverable by fixing the parser afterwards.",
        scaling="Exposure grows with the number of tools and with the privilege of the most powerful one, not with call volume.",
        quant=q_schema_validation,
    ),
    Mechanism(
        key="nested_retry_amplification", topic="agents",
        title="retry layers in an agent stack compose multiplicatively into a load multiplier",
        concepts=("retries", "load_amplification", "agents"),
        symptom="A brief dependency failure produced a load spike several times larger than normal traffic and outlasted the failure.",
        chain="Each layer that retries independently multiplies the attempts of the layer below it, and in an agent stack every attempt is a full model generation rather than a cheap call, so a transient failure converts into sustained amplified load.",
        metric="Total generations per user request, counted end to end across all layers.",
        signature="Generations per user request rises sharply during the dependency failure and equals the product of the per-layer attempt limits.",
        confounders=(
            "Client-side retries outside the system, which add another multiplier that internal metrics do not see.",
            "The agent itself retrying by generating a new plan, which is a retry that does not look like one.",
            "Queue backlog replaying work after recovery, which resembles amplification.",
        ),
        fixes=(
            "Retry at exactly one layer and make the others fail through.",
            "Give the whole request a single deadline that all layers respect, bounding total attempts by time.",
            "Add a retry budget that caps attempts as a fraction of overall traffic rather than per request.",
        ),
        rollback="Disable retries entirely at the inner layers if generations per request does not fall, since an unbounded multiplier is worse than no retry at all.",
        options=("retrying at exactly one layer", "enforcing a single request-wide deadline"),
        tradeoff="whether the layers can be made to agree on where retry responsibility sits",
        flip="the layers are owned by different teams and cannot be coordinated, at which point a shared deadline is the only enforceable bound",
        falsifier="generations per user request stays near one during the dependency failure",
        wrong_claim="Each layer retries at most three times, which is a conservative setting.",
        wrong_why="Three layers each retrying three times produces twenty-seven attempts, so a per-layer setting that reads as conservative is a large multiplier once composed.",
        threshold="Bound total generations per user request explicitly rather than bounding attempts per layer.",
        cost="Each amplified attempt is a full generation, so the wasted spend is orders of magnitude above an amplified network retry.",
        scaling="The multiplier is the product across layers, so adding one retrying layer multiplies rather than adds to the load.",
        quant=q_retry_amplification,
    ),
    Mechanism(
        key="agent_latency_budget", topic="agents",
        title="a per-request latency objective does not survive being multiplied by turn count",
        concepts=("latency", "agents", "slo"),
        symptom="An agent feature meets its per-generation latency target and misses its user-facing objective badly.",
        chain="A multi-turn loop pays generation latency and tool latency once per turn, so an objective written for a single generation is exceeded by a factor equal to the turn count before any component is slow.",
        metric="End-to-end task latency decomposed into per-turn generation time and per-turn tool time.",
        signature="Task latency equals turn count times the sum of the two per-turn terms, with no single component exceeding its own target.",
        confounders=(
            "One slow tool dominating the total, which is a component problem rather than a loop problem.",
            "Queue wait between turns, which adds a term outside both components.",
            "Turn count varying by task, so an average conceals the tail.",
        ),
        fixes=(
            "State the objective per task rather than per generation, and derive the per-turn budget from it.",
            "Cap turn count so the worst case is bounded rather than distributed.",
            "Stream partial results so perceived latency is decoupled from total task time.",
        ),
        rollback="Relax the turn cap if task success rate falls, and restate the objective rather than keeping both an unreachable cap and an unmet objective.",
        options=("deriving the per-turn budget from a task-level objective", "streaming partial results to decouple perceived latency"),
        tradeoff="whether the user needs the final answer quickly or needs visible progress",
        flip="the task genuinely requires all turns before anything meaningful can be shown, at which point streaming cannot help and only the turn budget matters",
        falsifier="task latency is close to a single turn's latency",
        wrong_claim="Each generation completes within its latency target, so the feature meets its objective.",
        wrong_why="The objective is per task and the loop performs many generations, so component targets being met says nothing about whether the aggregate is within budget.",
        threshold="Set the per-turn budget as the task objective divided by the maximum permitted turn count.",
        cost="A feature that meets component targets and misses the user objective consumes the full serving cost while delivering an unacceptable experience.",
        scaling="Latency grows linearly with turn count while the objective is fixed, so richer agent behaviour directly consumes the budget.",
        quant=q_tool_latency,
    ),
    Mechanism(
        key="constrained_decoding_scope", topic="agents",
        title="constrained decoding guarantees syntax and leaves every semantic check outstanding",
        concepts=("structured_output", "validation", "constrained_decoding"),
        symptom="Structured output always parses cleanly and occasionally contains values that are impossible or not permitted.",
        chain="A grammar restricts which tokens may be emitted at each position, so the output conforms to the schema's shape, but nothing in that mechanism examines whether the values are within range, whether the named tool exists or whether the caller is allowed to invoke it.",
        metric="Rejection counts split into syntactic failures and semantic failures, tracked separately.",
        signature="Syntactic failures fall to zero after constrained decoding is enabled while semantic failures are unchanged.",
        confounders=(
            "The grammar itself encoding some value constraints, which shifts part of the semantic work into it.",
            "Retries masking semantic rejections so their rate is not visible.",
            "Provider-side changes to the schema, which alter the syntactic surface without notice.",
        ),
        fixes=(
            "Keep semantic validation in place after enabling constrained decoding, and measure it separately.",
            "Encode range and enumeration constraints into the grammar where it can express them.",
            "Check authorisation at dispatch rather than trusting the tool name that was generated.",
        ),
        rollback="Restore any semantic check that was removed when constrained decoding was introduced, since the two address different failures.",
        options=("keeping semantic validation after the grammar", "encoding range constraints into the grammar itself"),
        tradeoff="whether the constraint is expressible in the grammar or requires runtime context",
        flip="the constraint depends on runtime state such as the caller's permissions, at which point no grammar can express it and only dispatch-time checking works",
        falsifier="semantic rejection counts fall alongside syntactic ones after the grammar is enabled",
        wrong_claim="Output is schema-constrained, so downstream validation is redundant and can be removed.",
        wrong_why="The schema constrains shape rather than meaning, so removing downstream validation removes exactly the checks the grammar never performed.",
        threshold="Track syntactic and semantic rejection rates separately and never let one stand in for the other.",
        cost="Removing semantic validation on the strength of a syntactic guarantee is a security regression disguised as a simplification.",
        scaling="Semantic surface grows with the number and privilege of tools, while syntactic surface does not, so the gap widens as the agent gains capabilities.",
        quant=q_grammar_constraint,
    ),
    Mechanism(
        key="tool_idempotency", topic="agents",
        title="retry safety is a property of the tool, not of the retry policy",
        concepts=("idempotency", "retries", "correctness"),
        symptom="Duplicate actions appear downstream after a period of elevated timeouts in the agent stack.",
        chain="A timed-out tool call may have executed successfully, and neither the retry layer nor the model can observe which, so a retry of a non-idempotent tool performs the action a second time.",
        metric="Duplicate downstream actions correlated with retry events and with timeout occurrences.",
        signature="Duplicates cluster around timeouts rather than around errors, since a timeout is the ambiguous case and an explicit error is not.",
        confounders=(
            "The model itself re-issuing a call after seeing an ambiguous result, which duplicates without any retry layer.",
            "Upstream client retries producing duplicate user requests.",
            "Downstream deduplication being present but keyed on the wrong field.",
        ),
        fixes=(
            "Require a caller-supplied idempotency key on every tool that mutates state.",
            "Classify timeouts as non-retryable for tools that cannot be made idempotent.",
            "Make the tool return its prior result on a repeated key rather than executing again.",
        ),
        rollback="Disable retries for the affected tool rather than tuning the retry policy, since the policy is not where the defect lives.",
        options=("requiring an idempotency key on mutating tools", "classifying timeouts as non-retryable for those tools"),
        tradeoff="whether the tool can be made to recognise a repeated request",
        flip="the tool is third-party and cannot accept a key, at which point refusing to retry on timeout is the only safe option",
        falsifier="duplicates occur independently of retry and timeout events",
        wrong_claim="The retry policy uses exponential backoff and a low attempt limit, so it is safe.",
        wrong_why="Backoff and attempt limits govern load rather than correctness, and retrying a non-idempotent action even once produces a duplicate.",
        threshold="Permit retries only on tools that accept an idempotency key or are provably read-only.",
        cost="Duplicate mutating actions require manual reconciliation downstream, which costs far more than the failed request would have.",
        scaling="Duplicate volume rises with concurrency and with timeout rate together, so the worst duplication occurs during the incidents that trigger the most retries.",
        quant=q_idempotency,
    ),
    Mechanism(
        key="agent_loop_detection", topic="agents",
        title="context exhaustion is not a loop detector, it is the absence of one",
        concepts=("agents", "loop_detection", "termination"),
        symptom="Agent runs occasionally repeat the same action many times and terminate only when the context window fills.",
        chain="Nothing in the loop compares the current state against previous states, so a model that re-derives the same next action continues until the growing prompt reaches the window limit and the request fails on length.",
        metric="Repeated identical or near-identical tool calls within one run, counted per run.",
        signature="The run terminates at the window limit rather than at a task outcome, and the call history contains a repeating cycle.",
        confounders=(
            "A legitimately repeated call with different arguments, such as pagination.",
            "A tool returning a non-deterministic result that genuinely warrants retrying.",
            "The model making slow but real progress that resembles repetition at the call level.",
        ),
        fixes=(
            "Detect repeated call signatures within a run and terminate or intervene explicitly.",
            "Cap turns so the failure is bounded and attributable rather than reaching the window.",
            "Feed the repetition back to the model as an observation so it can change strategy.",
        ),
        rollback="Loosen the repetition rule if legitimate repeated calls such as pagination are being terminated, and key the detector on arguments rather than tool name alone.",
        options=("detecting repeated call signatures and intervening", "capping turns so the failure is bounded"),
        tradeoff="whether repetition can be distinguished from legitimate iteration by the call signature",
        flip="legitimate iteration produces identical signatures, at which point only a turn cap can bound the run without false termination",
        falsifier="runs that reach the window limit contain no repeated call signature",
        wrong_claim="The context window bounds the loop, so a runaway agent cannot run indefinitely.",
        wrong_why="The window bounds the total but does not detect the condition, so the run consumes every generation up to the limit and then fails without producing a diagnosis.",
        threshold="Terminate on a repeated call signature before the window becomes the binding limit.",
        cost="Every generation in the repeating cycle is billed and produces nothing, and the run still ends in failure.",
        scaling="Wasted generations scale with window size, so a larger context window makes an undetected loop more expensive rather than safer.",
        quant=q_loop_detection,
    ),
    Mechanism(
        key="parallel_tool_scope", topic="agents",
        title="parallel tool execution shortens only the tool term of an agent loop",
        concepts=("agents", "parallelism", "latency"),
        symptom="Parallelising tool calls delivered a much smaller latency improvement than the number of parallelised calls suggested.",
        chain="A loop alternates generation and tool execution, and only the tool term can be parallelised, so the improvement is bounded by that term's share of total latency regardless of how many calls run together.",
        metric="Loop latency decomposed into generation time and tool time before any parallelisation.",
        signature="The measured improvement equals the tool term's share times the parallel speedup within it, and generation time is unchanged.",
        confounders=(
            "One tool dominating the parallel group, which caps the parallel speedup at that tool's latency.",
            "Added concurrency causing contention downstream, which lengthens each call.",
            "Fewer turns being needed after parallelisation, which is a separate and larger effect.",
        ),
        fixes=(
            "Decompose loop latency before choosing what to parallelise.",
            "Reduce turn count if generation dominates, since that addresses the larger term.",
            "Parallelise only where the tool term is a material share and the calls are genuinely independent.",
        ),
        rollback="Return to serial execution if downstream contention makes each parallel call slower than the serial total saved.",
        options=("decomposing loop latency before parallelising", "reducing turn count where generation dominates"),
        tradeoff="whether generation or tool execution is the larger share of loop latency",
        flip="tool latency grows until it dominates the loop, at which point parallelising it becomes the highest-value change",
        falsifier="tool execution accounts for most of the loop's latency",
        wrong_claim="We run four tools in parallel now, so that part of the agent is four times faster.",
        wrong_why="The tool term may be a small share of loop latency, so a fourfold improvement within it can be nearly invisible end to end, and the generation term is untouched.",
        threshold="Require the tool term's measured share before scheduling parallelisation work.",
        cost="Engineering effort spent on the smaller term leaves the dominant one untouched and the objective still unmet.",
        scaling="Generation time grows with turn count while tool parallelism is bounded by the slowest call, so the tool term's share falls as loops get longer.",
        quant=q_parallel_tools,
    ),
    Mechanism(
        key="tool_result_truncation", topic="agents",
        title="a truncated tool result is a data-loss event the model cannot detect",
        concepts=("truncation", "tool_calling", "context_window"),
        symptom="An agent confidently produces a wrong answer from a tool result that contained the correct information.",
        chain="When a tool result exceeds the space left in the window it is cut to fit, usually from the end, and the model receives a fragment with no indication that anything was removed, so it reasons over partial data as if it were complete.",
        metric="Truncation events per tool call, recorded with the original and retained sizes.",
        signature="Wrong answers correlate with truncation events on the specific call that supplied the evidence.",
        confounders=(
            "The tool genuinely returning incomplete data, which produces the same outcome without truncation.",
            "The model ignoring information that was present, which is a reasoning failure rather than a data-loss one.",
            "History summarisation removing the result in an earlier turn.",
        ),
        fixes=(
            "Signal truncation explicitly in the result so the model knows it is seeing a fragment.",
            "Summarise deliberately rather than cutting, so the retained portion is chosen rather than arbitrary.",
            "Store the full result externally and pass a reference the model can query further.",
        ),
        rollback="Return to passing full results if summarisation loses the information the task depends on, and reduce turn count instead to free window space.",
        options=("signalling truncation explicitly in the result", "summarising deliberately instead of cutting"),
        tradeoff="whether the important content can be identified before the result is shortened",
        flip="the important content cannot be identified in advance, at which point an external reference the model can query is the only lossless option",
        falsifier="the wrong answers occur on calls whose results were not truncated",
        wrong_claim="The tool returned the right data, so the model had everything it needed to answer correctly.",
        wrong_why="What the tool returned and what reached the model are different once truncation is in the path, and the model receives no signal that the two differ.",
        threshold="Require every truncation event to be recorded and surfaced to the model rather than applied silently.",
        cost="A confidently wrong answer derived from truncated evidence is more damaging than an explicit failure, because nothing downstream flags it.",
        scaling="Truncation frequency rises as history grows within a run, so it strikes hardest in the later turns where the task is closest to completion.",
        quant=q_output_truncation,
    ),
)
