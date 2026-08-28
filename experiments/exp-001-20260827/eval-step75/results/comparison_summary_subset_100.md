# step-75 subset evaluation summary
## Protocol
- Cases: 100
- Sampling: 10 categories × 10 cases, deterministic positions [1,6,11,16,21,26,31,36,41,46] within each category block
- Model under test: `/media/home/johnson/llm/models/exports/exp-001-20260827/step75-hf`
- Baseline comparator: `2026-08-13-domain-base-baseline` filtered to the same 100 ids
- Generation protocol: temperature=0, max_tokens=768, local vLLM OpenAI-compatible endpoint

## High-level result
- step-75 mean key-point coverage: 0.3160 vs base 0.3339 (delta -0.0179)
- step-75 numeric mean match fraction: 0.8000 vs base 0.8125 (delta -0.0125)
- step-75 numeric all-expected-matched: 1 vs base 1
- step-75 mean response chars: 3008.3 vs base 3049.9
- step-75 mean latency ms: 5243.4 vs base 5343.3
- step-75 code-fence responses: 0 vs base 13
- finish reasons: step-75 {'length': 100} vs base {'length': 99, 'stop': 1}

## Category notes (key-point coverage)
- Architecture Comparison: step-75 0.1934 vs base 0.2744 (delta -0.0810)
- Code: step-75 0.2632 vs base 0.2105 (delta +0.0527)
- Concept Understanding: step-75 0.5095 vs base 0.6652 (delta -0.1557)
- Knowledge: step-75 0.3976 vs base 0.5006 (delta -0.1030)
- Long-form Technical Analysis: step-75 0.3120 vs base 0.2840 (delta +0.0280)
- Performance Analysis: step-75 0.3433 vs base 0.3642 (delta -0.0209)
- Reasoning: step-75 0.2432 vs base 0.1626 (delta +0.0806)
- System Design: step-75 0.2584 vs base 0.2027 (delta +0.0557)
- Troubleshooting: step-75 0.2762 vs base 0.2296 (delta +0.0466)

## Interpretation
- On this 100-case stratified subset, step-75 does not beat the existing base baseline on the evaluator's heuristic key-point / numeric aggregates.
- step-75 is cleaner in formatting (0 code-fence responses on this slice) and slightly faster, but this is not sufficient evidence of better domain capability.
- Therefore the current evidence supports pipeline viability, not benchmark-proven capability gain.

## Methodological limitations
- Benchmark v0.1 is a curated synthetic scaffold awaiting domain-expert audit.
- Reference answers are outlines, not audited exhaustive key-point annotations.
- Numeric matching is a heuristic over extracted numbers and is not a validated pass rate.
- Generated code was not executed because no per-case sandbox fixtures are versioned.
- Open-ended answers require blinded human rubric scoring and inter-rater agreement.
- The 476 length-capped attempt-4 answers remain a generation limitation.
