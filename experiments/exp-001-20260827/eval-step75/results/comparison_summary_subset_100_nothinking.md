# Comparison summary: subset-100 no-thinking rerun

This rerun holds the benchmark subset fixed and explicitly sets chat_template_kwargs.enable_thinking=false for the step-75 model to remove the reasoning-template confounder.

## Base subset summary

- cases: 100
- generation_ok: 100
- nonempty: 100
- finish_reasons: {"length": 99, "stop": 1}
- code_fence_responses: 13
- mean_response_chars: 3049.9
- median_response_chars: 3067.5
- mean_latency_ms: 5343.3
- keypoint_cases: 81
- mean_keypoint_coverage: 0.3339
- median_keypoint_coverage: 0.2778
- numeric_cases: 10
- numeric_all_expected_matched: 1
- numeric_mean_match_fraction: 0.8125
- by_category_keypoint_coverage: {"Architecture Comparison": 0.2744, "Code": 0.2105, "Concept Understanding": 0.6652, "Knowledge": 0.5006, "Long-form Technical Analysis": 0.284, "Performance Analysis": 0.3642, "Reasoning": 0.1626, "System Design": 0.2027, "Troubleshooting": 0.2296}

## Prior step-75 summary

- cases: 100
- generation_ok: 100
- nonempty: 100
- finish_reasons: {"length": 100}
- code_fence_responses: 0
- mean_response_chars: 3008.3
- median_response_chars: 3054.0
- mean_latency_ms: 5243.4
- keypoint_cases: 81
- mean_keypoint_coverage: 0.316
- median_keypoint_coverage: 0.3077
- numeric_cases: 10
- numeric_all_expected_matched: 1
- numeric_mean_match_fraction: 0.8
- by_category_keypoint_coverage: {"Architecture Comparison": 0.1934, "Code": 0.2632, "Concept Understanding": 0.5095, "Knowledge": 0.3976, "Long-form Technical Analysis": 0.312, "Performance Analysis": 0.3433, "Reasoning": 0.2432, "System Design": 0.2584, "Troubleshooting": 0.2762}

## Step-75 no-thinking summary

- cases: 100
- generation_ok: 100
- nonempty: 100
- finish_reasons: {"length": 97, "stop": 3}
- code_fence_responses: 14
- mean_response_chars: 3053.7
- median_response_chars: 3103.0
- mean_latency_ms: 5234.3
- keypoint_cases: 81
- mean_keypoint_coverage: 0.3357
- median_keypoint_coverage: 0.2857
- numeric_cases: 10
- numeric_all_expected_matched: 1
- numeric_mean_match_fraction: 0.8125
- by_category_keypoint_coverage: {"Architecture Comparison": 0.2556, "Code": 0.3158, "Concept Understanding": 0.6439, "Knowledge": 0.5098, "Long-form Technical Analysis": 0.3028, "Performance Analysis": 0.3496, "Reasoning": 0.1674, "System Design": 0.2295, "Troubleshooting": 0.2291}

## Delta vs base

- mean_keypoint_coverage: 0.0018
- numeric_all_expected_matched: 0
- numeric_mean_match_fraction: 0.0
- mean_response_chars: 3.8
- mean_latency_ms: -109.0
- code_fence_responses: 1

## Top 10 regressions vs base

- aiinfra-0076 | Concept Understanding | delta=-0.142857 | base=0.5 | step75=0.35714285714285715
- aiinfra-0401 | Long-form Technical Analysis | delta=-0.142857 | base=0.2857142857142857 | step75=0.14285714285714285
- aiinfra-0331 | Architecture Comparison | delta=-0.130435 | base=0.2608695652173913 | step75=0.13043478260869565
- aiinfra-0206 | Performance Analysis | delta=-0.117647 | base=0.4117647058823529 | step75=0.29411764705882354
- aiinfra-0216 | Performance Analysis | delta=-0.1 | base=0.55 | step75=0.45
- aiinfra-0286 | Troubleshooting | delta=-0.1 | base=0.35 | step75=0.25
- aiinfra-0341 | Architecture Comparison | delta=-0.086957 | base=0.34782608695652173 | step75=0.2608695652173913
- aiinfra-0306 | Architecture Comparison | delta=-0.083333 | base=0.4166666666666667 | step75=0.3333333333333333
- aiinfra-0361 | Reasoning | delta=-0.083333 | base=0.25 | step75=0.16666666666666666
- aiinfra-0301 | Architecture Comparison | delta=-0.074074 | base=0.3333333333333333 | step75=0.25925925925925924

## Top 10 improvements vs base

- aiinfra-0431 | Long-form Technical Analysis | delta=0.222222 | base=0.2777777777777778 | step75=0.5
- aiinfra-0236 | Performance Analysis | delta=0.176471 | base=0.29411764705882354 | step75=0.47058823529411764
- aiinfra-0326 | Architecture Comparison | delta=0.115385 | base=0.2692307692307692 | step75=0.38461538461538464
- aiinfra-0406 | Long-form Technical Analysis | delta=0.111111 | base=0.2222222222222222 | step75=0.3333333333333333
- aiinfra-0481 | Code | delta=0.105263 | base=0.21052631578947367 | step75=0.3157894736842105
- aiinfra-0386 | Reasoning | delta=0.095238 | base=0.09523809523809523 | step75=0.19047619047619047
- aiinfra-0266 | Troubleshooting | delta=0.090909 | base=0.2727272727272727 | step75=0.36363636363636365
- aiinfra-0276 | Troubleshooting | delta=0.090909 | base=0.13636363636363635 | step75=0.22727272727272727
- aiinfra-0001 | Knowledge | delta=0.083333 | base=0.5 | step75=0.5833333333333334
- aiinfra-0371 | Reasoning | delta=0.083333 | base=0.08333333333333333 | step75=0.16666666666666666

