#!/usr/bin/env python3
"""Layered, non-authoritative evaluator for the frozen domain benchmark.

Automated checks are only authoritative for generation validity. Numeric
matching and lexical overlap remain diagnostics until the benchmark has audited
answers, unit contracts, code fixtures, and blinded rubric scores.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "which", "with", "one", "two", "than", "into", "their", "its", "has", "have",
    "does", "do", "how", "why", "when", "where", "versus", "include", "including",
}
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:,\d{3})*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def numbers(text):
    out = []
    for raw in NUMBER_RE.findall(text or ""):
        try:
            out.append(float(raw.rstrip("%").replace(",", "")))
        except ValueError:
            pass
    return out


def content_words(text):
    return {
        word.lower() for word in WORD_RE.findall(text or "")
        if word.lower() not in STOPWORDS and len(word) >= 4
    }


def numeric_diagnostic(reference, answer):
    expected = numbers(reference)
    observed = numbers(answer)
    if not expected:
        return {"status": "not_applicable", "expected_numbers": [], "observed_numbers": observed, "matched": None}
    matches = []
    for exp in expected:
        if exp == 0:
            matched = any(obs == 0 for obs in observed)
        else:
            matched = any(math.isclose(obs, exp, rel_tol=0.01, abs_tol=1e-9) for obs in observed)
        matches.append(matched)
    return {
        "status": "heuristic_only",
        "expected_numbers": expected,
        "observed_numbers": observed,
        "matched_expected_count": sum(matches),
        "expected_count": len(matches),
        "all_expected_matched": bool(matches) and all(matches),
    }


def keypoint_diagnostic(reference, answer):
    ref = content_words(reference)
    ans = content_words(answer)
    matched = sorted(ref & ans)
    return {
        "status": "heuristic_only",
        "reference_content_word_count": len(ref),
        "matched_content_words": matched,
        "coverage": (len(matched) / len(ref)) if ref else None,
    }


def evaluate_case(benchmark, generation):
    answer = generation.get("response_content") or ""
    verifier = benchmark.get("verifier")
    result = {
        "id": benchmark["id"],
        "category": benchmark["category"],
        "verifier": verifier,
        "generation_ok": bool(generation.get("ok")),
        "finish_reason": generation.get("finish_reason"),
        "nonempty_response": bool(answer.strip()),
        "response_chars": len(answer),
        "response_has_code_fence": "```" in answer,
    }
    if verifier == "numeric_tolerance":
        result["evaluation_tier"] = "calculation"
        result["numeric"] = numeric_diagnostic(benchmark.get("reference_answer", ""), answer)
    elif verifier in {"contains_key_points", "rubric_1_4", "unit_test_plus_rubric"}:
        result["evaluation_tier"] = (
            "code_execution_and_blind_rubric"
            if verifier == "unit_test_plus_rubric"
            else "open_ended_blind_rubric"
        )
        result["key_points"] = keypoint_diagnostic(benchmark.get("reference_answer", ""), answer)
    else:
        result["evaluation_tier"] = "code_execution"
    if verifier in {"rubric_1_4", "unit_test_plus_rubric"}:
        result["human_review"] = "needs_blind_rubric"
        result["rubric_dimensions"] = [
            "technical_correctness",
            "completeness",
            "operational_actionability",
            "constraints_and_risks",
            "unsupported_claims",
        ]
    if verifier in {"unit_test", "unit_test_plus_rubric"}:
        result["code_execution"] = "needs_sandbox_fixture"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    benchmark = load_jsonl(args.benchmark)
    generations = load_jsonl(args.generations)
    by_id = {row["id"]: row for row in generations}
    if len(by_id) != len(generations):
        raise SystemExit("duplicate generation IDs")

    missing = [row["id"] for row in benchmark if row["id"] not in by_id]
    if missing:
        raise SystemExit(f"missing generations: {missing[:5]}")

    cases = [evaluate_case(row, by_id[row["id"]]) for row in benchmark]
    by_category = defaultdict(list)
    for case in cases:
        by_category[case["category"]].append(case)

    truncated_cases = [c["id"] for c in cases if c["finish_reason"] == "length"]
    report = {
        "evaluator": "layered-domain-evaluator-v0.2",
        "authoritative_domain_score": None,
        "interpretation": "No domain capability score is claimed until blind rubric and sandbox evidence are available.",
        "generation_validity": {
            "status": "pass" if not truncated_cases and not missing and all(c["generation_ok"] and c["nonempty_response"] for c in cases) else "fail",
            "requirements": {
                "all_cases_generated": True,
                "all_responses_nonempty": True,
                "no_length_capped_responses": True,
            },
            "length_capped_case_ids": truncated_cases,
        },
        "benchmark_cases": len(benchmark),
        "generation_cases": len(generations),
        "missing_generation_cases": len(missing),
        "generation_ok": sum(c["generation_ok"] for c in cases),
        "nonempty_responses": sum(c["nonempty_response"] for c in cases),
        "finish_reasons": dict(Counter(c["finish_reason"] for c in cases)),
        "code_fence_responses": sum(c["response_has_code_fence"] for c in cases),
        "review_requirements": {
            "blind_rubric_cases": sum(c.get("human_review") == "needs_blind_rubric" for c in cases),
            "sandbox_fixture_cases": sum("code_execution" in c for c in cases),
            "numeric_cases": sum("numeric" in c for c in cases),
            "lexical_diagnostic_cases": sum("key_points" in c for c in cases),
        },
        "by_category": {},
        "limitations": [
            "Benchmark v0.1 is a curated synthetic scaffold awaiting domain-expert audit.",
            "Reference answers are outlines, not audited exhaustive key-point annotations.",
            "Numeric matching lacks audited unit/formula contracts and is not a validated pass rate.",
            "Generated code was not executed because no per-case sandbox fixtures are versioned.",
            "Open-ended answers require blinded human rubric scoring and inter-rater agreement.",
            "Lexical reference overlap is a diagnostic only and is sensitive to answer length and phrasing.",
        ],
        "cases": cases,
    }
    for category, rows in sorted(by_category.items()):
        report["by_category"][category] = {
            "cases": len(rows),
            "generation_ok": sum(r["generation_ok"] for r in rows),
            "nonempty": sum(r["nonempty_response"] for r in rows),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
            "code_fence_responses": sum(r["response_has_code_fence"] for r in rows),
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"DOMAIN_DIAGNOSTIC_DONE cases={len(cases)} generation_ok={report['generation_ok']} score=None")


if __name__ == "__main__":
    main()
