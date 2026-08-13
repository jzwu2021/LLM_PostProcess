#!/usr/bin/env python3
"""Run the frozen AI/LLM Infrastructure domain benchmark against an OpenAI-compatible endpoint.

This runner records raw responses and transport metrics only. It deliberately does not
claim an expert score: rubric, code sandbox, and citation checks require a separate
validated evaluator/manual review.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an AI/LLM Infrastructure engineering assistant. Answer the user's "
    "technical question directly and rigorously. State assumptions, units, formulas, "
    "trade-offs, uncertainty, and validation steps when relevant. Do not invent "
    "measurements or undocumented system facts."
)


def request_json(url, payload, timeout, retries):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response), attempt
        except Exception as exc:  # record and retry transient endpoint failures
            last = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    if last is None:
        raise RuntimeError("request failed without an exception")
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    benchmark_path = Path(args.benchmark)
    cases = [json.loads(line) for line in benchmark_path.read_text().splitlines() if line.strip()]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    endpoint = args.base.rstrip("/") + "/chat/completions"
    started = time.time()
    counts = Counter()
    latency = defaultdict(list)
    errors = []

    with out.open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case["question"]},
                ],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            started_case = time.perf_counter()
            record = {
                "index": index,
                "id": case["id"],
                "category": case["category"],
                "difficulty": case.get("difficulty"),
                "topic": case.get("topic"),
                "question": case["question"],
                "verifier": case.get("verifier"),
                "request": payload,
            }
            try:
                response, retries_used = request_json(
                    endpoint, payload, args.timeout, args.retries
                )
                elapsed_ms = (time.perf_counter() - started_case) * 1000
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content")
                usage = response.get("usage") or {}
                record.update(
                    {
                        "ok": True,
                        "retries_used": retries_used,
                        "latency_ms": round(elapsed_ms, 3),
                        "finish_reason": choice.get("finish_reason"),
                        "response_content": content,
                        "response_message": message,
                        "usage": usage,
                        "automatic_checks": {
                            "nonempty_response": bool(content and str(content).strip()),
                            "has_code_fence": "```" in (content or ""),
                        },
                    }
                )
                counts[case["category"]] += 1
                latency[case["category"]].append(elapsed_ms)
                print(f"{index}/{len(cases)} {case['id']} OK {elapsed_ms:.0f}ms", flush=True)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started_case) * 1000
                record.update(
                    {
                        "ok": False,
                        "latency_ms": round(elapsed_ms, 3),
                        "error": repr(exc),
                    }
                )
                errors.append({"id": case["id"], "category": case["category"], "error": repr(exc)})
                print(f"{index}/{len(cases)} {case['id']} ERROR {exc!r}", flush=True)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    def stats(values):
        if not values:
            return {"count": 0}
        values = sorted(values)
        return {
            "count": len(values),
            "mean_ms": round(sum(values) / len(values), 3),
            "p50_ms": round(values[(len(values) - 1) // 2], 3),
            "p95_ms": round(values[min(len(values) - 1, int(len(values) * 0.95))], 3),
            "max_ms": round(values[-1], 3),
        }

    summary = {
        "model": args.model,
        "endpoint": args.base,
        "benchmark": str(benchmark_path),
        "cases": len(cases),
        "successful_requests": sum(counts.values()),
        "failed_requests": len(errors),
        "categories": {category: counts[category] for category in sorted(counts)},
        "latency_ms_by_category": {
            category: stats(latency[category]) for category in sorted(latency)
        },
        "errors": errors,
        "decoding": {"temperature": 0.0, "max_tokens": args.max_tokens},
        "system_prompt": SYSTEM_PROMPT,
        "wall_time_s": round(time.time() - started, 3),
        "scoring_status": "raw_generation_only; rubric/code/numeric review pending",
        "limitations": [
            "The benchmark is a v0.1 authored scaffold and still needs domain-expert review.",
            "Open-ended rubric and code answers are not automatically scored by this runner.",
            "Runtime latency is reported separately and is not domain capability evidence.",
        ],
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        f"DOMAIN_BASELINE_DONE cases={len(cases)} successful={summary['successful_requests']} "
        f"failed={summary['failed_requests']} wall_time_s={summary['wall_time_s']}"
    )


if __name__ == "__main__":
    main()
