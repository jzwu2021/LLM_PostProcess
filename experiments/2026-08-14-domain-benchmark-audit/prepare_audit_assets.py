#!/usr/bin/env python3
"""Prepare benchmark audit assets without inventing expert judgments."""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:,\d{3})*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?")


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def numbers(text):
    result = []
    for raw in NUMBER_RE.findall(text or ""):
        try:
            result.append(float(raw.rstrip('%').replace(',', '')))
        except ValueError:
            pass
    return result


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--generations', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    benchmark = read_jsonl(args.benchmark)
    generations = {r['id']: r for r in read_jsonl(args.generations)}
    if len(benchmark) != 500 or len({r['id'] for r in benchmark}) != 500:
        raise SystemExit('expected 500 unique benchmark records')
    if set(generations) != {r['id'] for r in benchmark}:
        raise SystemExit('benchmark/generation IDs do not match')

    queue = []
    for row in benchmark:
        verifier = row['verifier']
        if verifier == 'numeric_tolerance':
            asset = 'numeric_metadata'
        elif verifier in {'unit_test', 'unit_test_plus_rubric'}:
            asset = 'code_sandbox_fixture'
        elif verifier == 'rubric_1_4':
            asset = 'blind_rubric_packet'
        else:
            asset = 'key_point_annotation'
        queue.append({
            'id': row['id'],
            'category': row['category'],
            'difficulty': row['difficulty'],
            'topic': row['topic'],
            'verifier': verifier,
            'evaluation_asset': asset,
            'audit_status': 'unreviewed',
            'provenance_status': row.get('provenance_status'),
            'evidence_status': 'pending',
            'contamination_review_status': 'pending',
            'reviewer_1': None,
            'reviewer_2': None,
        })
    write_jsonl(out / 'audit-queue.jsonl', queue)

    # Deterministic two-per-category calibration sample, preferring difficulty diversity.
    by_category = {}
    for row in benchmark:
        by_category.setdefault(row['category'], []).append(row)
    sample = []
    for category in sorted(by_category):
        candidates = sorted(by_category[category], key=lambda r: (r['difficulty'], r['id']))
        chosen = []
        for row in candidates:
            if not chosen or row['difficulty'] != chosen[0]['difficulty']:
                chosen.append(row)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            chosen = candidates[:2]
        for row in chosen:
            gen = generations[row['id']]
            sample.append({
                'id': row['id'],
                'category': row['category'],
                'difficulty': row['difficulty'],
                'topic': row['topic'],
                'question': row['question'],
                'reference_outline': row['reference_answer'],
                'verifier': row['verifier'],
                'model_response': gen.get('response_content', ''),
                'finish_reason': gen.get('finish_reason'),
                'rubric_version': 'domain-rubric-draft-v0.1',
                'review_status': 'pending_two_reviewer_calibration',
                'reviewer_scores': {'reviewer_1': None, 'reviewer_2': None},
                'adjudicated_score': None,
            })
    write_jsonl(out / 'calibration-sample.jsonl', sample)

    numeric = []
    code = []
    for row in benchmark:
        if row['verifier'] == 'numeric_tolerance':
            numeric.append({
                'id': row['id'], 'question': row['question'],
                'reference_answer': row['reference_answer'],
                'reference_numeric_candidates_draft': numbers(row['reference_answer']),
                'tolerance_draft': 0.01,
                'unit_normalization': None,
                'assumptions': None,
                'answer_key_status': 'expert_verification_required',
            })
        if row['verifier'] in {'unit_test', 'unit_test_plus_rubric'}:
            code.append({
                'id': row['id'], 'question': row['question'],
                'contract_draft': row['reference_answer'],
                'fixture_status': 'not_implemented',
                'network': 'disabled', 'filesystem': 'temporary_sandbox',
                'cpu_timeout_s': 5, 'memory_limit_mb': 256,
                'test_status': 'expert_fixture_authoring_required',
            })
    write_jsonl(out / 'numeric-metadata-draft.jsonl', numeric)
    write_jsonl(out / 'code-fixture-plan.jsonl', code)

    rubric = {
        'rubric_version': 'domain-rubric-draft-v0.1',
        'status': 'draft_pending_domain_expert_calibration',
        'score_scale': {'1': 'incorrect_or_unsafe', '2': 'partially_correct', '3': 'substantially_correct', '4': 'expert_level_for_scope'},
        'dimensions': [
            {'name': 'technical_correctness', 'score_required': True},
            {'name': 'required_component_coverage', 'score_required': True},
            {'name': 'tradeoff_constraint_reasoning', 'score_required': True},
            {'name': 'validation_measurement_plan', 'score_required': True},
            {'name': 'operational_safety_hallucination', 'score_required': True},
        ],
        'review_protocol': {
            'calibration_cases': 20,
            'independent_reviewers': 2,
            'adjudication': 'required_when_dimension_disagreement_exceeds_one_point',
            'agreement_metric': 'weighted_quadratic_kappa_after_calibration',
        },
    }
    (out / 'rubric-schema.json').write_text(json.dumps(rubric, ensure_ascii=False, indent=2) + '\n')
    summary = {
        'status': 'AUDIT_SCAFFOLD_ONLY',
        'records': len(benchmark),
        'calibration_cases': len(sample),
        'numeric_metadata_drafts': len(numeric),
        'code_fixture_plans': len(code),
        'categories': dict(Counter(r['category'] for r in benchmark)),
        'difficulty': dict(Counter(r['difficulty'] for r in benchmark)),
        'verifiers': dict(Counter(r['verifier'] for r in benchmark)),
        'all_records_require_expert_review': True,
        'authoritative_score_available': False,
    }
    (out / 'audit-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(f"AUDIT_SCAFFOLD_DONE records={len(benchmark)} calibration={len(sample)} numeric={len(numeric)} code={len(code)}")


if __name__ == '__main__':
    main()
