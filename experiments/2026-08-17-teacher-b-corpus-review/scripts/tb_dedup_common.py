"""Shared family map and record builders for the deduplicating teacher-B review stage.

A family is the (variant-normalised user prompt, assistant text) pair. The remaining
corpus carries 22 families, so most rows are near-duplicates that add no supervision
signal; those are rejected with a reference to the family's retained exemplars.
"""
import json
import re
from collections import defaultdict

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
RESULTS = f"{EXP}/results"

REVIEWED_PREFIX = 2580          # rows already reviewed before this stage
EXEMPLARS_PER_FAMILY = 3        # retained rewrites per family, rest rejected


def load_corpus():
    return [json.loads(l) for l in open(CORPUS) if l.strip()]


def msg(row, role):
    return next(x["content"] for x in row["messages"] if x["role"] == role)


def family_key(row):
    user = re.sub(r"\bvariant\s*\d+", "V", msg(row, "user"), flags=re.I)
    return (re.sub(r"\d+", "N", user).strip(), msg(row, "assistant").strip())


def build_plan():
    """Return (family_of_index, plan) where plan[index] is 'rewrite' or 'reject'."""
    corpus = load_corpus()
    members = defaultdict(list)
    for i, row in enumerate(corpus):
        members[family_key(row)].append(i)

    family_of = {}
    exemplars = {}
    plan = {}
    for fam_id, (key, idx) in enumerate(sorted(members.items(), key=lambda kv: min(kv[1]))):
        already = [i for i in idx if i < REVIEWED_PREFIX]
        remaining = [i for i in idx if i >= REVIEWED_PREFIX]
        # A family already covered by the reviewed prefix needs no new exemplars.
        slots = max(0, EXEMPLARS_PER_FAMILY - len(already))
        chosen = remaining[:slots]
        exemplars[fam_id] = {
            "size": len(idx),
            "reviewed_before_stage": len(already),
            "remaining": len(remaining),
            "exemplar_indices": chosen,
            "exemplar_ids": [corpus[i]["id"] for i in chosen] or [corpus[i]["id"] for i in already[:EXEMPLARS_PER_FAMILY]],
        }
        for i in idx:
            family_of[i] = fam_id
            if i >= REVIEWED_PREFIX:
                plan[i] = "rewrite" if i in chosen else "reject"
    return corpus, family_of, exemplars, plan


def reject_answer(corpus, index, fam_id, info):
    """Review output for a duplicate row: the finding is the duplication itself."""
    row = corpus[index]
    ids = ", ".join(info["exemplar_ids"])
    return (
        f"DUPLICATE OF FAMILY {fam_id:03d} - no additional supervision signal; recommend removal from the training mixture.\n\n"
        f"Finding. This record is one of {info['size']} rows in train.jsonl whose user turn differs from the others only by a "
        f"variant number and whose assistant turn is byte-identical across the whole family. Reviewing it as an independent item "
        f"would require inventing a distinct answer for a question that has already been answered, which manufactures stylistic "
        f"variety rather than domain signal.\n\n"
        f"Retained exemplars. {ids}. Those records carry the substantive review for this family; this record adds nothing beyond them.\n\n"
        f"Why duplicates are harmful here. Under a fixed step budget every duplicate consumes an optimiser step, so a family of "
        f"{info['size']} rows receives roughly {info['size']} times the weight of a singleton family. That concentrates training on one "
        f"question and its single rubric answer, and if that answer is defective the defect is amplified rather than averaged away. "
        f"The source assistant turn here is a grading rubric rather than an answer, so the amplified content is a description of what a "
        f"good answer would contain instead of the answer itself.\n\n"
        f"Recommended action. Drop this record from the supervised mixture and retain the exemplars. If variety within the family is "
        f"wanted, generate it from distinct underlying scenarios with distinct reference answers rather than by renumbering a variant, "
        f"because renumbering changes the prompt without changing the supervision target.\n\n"
        f"Evidence label. The family size and the byte-identity of the assistant turns are MEASURED by grouping train.jsonl; no "
        f"ESTIMATE is involved in this decision."
    )


def reject_record(corpus, index, fam_id, info):
    row = corpus[index]
    return {
        "source_id": row["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "reject",
        "source_user": msg(row, "user"),
        "source_assistant": msg(row, "assistant"),
        "corrected_answer": reject_answer(corpus, index, fam_id, info),
        "quality_dimensions": {
            "technical_correctness": 2,
            "instruction_coverage": 2,
            "operational_safety": 2,
        },
        "risks": [
            f"Record duplicates family {fam_id:03d}, which contains {info['size']} rows sharing one byte-identical assistant turn.",
            "Retaining the family at full size concentrates the training budget on a single question and its single rubric answer.",
            "The shared assistant turn is a grading rubric rather than an answer, so the amplified supervision target is a description of an answer.",
        ],
        "evidence_required": [
            "Family grouping over train.jsonl showing the variant-normalised prompt and byte-identical assistant turn across all members.",
            f"Retained exemplar records {', '.join(info['exemplar_ids'])} carrying the substantive review for this family.",
        ],
        "confidence": 0.9,
    }


def rewrite_record(corpus, index, head, body, qd, risks, evidence, confidence):
    row = corpus[index]
    return {
        "source_id": row["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": msg(row, "user"),
        "source_assistant": msg(row, "assistant"),
        "corrected_answer": head + "\n\n" + body,
        "quality_dimensions": {
            "technical_correctness": qd[0],
            "instruction_coverage": qd[1],
            "operational_safety": qd[2],
        },
        "risks": risks,
        "evidence_required": evidence,
        "confidence": confidence,
    }
