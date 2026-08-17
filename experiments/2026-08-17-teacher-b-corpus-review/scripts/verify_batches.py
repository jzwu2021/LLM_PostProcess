#!/usr/bin/env python3
"""Ad-hoc verification for teacher-B provisional review batches."""
import json, os, sys, glob

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
RES = os.path.join(EXP, "results")
CORP = {
    "train": os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl"),
    "validation": os.path.join(ROOT, "research/ai-infra-expert/corpus/validation.jsonl"),
}
REQ = ["source_id", "teacher_lane", "teacher_model", "calibration_status", "decision",
       "source_user", "source_assistant", "corrected_answer", "quality_dimensions",
       "risks", "evidence_required", "confidence"]
DIMS = ["technical_correctness", "instruction_coverage", "operational_safety"]

errs = []


def load_corpus(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            u = next(m["content"] for m in d["messages"] if m["role"] == "user")
            a = next(m["content"] for m in d["messages"] if m["role"] == "assistant")
            out.append((d["id"], u, a))
    return out


def main():
    seen = set()
    totals = {}
    for split in ("train", "validation"):
        corpus = load_corpus(CORP[split])
        cmap = {c[0]: c for c in corpus}
        files = sorted(glob.glob(os.path.join(RES, f"{split}-batch-*.jsonl")))
        seq = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                raw = f.read()
            if raw and not raw.endswith("\n"):
                errs.append(f"{fp}: missing trailing newline")
            lines = [l for l in raw.split("\n") if l.strip()]
            for ln, line in enumerate(lines, 1):
                try:
                    r = json.loads(line)
                except Exception as e:
                    errs.append(f"{fp}:{ln}: not valid JSON: {e}")
                    continue
                for k in REQ:
                    if k not in r:
                        errs.append(f"{fp}:{ln}: missing field {k}")
                if [k for k in REQ if k not in r]:
                    continue
                if r["teacher_lane"] != "teacher-B":
                    errs.append(f"{fp}:{ln}: bad teacher_lane")
                if r["teacher_model"] != "claude-opus-5-current":
                    errs.append(f"{fp}:{ln}: bad teacher_model")
                if r["calibration_status"] != "provisional":
                    errs.append(f"{fp}:{ln}: bad calibration_status")
                if r["decision"] not in ("keep", "rewrite", "reject"):
                    errs.append(f"{fp}:{ln}: bad decision {r['decision']!r}")
                if not isinstance(r["corrected_answer"], str) or not r["corrected_answer"].strip():
                    errs.append(f"{fp}:{ln}: empty corrected_answer")
                qd = r["quality_dimensions"]
                if not isinstance(qd, dict) or sorted(qd) != sorted(DIMS):
                    errs.append(f"{fp}:{ln}: bad quality_dimensions keys")
                else:
                    for k in DIMS:
                        v = qd[k]
                        if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 5):
                            errs.append(f"{fp}:{ln}: {k} not int 1-5")
                for k in ("risks", "evidence_required"):
                    v = r[k]
                    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                        errs.append(f"{fp}:{ln}: {k} not list[str]")
                c = r["confidence"]
                if not isinstance(c, (int, float)) or isinstance(c, bool) or not (0.0 <= c <= 1.0):
                    errs.append(f"{fp}:{ln}: confidence out of [0,1]")
                sid = r["source_id"]
                if sid in seen:
                    errs.append(f"{fp}:{ln}: duplicate source_id {sid}")
                seen.add(sid)
                if sid not in cmap:
                    errs.append(f"{fp}:{ln}: source_id {sid} not in {split} corpus")
                else:
                    _, u, a = cmap[sid]
                    if r["source_user"] != u:
                        errs.append(f"{fp}:{ln}: source_user mismatch for {sid}")
                    if r["source_assistant"] != a:
                        errs.append(f"{fp}:{ln}: source_assistant mismatch for {sid}")
                seq.append(sid)
        prefix = [c[0] for c in corpus[:len(seq)]]
        if seq != prefix:
            errs.append(f"{split}: sequence is not a strict prefix of corpus order")
        totals[split] = len(seq)

    print(f"train={totals['train']}/5399 validation={totals['validation']}/601 "
          f"total={totals['train']+totals['validation']}/6000")
    if errs:
        print("VERIFY_FAIL")
        for e in errs[:50]:
            print(" -", e)
        sys.exit(1)
    print("VERIFY_PASS")


if __name__ == "__main__":
    main()
