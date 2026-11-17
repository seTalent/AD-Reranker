"""Robustness of the conclusions to the judge (research point 2).

Scores the same answers three ways and recomputes the main comparisons on the test split:
  qwen    Qwen3.5-27B, binary-correctness prompt (judge_adr.jsonl; the judge used everywhere else)
  gemma   gemma-3-27b-it, same prompt (judge_gemma.jsonl; scripts/h200/chain_judge_gemma_h200.sh)
  rule    no model: format-aware matching in the style of the MMLongBench scorer (integers exact, floats
          within 1% relative, strings by ANLS >= 0.5, lists element-wise after sorting, "Not answerable")

usage: python scripts/rp2_judge_check.py results/MMLongBench/rp2-top10-mmlb-abl ... [--split test] [--md out.md]
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.analysis import doc_split, mcnemar   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, extracted_answer, judge_key, skey   # noqa: E402

VARIANTS = ["rerank", "ret@2", "scoreni20@3", "score20@3", "score20+f50@3", "oracle@2", "top10"]
READERS = {"qwen25vl7b": "answers_px1003520.jsonl", "qwen3vl8b": "answers_qwen3vl8b_px1003520.jsonl"}


# ----------------------------------------------------------------------------- rule-based scorer
def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, gt):
    p, g = norm(pred), norm(gt)
    if not p and not g:
        return 1.0
    d = _lev(p, g) / max(len(p), len(g))
    return 1 - d if d < 0.5 else 0.0


def norm(x):
    x = str(x).lower().strip()
    x = re.sub(r"[\"'`“”‘’]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip(" .,:;")


def numbers(x):
    return [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*\.?\d*", str(x))]


def as_items(x):
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    s = str(x).strip()
    if s.startswith("["):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(t) for t in v]
        except Exception:
            pass
        s = s.strip("[]")
    return [t for t in re.split(r",|;|\band\b", s) if t.strip()]


def unanswerable(x):
    return bool(re.search(r"not answerable|unanswerable|cannot be (determined|answered)|not (provided|mentioned|available)",
                          str(x).lower()))


def match_scalar(pred, gt, fmt):
    if fmt in ("Int", "Integer", "Float"):
        g, p = numbers(gt), numbers(pred)
        if not g or not p:
            return norm(pred) == norm(gt)
        g = g[0]
        tol = 0 if fmt != "Float" else max(abs(g) * 0.01, 1e-9)
        scales = (1, 100, 0.01) if fmt == "Float" else (1,)     # 0.123 vs 12.3 (%): same value
        return any(abs(x * k - g) <= tol for x in p for k in scales)
    return anls(pred, gt) >= 0.5 or (len(norm(gt)) >= 4 and norm(gt) in norm(pred))


def rule_correct(pred, gt, fmt):
    if fmt == "None" or norm(gt) == "not answerable":
        return unanswerable(pred)
    if unanswerable(pred):
        return False
    if fmt == "List":
        g, p = sorted(norm(x) for x in as_items(gt)), sorted(norm(x) for x in as_items(pred))
        if len(g) != len(p):
            return False
        return all(match_scalar(a, b, "Str" if not numbers(b) else "Float") for a, b in zip(p, g))
    return match_scalar(pred, gt, fmt)


# ----------------------------------------------------------------------------- comparison
def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def run(rd, reader, split):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    ans = _jsonl(os.path.join(rd, READERS[reader]))
    judges = {"qwen": _jsonl(os.path.join(rd, "judge_adr.jsonl")), "gemma": _jsonl(os.path.join(rd, "judge_gemma.jsonl"))}
    seen, score = set(), {v: {} for v in VARIANTS}       # score[var][q] = {scorer: 0/1}
    for s in rec:
        if (split and doc_split(s["doc_id"]) != split) or skey(s) in seen:
            continue
        seen.add(skey(s))
        for v in VARIANTS:
            x = s["variants"].get(v)
            a = ans.get(answer_key(s, x["ids"])) if x else None
            if a is None:
                continue
            jk = judge_key(s, a["response"])
            row = {}
            for name, j in judges.items():
                if jk in j:
                    row[name] = int(bool(j[jk]["verdict"]["equivalent"]))
            row["rule"] = int(rule_correct(extracted_answer(a["response"]), s["answer"], s.get("answer_format")))
            score[v][skey(s)] = row
    return score


def summarise(score):
    out = {}
    base = score["rerank"]
    for v, rows in score.items():
        r = {}
        for sc in ("qwen", "gemma", "rule"):
            qs = [q for q in rows if sc in rows[q] and q in base and sc in base[q]]
            if not qs:
                continue
            acc = sum(rows[q][sc] for q in qs) / len(qs)
            b = sum(rows[q][sc] and not base[q][sc] for q in qs)
            c = sum(base[q][sc] and not rows[q][sc] for q in qs)
            r[sc] = dict(n=len(qs), acc=100 * acc, delta=100 * (b - c) / len(qs), p=mcnemar(b, c))
        out[v] = r
    both = [(rows[q]["qwen"], rows[q]["gemma"]) for rows in score.values() for q in rows
            if "qwen" in rows[q] and "gemma" in rows[q]]
    rule = [(rows[q]["qwen"], rows[q]["rule"]) for rows in score.values() for q in rows if "qwen" in rows[q]]
    agree = dict(n=len(both), raw=100 * sum(a == b for a, b in both) / max(1, len(both)),
                 kappa=kappa([a for a, _ in both], [b for _, b in both]),
                 rule_raw=100 * sum(a == b for a, b in rule) / max(1, len(rule)),
                 rule_kappa=kappa([a for a, _ in rule], [b for _, b in rule]))
    return out, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--md")
    a = ap.parse_args()
    lines, dump = [], {}
    for rd in a.runs:
        for reader in READERS:
            if not os.path.exists(os.path.join(rd, READERS[reader])):
                continue
            res, agree = summarise(run(rd, reader, a.split))
            name = f"{os.path.basename(rd.rstrip('/'))} / {reader}"
            dump[name] = dict(results=res, agreement=agree)
            lines.append(f"\n### {name}  (judge agreement qwen-gemma {agree['raw']:.1f}%, kappa {agree['kappa']:.3f}, "
                         f"n={agree['n']}; qwen-rule {agree['rule_raw']:.1f}%, kappa {agree['rule_kappa']:.3f})\n")
            lines.append("| variant | Qwen judge | Gemma judge | rule |")
            lines.append("|---|---|---|---|")
            for v in VARIANTS:
                cells = []
                for sc in ("qwen", "gemma", "rule"):
                    x = res.get(v, {}).get(sc)
                    if not x:
                        cells.append("–")
                    elif v == "rerank":
                        cells.append(f"{x['acc']:.1f}")
                    else:
                        star = "*" if x["p"] < 0.05 else ""
                        cells.append(f"{x['acc']:.1f} ({x['delta']:+.1f}{star})")
                lines.append(f"| {v} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text + "\n")
        with open(a.md.replace(".md", ".json"), "w") as f:
            json.dump(dump, f, indent=1)


if __name__ == "__main__":
    main()
