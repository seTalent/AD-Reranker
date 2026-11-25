"""Accuracy vs. cost of each method, from the cached runs (no model calls).

Per test question and method, the tokens every component processes:
  reranker  prompt + generated tokens of the rerank call (recovery_cache/logit.jsonl, recorded with the rerank prompt);
            group-sampling methods (+fNN) add 6 sampled reranks (resample_n6_t0.7.jsonl has no token counts,
            so they are counted as 6 x the greedy rerank's tokens);
  verifier  prompt tokens and number of calls (one per dropped page) of the gap-conditioned check
            (verify_score_mm.jsonl for scoreNN, verify_score_noimg.jsonl for scoreniNN);
  reader    prompt tokens of the answer call and number of pages read.
Accuracy is the Qwen3.5-27B judge (judge_adr.jsonl). Methods on the accuracy/total-token Pareto front are marked.

usage: python scripts/rp2_cost_analysis.py <run dirs> [--split test] [--md out.md]
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.analysis import doc_split   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, judge_key, skey   # noqa: E402

METHODS = ["rerank", "ret@2", "scoreni20@3", "score20@3", "score20+f50@3", "oracle@2", "top10"]
READERS = {"qwen25vl7b": "answers_px1003520.jsonl", "qwen3vl8b": "answers_qwen3vl8b_px1003520.jsonl",
           "qwen25vl32b": "answers_qwen25vl32b_px1003520_ear-plain.jsonl"}
N_SAMPLES = 6


def verifier_file(method):
    base = method.split("@")[0].split("+")[0]
    if base.startswith("scoreni"):
        return "verify_score_noimg.jsonl"
    if base.startswith("score"):
        return "verify_score_mm.jsonl"
    return None


def run(rd, reader, split):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    cache = os.path.join(rd, "recovery_cache")
    logit = _jsonl(os.path.join(cache, "logit.jsonl"))
    ver = {f: _jsonl(os.path.join(cache, f)) for f in ("verify_score_mm.jsonl", "verify_score_noimg.jsonl")}
    ans = _jsonl(os.path.join(rd, READERS[reader]))
    jud = _jsonl(os.path.join(rd, "judge_adr.jsonl"))
    rows, seen = {m: [] for m in METHODS}, set()
    for s in rec:
        if doc_split(s["doc_id"]) != split or skey(s) in seen:
            continue
        seen.add(skey(s))
        lg = logit.get(skey(s))
        if lg is None or lg.get("error"):
            continue
        rr = lg["prompt_tokens"] + lg["completion_tokens"]
        for m in METHODS:
            v = s["variants"].get(m)
            a = ans.get(answer_key(s, v["ids"])) if v else None
            j = jud.get(judge_key(s, a["response"])) if a else None
            if j is None:
                continue
            vf = verifier_file(m)
            vr = ver[vf].get(s["job_key"]) if vf else None
            rows[m].append(dict(
                ok=int(bool(j["verdict"]["equivalent"])), rerank=rr * (1 + (N_SAMPLES if "+f" in m else 0)),
                ver=(vr or {}).get("prompt_tokens", 0) if vf else 0,
                calls=len((vr or {}).get("scores", {})) if vf else 0,
                read=a["prompt_tokens"] or 0, pages=len(v["ids"])))
    return rows


def summarise(rows):
    out = {}
    for m, xs in rows.items():
        if not xs:
            continue
        mean = lambda k: st.mean(x[k] for x in xs)
        out[m] = dict(n=len(xs), acc=100 * mean("ok"), rerank=mean("rerank"), ver=mean("ver"), calls=mean("calls"),
                      read=mean("read"), pages=mean("pages"))
        out[m]["total"] = out[m]["rerank"] + out[m]["ver"] + out[m]["read"]
    front = sorted(out, key=lambda m: (out[m]["total"], -out[m]["acc"]))
    best = -1
    for m in front:          # Pareto front: no cheaper method is at least as accurate
        out[m]["pareto"] = out[m]["acc"] > best
        best = max(best, out[m]["acc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--md")
    a = ap.parse_args()
    text, dump = [], {}
    for rd in a.runs:
        for reader in READERS:
            if not os.path.exists(os.path.join(rd, READERS[reader])):
                continue
            res = summarise(run(rd, reader, a.split))
            name = f"{os.path.basename(rd.rstrip('/'))} / {reader}"
            dump[name] = res
            base = res["rerank"]
            text += [f"\n### {name}  ({a.split}, {base['n']} questions; tokens per question)\n",
                     "| method | acc | Δacc | reader pages | reranker tok | verifier calls | verifier tok | reader tok | total tok | × rerank-only | Pareto |",
                     "|---|---|---|---|---|---|---|---|---|---|---|"]
            for m in METHODS:
                if m not in res:
                    continue
                r = res[m]
                text.append(f"| {m} | {r['acc']:.1f} | {r['acc'] - base['acc']:+.1f} | {r['pages']:.2f} | {r['rerank']:.0f} | "
                            f"{r['calls']:.2f} | {r['ver']:.0f} | {r['read']:.0f} | {r['total']:.0f} | "
                            f"{r['total'] / base['total']:.2f} | {'*' if r['pareto'] else ''} |")
    print("\n".join(text))
    if a.md:
        open(a.md, "w").write("\n".join(text) + "\n")
        json.dump(dump, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
