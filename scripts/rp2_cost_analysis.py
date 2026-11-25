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
from recovery.analysis import doc_split, mcnemar   # noqa: E402
from recovery import split_limit   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, judge_key, skey   # noqa: E402

METHODS = ["rerank", "ret@2", "scoreni20@3", "score20@3", "score20+f50@3", "oracle@2", "top10"]
READERS = {"qwen25vl7b": "answers_px1003520.jsonl", "qwen3vl8b": "answers_qwen3vl8b_px1003520.jsonl",
           "qwen25vl32b": "answers_qwen25vl32b_px1003520_ear-plain.jsonl"}
BUDGETS = "rerank,score20m1@3,score20m2@3,score20m3@3,score20m4@3,score20m6@3,score20@3,scoreni20m2@3,scoreni20m4@3,scoreni20@3,top10"
N_SAMPLES = 6


def verifier_file(method):
    base = method.split("@")[0].split("+")[0]
    if base.startswith("scoreni"):
        return "verify_score_noimg.jsonl"
    if base.startswith("score"):
        return "verify_score_mm.jsonl"
    return None


def run(rd, reader, split, methods=METHODS):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    cache = os.path.join(rd, "recovery_cache")
    logit = _jsonl(os.path.join(cache, "logit.jsonl"))
    ver = {f: _jsonl(os.path.join(cache, f)) for f in ("verify_score_mm.jsonl", "verify_score_noimg.jsonl")}
    ans = _jsonl(os.path.join(rd, READERS[reader]))
    jud = _jsonl(os.path.join(rd, "judge_adr.jsonl"))
    rows, seen = {m: [] for m in methods}, set()
    for s in rec:
        if doc_split(s["doc_id"]) != split or skey(s) in seen:
            continue
        seen.add(skey(s))
        lg = logit.get(skey(s))
        if lg is None or lg.get("error"):
            continue
        rr = lg["prompt_tokens"] + lg["completion_tokens"]
        for m in methods:
            v = s["variants"].get(m)
            a = ans.get(answer_key(s, v["ids"])) if v else None
            j = jud.get(judge_key(s, a["response"])) if a else None
            if j is None:
                continue
            vf = verifier_file(m)
            vr = ver[vf].get(s["job_key"]) if vf else None
            lim = split_limit(m.split("@")[0])
            n_all = len((vr or {}).get("scores", {})) if vf else 0
            n_calls = min(lim, n_all) if lim is not None else n_all
            # a budget-limited verifier only pays for its calls (prompt tokens split evenly over the checks)
            v_tok = ((vr or {}).get("prompt_tokens", 0) * n_calls / n_all) if vf and n_all else 0
            rows[m].append(dict(q=skey(s),
                ok=int(bool(j["verdict"]["equivalent"])), rerank=rr * (1 + (N_SAMPLES if "+f" in m else 0)),
                ver=v_tok, calls=n_calls,
                read=a["prompt_tokens"] or 0, pages=len(v["ids"])))
    return rows


def summarise(rows):
    out = {}
    base = {x["q"]: x["ok"] for x in rows.get("rerank", [])}
    for m, xs in rows.items():
        # a method is reported only if every question has an answer: a partial set (page lists that happen to
        # coincide with answered ones) would be a biased subset
        if not xs or len(xs) < len(base):
            continue
        mean = lambda k: st.mean(x[k] for x in xs)
        out[m] = dict(n=len(xs), acc=100 * mean("ok"), rerank=mean("rerank"), ver=mean("ver"), calls=mean("calls"),
                      read=mean("read"), pages=mean("pages"))
        out[m]["total"] = out[m]["rerank"] + out[m]["ver"] + out[m]["read"]
        b = sum(x["ok"] and not base[x["q"]] for x in xs)
        c = sum(base[x["q"]] and not x["ok"] for x in xs)
        out[m]["p"] = mcnemar(b, c)
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
    ap.add_argument("--methods", help="comma-separated variants (default: the main methods)")
    a = ap.parse_args()
    methods = a.methods.split(",") if a.methods else METHODS
    text, dump = [], {}
    for rd in a.runs:
        for reader in READERS:
            if not os.path.exists(os.path.join(rd, READERS[reader])):
                continue
            res = summarise(run(rd, reader, a.split, methods))
            name = f"{os.path.basename(rd.rstrip('/'))} / {reader}"
            dump[name] = res
            base = res["rerank"]
            text += [f"\n### {name}  ({a.split}, {base['n']} questions; tokens per question)\n",
                     "| method | acc | Δacc | reader pages | reranker tok | verifier calls | verifier tok | reader tok | total tok | × rerank-only | Pareto |",
                     "|---|---|---|---|---|---|---|---|---|---|---|"]
            for m in methods:
                if m not in res:
                    continue
                r = res[m]
                text.append(f"| {m} | {r['acc']:.1f} | {r['acc'] - base['acc']:+.1f}{'*' if r['p'] < 0.05 else ''} | {r['pages']:.2f} | {r['rerank']:.0f} | "
                            f"{r['calls']:.2f} | {r['ver']:.0f} | {r['read']:.0f} | {r['total']:.0f} | "
                            f"{r['total'] / base['total']:.2f} | {'*' if r['pareto'] else ''} |")
    print("\n".join(text))
    if a.md:
        open(a.md, "w").write("\n".join(text) + "\n")
        json.dump(dump, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
