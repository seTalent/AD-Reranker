"""Single-pass filter (no reranker) vs. rerank + recovery (results doc section 25).

filterNN@10[:TAG]: every Top-10 page gets the plain relevance check (score_all; the verifier's model and inputs
without the selected pages) and pages with P(Yes) >= NN/100 are read in retrieval order. τ is chosen on the dev
split two ways: best dev accuracy, and dev pages read closest to score20@3 of the same run (equal reader cost).
Compared on the test split with the Zero-Shot reranker (same run) and the trained AD-Reranker run of the same
dataset: accuracy, Δ vs. score20@3 (McNemar), pages read, gold recall / precision, verifier calls and tokens.

usage: python scripts/rp2_filter_compare.py <zero-shot run> <ad-reranker run> [--md out.md]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.analysis import doc_split, mcnemar   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, gold_pages, judge_key, skey   # noqa: E402

READERS = {"qwen25vl7b": "answers_px1003520.jsonl", "qwen3vl8b": "answers_qwen3vl8b_px1003520.jsonl"}
TAUS = [1, 2, 5, 10, 20, 30, 50, 70, 90]   # 1-5: low thresholds, to reach the method's page count


def load(rd, reader):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    ans, jud = _jsonl(os.path.join(rd, READERS[reader])), _jsonl(os.path.join(rd, "judge_adr.jsonl"))
    cache = os.path.join(rd, "recovery_cache")
    ver = {t: _jsonl(os.path.join(cache, f)) for t, f in (("", "verify_score_mm.jsonl"), ("all", "verify_score_all.jsonl"),
                                                           ("all:qwen27b", "verify_score_all_qwen27b.jsonl"))}
    out, seen = {}, set()
    for s in rec:
        if skey(s) in seen:
            continue
        seen.add(skey(s))
        G, cands = set(gold_pages(s)), s["cands"]
        r = dict(split=doc_split(s["doc_id"]), v={})
        for name, x in s["variants"].items():
            a = ans.get(answer_key(s, x["ids"]))
            j = jud.get(judge_key(s, a["response"])) if a else None
            if j is None:
                continue
            pages = {cands[i - 1] for i in x["ids"]}
            tok = 0
            if name.startswith("filter"):
                tok = (ver["all:qwen27b" if name.endswith(":qwen27b") else "all"].get(s["job_key"]) or {}).get("prompt_tokens", 0)
            elif name in ("score20@3", "score20m2@3"):
                vr = ver[""].get(s["job_key"]) or {}
                n_all = len(vr.get("scores") or {})
                share = 1.0 if name == "score20@3" or not n_all else min(2, n_all) / n_all
                tok = vr.get("prompt_tokens", 0) * share
            r["v"][name] = dict(ok=int(bool(j["verdict"]["equivalent"])), pages=len(x["ids"]), tok=tok,
                                rec=len(pages & G) / len(G) if G else None, prec=len(pages & G) / len(pages) if G else None)
        out[skey(s)] = r
    return out


def stats(rows, v):
    xs = [r["v"][v] for r in rows if v in r["v"]]
    if len(xs) < len(rows):
        return None
    m = lambda k: sum(x[k] for x in xs if x[k] is not None) / max(1, sum(x[k] is not None for x in xs))
    return dict(acc=100 * m("ok"), pages=m("pages"), rec=100 * m("rec"), prec=100 * m("prec"), tok=m("tok"))


def paired(rows, v, ref):
    b = sum(r["v"][v]["ok"] and not r["v"][ref]["ok"] for r in rows)
    c = sum(r["v"][ref]["ok"] and not r["v"][v]["ok"] for r in rows)
    return 100 * (b - c) / len(rows), mcnemar(b, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zs")
    ap.add_argument("adr")
    ap.add_argument("--md")
    a = ap.parse_args()
    text, dump = [], {}
    for reader in READERS:
        zs, adr = load(a.zs, reader), load(a.adr, reader)
        keys = [k for k in zs if k in adr and zs[k]["split"] == "test"]
        dev = [zs[k] for k in zs if zs[k]["split"] == "dev"]
        test = [zs[k] for k in keys]
        for k in keys:   # the AD-Reranker run's page lists, on the same questions
            for v in ("rerank", "score20@3", "score20m2@3"):
                if v in adr[k]["v"]:
                    zs[k]["v"]["adr:" + v] = adr[k]["v"][v]
        target = stats(dev, "score20@3")["pages"]
        name = f"{os.path.basename(os.path.dirname(a.zs.rstrip('/')))} / {reader}"
        text += [f"\n### {name}  (test {len(test)}; dev pages of Zero-Shot + score20@3: {target:.2f})\n",
                 "| method | acc | Δ vs ZS+score20@3 | Δ vs ADR+score20@3 | pages | gold recall | gold precision | verifier calls | verifier tok |",
                 "|---|---|---|---|---|---|---|---|---|"]
        dump[name] = {}

        def line(label, v, calls):
            st = stats(test, v)
            if st is None:
                return
            d1, p1 = paired(test, v, "score20@3")
            d2, p2 = paired(test, v, "adr:score20@3")
            dump[name][label] = dict(st, d_zs=d1, p_zs=p1, d_adr=d2, p_adr=p2)
            text.append(f"| {label} | {st['acc']:.1f} | {d1:+.1f}{'*' if p1 < 0.05 else ''} | {d2:+.1f}{'*' if p2 < 0.05 else ''} | "
                        f"{st['pages']:.2f} | {st['rec']:.1f} | {st['prec']:.1f} | {calls} | {st['tok']:.0f} |")

        line("Zero-Shot rerank", "rerank", "0")
        line("Zero-Shot + score20@3 (ours)", "score20@3", "≈7.5")
        line("Zero-Shot + score20m2@3 (ours, 2 checks)", "score20m2@3", "≤2")
        line("AD-Reranker rerank", "adr:rerank", "0")
        line("AD-Reranker + score20@3 (ours)", "adr:score20@3", "≈8.9")
        line("Top-10, no filter", "top10", "0")
        for tag, label in (("", "Qwen3.5-9B"), (":qwen27b", "Qwen3.5-27B")):
            cands = [f"filter{t}@10{tag}" for t in TAUS if stats(dev, f"filter{t}@10{tag}")]
            if not cands:
                continue
            best = max(cands, key=lambda v: stats(dev, v)["acc"])
            eq = min(cands, key=lambda v: abs(stats(dev, v)["pages"] - target))
            line(f"single-pass filter, {label} (best dev: {best.split('@')[0]})", best, "10")
            if eq != best:
                line(f"single-pass filter, {label} (equal pages: {eq.split('@')[0]})", eq, "10")
    print("\n".join(text))
    if a.md:
        open(a.md, "w").write("\n".join(text) + "\n")
        json.dump(dump, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
