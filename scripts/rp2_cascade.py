"""Offline analysis of a cascaded verifier from cached scores (no model calls).

Stage 1 scores every dropped page with the text-only verifier (score_noimg). Pages below `lo` are
rejected and pages at or above `hi` accepted without further checks; only pages in between are sent
to the multimodal verifier (score_mm), which accepts at `tau`. At most `b` pages are restored per
question, highest score first. Token cost: the text pass for every question with dropped pages, plus
the multimodal cost in proportion to the share of a question's pages that reach stage 2.

    python scripts/rp2_cascade.py results/MMLongBench/rp2-top10-mmlb-abl
"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recovery.analysis import doc_split
from recovery.pipeline import gold_pages, skey


def load(rd, mode):
    out = {}
    for line in open(os.path.join(rd, "recovery_cache", f"verify_{mode}.jsonl")):
        r = json.loads(line)
        if not r.get("error"):
            out[r["key"]] = r
    return out


def run(rec, txt, mm, lo, hi, tau, b):
    stats = dict(recall=[], added=0, added_gold=0, wf_num=0, wf_den=0, tokens=[], mm_share=[], pages=[])
    for s in rec:
        C, G = s["cands"], set(gold_pages(s))
        kept = [C[i - 1] for i in s["refiner_ids"]]
        D = [p for p in C if p not in kept]
        t = txt.get(s["job_key"], {})
        m = mm.get(s["job_key"], {})
        ts = {int(p): v for p, v in t.get("scores", {}).items()}
        ms = {int(p): v for p, v in m.get("scores", {}).items()}
        accept, sent = {}, []
        for p in D:
            x = ts.get(p, 0.0)
            if x >= hi:
                accept[p] = 1.0 + x
            elif x >= lo:
                sent.append(p)
                if ms.get(p, 0.0) >= tau:
                    accept[p] = ms[p]
        R = sorted(accept, key=lambda p: -accept[p])[:b]
        cost = t.get("prompt_tokens", 0)
        if D and m.get("prompt_tokens"):
            cost += m["prompt_tokens"] * len(sent) / len(D)
        stats["tokens"].append(cost)
        stats["mm_share"].append(len(sent) / len(D) if D else 0.0)
        final = set(kept) | set(R)
        stats["pages"].append(len(final))
        if G:
            stats["recall"].append(len(final & G) / len(G))
        stats["added"] += len(R)
        stats["added_gold"] += len(set(R) & G)
        if G & set(D):
            stats["wf_num"] += len(set(R) & G & set(D))
            stats["wf_den"] += len(G & set(D))
    return dict(recall=st.mean(stats["recall"]), pages=st.mean(stats["pages"]),
                precision=stats["added_gold"] / max(1, stats["added"]), wf_recovery=stats["wf_num"] / max(1, stats["wf_den"]),
                tokens=st.mean(stats["tokens"]), mm_share=st.mean(stats["mm_share"]))


def main(rd):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    txt, mm = load(rd, "score_noimg"), load(rd, "score_mm")
    splits = {sp: [s for s in rec if doc_split(s["doc_id"]) == sp] for sp in ("dev", "test")}
    grid = [(lo, hi) for lo in (0.0, 0.01, 0.02, 0.05, 0.1) for hi in (0.3, 0.5, 0.7, 0.9, 1.01) if lo < hi]
    ref = {sp: run(splits[sp], txt, mm, 0.0, 1.01, 0.2, 3) for sp in splits}     # all pages to the multimodal verifier
    txt_only = {sp: run(splits[sp], txt, {}, 0.2, 0.2, 9.9, 3) for sp in splits}  # text verifier alone at 0.2
    print("reference (multimodal for every page):", {k: round(v, 3) for k, v in ref["test"].items()})
    print("text verifier only (tau 0.2):        ", {k: round(v, 3) for k, v in txt_only["test"].items()})
    dev = {g: run(splits["dev"], txt, mm, g[0], g[1], 0.2, 3) for g in grid}
    # cheapest setting on dev whose recall stays within 1 point of the reference
    ok = [g for g in grid if dev[g]["recall"] >= ref["dev"]["recall"] - 0.01]
    best = min(ok, key=lambda g: dev[g]["tokens"])
    print("dev grid (lo, hi): recall / tokens / multimodal share")
    for g in grid:
        d = dev[g]
        print("  %-12s %.3f  %6.0f  %.2f%s" % (g, d["recall"], d["tokens"], d["mm_share"], "  <- chosen" if g == best else ""))
    print("chosen", best, "TEST:", {k: round(v, 3) for k, v in run(splits["test"], txt, mm, best[0], best[1], 0.2, 3).items()})


if __name__ == "__main__":
    main(sys.argv[1])
