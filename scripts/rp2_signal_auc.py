"""Compare recovery signals on pages the reranker dropped: AUC for separating gold pages, per split.

    python scripts/rp2_signal_auc.py results/MMLongBench/rp2-top10-mmlb-abl [resample_n6_t0.7.jsonl]
"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recovery.analysis import doc_split
from recovery.pipeline import _jsonl, gold_pages, skey


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    neg = sorted(neg)
    import bisect
    s = 0.0
    for p in pos:
        lo, hi = bisect.bisect_left(neg, p), bisect.bisect_right(neg, p)
        s += lo + 0.5 * (hi - lo)
    return s / (len(pos) * len(neg))


def ranks(d):
    order = sorted(d, key=lambda k: d[k])
    return {k: (i + 1) / len(order) for i, k in enumerate(order)}


def main(rd, resample_file):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    cache = os.path.join(rd, "recovery_cache")
    mm, txt = _jsonl(os.path.join(cache, "verify_score_mm.jsonl")), _jsonl(os.path.join(cache, "verify_score_noimg.jsonl"))
    rs = _jsonl(os.path.join(cache, resample_file))
    rows = {"dev": [], "test": []}
    tok = []
    for s in rec:
        r = rs.get(skey(s))
        if r is None:
            continue
        C, G = s["cands"], set(gold_pages(s))
        kept = [C[i - 1] for i in s["refiner_ids"]]
        D = [p for p in C if p not in kept]
        if not D:
            continue
        tok.append(r["prompt_tokens"] + r["completion_tokens"])
        vm = {int(p): v for p, v in mm.get(s["job_key"], {}).get("scores", {}).items()}
        vt = {int(p): v for p, v in txt.get(s["job_key"], {}).get("scores", {}).items()}
        feats = {p: dict(retrieval=-C.index(p), freq=r["freq"].get(str(C.index(p) + 1), 0.0),
                         text_verifier=vt.get(p, 0.0), mm_verifier=vm.get(p, 0.0)) for p in D}
        rf = ranks({p: f["freq"] + 1e-3 * f["retrieval"] for p, f in feats.items()})
        rt = ranks({p: f["text_verifier"] for p, f in feats.items()})
        for p, f in feats.items():
            f["freq+text_verifier"] = rf[p] + rt[p]
            rows[doc_split(s["doc_id"])].append((int(p in G), f))
    for sp, rr in rows.items():
        if not rr:
            continue
        print(f"== {sp}: dropped pages {len(rr)}, gold {sum(l for l, _ in rr)}")
        for name in ["retrieval", "freq", "text_verifier", "mm_verifier", "freq+text_verifier"]:
            print("  AUC %-20s %.3f" % (name, auc([f[name] for l, f in rr if l], [f[name] for l, f in rr if not l])))
        nz = [f["freq"] > 0 for _, f in rr]
        print("  dropped pages selected by >=1 sample: %.1f%%; gold among them %.1f%%; gold recalled %.1f%%" % (
            100 * st.mean(nz), 100 * sum(l for (l, f) in rr if f["freq"] > 0) / max(1, sum(nz)),
            100 * sum(l for (l, f) in rr if f["freq"] > 0) / max(1, sum(l for l, _ in rr))))
    print("resampling tokens per question (prompt+completion) mean %.0f" % st.mean(tok))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "resample_n6_t0.7.jsonl")
