"""Group-sampling ablations from cached samples (no model calls): how the way the group is drawn
(temperature, permutation), its size n (first n samples of a larger group) and the frequency threshold f
change the recovered pages. Selection uses EvidenceRecovery.select exactly as the recover stage does.

    python scripts/rp2_sampling_ablation.py results/MMLongBench/rp2-top10-mmlb-abl
"""
import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recovery import EvidenceRecovery
from recovery.analysis import doc_split
from recovery.pipeline import _jsonl, gold_pages, skey


def auc(pos, neg):
    import bisect
    if not pos or not neg:
        return float("nan")
    neg = sorted(neg)
    return sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p))
               for p in pos) / (len(pos) * len(neg))


def freq_of(row, n, k):
    sels = row["selections"][:n]
    return {p: sum(p in x for x in sels) / max(1, len(sels)) for p in range(1, k + 1)}


def evaluate(rec, txt, rows, n, f, tau=0.2, b=3):
    er = EvidenceRecovery.__new__(EvidenceRecovery)
    out = {}
    for sp in ("dev", "test"):
        rc, pos, neg, add, add_gold, wf_num, wf_den, div, tok = [], [], [], 0, 0, 0, 0, [], []
        for s in rec:
            if doc_split(s["doc_id"]) != sp or skey(s) not in rows:
                continue
            C, G = s["cands"], set(gold_pages(s))
            kept = [C[i - 1] for i in s["refiner_ids"]]
            D = [p for p in C if p not in kept]
            r = rows[skey(s)]
            fr = {C[i - 1]: x for i, x in freq_of(r, n, len(C)).items()}
            ctx = dict(kept=kept, dropped=D, rank={p: i for i, p in enumerate(C)}, edges={}, located=[])
            ver = {"score_noimg": txt.get(s["job_key"]), "freq": fr}
            # floor: f = votes/n - eps must not round above votes/n (1/6 -> f16, not f17)
            method = f"scoreni{round(tau * 100)}" + (f"+f{math.floor(f * 100)}" if f is not None else "")
            _, R = er.select(method, b, ctx, ver)
            if G:
                rc.append(len((set(kept) | set(R)) & G) / len(G))
            add += len(R)
            add_gold += len(set(R) & G)
            if G & set(D):
                wf_num += len(set(R) & G & set(D))
                wf_den += len(G & set(D))
            for p in D:
                (pos if p in G else neg).append(fr.get(p, 0.0))
            div.append(len({tuple(sorted(x)) for x in r["selections"][:n] if x}) > 1)
            share = n / max(1, len(r["selections"]))
            tok.append(r["completion_tokens"] * share)
        out[sp] = dict(recall=st.mean(rc), wf_recovery=wf_num / max(1, wf_den), precision=add_gold / max(1, add),
                       added=add / max(1, len(rc)), auc=auc(pos, neg), diverse=st.mean(div), decode_tokens=st.mean(tok))
    return out


def main(rd):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    txt = _jsonl(os.path.join(rd, "recovery_cache", "verify_score_noimg.jsonl"))
    base = evaluate(rec, txt, {skey(s): dict(selections=[], completion_tokens=0) for s in rec}, 1, None)
    print("text verifier alone (tau 0.2, b 3):", {sp: round(v["recall"], 4) for sp, v in base.items()})
    hdr = "%-28s %3s %5s | %-6s %-6s %-6s %-6s %-5s | %-6s %-6s %-6s %-6s %-5s %-6s"
    print(hdr % ("group", "n", "f", "dRec", "wfRec", "prec", "AUC", "div", "dRec", "wfRec", "prec", "AUC", "div", "tok"))
    print("%-38s | %-35s | test" % ("", "dev"))
    for path in sorted(glob.glob(os.path.join(rd, "recovery_cache", "resample_*.jsonl"))):
        rows = _jsonl(path)
        if not rows:
            continue
        name = os.path.basename(path).removeprefix("resample_").removesuffix(".jsonl")
        nmax = max(len(r["selections"]) for r in rows.values())
        for n in sorted({x for x in (2, 3, 6, 10) if x <= nmax} | {nmax}):
            for votes in sorted({max(1, round(x * n)) for x in (1 / 6, 1 / 3, 1 / 2, 2 / 3)}):
                f = votes / n - 1e-9        # restore a page that at least `votes` of the n samples keep
                res = evaluate(rec, txt, rows, n, f)
                d, t = res["dev"], res["test"]
                print("%-28s %3d %5s | %+.3f  %.3f  %.3f  %.3f  %.2f  | %+.3f  %.3f  %.3f  %.3f  %.2f  %5.0f" % (
                    name, n, f"{votes}/{n}", d["recall"] - base["dev"]["recall"], d["wf_recovery"], d["precision"], d["auc"],
                    d["diverse"], t["recall"] - base["test"]["recall"], t["wf_recovery"], t["precision"], t["auc"],
                    t["diverse"], t["decode_tokens"]))


if __name__ == "__main__":
    main(sys.argv[1])
