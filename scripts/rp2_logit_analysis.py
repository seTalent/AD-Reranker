"""Zero-sampling uncertainty signals (recovery/logit.py) against group sampling, offline.

For pages the greedy reranker dropped: AUC of each signal for separating gold pages, and the recall,
precision and pages added when the signal is united with the text verifier (tau 0.2, b 3) exactly as the
recover stage does it. Thresholds are picked on dev (budget-matched to the sampling-frequency union) and
applied unchanged to test. Optionally writes the chosen signal as a recover-stage frequency file.

    python scripts/rp2_logit_analysis.py results/MMLongBench/rp2-top10-mmlb-abl [--write first_cal]
"""
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recovery import EvidenceRecovery
from recovery.analysis import doc_split
from recovery.pipeline import _jsonl, gold_pages, skey

EPS = 1e-6


def auc(pos, neg):
    import bisect
    if not pos or not neg:
        return float("nan")
    neg = sorted(neg)
    return sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p))
               for p in pos) / (len(pos) * len(neg))


def normalise(d):
    z = sum(d.values())
    return {c: v / z for c, v in d.items()} if z > 0 else d


def signals(row, prior):
    """Per-candidate scores in [0, 1] from one logit row (keys: 1-based positions as int)."""
    k = row["k"]
    g = lambda name: {int(c): float(v) for c, v in row[name].items() if int(c) <= k}
    first, cf, slot = g("first"), g("cf"), g("slot")
    cal = normalise({c: math.exp(math.log(first[c] + EPS) - math.log(cf.get(c, 0) + EPS)) for c in first})
    pri = normalise({c: (first[c] + EPS) / (prior.get(c, 0) + EPS) for c in first})
    return dict(slot=slot, first=first, first_cal=cal, first_prior=pri)


def position_prior(rows, rec):
    """Mean first-identifier distribution per position on the dev split (zero-cost bias estimate)."""
    acc = defaultdict(list)
    for s in rec:
        r = rows.get(skey(s))
        if doc_split(s["doc_id"]) == "dev" and r and not r.get("error"):
            for c, v in r["first"].items():
                acc[int(c)].append(float(v))
    return {c: st.mean(v) for c, v in acc.items()}


def evaluate(rec, txt, sig_of, f, tau=0.2, b=3, splits=("dev", "test")):
    """Recall/precision of 'text verifier >= tau OR signal >= f' (f None: verifier alone)."""
    er = EvidenceRecovery.__new__(EvidenceRecovery)
    out, seen = {}, set()
    for sp in splits:
        rc, add, add_gold, wf_num, wf_den, n = [], 0, 0, 0, 0, 0
        for s in rec:
            if doc_split(s["doc_id"]) != sp or skey(s) in seen:
                continue
            seen.add(skey(s))
            fr = sig_of(s)
            if fr is None:
                continue
            C, G = s["cands"], set(gold_pages(s))
            kept = [C[i - 1] for i in s["refiner_ids"]]
            D = [p for p in C if p not in kept]
            ctx = dict(kept=kept, dropped=D, rank={p: i for i, p in enumerate(C)}, edges={}, located=[])
            method = f"scoreni{round(tau * 100)}" + (f"+f{math.floor(f * 100)}" if f is not None else "")
            ver = {"score_noimg": txt.get(s["job_key"]), "freq": {C[c - 1]: v for c, v in fr.items() if c <= len(C)}}
            _, R = er.select(method, b, ctx, ver)
            n += 1
            if G:
                rc.append(len((set(kept) | set(R)) & G) / len(G))
            add += len(R)
            add_gold += len(set(R) & G)
            if G & set(D):
                wf_num += len(set(R) & G & set(D))
                wf_den += len(G & set(D))
        out[sp] = dict(recall=st.mean(rc), precision=add_gold / max(1, add), added=add / max(1, n),
                       wf_recovery=wf_num / max(1, wf_den), n=n)
    return out


def dropped_auc(rec, sig_of, sp):
    pos, neg, seen = [], [], set()
    for s in rec:
        if doc_split(s["doc_id"]) != sp or skey(s) in seen:
            continue
        seen.add(skey(s))
        fr = sig_of(s)
        if fr is None:
            continue
        C, G = s["cands"], set(gold_pages(s))
        kept = {C[i - 1] for i in s["refiner_ids"]}
        for i, p in enumerate(C, 1):
            if p not in kept:
                (pos if p in G else neg).append(fr.get(i, 0.0))
    return auc(pos, neg), len(pos), len(neg)


def main(rd, write=None):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    cache = os.path.join(rd, "recovery_cache")
    txt = _jsonl(os.path.join(cache, "verify_score_noimg.jsonl"))
    mm = _jsonl(os.path.join(cache, "verify_score_mm.jsonl"))
    rows = {k: r for k, r in _jsonl(os.path.join(cache, "logit.jsonl")).items() if not r.get("error")}
    prior = position_prior(rows, rec)
    sig = {k: signals(r, prior) for k, r in rows.items()}
    agree = [r["sel"] == s["refiner_ids"] for s in rec for r in [rows.get(skey(s))] if r]
    lat = {n: st.median(r["latency"][n] for r in rows.values()) for n in ("greedy", "first", "cf")}
    print(f"{rd}: {len(rows)} questions with logits; greedy selection identical to rerank.json: "
          f"{100 * st.mean(agree):.1f}%; median latency greedy {lat['greedy']:.2f}s, first {lat['first']:.2f}s, "
          f"cf {lat['cf']:.2f}s (single requests under batch load)")
    print("dev position prior of the first identifier:", {c: round(v, 3) for c, v in sorted(prior.items())})

    getters = {name: (lambda name: lambda s: (sig.get(skey(s)) or {}).get(name))(name)
               for name in ("slot", "first", "first_cal", "first_prior")}
    for rf in ("resample_n6_t0.7.jsonl", "resample_perm_n6.jsonl"):
        rs = _jsonl(os.path.join(cache, rf))
        if rs:
            getters[rf.replace("resample_", "freq_").replace(".jsonl", "")] = \
                (lambda rs: lambda s: {int(c): v for c, v in rs[skey(s)]["freq"].items()} if skey(s) in rs else None)(rs)
    getters["retrieval"] = lambda s: {i: 1 - (i - 1) / len(s["cands"]) for i in range(1, len(s["cands"]) + 1)}

    print("\nAUC on dropped pages (gold vs non-gold)")
    for name, g in getters.items():
        a = {sp: dropped_auc(rec, g, sp) for sp in ("dev", "test")}
        print(f"  {name:22s} dev {a['dev'][0]:.3f}  test {a['test'][0]:.3f}  (test gold {a['test'][1]}, non-gold {a['test'][2]})")

    base = evaluate(rec, txt, lambda s: {}, None)
    print(f"\ntext verifier alone: dev rec {base['dev']['recall']:.4f} added {base['dev']['added']:.2f} | "
          f"test rec {base['test']['recall']:.4f} added {base['test']['added']:.2f} prec {base['test']['precision']:.3f}")
    ref = "freq_n6_t0.7" if "freq_n6_t0.7" in getters else None
    target = evaluate(rec, txt, getters[ref], 0.5 - 1e-9)["dev"]["added"] if ref else None
    if target:
        print(f"budget target (dev pages added by verifier ∪ freq_n6_t0.7 >= 3/6): {target:.3f}")
    print("%-22s %6s | %-7s %-6s %-6s | %-7s %-6s %-6s %-6s" % ("signal", "f", "dev dR", "added", "prec",
                                                               "test dR", "added", "prec", "wfRec"))
    chosen = {}
    for name in [n for n in getters if n != "retrieval"]:
        grid = [x / 100 for x in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90)]
        if name.startswith("freq_"):
            grid = [v / 6 - 1e-9 for v in range(1, 7)]
        res = [(f, evaluate(rec, txt, getters[name], f)) for f in grid]
        if target:
            f, r = min(res, key=lambda x: abs(x[1]["dev"]["added"] - target))
            chosen[name] = f
        for f, r in res:
            mark = "*" if target and f == chosen.get(name) else " "
            d, t = r["dev"], r["test"]
            print(f"{name:22s} {f:6.3f}{mark}| {d['recall'] - base['dev']['recall']:+.4f} {d['added']:.2f}   "
                  f"{d['precision']:.3f}  | {t['recall'] - base['test']['recall']:+.4f} {t['added']:.2f}   "
                  f"{t['precision']:.3f}  {t['wf_recovery']:.3f}")
    print("\nchosen on dev (budget-matched):", {k: round(v, 3) for k, v in chosen.items()})
    if write:
        out = os.path.join(cache, f"resample_logit_{write}.jsonl")
        with open(out, "w") as fo:
            for k, s in sig.items():
                fo.write(json.dumps(dict(key=k, freq={str(c): v for c, v in s[write].items()})) + "\n")
        print(f"wrote {out} (use +recovery.resample_file={os.path.basename(out)} and scoreni20+fNN@3)")


if __name__ == "__main__":
    args = sys.argv[1:]
    w = args[args.index("--write") + 1] if "--write" in args else None
    main(args[0], w)
