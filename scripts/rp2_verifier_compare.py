"""Verifier models compared on the same reranked candidates (results doc section 25).

Each verifier's P(Yes) on the pages the reranker dropped (recovery_cache/verify_score_mm[_TAG].jsonl; the
default Qwen3.5-9B has no tag):
  AUC        separating dropped gold pages from the other dropped pages
  unparsed   share of checks whose top tokens contain neither Yes nor No (scored 0)
  added      restored pages per question at threshold τ (budget b = 3); gold recall of the restored pages
  tokens     verifier prompt tokens per question
End to end (Qwen3.5-27B judge): accuracy of score20@3[:TAG] and of score{τ*}@3[:TAG], where τ* is chosen on
MMLongBench dev (Zero-Shot run) so that the verifier restores as many pages per question as the default verifier
does at τ = 0.2 (equal reader cost). `--pick` only prints τ* per tag.

usage: python scripts/rp2_verifier_compare.py <run dirs> --tags qwen9b_lr,qwen27b,internvl8b,gemma12b [--pick] [--md out.md]
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.analysis import doc_split, mcnemar   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, gold_pages, judge_key, skey   # noqa: E402

GRID = [x / 100 for x in range(5, 100, 5)]
READERS = {"qwen25vl7b": "answers_px1003520.jsonl", "qwen3vl8b": "answers_qwen3vl8b_px1003520.jsonl"}
B = 3


def auc(pos, neg):
    import bisect
    if not pos or not neg:
        return float("nan")
    neg = sorted(neg)
    s = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos)
    return s / (len(pos) * len(neg))


def load(rd, tag, split):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    ver = _jsonl(os.path.join(rd, "recovery_cache", "verify_score_mm" + (f"_{tag}" if tag else "") + ".jsonl"))
    rows, seen = [], set()
    for s in rec:
        if (split and doc_split(s["doc_id"]) != split) or skey(s) in seen:
            continue
        seen.add(skey(s))
        v = ver.get(s["job_key"])
        if v is None:
            continue
        cands, G = s["cands"], set(gold_pages(s))
        kept = {cands[i - 1] for i in s["variants"]["rerank"]["ids"]}
        sc = {int(p): x for p, x in (v.get("scores") or {}).items()}
        rows.append(dict(s=s, sc=sc, gold=G, dropped=[p for p in cands if p not in kept], tok=v.get("prompt_tokens", 0),
                         unparsed=v.get("n_unparsed", 0), calls=len(sc)))
    return rows


def restored(r, t):
    return [p for p in sorted(r["sc"], key=lambda p: -r["sc"][p]) if r["sc"][p] >= t][:B]


def signal(rows, t):
    pos = [x for r in rows for p, x in r["sc"].items() if p in r["gold"]]
    neg = [x for r in rows for p, x in r["sc"].items() if p not in r["gold"]]
    dg = sum(len(set(r["dropped"]) & r["gold"]) for r in rows)
    got = sum(len(set(restored(r, t)) & r["gold"]) for r in rows)
    calls = sum(r["calls"] for r in rows)
    return dict(auc=auc(pos, neg), added=st.mean(len(restored(r, t)) for r in rows),
                dropped_gold_recall=100 * got / max(1, dg), tokens=st.mean(r["tok"] for r in rows),
                unparsed=100 * sum(r["unparsed"] for r in rows) / max(1, calls))


def pick(ref_run, tag, base_added):
    rows = load(ref_run, tag, "dev")
    return min(GRID, key=lambda t: (abs(st.mean(len(restored(r, t)) for r in rows) - base_added), abs(t - 0.2)))


def accuracy(rd, reader, variant, split):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    ans, jud = _jsonl(os.path.join(rd, READERS[reader])), _jsonl(os.path.join(rd, "judge_adr.jsonl"))
    out, seen = {}, set()
    for s in rec:
        if doc_split(s["doc_id"]) != split or skey(s) in seen or variant not in s["variants"]:
            continue
        seen.add(skey(s))
        a = ans.get(answer_key(s, s["variants"][variant]["ids"]))
        j = jud.get(judge_key(s, a["response"])) if a else None
        if j is not None:
            out[skey(s)] = int(bool(j["verdict"]["equivalent"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tags", default="qwen9b_lr,qwen27b,internvl8b,gemma12b")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pick", action="store_true")
    ap.add_argument("--md")
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t]
    ref = a.runs[0]                                     # MMLongBench, Zero-Shot reranker: threshold selection
    base_added = st.mean(len(restored(r, 0.2)) for r in load(ref, "", "dev"))
    taus = {t: pick(ref, t, base_added) for t in tags}
    if a.pick:
        print(json.dumps(taus))
        return
    text = [f"τ* (equal restored pages on {os.path.basename(ref.rstrip('/'))} dev, default verifier at τ = 0.2: "
            f"{base_added:.2f} pages): " + ", ".join(f"{t} {x:g}" for t, x in taus.items()), ""]
    dump = dict(tau=taus, runs={})
    for rd in a.runs:
        name = os.path.basename(rd.rstrip("/"))
        text += [f"\n### {name}  ({a.split})\n",
                 "| verifier | AUC | unparsed % | verifier tok/question | τ | added pages | dropped-gold recall | "
                 + " | ".join(f"{r} acc (Δ vs rerank; vs default)" for r in READERS) + " |",
                 "|---|---|---|---|---|---|---|" + "---|" * len(READERS)]
        base_acc = {r: accuracy(rd, r, "rerank", a.split) for r in READERS}
        def_acc = {r: accuracy(rd, r, "score20@3", a.split) for r in READERS}
        dump["runs"][name] = {}
        for tag in [""] + tags:
            rows = load(rd, tag, a.split)
            if not rows:
                continue
            for t in sorted({0.2, taus.get(tag, 0.2)}):
                sig = signal(rows, t)
                var = f"score{round(100 * t)}@3" + (f":{tag}" if tag else "")
                cells = []
                for r in READERS:
                    acc = accuracy(rd, r, var, a.split)
                    qs = [q for q in acc if q in base_acc[r] and q in def_acc[r]]
                    if not qs:
                        cells.append("–")
                        continue
                    m = 100 * sum(acc[q] for q in qs) / len(qs)
                    b = sum(acc[q] and not base_acc[r][q] for q in qs)
                    c = sum(base_acc[r][q] and not acc[q] for q in qs)
                    b2 = sum(acc[q] and not def_acc[r][q] for q in qs)
                    c2 = sum(def_acc[r][q] and not acc[q] for q in qs)
                    d1, d2 = 100 * (b - c) / len(qs), 100 * (b2 - c2) / len(qs)
                    cells.append(f"{m:.1f} ({d1:+.1f}{'*' if mcnemar(b, c) < 0.05 else ''}; "
                                 f"{d2:+.1f}{'*' if mcnemar(b2, c2) < 0.05 else ''})")
                    dump["runs"][name].setdefault(var, {})[r] = dict(acc=m, d_rerank=d1, p_rerank=mcnemar(b, c),
                                                                    d_default=d2, p_default=mcnemar(b2, c2), n=len(qs))
                dump["runs"][name].setdefault(var, {}).update(signal=sig)
                text.append(f"| {tag or 'qwen9b (default)'} | {sig['auc']:.3f} | {sig['unparsed']:.1f} | {sig['tokens']:.0f} | "
                            f"{t:g} | {sig['added']:.2f} | {sig['dropped_gold_recall']:.1f} | " + " | ".join(cells) + " |")
    print("\n".join(text))
    if a.md:
        open(a.md, "w").write("\n".join(text) + "\n")
        json.dump(dump, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
