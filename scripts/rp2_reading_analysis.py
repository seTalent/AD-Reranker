"""Evidence-aware reading (recovery/reading.py): prompts and answer-level fallback.

For each run and reader:
  1. accuracy of rerank / score20@3 / oracle@2 under each prompt (plain, tag, text, tag+text, text_all),
     paired against the plain rerank answer (McNemar, test split);
  2. where the gains and losses come from: questions whose recovered pages include gold evidence vs.
     questions where only non-evidence pages were added;
  3. answer-level fallback: answer with the reranked pages (A0) and with the recovered pages (A1); keep A1
     when both answers agree, otherwise keep A1 only if the reader is not much less confident in it:
     conf(A1) - conf(A0) >= delta, conf = mean token log-probability of the "Answer" value.
     delta is chosen on dev (grid), and test is reported with it; delta = -inf is "always A1".
     The fallback is also paired against the method alone under the same prompt (McNemar), and the
     distractor group's gain/loss vs. the plain rerank answer is shown after the fallback.

usage: python scripts/rp2_reading_analysis.py results/MMLongBench/rp2-top10-mmlb-abl ... [--md out.md]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.analysis import doc_split, mcnemar   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, gold_pages, judge_key, skey   # noqa: E402
from recovery.reading import answer_text, normalise   # noqa: E402

MODES = ["plain", "tag", "text", "tag-text", "text_all"]
READERS = {"qwen25vl7b": "answers_px1003520_ear-{m}.jsonl", "qwen3vl8b": "answers_qwen3vl8b_px1003520_ear-{m}.jsonl"}
METHOD = "score20@3"
GRID = [-9.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]


def load(rd, reader, judge):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    jud = _jsonl(os.path.join(rd, judge))
    ans = {m: _jsonl(os.path.join(rd, READERS[reader].format(m=m))) for m in MODES}
    rows, seen = [], set()
    for s in rec:
        if skey(s) in seen:
            continue
        seen.add(skey(s))
        r = dict(q=skey(s), split=doc_split(s["doc_id"]))
        cands, G = s["cands"], set(gold_pages(s))
        added = {cands[i - 1] for i in s["variants"][METHOD]["added"]}
        r["group"] = "none" if not added else ("gold" if added & G else "distractor")
        for m in MODES:
            for v in ("rerank", METHOD, "oracle@2"):
                a = ans[m].get(answer_key(s, s["variants"][v]["ids"]))
                j = jud.get(judge_key(s, a["response"])) if a else None
                if j is not None:
                    r[(m, v)] = dict(ok=int(bool(j["verdict"]["equivalent"])), conf=a.get("conf_ans"),
                                     ans=normalise(answer_text(a["response"])), tok=a.get("prompt_tokens"))
        rows.append(r)
    return rows


def acc(rows, key):
    xs = [r[key]["ok"] for r in rows if key in r]
    return 100 * sum(xs) / len(xs) if xs else float("nan"), len(xs)


def paired(rows, key, base=("plain", "rerank")):
    qs = [r for r in rows if key in r and base in r]
    b = sum(r[key]["ok"] and not r[base]["ok"] for r in qs)
    c = sum(r[base]["ok"] and not r[key]["ok"] for r in qs)
    return 100 * (b - c) / max(1, len(qs)), mcnemar(b, c), b, c


def fallback(r, mode, delta):
    a0, a1 = r.get(("plain", "rerank")), r.get((mode, METHOD))
    if a0 is None or a1 is None:
        return None
    if a1["ans"] == a0["ans"] or a0["conf"] is None or a1["conf"] is None:
        return a1["ok"], False
    use1 = a1["conf"] - a0["conf"] >= delta
    return (a1["ok"] if use1 else a0["ok"]), not use1


def fb_acc(rows, mode, delta):
    xs = [fallback(r, mode, delta) for r in rows]
    xs = [x for x in xs if x is not None]
    return 100 * sum(o for o, _ in xs) / max(1, len(xs)), 100 * sum(f for _, f in xs) / max(1, len(xs))


def report(rd, reader, judge):
    rows = load(rd, reader, judge)
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    out = [f"\n### {os.path.basename(rd.rstrip('/'))} / {reader}  (test {len(test)} questions; judge {judge})\n",
           "| prompt | rerank | score20@3 (Δ vs plain rerank) | oracle@2 | gold-group gain/loss | distractor-group gain/loss | + fallback (δ from dev) | fallback vs score20@3: gain/loss | distractor group after fallback | fallback rate |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for m in MODES:
        if not any((m, METHOD) in r for r in test):
            continue
        a_r, _ = acc(test, (m, "rerank"))
        a_m, n = acc(test, (m, METHOD))
        a_o, _ = acc(test, (m, "oracle@2"))
        d, p, _, _ = paired(test, (m, METHOD))
        grp = {}
        for g in ("gold", "distractor"):
            sub = [r for r in test if r["group"] == g]
            _, _, b, c = paired(sub, (m, METHOD))
            grp[g] = f"{b}/{c}"
        best = max(GRID, key=lambda x: (fb_acc(dev, m, x)[0], -abs(x)))
        fa, rate = fb_acc(test, m, best)
        fb = [fallback(r, m, best) for r in test]
        base = [r[("plain", "rerank")]["ok"] for r, x in zip(test, fb) if x is not None]
        b = sum(x[0] and not y for x, y in zip([x for x in fb if x is not None], base))
        c = sum(y and not x[0] for x, y in zip([x for x in fb if x is not None], base))
        # fallback against the method alone (same prompt): what the fallback itself changes
        pairs = [(x[0], r[(m, METHOD)]["ok"], r["group"]) for r, x in zip(test, fb) if x is not None and (m, METHOD) in r]
        fb_b = sum(f and not o for f, o, _ in pairs)
        fb_c = sum(o and not f for f, o, _ in pairs)
        fb_p = mcnemar(fb_b, fb_c)
        dis = [(x[0], r[("plain", "rerank")]["ok"]) for r, x in zip(test, fb)
               if x is not None and r["group"] == "distractor" and ("plain", "rerank") in r]
        dis_gl = f"{sum(f and not o for f, o in dis)}/{sum(o and not f for f, o in dis)}"
        star = "*" if p < 0.05 else ""
        fstar = "*" if mcnemar(b, c) < 0.05 else ""
        out.append(f"| {m} | {a_r:.1f} | {a_m:.1f} ({d:+.1f}{star}) | {a_o:.1f} | {grp['gold']} | {grp['distractor']} | "
                   f"{fa:.1f} ({100 * (b - c) / max(1, len(base)):+.1f}{fstar}, δ={best:g}) | "
                   f"{fb_b}/{fb_c}{'*' if fb_p < 0.05 else ''} (p={fb_p:.3f}) | {dis_gl} | {rate:.1f}% |")
        res[m] = dict(rerank=a_r, method=a_m, oracle=a_o, delta=d, p=p, groups=grp, fallback=fa, fallback_delta=best,
                      fallback_rate=rate, fallback_gain=100 * (b - c) / max(1, len(base)), fallback_p=mcnemar(b, c), n=n,
                      fallback_vs_method=dict(gain=fb_b, loss=fb_c, p=fb_p), distractor_after_fallback=dis_gl)
    return "\n".join(out), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--judge", default="judge_adr.jsonl")
    ap.add_argument("--md")
    a = ap.parse_args()
    text, dump = [], {}
    for rd in a.runs:
        for reader in READERS:
            if not os.path.exists(os.path.join(rd, READERS[reader].format(m="plain"))):
                continue
            t, res = report(rd, reader, a.judge)
            text.append(t)
            dump[f"{os.path.basename(rd.rstrip('/'))}/{reader}"] = res
    print("\n".join(text))
    if a.md:
        open(a.md, "w").write("\n".join(text) + "\n")
        json.dump(dump, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
