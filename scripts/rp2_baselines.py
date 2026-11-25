"""Related-work baselines in the recovery slot vs. the method (results doc section 27).

All baselines start from the same reranked candidates and are compared on the test split (Qwen3.5-27B judge).
Each family has a free parameter; it is chosen on the dev split of the same run in two ways:
  equal pages  the setting whose mean pages read on dev is closest to score20@3 (equal reader cost)
  best dev     the setting with the highest dev accuracy
Families
  conf→top10 / conf→ret@2  reader-side trigger (Self-RAG / FLARE-like): answer with the reranked pages; if the
                           reader's answer confidence is below δ or it answers "Not answerable", answer again with
                           Top-10 / the reranked pages + 2 retrieved pages (pages read = both calls)
  suff     CRAG-style sufficiency check, then + top-2 dropped pages by retrieval rank (suff30/50/70@2)
  gate     the sufficiency check as a gate in front of score20 (gate30/50/70@3)
  mmr      MMR (mmr5/7/9@1, mmr7@2);  rs  retrieval-score threshold (rs50/70/90@3)
Reported: accuracy, Δ vs rerank and vs score20@3 (McNemar), mean pages read, and the extra model calls.

usage: python scripts/rp2_baselines.py <run dirs> [--md out.md]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.analysis import doc_split, mcnemar   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, judge_key, skey   # noqa: E402
from recovery.reading import answer_text, normalise   # noqa: E402

READERS = {"qwen25vl7b": ("answers_px1003520.jsonl", "answers_px1003520_ear-plain.jsonl"),
           "qwen3vl8b": ("answers_qwen3vl8b_px1003520.jsonl", "answers_qwen3vl8b_px1003520_ear-plain.jsonl")}
FAMILIES = {"suff": ["suff30@2", "suff50@2", "suff70@2"], "gate": ["gate30@3", "gate50@3", "gate70@3"],
            "mmr": ["mmr5@1", "mmr7@1", "mmr9@1", "mmr7@2"], "rs": ["rs50@3", "rs70@3", "rs90@3"]}
OURS = ["score20@3", "score20m2@3"]


def load(rd, reader):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    ans, conf = (_jsonl(os.path.join(rd, f)) for f in READERS[reader])
    jud = _jsonl(os.path.join(rd, "judge_adr.jsonl"))
    rows, seen = [], set()
    for s in rec:
        if skey(s) in seen:
            continue
        seen.add(skey(s))
        r = dict(split=doc_split(s["doc_id"]), ok={}, pages={})
        for v, x in s["variants"].items():
            a = ans.get(answer_key(s, x["ids"]))
            j = jud.get(judge_key(s, a["response"])) if a else None
            if j is not None:
                r["ok"][v] = int(bool(j["verdict"]["equivalent"]))
                r["pages"][v] = len(x["ids"])
        c = conf.get(answer_key(s, s["variants"]["rerank"]["ids"]))    # plain prompt with log-probs (section 20)
        if c and c.get("conf_ans") is not None:
            r["conf"], r["na"] = c["conf_ans"], normalise(answer_text(c["response"])).startswith("not answerable")
        if "rerank" in r["ok"]:
            rows.append(r)
    return rows


def static(rows, v):
    xs = [r for r in rows if v in r["ok"]]
    if len(xs) < len(rows):            # every question needs an answer for this page list
        return None
    return [(r["ok"][v], r["pages"][v], r) for r in xs]


def trigger(rows, v, d):
    xs = []
    for r in rows:
        if "conf" not in r or v not in r["ok"]:
            return None
        t = r["conf"] < d or r["na"]
        xs.append((r["ok"][v] if t else r["ok"]["rerank"], r["pages"]["rerank"] + (r["pages"][v] if t else 0), r))
    return xs


def mean(xs, i):
    return sum(x[i] for x in xs) / len(xs)


def candidates(rows, fam):
    if fam.startswith("conf"):
        v = "top10" if fam.endswith("top10") else "ret@2"
        cs = sorted(r["conf"] for r in rows if "conf" in r)
        if not cs:
            return []
        qs = sorted({cs[min(len(cs) - 1, int(q / 50 * len(cs)))] for q in range(0, 51)} | {-1e9})
        return [(f"δ={d:.3f}", lambda rs, d=d, v=v: trigger(rs, v, d)) for d in qs]
    return [(v, lambda rs, v=v: static(rs, v)) for v in FAMILIES[fam]]


def report(rd, reader):
    rows = load(rd, reader)
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    base = {id(r): r["ok"]["rerank"] for r in test}
    ours = static(test, "score20@3")
    target = mean(static(dev, "score20@3"), 1)
    lines, dump = [], {}

    def row(name, xs):
        b = sum(x[0] and not base[id(x[2])] for x in xs)
        c = sum(base[id(x[2])] and not x[0] for x in xs)
        o = {id(x[2]): x[0] for x in ours}
        b2 = sum(x[0] and not o[id(x[2])] for x in xs)
        c2 = sum(o[id(x[2])] and not x[0] for x in xs)
        n = len(xs)
        d1, d2 = 100 * (b - c) / n, 100 * (b2 - c2) / n
        dump[name] = dict(acc=100 * mean(xs, 0), pages=mean(xs, 1), d_rerank=d1, p_rerank=mcnemar(b, c),
                          d_ours=d2, p_ours=mcnemar(b2, c2))
        return (f"| {name} | {100 * mean(xs, 0):.1f} | {d1:+.1f}{'*' if mcnemar(b, c) < 0.05 else ''} | "
                f"{d2:+.1f}{'*' if mcnemar(b2, c2) < 0.05 else ''} | {mean(xs, 1):.2f} |")

    for v in ["rerank", "ret@2", "top10"] + OURS:
        xs = static(test, v)
        if xs:
            lines.append(row(v, xs))
    for fam in ["conf→top10", "conf→ret@2", "suff", "gate", "mmr", "rs"]:
        cands = [(n, f) for n, f in candidates(dev, fam) if f(dev)]
        if not cands:
            continue
        eq = min(cands, key=lambda nf: abs(mean(nf[1](dev), 1) - target))
        best = max(cands, key=lambda nf: mean(nf[1](dev), 0))
        for how, (n, f) in (("equal pages", eq), ("best dev", best)):
            if fam.startswith("conf"):   # the threshold is a dev quantile: apply the same δ to test
                d = float(n.split("=")[1])
                xs = trigger(test, "top10" if fam.endswith("top10") else "ret@2", d)
            else:
                xs = static(test, n)
            if xs:
                lines.append(row(f"{fam} ({how}: {n})", xs))
    head = [f"\n### {os.path.basename(rd.rstrip('/'))} / {reader}  (test {len(test)}; dev target pages {target:.2f})\n",
            "| method | acc | Δ vs rerank | Δ vs score20@3 | pages read |", "|---|---|---|---|---|"]
    return head + lines, dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md")
    a = ap.parse_args()
    text, dump = [], {}
    for rd in a.runs:
        for reader in READERS:
            t, d = report(rd, reader)
            text += t
            dump[f"{os.path.basename(rd.rstrip('/'))}/{reader}"] = d
    print("\n".join(text))
    if a.md:
        open(a.md, "w").write("\n".join(text) + "\n")
        json.dump(dump, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
