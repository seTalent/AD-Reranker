"""Where the method helps: rerank vs. score20@3 by question group, for every run and reader (results doc section 24).

Groups (test split, Qwen3.5-27B judge):
  evidence    single-page / multi-page / unanswerable (no gold page)
  misfilter   whether the reranker dropped at least one gold page that was in the candidate pool
  kept        how many pages the reranker kept (1 / 2 / 3+)
  source      evidence source (MMLongBench: Pure-text, Chart, Table, ...; LongDocURL: Text, Layout, Table, Figure);
              a question with several sources counts in each
  task        LongDocURL task tag (Understanding / Reasoning / Locating)
  doctype     MMLongBench document type
  format      answer format
  doclen      document length in pages (terciles of the run)
Per group: n, accuracy of both, Δ, gains/losses, McNemar p, gold recall of both.

usage: python scripts/rp2_breakdown_all.py <run dirs> [--method score20@3] [--md out.md]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.analysis import doc_split, mcnemar   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, as_list, gold_pages, judge_key, skey   # noqa: E402

READERS = {"qwen25vl7b": "answers_px1003520.jsonl", "qwen3vl8b": "answers_qwen3vl8b_px1003520.jsonl",
           "qwen25vl32b": "answers_qwen25vl32b_px1003520_ear-plain.jsonl"}
DIMS = ["evidence", "misfilter", "kept", "source", "task", "doctype", "format", "doclen"]


def doc_pages(s, extract_root):
    if s.get("total_pages"):
        return int(s["total_pages"])
    doc = os.path.splitext(s["doc_id"])[0]
    return len(glob.glob(os.path.join(extract_root, glob.escape(doc) + "_*.png"))) or None


def groups(s, method, npages, cuts):
    G, cands = set(gold_pages(s)), s["cands"]
    kept = {cands[i - 1] for i in s["variants"]["rerank"]["ids"]}
    unans = str(s.get("answer", "")).startswith("Not answerable") or not G
    out = dict(evidence=["unanswerable" if unans else "single-page" if len(G) == 1 else "multi-page"],
               misfilter=["no gold in pool" if not (G & set(cands)) else
                          "gold dropped" if (G & set(cands)) - kept else "no gold dropped"],
               kept=[str(min(len(kept), 3)) + ("+" if len(kept) >= 3 else "")],
               source=as_list(s.get("evidence_sources", [])) or ["none"],
               format=[str(s.get("answer_format"))])
    if s.get("task_tag"):
        out["task"] = [s["task_tag"]]
    if s.get("doc_type"):
        out["doctype"] = [s["doc_type"].split("/")[0].strip()]
    if npages:
        out["doclen"] = [f"≤{cuts[0]} pages" if npages <= cuts[0] else f"{cuts[0] + 1}–{cuts[1]} pages"
                         if npages <= cuts[1] else f">{cuts[1]} pages"]
    return out


def run(rd, reader, method, split, extract_root):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    ans, jud = _jsonl(os.path.join(rd, READERS[reader])), _jsonl(os.path.join(rd, "judge_adr.jsonl"))
    rows, seen = [], set()
    for s in rec:
        if doc_split(s["doc_id"]) != split or skey(s) in seen:
            continue
        seen.add(skey(s))
        oks = []
        for v in ("rerank", method):
            a = ans.get(answer_key(s, s["variants"][v]["ids"]))
            j = jud.get(judge_key(s, a["response"])) if a else None
            oks.append(None if j is None else int(bool(j["verdict"]["equivalent"])))
        if None in oks:
            continue
        G, cands = set(gold_pages(s)), s["cands"]
        rec_ = [len(G & {cands[i - 1] for i in s["variants"][v]["ids"]}) / len(G) if G else None
                for v in ("rerank", method)]
        rows.append((s, oks, rec_, doc_pages(s, extract_root)))
    lens = sorted(n for *_, n in rows if n)
    cuts = (lens[len(lens) // 3], lens[2 * len(lens) // 3]) if lens else (0, 0)
    table = defaultdict(lambda: defaultdict(list))
    for s, oks, rc, n in rows:
        for dim, names in groups(s, method, n, cuts).items():
            for g in names:
                table[dim][g].append((oks, rc))
    return table, len(rows)


def fmt(xs):
    n = len(xs)
    a0 = 100 * sum(o[0] for o, _ in xs) / n
    a1 = 100 * sum(o[1] for o, _ in xs) / n
    b = sum(o[1] and not o[0] for o, _ in xs)
    c = sum(o[0] and not o[1] for o, _ in xs)
    r = [x for _, x in xs if x[0] is not None]
    rr = (f"{100 * sum(x[0] for x in r) / len(r):.1f} → {100 * sum(x[1] for x in r) / len(r):.1f}" if r else "–")
    p = mcnemar(b, c)
    return dict(n=n, rerank=a0, method=a1, delta=a1 - a0, gain=b, loss=c, p=p, recall=rr), \
        f"{n} | {a0:.1f} | {a1:.1f} | {a1 - a0:+.1f}{'*' if p < 0.05 else ''} | {b}/{c} | {rr}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--method", default="score20@3")
    ap.add_argument("--split", default="test")
    ap.add_argument("--md")
    a = ap.parse_args()
    text, dump = [], {}
    for rd in a.runs:
        ds = os.path.basename(os.path.dirname(rd.rstrip("/")))
        extract_root = os.path.join("tmp", ds)
        for reader in READERS:
            if not os.path.exists(os.path.join(rd, READERS[reader])):
                continue
            table, n = run(rd, reader, a.method, a.split, extract_root)
            name = f"{os.path.basename(rd.rstrip('/'))} / {reader}"
            text += [f"\n### {name}  ({a.split}, {n} questions; rerank vs {a.method})\n",
                     "| dimension | group | n | rerank | method | Δ | gain/loss | gold recall |", "|---|---|---|---|---|---|---|---|"]
            dump[name] = {}
            for dim in DIMS:
                for g, xs in sorted(table.get(dim, {}).items(), key=lambda kv: -len(kv[1])):
                    if len(xs) < 20:
                        continue
                    d, line = fmt(xs)
                    dump[name].setdefault(dim, {})[g] = d
                    text.append(f"| {dim} | {g} | {line} |")
    print("\n".join(text))
    if a.md:
        open(a.md, "w").write("\n".join(text) + "\n")
        json.dump(dump, open(a.md.replace(".md", ".json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
