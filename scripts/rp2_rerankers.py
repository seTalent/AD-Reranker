"""Recovery on top of different rerankers (results doc sections 25-26).

Runs of one dataset that differ only in the reranker (Zero-Shot Qwen2.5-VL-7B, the trained AD-Reranker,
Qwen2.5-VL-32B zero-shot, MM-R5 keeping its first 2 pages). Per reranker and reader (test split, Qwen3.5-27B judge):
accuracy, pages read and gold recall of rerank-only / + score20@3 / + score20m2@3 / + oracle@2, the gain over
rerank-only (McNemar), and the reranker-only accuracy compared with Zero-Shot + score20@3 on the same questions.

usage: python scripts/rp2_rerankers.py <name>=<run dir> ... [--md out.md]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.analysis import doc_split, mcnemar   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, gold_pages, judge_key, skey   # noqa: E402

READERS = {"qwen25vl7b": "answers_px1003520.jsonl", "qwen3vl8b": "answers_qwen3vl8b_px1003520.jsonl"}
VARIANTS = ["rerank", "score20@3", "score20m2@3", "oracle@2"]


def load(rd, reader):
    rec = json.load(open(os.path.join(rd, "recover.json")))
    ans, jud = _jsonl(os.path.join(rd, READERS[reader])), _jsonl(os.path.join(rd, "judge_adr.jsonl"))
    out, seen = {}, set()
    for s in rec:
        if doc_split(s["doc_id"]) != "test" or skey(s) in seen:
            continue
        seen.add(skey(s))
        G, cands, r = set(gold_pages(s)), s["cands"], {}
        for v in VARIANTS:
            x = s["variants"].get(v)
            a = ans.get(answer_key(s, x["ids"])) if x else None
            j = jud.get(judge_key(s, a["response"])) if a else None
            if j is not None:
                pages = {cands[i - 1] for i in x["ids"]}
                r[v] = dict(ok=int(bool(j["verdict"]["equivalent"])), pages=len(x["ids"]),
                            rec=len(pages & G) / len(G) if G else None)
        out[skey(s)] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=run_dir; the first is the reference (Zero-Shot)")
    ap.add_argument("--md")
    a = ap.parse_args()
    runs = [x.split("=", 1) for x in a.runs]
    text, dump = [], {}
    for reader in READERS:
        data = {n: load(rd, reader) for n, rd in runs}
        ref = data[runs[0][0]]
        text += [f"\n### {reader}\n", "| reranker | variant | acc | Δ vs its rerank | Δ vs Zero-Shot + score20@3 | pages | gold recall |",
                 "|---|---|---|---|---|---|---|"]
        for n, _ in runs:
            d = data[n]
            for v in VARIANTS:
                qs = [q for q in d if v in d[q] and "rerank" in d[q] and q in ref and "score20@3" in ref[q]]
                if not qs:
                    continue
                acc = 100 * sum(d[q][v]["ok"] for q in qs) / len(qs)
                b = sum(d[q][v]["ok"] and not d[q]["rerank"]["ok"] for q in qs)
                c = sum(d[q]["rerank"]["ok"] and not d[q][v]["ok"] for q in qs)
                b2 = sum(d[q][v]["ok"] and not ref[q]["score20@3"]["ok"] for q in qs)
                c2 = sum(ref[q]["score20@3"]["ok"] and not d[q][v]["ok"] for q in qs)
                rc = [d[q][v]["rec"] for q in qs if d[q][v]["rec"] is not None]
                pg = sum(d[q][v]["pages"] for q in qs) / len(qs)
                d1, d2 = 100 * (b - c) / len(qs), 100 * (b2 - c2) / len(qs)
                dump.setdefault(reader, {}).setdefault(n, {})[v] = dict(acc=acc, d_rerank=d1, p_rerank=mcnemar(b, c),
                                                                       d_ref=d2, p_ref=mcnemar(b2, c2), pages=pg, n=len(qs))
                text.append(f"| {n} | {v} | {acc:.1f} | {d1:+.1f}{'*' if mcnemar(b, c) < 0.05 else ''} | "
                            f"{d2:+.1f}{'*' if mcnemar(b2, c2) < 0.05 else ''} | {pg:.2f} | {100 * sum(rc) / max(1, len(rc)):.1f} |")
    print("\n".join(text))
    if a.md:
        open(a.md, "w").write("\n".join(text) + "\n")
        json.dump(dump, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
