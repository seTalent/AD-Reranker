"""Summarise the +stage=probe_v2 latency probes (results doc section 22).

Per run: median sequential wall time of each component (seconds per question, one request in flight) and
seconds per question at throughput (batch of other questions, 32 requests in flight); then the end-to-end
pipeline of each method per question = rerank + page representation/graph + verification (if the method
uses it) + reading its page set, as the median over questions of the per-question sums
(the reranker / verifier times come from the 7B probe of the same questions).

usage: python scripts/rp2_latency_summary.py <run dirs> [--md out.md]
"""
import argparse
import json
import os
import statistics as st

READERS = ["qwen25vl7b", "qwen3vl8b", "qwen25vl32b"]
METHODS = {"rerank": None, "ret@2": None, "scoreni20@3": "verify:score_noimg", "score20@3": "verify:score_mm",
           "top10": None}


def rows(rd, tag):
    p = os.path.join(rd, f"probe_v2_{tag}.jsonl")
    return {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md")
    a = ap.parse_args()
    text, dump = [], {}
    for rd in a.runs:
        name = os.path.basename(rd.rstrip("/"))
        base = rows(rd, "qwen25vl7b")
        if not base:
            continue
        summ = {t: json.load(open(os.path.join(rd, f"probe_v2_{t}.json"))) for t in READERS
                if os.path.exists(os.path.join(rd, f"probe_v2_{t}.json"))}
        comp = ["rerank", "sample_n6", "context", "verify:score_mm", "verify:score_noimg"]
        s7 = summ.get("qwen25vl7b", {})
        text += [f"\n### {name}  (sequential n = {len(base)}; batch n = {s7.get('n_batch', '?')})\n",
                 "| component | sequential median (s) | batch s/question |", "|---|---|---|"]
        for c in comp:
            sq = s7.get("sequential_seconds", {}).get(c, {}).get("median")
            bt = s7.get("batch_seconds_per_question", {}).get(c)
            if sq is not None:
                text.append(f"| {c} | {sq:.3f} | {'–' if bt is None else f'{bt:.3f}'} |")
        text += ["", "| method | " + " | ".join(f"{t} read (seq / batch)" for t in summ) + " | "
                 + " | ".join(f"{t} end-to-end seq (× rerank-only)" for t in summ) + " |",
                 "|---|" + "---|" * (2 * len(summ))]
        dump[name] = {}
        e2e_base = {}
        for m, vkey in METHODS.items():
            reads, e2e = [], []
            for t in summ:
                rr = rows(rd, t)
                sq = summ[t]["sequential_seconds"].get(f"read:{m}", {}).get("median")
                bt = summ[t]["batch_seconds_per_question"].get(f"read:{m}")
                reads.append("–" if sq is None else f"{sq:.2f} / {'–' if bt is None else f'{bt:.2f}'}")
                tot = [base[k]["rerank"] + base[k]["context"] + (base[k][vkey] if vkey else 0) + rr[k][f"read:{m}"]
                       for k in base if k in rr and f"read:{m}" in rr[k] and (not vkey or vkey in base[k])]
                med = st.median(tot) if tot else None
                if m == "rerank":
                    e2e_base[t] = med
                e2e.append("–" if med is None else f"{med:.2f} ({med / e2e_base[t]:.2f}×)")
                dump[name].setdefault(m, {})[t] = dict(read_seq=sq, read_batch=bt, e2e_seq=med)
            text.append(f"| {m} | " + " | ".join(reads) + " | " + " | ".join(e2e) + " |")
        dump[name]["components"] = {c: dict(seq=s7.get("sequential_seconds", {}).get(c, {}).get("median"),
                                            batch=s7.get("batch_seconds_per_question", {}).get(c)) for c in comp}
    print("\n".join(text))
    if a.md:
        open(a.md, "w").write("\n".join(text) + "\n")
        json.dump(dump, open(a.md.replace(".md", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
