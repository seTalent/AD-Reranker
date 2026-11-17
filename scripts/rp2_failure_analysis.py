"""Why recovered recall does not all turn into accuracy (results doc, section 19).

Test questions are grouped by what the method added: no page / at least one gold page / only non-gold pages /
pages for a question without gold pages. For each group: gains (rerank wrong -> method right) and losses.
For gold-added questions the method still gets wrong, the share the oracle page set (kept + dropped gold
pages, `oracle@2`) answers correctly separates "distracted by the extra pages" from "cannot read it anyway".

usage: python scripts/rp2_failure_analysis.py <run_dir> [--variant score20@3] [--answers answers_px1003520.jsonl]
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.analysis import doc_split   # noqa: E402
from recovery.pipeline import _jsonl, answer_key, gold_pages, judge_key, skey   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--variant", default="score20@3")
    ap.add_argument("--answers", default="answers_px1003520.jsonl")
    ap.add_argument("--judge", default="judge_adr.jsonl")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    rec = json.load(open(os.path.join(a.run, "recover.json")))
    ans, jud = _jsonl(os.path.join(a.run, a.answers)), _jsonl(os.path.join(a.run, a.judge))

    def ok(s, name):
        v = s["variants"].get(name)
        r = ans.get(answer_key(s, v["ids"])) if v else None
        j = jud.get(judge_key(s, r["response"])) if r else None
        return None if j is None else bool(j["verdict"]["equivalent"])

    groups, oracle, seen = {}, Counter(), set()
    for s in rec:
        if doc_split(s["doc_id"]) != a.split or skey(s) in seen:
            continue
        seen.add(skey(s))
        c0, c1 = ok(s, "rerank"), ok(s, a.variant)
        if c0 is None or c1 is None:
            continue
        G, cands = set(gold_pages(s)), s["cands"]
        added = {cands[i - 1] for i in s["variants"][a.variant]["added"]}
        g = ("no page added" if not added else "gold page added" if added & G else
             "no gold pages (unanswerable)" if not G else "only non-gold pages added")
        groups.setdefault(g, Counter())[("R" if c0 else "r") + ("M" if c1 else "m")] += 1
        if g == "gold page added" and not c1:
            co = ok(s, "oracle@2")
            if co is not None:
                oracle["oracle right" if co else "oracle wrong"] += 1
    print(f"{a.run}  {a.variant}  {a.answers}  ({a.split})")
    print(f"{'group':30s} {'n':>5s} {'gain':>5s} {'loss':>5s} {'both right':>11s} {'both wrong':>11s}")
    for g, t in sorted(groups.items()):
        print(f"{g:30s} {sum(t.values()):5d} {t['rM']:5d} {t['Rm']:5d} {t['RM']:11d} {t['rm']:11d}")
    n = sum(oracle.values())
    print(f"gold page added but still wrong: {n}; oracle page set answers {oracle['oracle right']} "
          f"({100 * oracle['oracle right'] / max(1, n):.1f}%) of them correctly")


if __name__ == "__main__":
    main()
