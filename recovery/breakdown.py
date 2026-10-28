"""Additional metrics for selected variants: accuracy/recall by evidence source, evidence count and
document type; page-selection precision/recall/F1, MRR and nDCG of the reader input; answer flips
relative to the reranked input."""
import ast
import json
import math
import os
from collections import defaultdict

from recovery.analysis import doc_split, mean
from recovery.pipeline import _jsonl, answer_key, answers_file, judge_key, skey


def _ranking(order, gold):
    """MRR and nDCG of the ordered reader input against the gold pages."""
    rel = [1.0 if p in gold else 0.0 for p in order]
    mrr = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), len(order))))
    return mrr, (dcg / ideal if ideal else 0.0)


def breakdown(cfg, dataset):
    rd = dataset.config.result_dir
    rec = json.load(open(os.path.join(rd, "recover.json")))
    answers = _jsonl(os.path.join(rd, answers_file(cfg)))
    judge = _jsonl(os.path.join(rd, cfg.judge.get("out", "judge.jsonl")))
    if cfg.get("split"):
        rec = [s for s in rec if doc_split(s["doc_id"]) == cfg.split]
    variants = list(cfg.get("breakdown_variants", ["rerank", "score20@3", "top10"]))
    correct = defaultdict(dict)
    out = dict(selection={}, by_source={}, by_evidence={}, by_doctype={}, flips={})
    for v in variants:
        sel = defaultdict(list)
        groups = {k: defaultdict(lambda: defaultdict(list)) for k in ("by_source", "by_evidence", "by_doctype")}
        for s in rec:
            ids = s["variants"][v]["ids"]
            a = answers.get(answer_key(s, ids))
            j = judge.get(judge_key(s, a["response"])) if a else None
            if j is None:
                continue
            ok = float(j["verdict"]["equivalent"])
            correct[v][skey(s)] = ok
            gold = {p - 1 for p in ast.literal_eval(s["evidence_pages"])}
            order = [s["cands"][i - 1] for i in ids]
            n_gold = len(gold)
            ev = "unanswerable" if str(s["answer"]).startswith("Not answerable") else (
                "single-page" if n_gold == 1 else "multi-page" if n_gold > 1 else "no-annotation")
            cats = dict(by_source=ast.literal_eval(s["evidence_sources"]) or ["none"],
                        by_evidence=[ev], by_doctype=[s["doc_type"]])
            if gold:
                hit = len(set(order) & gold)
                p, r = hit / max(1, len(order)), hit / n_gold
                mrr, ndcg = _ranking(order, gold)
                sel["precision"].append(p)
                sel["recall"].append(r)
                sel["f1"].append(2 * p * r / (p + r) if p + r else 0.0)
                sel["mrr"].append(mrr)
                sel["ndcg"].append(ndcg)
            for g, keys in cats.items():
                for c in keys:
                    groups[g][c]["acc"].append(ok)
                    if gold:
                        groups[g][c]["recall"].append(len(set(order) & gold) / n_gold)
        out["selection"][v] = {k: mean(x) for k, x in sel.items()}
        for g in groups:
            out[g][v] = {c: dict(n=len(x["acc"]), acc=mean(x["acc"]), recall=mean(x["recall"]))
                         for c, x in sorted(groups[g].items())}
    base = correct.get("rerank", {})
    for v in variants:
        if v == "rerank":
            continue
        ks = [k for k in correct[v] if k in base]
        out["flips"][v] = dict(fixed=sum(base[k] < correct[v][k] for k in ks),
                               broken=sum(base[k] > correct[v][k] for k in ks), n=len(ks))
    name = f"breakdown_{cfg.get('split', 'all')}_px{cfg.models.answer.all_pixels}.json"
    with open(os.path.join(rd, name), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out
