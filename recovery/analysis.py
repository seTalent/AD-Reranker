"""Metrics for the evidence-recovery experiments (called by recovery.pipeline.eval_stage)."""
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict

from recovery import v2_eval
from recovery.pagerep import PageStore
from recovery.verifier import gap_support

MAIN = ["top4", "top10", "rerank", "ret@2", "adj@2", "graph@2", "rel@2", "ours_ret@2", "ours_all@2",
        "ours_noprov@2", "ours_tail@2", "ours@2", "oracle@2"]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def cluster_bootstrap(diff, doc_of, iters=2000, seed=0):
    docs = defaultdict(list)
    for q, d in diff.items():
        docs[doc_of[q]].append(d)
    keys, rng, stats = sorted(docs), random.Random(seed), []
    for _ in range(iters):
        vals = [x for _ in keys for x in docs[rng.choice(keys)]]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    return stats[int(0.025 * iters)], stats[int(0.975 * iters) - 1]


def doc_split(doc_id):
    """Fixed document-level split: ~5/16 of documents for development, the rest for test."""
    import hashlib
    return "dev" if hashlib.sha1(doc_id.encode()).hexdigest()[0] in "01234" else "test"


def analyze(rec, answers, judge, result_dir, cfg):
    from recovery.pipeline import answer_key, judge_key, skey
    q_of = {skey(s): s for s in rec}
    if cfg.get("split"):
        q_of = {k: s for k, s in q_of.items() if doc_split(s["doc_id"]) == cfg.split}
    doc_of = {k: s["doc_id"] for k, s in q_of.items()}
    split = {}
    for k, s in q_of.items():
        cands, G = s["cands"], {p - 1 for p in eval(s["evidence_pages"])}
        kept = {cands[i - 1] for i in s["refiner_ids"]}
        split[k] = dict(G=G, C=set(cands), kept=kept, D=set(cands) - kept)
    wf = {k for k in q_of if split[k]["G"] & split[k]["D"]}

    table, score = {}, {}
    names = sorted({n for s in rec for n in s["variants"]})
    for var in names:
        rows, sc, ev, cov, npg, add, ptok = [], {}, [], [], [], [], []
        add_gold = rec_num = rec_den = 0
        for k, s in q_of.items():
            v = s["variants"].get(var)
            a = answers.get(answer_key(s, v["ids"])) if v else None
            j = judge.get(judge_key(s, a["response"])) if a else None
            if j is None:
                continue
            rows.append(dict(s, llm_judge=j["verdict"]))
            sc[k] = 1.0 if j["verdict"]["equivalent"] else 0.0
            pages = {s["cands"][i - 1] for i in v["ids"]}
            R = {s["cands"][i - 1] for i in v["added"]}
            G = split[k]["G"]
            if G:
                ev.append(len(pages & G) / len(G))
                cov.append(float(G <= pages))
            npg.append(len(pages))
            if a.get("prompt_tokens") is not None:
                ptok.append(a["prompt_tokens"])
            add.append(len(R))
            add_gold += len(R & G)
            if k in wf:
                rec_num += len(R & G & split[k]["D"])
                rec_den += len(G & split[k]["D"])
        if not rows:
            continue
        acc, f1 = v2_eval.metrics.acc_and_f1(rows)
        if cfg.judge.prompt != "v2":
            f1 = float("nan")    # binary correctness has no abstention verdict, so F1 is undefined
        score[var] = sc
        is_ans = lambda k: not str(q_of[k]["answer"]).startswith("Not answerable")
        n_gold = lambda k: len(split[k]["G"])
        table[var] = dict(n=len(rows), acc=acc, f1=f1, page_recall=mean(ev), full_coverage=mean(cov),
                          pages=mean(npg), reader_prompt_tokens=mean(ptok), added=mean(add),
                          added_precision=add_gold / sum(add) if sum(add) else float("nan"),
                          wf_recovery=rec_num / rec_den if rec_den else float("nan"),
                          wf_acc=mean(sc[k] for k in wf if k in sc),
                          single_acc=mean(sc[k] for k in sc if n_gold(k) == 1),
                          multi_acc=mean(sc[k] for k in sc if n_gold(k) > 1 and is_ans(k)),
                          unans_acc=mean(sc[k] for k in sc if not is_ans(k)))

    paired, base = {}, score.get("rerank", {})
    for var, sc in score.items():
        common = [k for k in sc if k in base]
        if var == "rerank" or not common:
            continue
        win = sum(sc[k] > base[k] for k in common)
        loss = sum(sc[k] < base[k] for k in common)
        lo, hi = cluster_bootstrap({k: sc[k] - base[k] for k in common}, doc_of)
        changed = [k for k in common if q_of[k]["variants"][var]["added"]]
        paired[var] = dict(n=len(common), delta=mean(sc[k] - base[k] for k in common), ci95=[lo, hi], win=win,
                           loss=loss, mcnemar_p=mcnemar(win, loss), changed=len(changed),
                           delta_changed=mean(sc[k] - base[k] for k in changed))

    ans = [k for k in q_of if split[k]["G"]]
    pool = [k for k in ans if split[k]["G"] & split[k]["C"]]
    motivation = dict(
        questions=len(q_of), with_gold=len(ans),
        retrieval_recall=mean(len(split[k]["G"] & split[k]["C"]) / len(split[k]["G"]) for k in ans),
        all_gold_retrieved=mean(float(split[k]["G"] <= split[k]["C"]) for k in ans),
        kept_pages=mean(len(split[k]["kept"]) for k in q_of),
        refiner_fallback=mean(float(q_of[k].get("refiner_fallback", False)) for k in q_of),
        kept_recall_in_pool=mean(len(split[k]["G"] & split[k]["kept"]) / len(split[k]["G"] & split[k]["C"]) for k in pool),
        wrongful_filter_rate=mean(float(k in wf) for k in pool),
        wrongful_filter_single=mean(float(k in wf) for k in pool if len(split[k]["G"]) == 1),
        wrongful_filter_multi=mean(float(k in wf) for k in pool if len(split[k]["G"]) > 1),
        wf_questions=len(wf), dropped_gold_pages=sum(len(split[k]["G"] & split[k]["D"]) for k in q_of))

    reach, kinds = Counter(), {"gold": Counter(), "other": Counter()}
    m = cfg.recovery.candidate_limit
    for k, s in q_of.items():
        located = s["located"]
        for pos, kind in s["dropped_relation"].items():
            page = s["cands"][int(pos) - 1]
            g = "gold" if page in split[k]["G"] else "other"
            reach[(g, "dropped")] += 1
            reach[(g, "linked")] += int(pos) in located
            reach[(g, "in_U")] += int(pos) in located[:m]
            kinds[g][kind] += 1
    graph = {f"{g}_{x}": reach[(g, x)] / max(1, reach[(g, "dropped")]) for g in ("gold", "other") for x in ("linked", "in_U")}
    graph.update(dropped_gold=reach[("gold", "dropped")], dropped_other=reach[("other", "dropped")],
                 strongest_relation={g: dict(c) for g, c in kinds.items()})

    store = PageStore(cfg.dataset.extract_path, cfg.recovery.mineru_dir)
    by_key = {s["job_key"]: s for s in rec}
    vstats = {}
    for mode in ("gap_graph", "gap_ret", "gap_all", "rel_graph"):
        path = os.path.join(result_dir, "recovery_cache", f"verify_{mode}.jsonl")
        if not os.path.exists(path):
            continue
        rows = {}
        for line in open(path):
            r = json.loads(line)
            if not r.get("error") and r["key"] in by_key:
                rows[r["key"]] = r
        called = [r for r in rows.values() if not r.get("skipped")]
        parsed = n_req = n_miss = prov_ok = prov_all = 0
        if mode.startswith("gap"):
            for r in called:
                s = by_key[r["key"]]
                res = gap_support(r["raw"], {c: int(p) for c, p in r["cand_ids"].items()},
                                  store.get(s["doc_id"].rsplit(".pdf", 1)[0])[0])
                if res is None:
                    continue
                parsed += 1
                n_req += res["n_req"]
                n_miss += len(res["missing"])
                prov_all += sum(x["how"] not in ("not-missing", "unknown-candidate") for x in res["audit"])
                prov_ok += sum(x["ok"] for x in res["audit"])
        vstats[mode] = dict(questions=len(rows), called=len(called), parsed=parsed,
                            requirements=n_req / max(1, parsed), missing=n_miss / max(1, parsed),
                            provenance_pass=prov_ok / max(1, prov_all),
                            prompt_tokens=mean(r["prompt_tokens"] for r in called),
                            completion_tokens=mean(r["completion_tokens"] for r in called),
                            graph_ms=1000 * mean(r["t_graph"] for r in rows.values()))
    probe = []
    if os.path.exists(os.path.join(result_dir, "probe.jsonl")):
        probe = [json.loads(l) for l in open(os.path.join(result_dir, "probe.jsonl"))]
    latency = {k: dict(mean=mean(r[k] for r in probe), median=statistics.median(r[k] for r in probe))
               for k in (probe[0] if probe else {}) if isinstance(probe[0][k], float)}
    return dict(variants=table, paired=paired, motivation=motivation, graph=graph, verifier=vstats,
                latency=latency, n_probe=len(probe), judge=dict(cfg.judge))
