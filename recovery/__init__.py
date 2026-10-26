"""Post-rerank evidence recovery (thesis research point 2).

The reranker keeps S from the top-k candidates C and drops D = C \\ S. Recovery
1. builds a page association graph over C and locates dropped pages linked to S;
2. asks one text-only verifier call which requirements S misses and which candidates
   fill them with a verbatim quote, then checks each quote against the page (V(z));
3. greedily restores at most b pages that cover the most missing requirements, inserted
   next to their related kept page. The reranked order of S is never changed.

Methods (baselines and ablations share the same inputs):
  none        reranked pages only
  ret         + top-b dropped pages by retrieval rank
  adj         + dropped pages adjacent to a kept page
  graph       + top-b graph-linked pages, no verification        (w/o gap detection)
  rel         + graph candidates judged "useful" by the verifier   (w/o requirement modelling)
  ours_ret    + gap verification over top-m dropped by retrieval   (w/o graph)
  ours_all    + gap verification over all dropped pages            (w/o candidate narrowing)
  ours_noprov + method without the provenance check                (w/o V(z))
  ours_tail   + method, restored pages appended at the end         (placement)
  ours        the method
  oracle      + dropped gold evidence pages                        (upper bound, analysis only)
"""
import hashlib
import json
import os
import time

from models.local_llm import LocalLLM
from recovery import verifier as V
from recovery.graph import anchor, build_graph, locate
from recovery.pagerep import PageStore

METHODS = ("none", "ret", "adj", "graph", "rel", "ours_ret", "ours_all", "ours_noprov", "ours_tail", "ours", "oracle")
VERIFY_MODE = {"ours": "gap_graph", "ours_noprov": "gap_graph", "ours_tail": "gap_graph",
               "ours_ret": "gap_ret", "ours_all": "gap_all", "rel": "rel_graph"}


def greedy(F, order, b):
    """Budgeted max coverage by marginal gain; ties keep candidate order."""
    covered, R = set(), []
    while len(R) < b:
        best = None
        for u in order:
            if u in R:
                continue
            gain = len(F.get(u, set()) - covered)
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, u)
        if best is None:
            break
        R.append(best[1])
        covered |= F[best[1]]
    return R


def insert(kept, recovered, edges, tail=False):
    """Each restored page sits next to its anchor kept page, in physical page order;
    kept pages keep their reranked order; unanchored pages go last."""
    before, after, loose = {}, {}, []
    for u in recovered:
        a = None if tail else anchor(edges, u, kept)
        if a is None:
            loose.append(u)
        else:
            (before if u < a else after).setdefault(a, []).append(u)
    out = []
    for k in kept:
        out += sorted(before.get(k, [])) + [k] + sorted(after.get(k, []))
    return out + sorted(loose)


def job_key(doc_id, question, cands, kept):
    return hashlib.sha1(json.dumps([doc_id, question, list(cands), list(kept)]).encode()).hexdigest()[:16]


class EvidenceRecovery:
    def __init__(self, cfg, extract_path, cache_dir):
        self.cfg = cfg
        self.store = PageStore(extract_path, cfg.mineru_dir)
        v = cfg.verifier
        self.llm = LocalLLM(v.base_url, v.model, v.concurrency)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------ per-question context
    def context(self, doc, question, cands, kept):
        """cands: 0-based pages in retrieval order; kept: reranked 0-based pages."""
        t0 = time.perf_counter()
        reps, df, n = self.store.get(doc)
        t1 = time.perf_counter()
        dropped = [p for p in cands if p not in kept]
        rank = {p: i for i, p in enumerate(cands)}
        edges = build_graph(cands, reps, question, df, n)
        located = locate(edges, kept, dropped, rank)
        return dict(question=question, cands=list(cands), kept=list(kept), dropped=dropped, rank=rank,
                    reps=reps, edges=edges, located=located, t_doc=t1 - t0, t_graph=time.perf_counter() - t1)

    def pool(self, ctx, mode):
        m = self.cfg.candidate_limit
        if mode in ("gap_graph", "rel_graph"):
            return ctx["located"][:m]
        if mode == "gap_ret":
            return ctx["dropped"][:m]
        return list(ctx["dropped"])

    # ------------------------------------------------------------ verification (cached)
    def _cache(self, mode):
        path = os.path.join(self.cache_dir, f"verify_{mode}.jsonl")
        rows = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if not r.get("error"):
                        rows[r["key"]] = r
        return path, rows

    def verify(self, items, mode):
        """items: {key: ctx}. Returns {key: record} for verification `mode`."""
        path, done = self._cache(mode)
        todo, inputs = [], []
        for key, ctx in items.items():
            if key in done:
                continue
            pool = self.pool(ctx, mode)
            if not pool or not ctx["kept"]:
                done[key] = dict(key=key, mode=mode, skipped=True, pool=pool, raw="", prompt_tokens=0,
                                 completion_tokens=0, latency=0.0, t_graph=ctx["t_graph"], error=None)
                todo.append(done[key])
                continue
            edges = {} if mode == "gap_ret" else ctx["edges"]       # w/o-graph ablation sees no relations
            vin = V.build_input(ctx["question"], ctx["kept"], pool, ctx["reps"], edges,
                                self.cfg.page_chars, self.cfg.total_chars)
            msgs = V.messages("gap" if mode.startswith("gap") else "rel", ctx["question"],
                              vin["kept_text"], vin["cand_text"], vin["relation_text"])
            inputs.append((key, pool, vin, msgs, ctx["t_graph"]))
        if inputs:
            outs = self.llm.chat_many([x[3] for x in inputs], self.cfg.verifier.max_tokens,
                                      schema=V.GAP_SCHEMA if mode.startswith("gap") else V.REL_SCHEMA,
                                      thinking=self.cfg.verifier.thinking)
            for (key, pool, vin, msgs, tg), r in zip(inputs, outs):
                rec = dict(key=key, mode=mode, skipped=False, pool=pool, cand_ids=vin["cand_ids"],
                           kept_ids=vin["kept_ids"], truncated=vin["truncated"], input_chars=len(msgs[1]["content"]),
                           raw=r["text"], prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
                           finish=r["finish_reason"], latency=r["latency"], t_graph=tg, error=r["error"])
                todo.append(rec)
                if not rec["error"]:
                    done[key] = rec
        with open(path, "a") as f:
            for rec in todo:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return done

    # ------------------------------------------------------------ selection
    def select(self, method, b, ctx, ver, gold=()):
        kept, dropped, edges = ctx["kept"], ctx["dropped"], ctx["edges"]
        if method == "none":
            return list(kept), []
        if method == "ret":
            R = dropped[:b]
        elif method == "adj":
            R = [d for d in dropped if any(abs(d - k) == 1 for k in kept)][:b]
        elif method == "graph":
            R = ctx["located"][:b]
        elif method == "oracle":
            R = [d for d in dropped if d in set(gold)][:b]
        elif method == "rel":
            v = ver.get("rel_graph")
            ranked = [] if not v or v.get("skipped") else (
                V.rel_ranking(v["raw"], {k: int(p) for k, p in v["cand_ids"].items()}) or [])
            R = ranked[:b]
        else:
            v = ver.get(VERIFY_MODE[method])
            F = {}
            if v and not v.get("skipped"):
                res = V.gap_support(v["raw"], {k: int(p) for k, p in v["cand_ids"].items()}, ctx["reps"],
                                    check=method != "ours_noprov")
                F = res["F"] if res else {}
            order = self.pool(ctx, VERIFY_MODE[method])
            R = greedy(F, order, b)
        return insert(kept, R, edges, tail=method == "ours_tail"), R
