"""Candidate-pool page association graph G_C and recovery-candidate location.

Edges carry provenance records (type, z_u, z_v, detail). Relation types by priority:
explicit object reference > cross-page continuation > shared question entity >
shared rare entities > plain adjacency.
"""
from dataclasses import asdict, dataclass
from itertools import combinations

from recovery.pagerep import TERMINAL, content_terms

PRIORITY = {"ref": 4.0, "cont": 3.0, "qent": 2.0, "ent": 1.0, "adj": 0.5}
NAMES = {"ref": "explicit reference", "cont": "cross-page continuation", "qent": "shared question entity",
         "ent": "shared rare entities", "adj": "adjacent pages"}


@dataclass
class Relation:
    kind: str
    u: int
    v: int
    zu: str
    zv: str
    detail: str

    def as_dict(self):
        return asdict(self)


def continuation_cue(a, b):
    """Cue that page b continues page a (b = a + 1)."""
    la, fb = a.last_content(), b.first_content()
    if la is None or fb is None:
        return None
    if la.kind == "table" and fb.kind == "table":
        return "table continues"
    head = " ".join(x.text.lower() for x in b.blocks[:2])
    if "continued" in head or "cont'd" in head:
        return "continued marker"
    if la.kind == "text" and len(la.text) >= 40 and not la.text.rstrip().endswith(TERMINAL):
        return "sentence continues"
    if fb.kind == "text" and fb.text[:1].islower():
        return "lowercase start"
    return None


def block_with(rep, term):
    for b in rep.blocks:
        if term in content_terms(b.text):
            return b.bid
    return rep.blocks[0].bid if rep.blocks else ""


def relations(ru, rv, qterms, df, n_pages):
    out = []
    for a, b in ((ru, rv), (rv, ru)):
        for label, bids in a.mentions.items():
            if label in b.defs:
                out.append(Relation("ref", a.page, b.page, bids[0], b.defs[label], label))
    if abs(ru.page - rv.page) == 1:
        a, b = (ru, rv) if ru.page < rv.page else (rv, ru)
        la, fb = a.last_content(), b.first_content()
        if la is not None and fb is not None:
            cue = continuation_cue(a, b)
            out.append(Relation("cont" if cue else "adj", a.page, b.page, la.bid, fb.bid, cue or "adjacent"))
    shared = set(ru.terms) & set(rv.terms)
    limit_q, limit_e = max(2.0, 0.3 * n_pages), max(2.0, 0.1 * n_pages)
    q = sorted((t for t in shared if t in qterms and df[t] <= limit_q), key=lambda t: (df[t], -len(t), t))
    if q:
        out.append(Relation("qent", ru.page, rv.page, block_with(ru, q[0]), block_with(rv, q[0]), ", ".join(q[:3])))
    e = sorted((t for t in shared if t not in qterms and " " not in t and not t.isdigit()
                and len(t) >= 4 and df[t] <= limit_e), key=lambda t: (df[t], -len(t), t))
    if len(e) >= 3:
        out.append(Relation("ent", ru.page, rv.page, block_with(ru, e[0]), block_with(rv, e[0]), ", ".join(e[:3])))
    return out


def build_graph(pool, reps, question, df, n_pages):
    """Edges among candidate-pool pages: {(u, v): [Relation]} with u < v."""
    qterms = set(content_terms(question))
    edges = {}
    for u, v in combinations(sorted(set(pool)), 2):
        rels = relations(reps[u], reps[v], qterms, df, n_pages)
        if rels:
            edges[(u, v)] = rels
    return edges


def links(edges, u, kept):
    """Relations between page u and kept pages, strongest first."""
    out = []
    for (a, b), rels in edges.items():
        if (a == u and b in kept) or (b == u and a in kept):
            out.extend(rels)
    return sorted(out, key=lambda r: -PRIORITY[r.kind])


def locate(edges, kept, dropped, retrieval_rank):
    """U0: dropped pages directly linked to a kept page, ordered by relation priority."""
    kept, scored = set(kept), {}
    for u in dropped:
        rels = links(edges, u, kept)
        if rels:
            kinds = {r.kind for r in rels}
            scored[u] = (max(PRIORITY[k] for k in kinds), len(kinds))
    return sorted(scored, key=lambda u: (-scored[u][0], -scored[u][1], retrieval_rank[u]))


def anchor(edges, u, kept_order):
    """Kept page with the strongest relation to u (ties: earlier in the reranked list)."""
    pos = {p: i for i, p in enumerate(kept_order)}
    rels = links(edges, u, set(kept_order))
    if not rels:
        return None
    best = max(PRIORITY[r.kind] for r in rels)
    cands = {r.u if r.v == u else r.v for r in rels if PRIORITY[r.kind] == best}
    return min(cands, key=lambda p: pos[p])
