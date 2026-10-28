"""Information-gap verification: one text-only request per question that returns
requirement coverage and "missing requirement -> candidate page -> verbatim quote"
support, followed by a program-side provenance check V(z)."""
import json
import re
from difflib import SequenceMatcher

from recovery.graph import NAMES, links
from recovery.pagerep import content_terms, render

SYSTEM = ("You audit the evidence given to a document question-answering model. "
          "You only use the page text provided and you copy quotes verbatim.")

GAP_TASK = """Task.
1. Write the information requirements needed to answer the question: atomic facts, values or relations, each with its conditions (entity, year/period, unit, scope). Use at most 6 requirements; ids h1, h2, ...
2. Using ONLY the kept pages, set each requirement's status:
   - "covered": a kept page states it explicitly under the same conditions (give the kept block id in kept_evidence);
   - "missing": the kept pages contain related content but this item is absent, or they contain nothing about it;
   - "unknown": the kept text cannot settle it (e.g. it depends on a chart, image or table that the text does not capture).
   Treat equivalent wording as covered. If the question asks about something the document may not contain, do not invent requirements to be filled.
3. For each MISSING requirement only, list candidate pages that explicitly provide it under the same conditions. Give the block id and a short quote (5-30 words) copied verbatim from that block. Do not cite a candidate that is only on the same topic, only shares an entity, repeats covered information, or conflicts with the kept pages. A candidate may support several requirements; a requirement may have no support.
Return JSON only."""

GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "need": {"type": "string"}, "conditions": {"type": "string"},
            "status": {"type": "string", "enum": ["covered", "missing", "unknown"]},
            "kept_evidence": {"type": "string"}},
            "required": ["id", "need", "conditions", "status", "kept_evidence"]}},
        "support": {"type": "array", "items": {"type": "object", "properties": {
            "requirement": {"type": "string"}, "candidate": {"type": "string"},
            "block": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["requirement", "candidate", "block", "quote"]}},
    },
    "required": ["requirements", "support"],
}

REL_TASK = """Task. Decide which candidate pages are useful for answering the question. List the useful candidates from most to least useful (at most 3); give an empty list if none is useful.
Return JSON only."""

REL_SCHEMA = {
    "type": "object",
    "properties": {"useful": {"type": "array", "items": {"type": "object", "properties": {
        "candidate": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["candidate", "reason"]}}},
    "required": ["useful"],
}


MM_SYSTEM = ("You audit the evidence given to a document question-answering model. You see each page as an "
             "image together with its extracted text. Text quotes must be copied verbatim from the extracted text.")

MM_TASK = """Task.
1. Write the information requirements needed to answer the question: atomic facts, values or relations, each with its conditions (entity, year/period, unit, scope). Use at most 5 requirements; ids h1, h2, ...
2. Using ONLY the kept pages (images and text), set each requirement's status: "covered" (a kept page shows it under the same conditions), "missing" (no kept page shows it), or "unknown" (the kept pages are unreadable for it). Treat equivalent wording as covered. If the question asks about something the document may not contain, do not invent requirements.
3. For each MISSING requirement only, list the candidate pages that show it. Use source "text" with the block id and a short verbatim quote (5-25 words) from that block's extracted text, or source "visual" when the information is only in a chart, figure, photo or table image of the page; then put a short description of what the image shows in quote and leave block empty. Do not cite a candidate that is only on the same topic, repeats covered information, or conflicts with the kept pages.
Return JSON only."""

MM_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "need": {"type": "string"},
            "status": {"type": "string", "enum": ["covered", "missing", "unknown"]}},
            "required": ["id", "need", "status"]}},
        "support": {"type": "array", "items": {"type": "object", "properties": {
            "requirement": {"type": "string"}, "candidate": {"type": "string"},
            "source": {"type": "string", "enum": ["text", "visual"]},
            "block": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["requirement", "candidate", "source", "block", "quote"]}},
    },
    "required": ["requirements", "support"],
}


def messages_mm(question, vin, image_of):
    """Multimodal verifier request: every kept / candidate page as image + its block-id text."""
    content = [{"type": "text", "text": f"Question: {question}\n\nKEPT PAGES (already given to the answering model):"}]
    for group, ids, texts in (("kept", vin["kept_ids"], vin["kept_parts"]), ("cand", vin["cand_ids"], vin["cand_parts"])):
        if group == "cand":
            content.append({"type": "text", "text": "\nCANDIDATE PAGES (removed by the reranker):"})
        for (pid, p), txt in zip(ids.items(), texts):
            content.append({"type": "text", "text": f'<page id="{pid}" physical_page="{p + 1}">'})
            content.append({"type": "image_url", "image_url": {"url": image_of(p)}})
            content.append({"type": "text", "text": f"Extracted text:\n{txt}\n</page>"})
    content.append({"type": "text", "text": f"\nRELATIONS between candidate and kept pages:\n"
                                            f"{vin['relation_text'] or '(none)'}\n\n{MM_TASK}"})
    return [{"role": "system", "content": MM_SYSTEM}, {"role": "user", "content": content}]


SCORE_SYSTEM = ("You check whether a document page adds information that is needed to answer a question and "
                "is not already provided by the pages selected so far.")
SCORE_ASK = ("Does the CANDIDATE page contain information that is needed to answer the question and is missing "
             "from the SELECTED pages? Answer with one word: Yes or No.")


REL_SYSTEM = "You check whether a document page contains information needed to answer a question."
REL_ASK = ("Does the CANDIDATE page contain information that is needed to answer the question? "
           "Answer with one word: Yes or No.")


def messages_score(question, kept, cand, texts, relations, image_of, gap=True, images=True):
    """One gap-conditioned check of a dropped page. The question and the selected pages form a
    prefix shared by all candidates of the question (prefix-cached by the server).
    Ablations: gap=False drops the selected pages (plain relevance); images=False sends text only;
    relations=None omits the relation hints."""
    def img(p):
        return [{"type": "image_url", "image_url": {"url": image_of(p)}}] if images else []

    if gap:
        content = [{"type": "text", "text": f"Question: {question}\n\nSELECTED PAGES:"}]
        for i, p in enumerate(kept, 1):
            content += ([{"type": "text", "text": f'<page id="S{i}" physical_page="{p + 1}">'}] + img(p)
                        + [{"type": "text", "text": f"Extracted text:\n{texts[p]}\n</page>"}])
    else:
        content = [{"type": "text", "text": f"Question: {question}\n"}]
    rel = ""
    if gap and relations is not None:
        rel = "Relations of the candidate to selected pages:\n" + ("\n".join(relations) or "(none)") + "\n\n"
    content += ([{"type": "text", "text": f'\nCANDIDATE PAGE:\n<page id="C" physical_page="{cand + 1}">'}] + img(cand)
                + [{"type": "text", "text": f"Extracted text:\n{texts[cand]}\n</page>\n{rel}{SCORE_ASK if gap else REL_ASK}"}])
    return [{"role": "system", "content": SCORE_SYSTEM if gap else REL_SYSTEM}, {"role": "user", "content": content}]


def yes_probability(first_token_logprobs):
    """P(Yes) normalised over the Yes/No answer tokens; None if neither appears in the top tokens."""
    import math
    if not first_token_logprobs:
        return None
    yes = [v for t, v in first_token_logprobs.items() if t.strip().lower() == "yes"]
    no = [v for t, v in first_token_logprobs.items() if t.strip().lower() == "no"]
    if not yes and not no:
        return None
    py = sum(math.exp(v) for v in yes)
    pn = sum(math.exp(v) for v in no)
    return py / (py + pn)


def messages(mode, question, kept_text, cand_text, relation_text):
    task = GAP_TASK if mode == "gap" else REL_TASK
    user = (f"Question: {question}\n\n"
            f"KEPT PAGES (already given to the answering model):\n{kept_text}\n\n"
            f"CANDIDATE PAGES (removed by the reranker):\n{cand_text}\n\n"
            f"RELATIONS between candidate and kept pages:\n{relation_text or '(none)'}\n\n{task}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _relation_lines(cand_ids, kept_ids, edges):
    kept_of = {p: k for k, p in kept_ids.items()}
    lines = []
    for cid, u in cand_ids.items():
        for r in links(edges, u, set(kept_of)):
            other = r.v if r.u == u else r.u
            zc, zk = (r.zu, r.zv) if r.u == u else (r.zv, r.zu)
            lines.append(f"- {cid} <-> {kept_of[other]}: {NAMES[r.kind]} ({r.detail}); "
                         f"{kept_of[other]}.{zk} ~ {cid}.{zc}")
    return "\n".join(lines)


def _evidence_blocks(edges, page):
    out = set()
    for rels in edges.values():
        for r in rels:
            if r.u == page:
                out.add(r.zu)
            if r.v == page:
                out.add(r.zv)
    return out


def build_input(question, kept, cands, reps, edges, page_chars=6000, total_chars=60000):
    """Render kept pages S (reranked order) and candidates U with block ids.
    Kept pages are given budget priority; edges={} hides all relation information."""
    qterms = set(content_terms(question))
    qblocks = lambda r: {b.bid for b in r.blocks if qterms & set(content_terms(b.text))}
    kept_ids = {f"S{i}": p for i, p in enumerate(kept, 1)}
    cand_ids = {f"C{i}": p for i, p in enumerate(cands, 1)}
    kept_budget = min(page_chars, (total_chars * 0.6) / max(1, len(kept)))
    parts, used, truncated = [], 0, []
    for pid, p in kept_ids.items():
        txt, cut = render(reps[p], pid, int(kept_budget), _evidence_blocks(edges, p) | qblocks(reps[p]))
        parts.append(f'<page id="{pid}" physical_page="{p + 1}">\n{txt}\n</page>')
        used += len(txt)
        truncated += [pid] if cut else []
    cand_budget = min(page_chars, max(1500, (total_chars - used) / max(1, len(cands))))
    cparts = []
    for pid, p in cand_ids.items():
        txt, cut = render(reps[p], pid, int(cand_budget), _evidence_blocks(edges, p) | qblocks(reps[p]))
        cparts.append(f'<page id="{pid}" physical_page="{p + 1}">\n{txt}\n</page>')
        truncated += [pid] if cut else []
    return dict(kept_ids=kept_ids, cand_ids=cand_ids, kept_text="\n\n".join(parts),
                cand_text="\n\n".join(cparts), relation_text=_relation_lines(cand_ids, kept_ids, edges),
                truncated=truncated, kept_parts=[_strip_page(x) for x in parts],
                cand_parts=[_strip_page(x) for x in cparts])


def _strip_page(rendered):
    """Text of a rendered <page> element without its tags (used by the multimodal request)."""
    return "\n".join(rendered.split("\n")[1:-1])


def _norm(s):
    return re.sub(r"[^a-z0-9%.$]+", " ", s.lower()).strip()


def quote_found(quote, text):
    q, t = _norm(quote), _norm(text)
    if len(q) < 4 or not t:
        return None
    if q in t:
        return "exact"
    m = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return "fuzzy" if m.size >= 0.8 * len(q) else None


def check_provenance(rep, block_id, quote):
    """V(z): the quote must occur on the cited candidate page (cited block first)."""
    blk = rep.block(block_id) if block_id else None
    if blk is not None:
        how = quote_found(quote, blk.text)
        if how:
            return True, how
    for b in rep.blocks:
        how = quote_found(quote, b.text)
        if how:
            return True, f"{how}-other-block"
    return False, "not-found"


_CID = re.compile(r"\b(C\d+)\b")
_BID = re.compile(r"\b(?:C\d+\.)?(b\d+)\b")


def parse_json(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                return None
    return None


def gap_support(raw, cand_ids, reps, check=True):
    """F(u): missing requirements each candidate page is verified to fill."""
    data = parse_json(raw)
    if not isinstance(data, dict):
        return None
    reqs = [r for r in data.get("requirements", []) if isinstance(r, dict)]
    missing = {str(r.get("id")) for r in reqs if r.get("status") == "missing"}
    F, audit = {}, []
    for s in data.get("support", []):
        if not isinstance(s, dict):
            continue
        cm = _CID.search(str(s.get("candidate", ""))) or _CID.search(str(s.get("block", "")))
        req = str(s.get("requirement"))
        if not cm or cm.group(1) not in cand_ids:
            audit.append(dict(s, ok=False, how="unknown-candidate"))
            continue
        u = cand_ids[cm.group(1)]
        if req not in missing:
            audit.append(dict(s, ok=False, how="not-missing"))
            continue
        bm = _BID.search(str(s.get("block", "")))
        if not check:
            ok, how = True, "unchecked"
        elif s.get("source") == "visual":    # cannot be quoted; the page must at least carry a figure/table
            ok = reps[u].has_visual
            how = "visual" if ok else "visual-no-figure"
        else:
            ok, how = check_provenance(reps[u], bm.group(1) if bm else None, str(s.get("quote", "")))
        audit.append(dict(s, ok=ok, how=how))
        if ok:
            F.setdefault(u, set()).add(req)
    return dict(F=F, missing=sorted(missing), status={str(r.get("id")): r.get("status") for r in reqs},
                audit=audit, n_req=len(reqs))


def rel_ranking(raw, cand_ids):
    data = parse_json(raw)
    if not isinstance(data, dict):
        return None
    ranked = []
    for x in data.get("useful", []):
        m = _CID.search(str(x.get("candidate", ""))) if isinstance(x, dict) else None
        if m and m.group(1) in cand_ids and cand_ids[m.group(1)] not in ranked:
            ranked.append(cand_ids[m.group(1)])
    return ranked
