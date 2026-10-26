"""Structured page representation T(p) used by evidence recovery.

Blocks (heading / text / table / caption) with ids, object labels (Figure 3, Table 2,
Note 7, ...) defined or mentioned on the page, and content terms. The source is the cached
MinerU markdown of the page when available, otherwise the PyMuPDF text written by
scripts/extract.py. Page indices are 0-based, as in my_datasets.BaseDataset.
"""
import html
import os
import re
from collections import Counter
from dataclasses import dataclass, field

LABEL_KINDS = r"figure|fig\.|table|exhibit|chart|appendix|note|schedule|item|section"
LABEL_NUM = r"([0-9]+[a-z]?(?:[.\-][0-9]+)*|[ivx]+)"
CAPTION_RE = re.compile(rf"^\W{{0,3}}({LABEL_KINDS})\s*{LABEL_NUM}\b", re.I)
MENTION_RE = re.compile(rf"\b({LABEL_KINDS})\s*{LABEL_NUM}\b", re.I)
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*[a-z0-9]|[a-z0-9]", re.I)
TERMINAL = tuple('.!?:;"”)]%')
STOP = set("""a an the and or of to in on for by with from at as is are was were be been being this that these those it its
which who whom what when where why how than then there their they them we our you your he she his her not no yes do does did
can could should would may might will shall must about into over under between among per also such more most less least other
any all each both either neither one two three four five six seven eight nine ten first second third report document page pages
according shown show shows following based total number many much what's how many list provide answer question""".split())


@dataclass
class Block:
    bid: str
    kind: str
    text: str


@dataclass
class PageRep:
    page: int
    source: str
    blocks: list = field(default_factory=list)
    defs: dict = field(default_factory=dict)        # label -> defining block id (caption / heading)
    mentions: dict = field(default_factory=dict)    # label -> [block ids] referring to it
    terms: Counter = field(default_factory=Counter)

    @property
    def chars(self):
        return sum(len(b.text) for b in self.blocks)

    def block(self, bid):
        return next((b for b in self.blocks if b.bid == bid), None)

    def first_content(self):
        return next((b for b in self.blocks if b.kind != "heading"), None)

    def last_content(self):
        return next((b for b in reversed(self.blocks) if b.kind != "heading"), None)


def norm_label(kind, num):
    kind = kind.lower().rstrip(".")
    return f"{'figure' if kind == 'fig' else kind} {num.lower()}"


def table_to_text(chunk):
    s = re.sub(r"</t[dh]>", " | ", chunk, flags=re.I)
    s = re.sub(r"</tr>", "\n", s, flags=re.I)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    rows = [re.sub(r"[ \t]+", " ", r).strip(" |") for r in s.split("\n")]
    return "\n".join(r for r in rows if r)


def split_markdown(md):
    blocks = []
    for chunk in re.split(r"\n\s*\n", md):
        chunk = chunk.strip()
        if not chunk or re.fullmatch(r"!\[[^\]]*\]\([^)]*\)", chunk):
            continue
        if "<table" in chunk.lower():
            blocks.append(("table", table_to_text(chunk)))
        elif chunk.startswith("#"):
            blocks.append(("heading", chunk.lstrip("#").strip()))
        else:
            chunk = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", chunk).strip()
            if chunk:
                blocks.append(("caption" if CAPTION_RE.match(chunk) else "text", chunk))
    return blocks


def split_plain(txt):
    blocks = []
    for chunk in re.split(r"\n\s*\n", txt):
        chunk = re.sub(r"\s*\n\s*", " ", chunk).strip()
        if chunk:
            blocks.append(("caption" if CAPTION_RE.match(chunk) else "text", chunk))
    return blocks


def content_terms(text):
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    toks = [t for t in toks if t not in STOP and (len(t) >= 3 or t.isdigit())]
    terms = Counter(toks)
    terms.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    return terms


def build(page, kinds_texts, source):
    rep = PageRep(page=page, source=source)
    for i, (kind, text) in enumerate(kinds_texts, 1):
        b = Block(f"b{i}", kind, text)
        rep.blocks.append(b)
        m = CAPTION_RE.match(text)
        if m and kind in ("caption", "heading"):
            rep.defs.setdefault(norm_label(m.group(1), m.group(2)), b.bid)
        for mm in MENTION_RE.finditer(text):
            if m and mm.start() == m.start() and kind in ("caption", "heading"):
                continue
            rep.mentions.setdefault(norm_label(mm.group(1), mm.group(2)), []).append(b.bid)
        rep.terms.update(content_terms(text))
    return rep


class PageStore:
    """Page representations and per-document term frequencies, cached per document."""

    def __init__(self, extract_path, mineru_dir=None):
        self.extract_path = extract_path
        self.mineru_dir = mineru_dir
        self._docs = {}

    def n_pages(self, doc):
        n = 0
        while os.path.exists(os.path.join(self.extract_path, f"{doc}_{n}.png")):
            n += 1
        return n

    def load_page(self, doc, page):
        if self.mineru_dir:
            p = os.path.join(self.mineru_dir, doc, f"MinerU_Page{page + 1}", f"{page + 1}.md")
            if os.path.exists(p):
                with open(p, errors="replace") as f:
                    blocks = split_markdown(f.read())
                if sum(len(t) for _, t in blocks) >= 20:
                    return build(page, blocks, "mineru")
        t = os.path.join(self.extract_path, f"{doc}_{page}.txt")
        txt = open(t, errors="replace").read() if os.path.exists(t) else ""
        return build(page, split_plain(txt), "pymupdf")

    def get(self, doc):
        """(reps, document frequency of terms, number of pages)."""
        if doc not in self._docs:
            n = self.n_pages(doc)
            reps = [self.load_page(doc, i) for i in range(n)]
            df = Counter()
            for r in reps:
                df.update(set(r.terms))
            self._docs[doc] = (reps, df, n)
        return self._docs[doc]


def render(rep, pid, char_budget, keep=()):
    """Page text with block ids '[pid.bk]'. Over budget, keep relation-evidence /
    question-term blocks, then headings and captions, preserving page order."""
    lines = [(b, f"[{pid}.{b.bid}] ({b.kind}) {b.text}") for b in rep.blocks]
    if sum(len(s) for _, s in lines) <= char_budget:
        return "\n".join(s for _, s in lines), False
    keep = set(keep)
    order = sorted(range(len(lines)), key=lambda i: (
        0 if lines[i][0].bid in keep else 1 if lines[i][0].kind in ("heading", "caption") else 2, i))
    chosen, used = {}, 0
    for i in order:
        s = lines[i][1]
        if used + len(s) > char_budget:
            s = s[: max(0, char_budget - used)]
            if len(s) < 80:
                continue
            s += " ..."
        chosen[i] = s
        used += len(s)
        if used >= char_budget:
            break
    return "\n".join(chosen[i] for i in sorted(chosen)) + "\n[... page truncated ...]", True
