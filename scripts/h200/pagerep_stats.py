"""Coverage statistics of the page representations T(p) over all documents."""
import statistics
import time
from collections import Counter

from erc.data import load
from erc.pagerep import DocCache

t0 = time.time()
samples = load(require_retrieval=False)
docs = {s.stem: s.n_pages for s in samples}
dc, src, chars, defs, ments, kinds = DocCache(), Counter(), [], 0, 0, Counter()
for stem, n in docs.items():
    reps, _ = dc.get(stem, n)
    for r in reps:
        src[r.source] += 1
        chars.append(r.chars)
        defs += len(r.defs)
        ments += sum(len(v) for v in r.mentions.values())
        kinds.update(b.kind for b in r.blocks)
print(f"documents {len(docs)}, pages {sum(docs.values())}, built in {time.time() - t0:.1f}s")
print("source", dict(src))
print(f"chars/page mean {statistics.mean(chars):.0f} median {statistics.median(chars):.0f} "
      f"empty {sum(c == 0 for c in chars)}")
print(f"object definitions {defs}, mentions {ments}, blocks {dict(kinds)}")
