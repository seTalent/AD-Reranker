"""Pilot go/no-go: how well do answer-sensitivity signals separate gold pages among dropped pages?"""
import ast, json, statistics as st, sys
sys.path.insert(0, ".")
from recovery.pipeline import skey
from recovery.analysis import doc_split
from recovery.sensitivity import load_sense

rd = "results/MMLongBench/rp2-top10-mmlb-abl"
rec = {skey(s): s for s in json.load(open(rd + "/recover.json"))}
sense = load_sense(rd)
ver = {}
for l in open(rd + "/recovery_cache/verify_score_mm.jsonl"):
    r = json.loads(l)
    if not r.get("error"):
        ver[r["key"]] = r


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / (len(pos) * len(neg))


def ranks(d):
    order = sorted(d, key=lambda k: d[k])
    return {k: (i + 1) / len(order) for i, k in enumerate(order)}


rows = []    # (label, features)
qrows = []   # per question: (wf, base_conf, base_abstain)
for k, z in sense.items():
    s = rec.get(k)
    if s is None or "base" not in z:
        continue
    C = s["cands"]
    G = {p - 1 for p in ast.literal_eval(s["evidence_pages"])}
    kept = {C[i - 1] for i in s["refiner_ids"]}
    v = ver.get(s["job_key"], {})
    vs = {int(p): x for p, x in v.get("scores", {}).items()}
    b = z["base"]
    feats = {}
    for pos, pr in z["probes"].items():
        page = C[int(pos) - 1]
        gain = pr["conf"] - b["conf"]
        feats[page] = dict(flip=float(pr["flip"]), gain=gain, mingain=pr["min_conf"] - b["min_conf"],
                           flipgain=(gain if pr["flip"] else -10 + gain), conf=pr["conf"],
                           verifier=vs.get(page, 0.0), retrieval=-C.index(page))
    if feats:
        # within-question rank fusion of answer gain and verifier
        rg = ranks({p: f["gain"] for p, f in feats.items()})
        rv = ranks({p: f["verifier"] for p, f in feats.items()})
        for p, f in feats.items():
            f["fuse_gain_verifier"] = rg[p] + rv[p]
            f["fuse_flipgain_verifier"] = (1.0 if f["flip"] else 0.0) + rg[p] + rv[p]
            rows.append((page_label := int(p in G), f))
    qrows.append((bool(G & (set(C) - kept)), b["conf"], b["abstain"]))

print("questions probed", len(qrows), "dropped pages", len(rows), "gold among them", sum(l for l, _ in rows))
for name in ["retrieval", "verifier", "flip", "gain", "mingain", "flipgain", "conf", "fuse_gain_verifier",
             "fuse_flipgain_verifier"]:
    pos = [f[name] for l, f in rows if l]
    neg = [f[name] for l, f in rows if not l]
    print("  AUC %-24s %.3f" % (name, auc(pos, neg)))
wf = [c for w, c, _ in qrows if w]
nwf = [c for w, c, _ in qrows if not w]
print("gate: AUC(-base_conf predicts wrongful filter) %.3f  (mean conf wf %.3f vs others %.3f)" % (
    auc([-c for c in wf], [-c for c in nwf]), st.mean(wf), st.mean(nwf)))
tok = [sum(p["prompt_tokens"] for p in z["probes"].values()) + z["base"]["prompt_tokens"] for z in sense.values() if "base" in z]
print("probe prompt tokens/question mean %.0f median %.0f" % (st.mean(tok), st.median(tok)))
