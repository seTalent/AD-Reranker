"""Sequential latency probe: one request in flight, per component, on a fixed random subset."""
import json
import os
import random
import time

from recovery import EvidenceRecovery, job_key
from recovery.pipeline import skey


def probe(cfg, dataset):
    from models.qwen import AnswerModel, RefinerModel
    rd = dataset.config.result_dir
    rec = json.load(open(os.path.join(rd, "recover.json")))
    rng = random.Random(0)
    subset = rng.sample(rec, min(cfg.get("probe_n", 100), len(rec)))
    path = os.path.join(rd, "probe.jsonl")
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    refiner, reader = RefinerModel(cfg.models.refiner), AnswerModel(cfg.models.answer)
    with open(path, "a") as f:
        for s in subset:
            if skey(s) in done:
                continue
            q, _, ims = dataset.load_sample_retrieval_data(s)
            r = dict(key=skey(s))
            t0 = time.perf_counter()
            ids, _, _ = refiner.batch_select([q], [ims], [[]])
            r["rerank"] = time.perf_counter() - t0
            cands = s["cands"]
            kept = [cands[i - 1] for i in ids[0]]
            er = EvidenceRecovery(cfg.recovery, dataset.config.extract_path, os.path.join(rd, "probe_cache"))
            ctx = er.context(dataset.EXTRACT_DOCUMENT_ID(s), q, cands, kept)
            r["doc_repr"], r["graph"] = ctx["t_doc"], ctx["t_graph"]
            for mode in cfg.get("probe_modes", ["gap_graph"]):
                k = job_key(s["doc_id"], q, cands, kept) + f"-probe{time.time_ns()}"
                t0 = time.perf_counter()
                er.verify({k: ctx}, mode)
                r[f"verify:{mode}"] = time.perf_counter() - t0
            for name in cfg.get("probe_variants", ["rerank", f"ours@{cfg.recovery.budget}", "top10"]):
                t0 = time.perf_counter()
                reader.batch_answer([q], [ims], [s["variants"][name]["ids"]], [""])
                r[f"read:{name}"] = time.perf_counter() - t0
            f.write(json.dumps(r) + "\n")
            f.flush()
