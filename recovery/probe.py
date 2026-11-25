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


def probe_v2(cfg, dataset):
    """Wall time per component on a fixed random test subset (results doc section 23).

    sequential (`probe_n`, default 100): one question at a time, each component alone:
      rerank (greedy, the reranker of this run), group sampling (n = 6, T = 0.7, for the +f methods),
      page representation + relation graph, multimodal / text-only gap-conditioned verification
      (all dropped pages of the question sent together, as in the pipeline), and reading the page set of
      every method in `probe_variants`.
    batch (`probe_batch_n`, default 300, other questions than the sequential ones): the same components at the
      clients' usual concurrency (32 requests in flight) -> seconds per question at throughput.
    Verification uses fresh cache keys, so nothing comes from the recovery cache. `probe_parts` limits the
    components (e.g. [read] for a second reader); `probe_tag` names the output files."""
    import statistics as st
    import subprocess
    from models.qwen import AnswerModel, RefinerModel
    from recovery.analysis import doc_split
    rd = dataset.config.result_dir
    parts = list(cfg.get("probe_parts", ["rerank", "sample", "context", "verify", "read"]))
    variants = list(cfg.get("probe_variants", ["rerank", "ret@2", "scoreni20@3", "score20@3", "top10"]))
    modes = list(cfg.get("probe_modes", ["score_mm", "score_noimg"]))
    tag = cfg.get("probe_tag", cfg.models.answer.get("vllm_model", "qwen25vl7b"))
    rec = [s for s in json.load(open(os.path.join(rd, "recover.json"))) if doc_split(s["doc_id"]) == "test"]
    rec = list({skey(s): s for s in rec}.values())
    rng = random.Random(0)
    n_seq, n_bat = int(cfg.get("probe_n", 100)), int(cfg.get("probe_batch_n", 300))
    drawn = rng.sample(rec, min(n_seq + n_bat, len(rec)))
    seq, pool = drawn[:n_seq], drawn[n_seq:]   # disjoint: the batch run cannot reuse the sequential run's prefixes
    refiner = RefinerModel(cfg.models.refiner) if {"rerank", "sample"} & set(parts) else None
    reader = AnswerModel(cfg.models.answer) if "read" in parts else None
    er = EvidenceRecovery(cfg.recovery, dataset.config.extract_path, os.path.join(rd, "probe_cache"))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()

    def load(s):
        q, _, ims = dataset.load_sample_retrieval_data(s)
        cands = s["cands"]
        kept = [cands[i - 1] for i in s["refiner_ids"]]
        return q, ims, cands, kept

    def fresh(s, ctx):
        return job_key(s["doc_id"], ctx["question"], ctx["cands"], ctx["kept"]) + f"-p{time.time_ns()}"

    # ---------------------------------------------------------------- sequential
    path = os.path.join(rd, f"probe_v2_{tag}.jsonl")
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    with open(path, "a") as f:
        for s in seq:
            if skey(s) in done:
                continue
            q, ims, cands, kept = load(s)
            r = dict(key=skey(s), n_dropped=len(cands) - len(kept))
            if "rerank" in parts:
                t0 = time.perf_counter()
                refiner.batch_select([q], [ims], [[]])
                r["rerank"] = time.perf_counter() - t0
            if "sample" in parts:
                from models.local_llm import prepare_images, to_openai_messages
                m = refiner.process_message(q, None, None, ims, None)
                t0 = time.perf_counter()
                refiner.llm.sample_many([to_openai_messages(m, prepare_images([m]))], refiner.config.max_new_tokens,
                                        n=6, temperature=0.7, extra=refiner.vllm_extra())
                r["sample_n6"] = time.perf_counter() - t0
            ctx = er.context(dataset.EXTRACT_DOCUMENT_ID(s), q, cands, kept)
            if "context" in parts:
                r["context"] = ctx["t_doc"] + ctx["t_graph"]
            if "verify" in parts:
                for mode in modes:
                    t0 = time.perf_counter()
                    out = er.verify({fresh(s, ctx): ctx}, mode)
                    r[f"verify:{mode}"] = time.perf_counter() - t0
                    rr = list(out.values())[-1] if out else {}
                    r[f"verify_tokens:{mode}"] = rr.get("prompt_tokens", 0)
            if "read" in parts:
                for name in variants:
                    t0 = time.perf_counter()
                    reader.batch_answer([q], [ims], [s["variants"][name]["ids"]], [""])
                    r[f"read:{name}"] = time.perf_counter() - t0
            f.write(json.dumps(r) + "\n")
            f.flush()
    rows = [json.loads(l) for l in open(path)]
    seq_out = {k: dict(median=st.median(v), mean=st.mean(v), n=len(v))
               for k in sorted({k for r in rows for k in r if k not in ("key", "n_dropped")})
               for v in [[r[k] for r in rows if k in r]]}

    # ---------------------------------------------------------------- batch (throughput)
    bat = {}
    if cfg.get("probe_batch", True):
        loaded = [load(s) for s in pool]
        qs, imss = [x[0] for x in loaded], [x[1] for x in loaded]
        if "rerank" in parts:
            t0 = time.perf_counter()
            refiner.batch_select(qs, imss, [[] for _ in qs])
            bat["rerank"] = (time.perf_counter() - t0) / len(qs)
        ctxs = [er.context(dataset.EXTRACT_DOCUMENT_ID(s), x[0], x[2], x[3]) for s, x in zip(pool, loaded)]
        if "verify" in parts:
            for mode in modes:
                items = {fresh(s, c): c for s, c in zip(pool, ctxs)}
                t0 = time.perf_counter()
                er.verify(items, mode)
                bat[f"verify:{mode}"] = (time.perf_counter() - t0) / len(qs)
        if "read" in parts:
            for name in variants:
                t0 = time.perf_counter()
                reader.batch_answer(qs, imss, [s["variants"][name]["ids"] for s in pool], ["" for _ in qs])
                bat[f"read:{name}"] = (time.perf_counter() - t0) / len(qs)
    out = dict(tag=tag, n_seq=len(rows), n_batch=len(pool), parts=parts, sequential_seconds=seq_out,
               batch_seconds_per_question=bat, gpu_at_start=gpu)
    print("[probe_v2]", json.dumps(out, indent=1))
    with open(os.path.join(rd, f"probe_v2_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
