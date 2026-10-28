"""Stage-wise runner for the evidence-recovery experiments.

rerank  -> results/<dataset>/<run-name>/rerank.json     refiner selections on the top-k candidates
recover -> .../recover.json                              reader page lists for every variant in recovery.plan
answer  -> .../answers.jsonl                             one answer per distinct (question, page list)
eval    -> .../judge.jsonl, eval.json                    local judge verdicts and metrics

Every stage reuses earlier outputs, so ablations do not rerun the reranker, and a variant
that leaves the page list unchanged reuses the same answer.
"""
import hashlib
import json
import os
import time
from collections import defaultdict

from tqdm import tqdm


def skey(sample):
    return f"{sample['doc_id']}||{sample['question']}"


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _jsonl(path):
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if not r.get("error"):
                        rows[r["key"]] = r
    return rows


def chunk_size(model_cfg):
    return model_cfg.decode_batch_size if model_cfg.get("backend", "hf") == "hf" else model_cfg.get("vllm_chunk", 256)


# ---------------------------------------------------------------------------- rerank
def rerank_stage(cfg, dataset):
    from models.qwen import RefinerModel
    os.makedirs(dataset.config.result_dir, exist_ok=True)
    path = os.path.join(dataset.config.result_dir, "rerank.json")
    samples = dataset.load_data(use_retrieval=True)
    if cfg.get("limit"):
        samples = samples[: cfg.limit]
    done = {skey(s): s for s in json.load(open(path))} if os.path.exists(path) else {}
    todo = [s for s in samples if skey(s) not in done]
    print(f"[rerank] {len(samples)} samples, {len(done)} cached, {len(todo)} to run")
    if todo:
        refiner = RefinerModel(cfg.models.refiner)
        n = chunk_size(cfg.models.refiner)
        for i in tqdm(range(0, len(todo), n)):
            batch = todo[i:i + n]
            qs, ims = [], []
            for s in batch:
                q, _, im = dataset.load_sample_retrieval_data(s)
                qs.append(q)
                ims.append(im)
            t0 = time.time()
            ids, _, _ = refiner.batch_select(qs, ims, [[] for _ in batch])
            dt = (time.time() - t0) / len(batch)
            for s, sel, raw, fb in zip(batch, ids, refiner.last_outputs, refiner.last_fallbacks):
                s["cands"] = list(s[dataset.config.r_image_key][: dataset.config.top_k])
                s.update(refiner_ids=sel, refiner_output=raw, refiner_fallback=fb, refiner_sec=dt)
                done[skey(s)] = s
            _dump(path, [done[skey(s)] for s in samples if skey(s) in done])
    return [done[skey(s)] for s in samples]


# ---------------------------------------------------------------------------- recover
def parse_plan(entry):
    name, _, b = entry.partition("@")
    return name, int(b or 0)


def recover_stage(cfg, dataset):
    from recovery import EvidenceRecovery, job_key, mode_of
    from recovery.graph import links
    rer = json.load(open(os.path.join(dataset.config.result_dir, "rerank.json")))
    rec = EvidenceRecovery(cfg.recovery, dataset.config.extract_path,
                           os.path.join(dataset.config.result_dir, "recovery_cache"))
    plan = [parse_plan(p) for p in cfg.recovery.plan]
    ctxs, keys = {}, []
    for s in rer:
        cands = s["cands"]
        kept = [cands[i - 1] for i in s["refiner_ids"]]
        k = job_key(s["doc_id"], s["question"], cands, kept)
        ctxs[k] = rec.context(dataset.EXTRACT_DOCUMENT_ID(s), s[dataset.config.question_key], cands, kept)
        keys.append(k)
    modes = sorted({mode_of(m) for m, _ in plan if mode_of(m)})
    ver = {}
    for mode in modes:
        t0 = time.time()
        ver[mode] = rec.verify(ctxs, mode)
        print(f"[recover] verification {mode}: {time.time() - t0:.0f}s")
    out = []
    for s, k in zip(rer, keys):
        ctx, cands = ctxs[k], s["cands"]
        gold = [p - 1 for p in eval(s["evidence_pages"])]
        variants = {"top4": dict(ids=list(range(1, min(4, len(cands)) + 1)), added=[]),
                    "top10": dict(ids=list(range(1, len(cands) + 1)), added=[])}
        for m, b in plan:
            pages, added = rec.select(m, b, ctx, {mode: ver[mode].get(k) for mode in modes}, gold)
            variants["rerank" if m == "none" else f"{m}@{b}"] = dict(
                ids=[cands.index(p) + 1 for p in pages], added=[cands.index(p) + 1 for p in added])
        strongest = {}
        for d in ctx["dropped"]:
            rels = links(ctx["edges"], d, set(ctx["kept"]))
            strongest[cands.index(d) + 1] = rels[0].kind if rels else "none"
        out.append(dict({x: s[x] for x in s if x not in ("refiner_output",)}, job_key=k, variants=variants,
                        located=[cands.index(p) + 1 for p in ctx["located"]], dropped_relation=strongest,
                        t_graph=ctx["t_graph"]))
    _dump(os.path.join(dataset.config.result_dir, "recover.json"), out)
    print(f"[recover] {len(out)} samples x {len(out[0]['variants'])} variants")
    return out


# ---------------------------------------------------------------------------- answer
def answer_key(sample, ids):
    return f"{skey(sample)}||{','.join(map(str, ids))}"


def answers_file(cfg):
    """Answers depend on the reader's page resolution; the AD-Reranker default keeps the original name."""
    px = cfg.models.answer.all_pixels
    return "answers.jsonl" if px == 401408 else f"answers_px{px}.jsonl"


def answer_stage(cfg, dataset):
    from models.qwen import AnswerModel
    rec = json.load(open(os.path.join(dataset.config.result_dir, "recover.json")))
    path = os.path.join(dataset.config.result_dir, answers_file(cfg))
    done = _jsonl(path)
    wanted = cfg.get("variants")
    jobs = {}
    for s in rec:
        for name, v in s["variants"].items():
            if wanted and name not in wanted:
                continue
            k = answer_key(s, v["ids"])
            if k not in done:
                jobs.setdefault(k, (s, v["ids"]))
    jobs = sorted(jobs.items(), key=lambda kv: (len(kv[1][1]), kv[0]))
    print(f"[answer] {len(done)} cached, {len(jobs)} to run")
    if not jobs:
        return
    model = AnswerModel(cfg.models.answer)
    n = chunk_size(cfg.models.answer)
    with open(path, "a") as f:
        for i in tqdm(range(0, len(jobs), n)):
            batch = jobs[i:i + n]
            qs, ims, ids = [], [], []
            for _, (s, sel) in batch:
                q, _, im = dataset.load_sample_retrieval_data(s)
                qs.append(q)
                ims.append(im)
                ids.append(sel)
            outs, _ = model.batch_answer(qs, ims, ids, [""] * len(batch))
            usage = getattr(model, "last_usage", [(None, None)] * len(batch))
            for (k, (s, sel)), o, u in zip(batch, outs, usage):
                f.write(json.dumps(dict(key=k, ids=sel, response=o, prompt_tokens=u[0],
                                        completion_tokens=u[1]), ensure_ascii=False) + "\n")
            f.flush()


# ---------------------------------------------------------------------------- eval
def extracted_answer(response):
    """The "Answer" field that MyModel.predict_dataset stores and eval_dataset judges."""
    try:
        return json.loads(response)["Answer"]
    except Exception:
        return response


def judge_key(sample, response):
    return f"{skey(sample)}||{hashlib.sha1(response.encode()).hexdigest()[:16]}"


def eval_stage(cfg, dataset):
    from models.local_llm import LocalLLM
    from recovery import v2_eval
    from recovery.verifier import parse_json
    rd = dataset.config.result_dir
    rec = json.load(open(os.path.join(rd, "recover.json")))
    answers = _jsonl(os.path.join(rd, answers_file(cfg)))
    jc = cfg.judge
    path = os.path.join(rd, jc.get("out", "judge.jsonl"))   # one file per judge setting
    done = _jsonl(path)
    todo = {}
    for s in rec:
        for v in s["variants"].values():
            a = answers.get(answer_key(s, v["ids"]))
            if a is None:
                continue
            k = judge_key(s, a["response"])
            if k not in done:
                todo.setdefault(k, (s, a["response"]))
    if jc.get("subset"):    # fixed random subset of all responses, e.g. to calibrate a cheaper judge setting
        import random
        every = sorted({judge_key(s, answers[answer_key(s, v["ids"])]["response"]) for s in rec
                        for v in s["variants"].values() if answer_key(s, v["ids"]) in answers})
        chosen = set(random.Random(0).sample(every, min(jc.subset, len(every))))
        todo = {k: x for k, x in todo.items() if k in chosen}
    print(f"[eval] {len(done)} cached verdicts, {len(todo)} to judge with {jc.model} "
          f"({jc.prompt} prompt, thinking={jc.thinking}) -> {os.path.basename(path)}")
    if todo:
        llm = LocalLLM(jc.base_url, jc.model, jc.concurrency)
        items = list(todo.items())
        step = jc.get("batch", 1024)    # verdicts are saved after each batch
        for i in tqdm(range(0, len(items), step)):
            batch = items[i:i + step]
            msgs = [v2_eval.judge_messages(s, resp) if jc.prompt == "v2" else
                    [{"role": "user", "content": cfg.models.answer.eval_prompt.format(   # as MyModel.eval_dataset
                        question=s["question"], answer=extracted_answer(resp), gt=s["answer"])}] for _, (s, resp) in batch]
            schema = v2_eval.SCHEMA if jc.prompt == "v2" else None
            outs = llm.chat_many(msgs, jc.max_tokens, schema=schema, thinking=jc.thinking)
            retry = [j for j, o in enumerate(outs) if not v2_eval.verdict(parse_json(o["text"]), jc.prompt)]
            if retry:   # unparseable after thinking (e.g. truncated): judge again without thinking
                again = llm.chat_many([msgs[j] for j in retry], 512, schema=schema, thinking=False)
                for j, o in zip(retry, again):
                    outs[j] = dict(o, retried=True)
            with open(path, "a") as f:
                for (k, (s, resp)), o in zip(batch, outs):
                    v = v2_eval.verdict(parse_json(o["text"]), jc.prompt)
                    f.write(json.dumps(dict(key=k, verdict=v, raw=o["text"], retried=o.get("retried", False),
                                            prompt_tokens=o["prompt_tokens"], completion_tokens=o["completion_tokens"],
                                            error=o["error"] or (None if v else "unparseable verdict")),
                                       ensure_ascii=False) + "\n")
                    if v:
                        done[k] = dict(key=k, verdict=v)
    if jc.get("subset"):
        return None
    from recovery.analysis import analyze
    res = analyze(rec, answers, _jsonl(path), rd, cfg)
    ref = jc.get("compare_with")     # agreement with another judge setting on the shared responses
    if ref and os.path.exists(os.path.join(rd, ref)):
        a, b = _jsonl(path), _jsonl(os.path.join(rd, ref))
        common = [k for k in a if k in b]
        res["judge_agreement"] = dict(
            reference=ref, n=len(common),
            equivalent=sum(a[k]["verdict"]["equivalent"] == b[k]["verdict"]["equivalent"] for k in common) / max(1, len(common)),
            abstained=sum(a[k]["verdict"]["abstained"] == b[k]["verdict"]["abstained"] for k in common) / max(1, len(common)),
            positive_rate=[sum(a[k]["verdict"]["equivalent"] for k in common) / max(1, len(common)),
                           sum(b[k]["verdict"]["equivalent"] for k in common) / max(1, len(common))])
    _dump(os.path.join(rd, jc.get("eval_out", "eval.json")), res)
    return res
