"""Selection uncertainty of the reranker by group sampling at inference.

AD-Reranker is trained with GRPO, which samples a group of G selections per question and compares them.
Here the same idea is used at test time without training: the zero-shot reranker is sampled n times
(temperature T) on the same candidates, and the fraction of samples that select each candidate is its
selection frequency. Pages the greedy reranker dropped but that some samples keep are recovery candidates.

Cache: <result_dir>/recovery_cache/resample_n<n>_t<T>.jsonl, one row per question:
  selections: [[positions], ...] (1-based candidate positions), freq: {position: fraction}, fallbacks
"""
import json
import os
import time

from recovery.pipeline import skey


def resample_stage(cfg, dataset, split=None):
    from models.local_llm import prepare_images, to_openai_messages
    from models.qwen import RefinerModel, parse_selection
    from recovery.analysis import doc_split
    rd = dataset.config.result_dir
    n = int(cfg.get("resample_n", 6))
    temp = float(cfg.get("resample_t", 0.7))
    path = os.path.join(rd, "recovery_cache", f"resample_n{n}_t{temp:g}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
    rer = json.load(open(os.path.join(rd, "rerank.json")))
    if split:
        rer = [s for s in rer if doc_split(s["doc_id"]) == split]
    todo = [s for s in rer if skey(s) not in done]
    print(f"[resample] {len(rer)} questions, {len(done)} cached, {len(todo)} to sample (n={n}, T={temp})")
    refiner = RefinerModel(cfg.models.refiner)
    chunk = int(cfg.get("resample_chunk", 64))
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        msgs, ims_all = [], []
        for s in batch:
            q, _, ims = dataset.load_sample_retrieval_data(s)
            msgs.append(refiner.process_message(q, None, None, ims, None))
            ims_all.append(ims)
        images = prepare_images(msgs)
        t0 = time.time()
        outs = refiner.llm.sample_many([to_openai_messages(m, images) for m in msgs], refiner.config.max_new_tokens,
                                       n=n, temperature=temp, extra=refiner.vllm_extra())
        dt = time.time() - t0
        with open(path, "a") as f:
            for s, ims, o in zip(batch, ims_all, outs):
                sels, fbs = [], 0
                for text in o["texts"]:
                    _, answer = refiner._extract_info(text)
                    pos, fb = parse_selection(answer, len(ims))
                    fbs += fb
                    sels.append([] if fb else pos)   # a fallback ("select all") carries no selection signal
                k = len(ims)
                freq = {str(p): sum(p in x for x in sels) / max(1, len(sels)) for p in range(1, k + 1)}
                f.write(json.dumps(dict(key=skey(s), selections=sels, freq=freq, fallbacks=fbs,
                                        prompt_tokens=o["prompt_tokens"], completion_tokens=o["completion_tokens"],
                                        latency=o["latency"], error=o["error"]), ensure_ascii=False) + "\n")
        print(f"[resample] {min(i + chunk, len(todo))}/{len(todo)} in {dt:.1f}s", flush=True)
    return path
