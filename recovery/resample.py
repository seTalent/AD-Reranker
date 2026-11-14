"""Selection uncertainty of the reranker by group sampling at inference.

AD-Reranker is trained with GRPO, which samples a group of G selections per question and compares them.
Here the same idea is used at test time without training: the reranker is sampled n times on the same
candidates, and the fraction of samples that select each candidate is its selection frequency. Pages the
greedy reranker dropped but that some samples keep are recovery candidates.

Two ways to draw the group:
  temperature  n samples at temperature T from one prompt (shared prefill)          resample_n<n>_t<T>.jsonl
  permutation  n greedy decodes, each over a random order of the same candidates    resample_perm_n<n>.jsonl
               (positional bias of listwise rerankers; cf. permutation self-consistency, Tang et al. 2024)

Cache: one row per question: selections [[positions], ...] in the ORIGINAL 1-based candidate positions,
freq {position: fraction}, fallbacks.
"""
import json
import os
import random
import time

from recovery.pipeline import _jsonl, skey


def cache_name(n, temp, perm):
    return f"resample_perm_n{n}.jsonl" if perm else f"resample_n{n}_t{temp:g}.jsonl"


def permutations(key, k, n):
    """n orders of k candidates, the first being the identity; deterministic per question."""
    rng = random.Random(key)
    out = [list(range(k))]
    while len(out) < n:
        p = list(range(k))
        rng.shuffle(p)
        out.append(p)
    return out


def resample_stage(cfg, dataset, split=None):
    from models.local_llm import prepare_images, to_openai_messages
    from models.qwen import RefinerModel, parse_selection
    from recovery.analysis import doc_split
    rd = dataset.config.result_dir
    n = int(cfg.get("resample_n", 6))
    temp = float(cfg.get("resample_t", 0.7))
    perm = bool(cfg.get("resample_perm", False))
    path = os.path.join(rd, "recovery_cache", cache_name(n, temp, perm))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set(_jsonl(path))   # rows that ended in a server error are sampled again
    rer = json.load(open(os.path.join(rd, "rerank.json")))
    if split:
        rer = [s for s in rer if doc_split(s["doc_id"]) == split]
    todo, seen = [], set(done)
    for s in rer:
        if skey(s) not in seen:
            seen.add(skey(s))
            todo.append(s)
    print(f"[resample] {len(rer)} questions, {len(done)} cached, {len(todo)} to sample "
          f"(n={n}, {'permutation' if perm else f'T={temp}'}) -> {path}")
    refiner = RefinerModel(cfg.models.refiner)
    chunk = int(cfg.get("resample_chunk", 64))
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        msgs, owners = [], []
        for s in batch:
            q, _, ims = dataset.load_sample_retrieval_data(s)
            orders = permutations(skey(s), len(ims), n) if perm else [list(range(len(ims)))]
            for o in orders:
                msgs.append(refiner.process_message(q, None, None, [ims[j] for j in o], None))
                owners.append((s, len(ims), o))
        images = prepare_images(msgs)
        t0 = time.time()
        reqs = [to_openai_messages(m, images) for m in msgs]
        if perm:   # one greedy decode per order
            outs = refiner.llm.sample_many(reqs, refiner.config.max_new_tokens, n=1, temperature=0.0,
                                           extra=refiner.vllm_extra())
        else:
            outs = refiner.llm.sample_many(reqs, refiner.config.max_new_tokens, n=n, temperature=temp,
                                           extra=refiner.vllm_extra())
        dt = time.time() - t0
        rows = {}
        for (s, k, order), o in zip(owners, outs):
            r = rows.setdefault(skey(s), dict(key=skey(s), selections=[], fallbacks=0, prompt_tokens=0,
                                              completion_tokens=0, latency=0.0, error=None, k=k))
            for text in o["texts"]:
                _, answer = refiner._extract_info(text)
                pos, fb = parse_selection(answer, k)
                r["fallbacks"] += fb
                # a fallback ("select all") carries no selection signal; map shown positions back
                r["selections"].append([] if fb else [order[p - 1] + 1 for p in pos])
            r["prompt_tokens"] += o["prompt_tokens"]
            r["completion_tokens"] += o["completion_tokens"]
            r["latency"] = max(r["latency"], o["latency"])
            r["error"] = r["error"] or o["error"]
        with open(path, "a") as f:
            for r in rows.values():
                k = r.pop("k")
                sels = r["selections"]
                r["freq"] = {str(p): sum(p in x for x in sels) / max(1, len(sels)) for p in range(1, k + 1)}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[resample] {min(i + chunk, len(todo))}/{len(todo)} in {dt:.1f}s", flush=True)
    return path
