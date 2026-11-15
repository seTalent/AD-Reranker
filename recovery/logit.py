"""Reranker uncertainty without extra sampling: read it from the reranker's own output distribution.

Group sampling (recovery/resample.py) costs about one more rerank of GPU time per question. The signals
here need no sampling:

  slot   P(candidate c is written in an <answer> slot), max over slots, from the top-20 log-probabilities
         of the same greedy decode that produces the selection. No extra request.
  first  FIRST-style (Reddy et al., EMNLP 2024): the distribution of the first identifier when the answer
         is forced to start immediately ("<think>\n</think>\n<answer>["). One decoded token; the prompt is
         the rerank prompt, so the server's prefix cache covers the prefill.
  cf     content-free prior of `first` (CapCal-style calibration of positional bias): the same question
         with blank placeholder pages at a tiny resolution. Calibrated score log p_first(c) - log p_cf(c).

Identifiers are written digit by digit, so "10" starts like "1". Where the first digit is "1", the split
between candidates 1 and 10 uses P("0") at the next position (a second forced token for `first` / `cf`).

Cache recovery_cache/logit.jsonl, one row per question:
  key, k, sel (greedy selection of this decode), slot / first / cf {position: probability},
  prompt_tokens, completion_tokens, latency {greedy, first, cf}, error.
"""
import json
import math
import os
import time

from PIL import Image

from recovery.pipeline import _jsonl, skey

DIGITS = set("0123456789")
FORCED = "<think>\n</think>\n<answer>["


def _p(top, tok):
    """Probability of a digit token in a top-logprob dict (token strings may carry whitespace)."""
    return sum(math.exp(lp) for t, lp in (top or {}).items() if t.strip() == tok)


def split_first_digit(top_here, top_next, k):
    """Probability of each candidate 1..k at an identifier slot. top_next is the distribution after "1"."""
    out = {c: _p(top_here, str(c)) for c in range(1, min(k, 9) + 1)}
    if k >= 10:
        p0 = _p(top_next, "0") if top_next is not None else 0.0
        out[10] = out.get(1, 0.0) * p0
        out[1] = out.get(1, 0.0) * (1 - p0)
    return out


def slot_probs(tokens, tops, k):
    """Max over the identifier slots of the <answer> span of P(candidate)."""
    text, start = "", None
    for i, (tok, _) in enumerate(tokens):
        text += tok
        if start is None and "<answer>" in text:
            start = i + 1
            break
    best = {c: 0.0 for c in range(1, k + 1)}
    if start is None:
        return best
    for i in range(start, len(tokens)):
        tok = tokens[i][0].strip()
        if "</answer>" in tokens[i][0] or tok.startswith("</"):
            break
        prev = tokens[i - 1][0].strip() if i > start else ""
        if len(tok) == 1 and tok in DIGITS and not (prev and prev[-1] in DIGITS):
            nxt = tops[i + 1] if tok == "1" and i + 1 < len(tops) else None
            for c, p in split_first_digit(tops[i], nxt, k).items():
                best[c] = max(best[c], p)
    return best


def _blank(path, size=56):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.new("RGB", (size, size), (255, 255, 255)).save(path)
    return path


def logit_stage(cfg, dataset, split=None):
    from models.local_llm import prepare_images, to_openai_messages
    from models.qwen import RefinerModel, parse_selection
    from recovery.analysis import doc_split
    rd = dataset.config.result_dir
    path = os.path.join(rd, "recovery_cache", cfg.get("logit_file", "logit.jsonl"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {k for k, r in _jsonl(path).items() if not r.get("error")}
    rer = json.load(open(os.path.join(rd, "rerank.json")))
    if split:
        rer = [s for s in rer if doc_split(s["doc_id"]) == split]
    todo, seen = [], set(done)
    for s in rer:
        if skey(s) not in seen:
            seen.add(skey(s))
            todo.append(s)
    print(f"[logit] {len(rer)} questions, {len(done)} cached, {len(todo)} to score -> {path}")
    refiner = RefinerModel(cfg.models.refiner)
    llm, extra = refiner.llm, refiner.vllm_extra()
    forced = dict(extra, continue_final_message=True, add_generation_prompt=False)
    tiny = int(cfg.get("logit_cf_pixels", 56 * 56))
    forced_cf = dict(forced, mm_processor_kwargs={"min_pixels": tiny, "max_pixels": tiny})
    blank = _blank(os.path.join(os.path.dirname(path), "blank.png"))
    chunk = int(cfg.get("logit_chunk", 64))
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        msgs, cf_msgs, ks = [], [], []
        for s in batch:
            q, _, ims = dataset.load_sample_retrieval_data(s)
            msgs.append(refiner.process_message(q, None, None, ims, None))
            cf_msgs.append(refiner.process_message(q, None, None, [blank] * len(ims), None))
            ks.append(len(ims))
        images = prepare_images(msgs)
        reqs = [to_openai_messages(m, images) for m in msgs]
        cf_reqs = [to_openai_messages(m, {blank: os.path.abspath(blank)}) for m in cf_msgs]
        with_prefix = lambda rs, pre: [r + [{"role": "assistant", "content": pre}] for r in rs]
        t0 = time.time()
        greedy = llm.chat_many(reqs, refiner.config.max_new_tokens, extra=extra, logprobs=20)
        t1 = time.time()
        first = llm.chat_many(with_prefix(reqs, FORCED), 1, extra=forced, logprobs=20)
        first1 = llm.chat_many(with_prefix(reqs, FORCED + "1"), 1, extra=forced, logprobs=20)
        t2 = time.time()
        cf = llm.chat_many(with_prefix(cf_reqs, FORCED), 1, extra=forced_cf, logprobs=20)
        cf1 = llm.chat_many(with_prefix(cf_reqs, FORCED + "1"), 1, extra=forced_cf, logprobs=20)
        t3 = time.time()
        with open(path, "a") as f:
            for s, k, g, a, a1, c, c1 in zip(batch, ks, greedy, first, first1, cf, cf1):
                err = g["error"] or a["error"] or a1["error"] or c["error"] or c1["error"]
                row = dict(key=skey(s), k=k, error=err, prompt_tokens=g["prompt_tokens"],
                           completion_tokens=g["completion_tokens"],
                           latency=dict(greedy=g["latency"], first=max(a["latency"], a1["latency"]),
                                        cf=max(c["latency"], c1["latency"])))
                if not err:
                    _, answer = refiner._extract_info(g["text"])
                    pos, fb = parse_selection(answer, k)
                    top1 = lambda o: (o.get("top_logprobs") or [None])[0]
                    row.update(sel=[] if fb else pos,
                               slot=slot_probs(g["token_logprobs"] or [], g["top_logprobs"] or [], k),
                               first=split_first_digit(top1(a), top1(a1), k),
                               cf=split_first_digit(top1(c), top1(c1), k))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[logit] {min(i + chunk, len(todo))}/{len(todo)} greedy {t1 - t0:.1f}s first {t2 - t1:.1f}s "
              f"cf {t3 - t2:.1f}s", flush=True)
    return path


def latency_probe(cfg, dataset):
    """Sequential single-question wall time of the uncertainty signals, one question at a time:
    rerank (greedy), then the FIRST-style token (prefix cached by the rerank request), temperature
    sampling n=6 and permutation sampling (6 greedy orders sent concurrently)."""
    import statistics as st
    from models.local_llm import prepare_images, to_openai_messages
    from models.qwen import RefinerModel
    from recovery.analysis import doc_split
    from recovery.resample import permutations
    rd = dataset.config.result_dir
    n_q = int(cfg.get("probe_n", 50))
    rer = [s for s in json.load(open(os.path.join(rd, "rerank.json"))) if doc_split(s["doc_id"]) == "test"]
    rer = list({skey(s): s for s in rer}.values())[:n_q]
    refiner = RefinerModel(cfg.models.refiner)
    llm, extra = refiner.llm, refiner.vllm_extra()
    forced = dict(extra, continue_final_message=True, add_generation_prompt=False)
    blank = _blank(os.path.join(rd, "recovery_cache", "blank.png"))
    tiny = int(cfg.get("logit_cf_pixels", 56 * 56))
    forced_cf = dict(forced, mm_processor_kwargs={"min_pixels": tiny, "max_pixels": tiny})
    T = {k: [] for k in ("rerank", "first", "cf", "temp_n6", "perm_n6")}
    for s in rer:
        q, _, ims = dataset.load_sample_retrieval_data(s)
        m = refiner.process_message(q, None, None, ims, None)
        req = to_openai_messages(m, prepare_images([m]))
        orders = permutations(skey(s), len(ims), 6)
        perm = [refiner.process_message(q, None, None, [ims[j] for j in o], None) for o in orders]
        perm_reqs = [to_openai_messages(x, prepare_images([x])) for x in perm]
        cfm = refiner.process_message(q, None, None, [blank] * len(ims), None)
        cf_req = to_openai_messages(cfm, {blank: os.path.abspath(blank)})
        t = time.perf_counter(); llm.chat_many([req], refiner.config.max_new_tokens, extra=extra); T["rerank"].append(time.perf_counter() - t)
        t = time.perf_counter()
        llm.chat_many([req + [{"role": "assistant", "content": FORCED}], req + [{"role": "assistant", "content": FORCED + "1"}]],
                      1, extra=forced, logprobs=20)
        T["first"].append(time.perf_counter() - t)
        t = time.perf_counter()
        llm.chat_many([cf_req + [{"role": "assistant", "content": FORCED}], cf_req + [{"role": "assistant", "content": FORCED + "1"}]],
                      1, extra=forced_cf, logprobs=20)
        T["cf"].append(time.perf_counter() - t)
        t = time.perf_counter(); llm.sample_many([req], refiner.config.max_new_tokens, n=6, temperature=0.7, extra=extra); T["temp_n6"].append(time.perf_counter() - t)
        t = time.perf_counter(); llm.sample_many(perm_reqs, refiner.config.max_new_tokens, n=1, temperature=0.0, extra=extra); T["perm_n6"].append(time.perf_counter() - t)
    out = {k: dict(median=st.median(v), mean=st.mean(v)) for k, v in T.items()}
    print("[logit-probe]", json.dumps(out))
    with open(os.path.join(rd, "logit_latency_probe.json"), "w") as f:
        json.dump(dict(n=len(rer), seconds=out), f, indent=1)
    return out
