"""Answer-sensitivity probing of pages dropped by the reranker.

For a question with reranked pages S and dropped pages D, the reader answers from S alone and from
S + {u} for every u in D. A page matters for the answer if adding it changes the reader's answer and/or
its confidence in the answer. Probes use a low page resolution and stop after the answer field, so a
probe costs one short prefill-dominated request; requests of one question share the S prefix.

Cache: <result_dir>/recovery_cache/sense_px<pixels>.jsonl, one row per question:
  base: {answer, conf, min_conf, abstain}, probes: {page: {answer, conf, min_conf, abstain, flip}}
"""
import json
import math
import os
import re
import time

from recovery.pipeline import _jsonl, skey

_ANS = re.compile(r'"Answer"\s*:\s*("(?:[^"\\]|\\.)*"|[^,}\n]+)')
_ABST = re.compile(r"not answerable|cannot be (determined|answered)|does not (provide|contain|mention|specify)|"
                   r"no (information|data)|not (provided|mentioned|specified|available)|unable to|insufficient", re.I)


def answer_span(tokens):
    """(answer text, mean logprob, min logprob) of the "Answer" value; falls back to the first tokens."""
    text = "".join(t for t, _ in tokens)
    m = _ANS.search(text)
    if m:
        lo, hi = m.span(1)
        value = m.group(1).strip().strip('"')
    else:
        lo, hi, value = 0, min(len(text), 40), text[:40]
    pos, lps = 0, []
    for t, lp in tokens:
        a, b = pos, pos + len(t)
        if b > lo and a < hi and t.strip(' "'):
            lps.append(lp)
        pos = b
    if not lps:
        lps = [lp for _, lp in tokens[:8]] or [0.0]
    return value, sum(lps) / len(lps), min(lps)


def norm_answer(a):
    a = str(a).lower().strip()
    a = re.sub(r"[\s,]+", " ", a)
    a = re.sub(r"[^\w%. ]", "", a).strip(" .")
    try:
        return f"{float(a.rstrip('%')):g}"
    except ValueError:
        return a


def probe_record(out):
    value, conf, mn = answer_span(out.get("token_logprobs") or [])
    return dict(answer=value, conf=conf, min_conf=mn, abstain=bool(_ABST.search(value)),
                prompt_tokens=out["prompt_tokens"], completion_tokens=out["completion_tokens"],
                latency=out["latency"], error=out["error"])


def sense_stage(cfg, dataset, split=None):
    """Probe every question (or one dev/test split) of the reranked run; resumable."""
    from omegaconf import OmegaConf
    from models.qwen import AnswerModel
    from models.local_llm import prepare_images, to_openai_messages
    from recovery.analysis import doc_split
    rd = dataset.config.result_dir
    px = int(cfg.get("sense_pixels", 401408))
    max_tokens = int(cfg.get("sense_max_tokens", 48))
    path = os.path.join(rd, "recovery_cache", f"sense_px{px}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {r["key"] for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    rer = json.load(open(os.path.join(rd, "rerank.json")))
    if split:
        rer = [s for s in rer if doc_split(s["doc_id"]) == split]
    todo = [s for s in rer if skey(s) not in done]
    print(f"[sense] {len(rer)} questions, {len(done)} cached, {len(todo)} to probe at {px} pixels/page")
    acfg = OmegaConf.merge(cfg.models.answer, {"all_pixels": px})
    reader = AnswerModel(acfg)
    chunk = int(cfg.get("sense_chunk", 32))
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        msgs, owners = [], []
        for s in batch:
            q, _, ims = dataset.load_sample_retrieval_data(s)
            kept = list(s["refiner_ids"])
            dropped = [p for p in range(1, len(s["cands"]) + 1) if p not in kept]
            if not dropped:      # reranker kept everything (fallback): nothing to probe
                continue
            for extra in [None] + dropped:
                ids = kept + ([extra] if extra else [])
                msgs.append(reader.create_image_message(q, "", ids, ims))
                owners.append((s, extra))
        if not msgs:
            continue
        images = prepare_images(msgs)
        t0 = time.time()
        outs = reader.llm.chat_many([to_openai_messages(m, images) for m in msgs], max_tokens,
                                    extra=reader.vllm_extra(), logprobs=1)
        dt = time.time() - t0
        rows = {}
        for (s, extra), o in zip(owners, outs):
            r = rows.setdefault(skey(s), dict(key=skey(s), kept=list(s["refiner_ids"]), probes={}, wall=dt / len(batch)))
            rec = probe_record(o)
            if extra is None:
                r["base"] = rec
            else:
                r["probes"][str(extra)] = rec
        with open(path, "a") as f:
            for r in rows.values():
                if "base" not in r:
                    continue
                b = norm_answer(r["base"]["answer"])
                for p in r["probes"].values():
                    p["flip"] = norm_answer(p["answer"]) != b
                r["error"] = r["base"]["error"] or next((p["error"] for p in r["probes"].values() if p["error"]), None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[sense] {min(i + chunk, len(todo))}/{len(todo)} questions, {len(msgs)} probes in {dt:.1f}s", flush=True)
    return path


def load_sense(rd, px=401408):
    return _jsonl(os.path.join(rd, "recovery_cache", f"sense_px{px}.jsonl"))


def page_scores(row, kind="gain"):
    """Candidate-page scores from one question's probes (candidate position -> score)."""
    b = row["base"]
    out = {}
    for p, r in row["probes"].items():
        gain = r["conf"] - b["conf"]
        if kind == "gain":
            out[int(p)] = gain
        elif kind == "flipgain":
            out[int(p)] = gain if r["flip"] else -10 + gain
        elif kind == "conf":
            out[int(p)] = r["conf"]
        elif kind == "flip":
            out[int(p)] = float(r["flip"])
    return out


def conf_prob(lp):
    return math.exp(lp)
