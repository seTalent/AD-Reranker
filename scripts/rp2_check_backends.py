"""Check that backend=vllm reproduces the original transformers path on N samples.

    python scripts/rp2_check_backends.py --config-name mmlb dataset.top_k=10 +n=48 \
        models.refiner.model_id=... models.answer.model_id=...

Refiner: same messages, compare selected positions. Answer: both backends answer the
HF refiner's selection, compare responses and the extracted "Answer" field.
"""
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_datasets.base_dataset import BaseDataset
import hydra
from omegaconf import OmegaConf


def extracted(resp):
    try:
        return str(json.loads(resp)["Answer"]).strip().lower()
    except Exception:
        return resp.strip().lower()


@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    from models.qwen import AnswerModel, RefinerModel
    dataset = BaseDataset(cfg.dataset)
    samples = dataset.load_data(use_retrieval=True)[: cfg.n]
    qs, ims = [], []
    for s in samples:
        q, _, im = dataset.load_sample_retrieval_data(s)
        qs.append(q)
        ims.append(im)
    out = {}
    for backend in ("hf", "vllm"):
        rc = OmegaConf.merge(cfg.models.refiner, {"backend": backend})
        ac = OmegaConf.merge(cfg.models.answer, {"backend": backend})
        refiner = RefinerModel(rc)
        sel, raw = [], []
        bs = rc.decode_batch_size if backend == "hf" else len(samples)
        for i in range(0, len(samples), bs):
            ids, _, _ = refiner.batch_select(qs[i:i + bs], ims[i:i + bs], [[]] * len(qs[i:i + bs]))
            sel += ids
            raw += refiner.last_outputs
        del refiner
        out[backend] = dict(sel=sel, raw=raw)
        if backend == "hf":
            import torch
            torch.cuda.empty_cache()
        answer = AnswerModel(ac)
        ans = []
        bs = ac.decode_batch_size if backend == "hf" else len(samples)
        for i in range(0, len(samples), bs):
            a, _ = answer.batch_answer(qs[i:i + bs], ims[i:i + bs], out["hf"]["sel"][i:i + bs], [""] * len(qs[i:i + bs]))
            ans += a
        del answer
        out[backend]["ans"] = ans
    n = len(samples)
    same_sel = sum(a == b for a, b in zip(out["hf"]["sel"], out["vllm"]["sel"]))
    same_raw = sum(a == b for a, b in zip(out["hf"]["raw"], out["vllm"]["raw"]))
    same_ans = sum(a == b for a, b in zip(out["hf"]["ans"], out["vllm"]["ans"]))
    same_ext = sum(extracted(a) == extracted(b) for a, b in zip(out["hf"]["ans"], out["vllm"]["ans"]))
    summary = dict(n=n, refiner_same_selection=same_sel, refiner_same_text=same_raw,
                   answer_same_text=same_ans, answer_same_extracted=same_ext)
    print(json.dumps(summary, indent=1))
    os.makedirs(dataset.config.result_dir, exist_ok=True)
    with open(os.path.join(dataset.config.result_dir, "backend_check.json"), "w") as f:
        json.dump(dict(summary=summary, detail=out, questions=qs), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
