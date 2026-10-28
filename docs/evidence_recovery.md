# Post-rerank evidence recovery (research point 2)

> **Final method and results**: gap-conditioned page verification (`scoreNN@b`, multimodal
> Qwen3.5-9B served with `scripts/serve_vllm.sh verifier_mm`), threshold 0.20, budget 3, reader at
> 1,280 visual tokens/page (`models.answer.all_pixels=1003520`). On the MMLongBench test split it
> raises accuracy from 30.0 to 32.5 and gold-page recall from 61.5 to 75.7 over the reranked input.
> See [evidence_recovery_results.md](evidence_recovery_results.md). The listwise verifier described
> below (`ours`) was the first version and is kept as a baseline.
>
> ```bash
> python scripts/rp2.py $COMMON +stage=recover "recovery.plan=[none,score20@3]"
> python scripts/rp2.py $COMMON +stage=answer models.answer.all_pixels=1003520
> python scripts/rp2.py $COMMON +stage=eval models.answer.all_pixels=1003520 judge.prompt=adr judge.thinking=false
> ```

A listwise MLLM reranker filters the top-k retrieved pages before answering. It also drops
pages that the answer needs. This component runs **after** the reranker, never re-ranks
or re-retrieves, and restores a few of the dropped pages:

1. **Candidate-pool association graph.** Over the k candidates, pages are linked by explicit
   object references (a page mentions *Table 4* and another page carries its caption),
   cross-page continuation (a sentence or table running over the page break), shared
   question entities and shared rare entities. Dropped pages linked to a kept page are the
   recovery candidates U (at most m = 4), ordered by relation strength.
2. **One information-gap verification.** A text-only verifier (Qwen3.5-9B, served locally)
   reads the question, the kept pages and the candidates (structured OCR with block ids) and
   returns the requirements of the question, whether the kept pages cover each one
   (covered / missing / unknown), and for missing ones the candidate block and a verbatim
   quote that fills it. A quote must be found on the cited page (provenance check V(z)).
3. **Budgeted recovery.** At most b pages are restored by greedy maximum coverage of the
   missing requirements and inserted next to their related kept page. Kept pages keep their
   reranked order. No missing requirement or no verified support means nothing is added.

Code: `recovery/` (`pagerep.py` page representation, `graph.py`, `verifier.py`,
`__init__.py` selection and baselines, `pipeline.py` experiment stages, `analysis.py`).

## Pipeline used for the thesis experiments

| Stage | Setting |
|---|---|
| Data | MMLongBench (134 PDFs); questions/gold from MMLongBench-Doc-V2 `samples.json` (1,071 questions, same schema as V1, corrected annotations) |
| Extraction / retrieval | `scripts/extract.py`, `scripts/retrieve.py` (ColPali), **top-10** candidates |
| Reranker | `RefinerModel`, Qwen2.5-VL-7B-Instruct **zero-shot** (`use_grpo_model: False`), refiner prompt with k = 10 |
| Recovery | `recovery/`, b = 2, m = 4, verifier Qwen3.5-9B (no thinking, JSON schema) |
| Answer | `AnswerModel`, Qwen2.5-VL-7B-Instruct, answer prompt, 512 visual tokens per page |
| Judge | MMLongBench-Doc-V2 official judge prompt/schema/metrics with a local Qwen3.5-27B |

All models run on local GPUs (vLLM); no external API is called.

## Running

```bash
# model servers (each on a free GPU)
MODELS_DIR=/path/to/models VLLM_BIN=/path/to/vllm MEDIA_ROOT=$PWD bash scripts/serve_vllm.sh vlm 0 0.40
MODELS_DIR=... VLLM_BIN=... bash scripts/serve_vllm.sh verifier 1 0.35 20G
MODELS_DIR=... VLLM_BIN=... bash scripts/serve_vllm.sh judge 2 0.55 16G

COMMON="--config-name mmlb run-name=rp2-top10 dataset.top_k=10 \
  models.refiner.backend=vllm models.answer.backend=vllm \
  models.refiner.model_id=/path/Qwen2.5-VL-7B-Instruct models.answer.model_id=/path/Qwen2.5-VL-7B-Instruct \
  recovery.mineru_dir=/path/to/mineru_cache"
python scripts/rp2.py $COMMON +stage=rerank
python scripts/rp2.py $COMMON +stage=recover
python scripts/rp2.py $COMMON +stage=answer
python scripts/rp2.py $COMMON +stage=eval
python scripts/rp2.py $COMMON +stage=report
```

The end-to-end entry also supports the component:
`python scripts/predict.py --config-name mmlb dataset.top_k=10 recovery.enable=true recovery.method=ours`.

## Variants (all computed from the same reranking run)

| Name | Meaning |
|---|---|
| `top4`, `top10` | no reranker, first 4 / 10 retrieved pages |
| `rerank` | reranked pages only (baseline) |
| `ret@b` | + top-b dropped pages by retrieval rank |
| `adj@b` | + dropped pages adjacent to a kept page |
| `graph@b` | + strongest graph-linked pages, no verification (w/o gap detection) |
| `rel@b` | + graph candidates judged "useful" (w/o requirement modelling) |
| `ours_ret@b` | gap verification over the top-m dropped pages by retrieval, no relations (w/o graph) |
| `ours_all@b` | gap verification over all dropped pages (w/o candidate narrowing) |
| `ours_noprov@b` | method without the provenance check |
| `ours_tail@b` | method, restored pages appended at the end |
| `ours@b` | the method |
| `oracle@b` | + dropped gold evidence pages (upper bound, analysis only) |

## Changes to the existing pipeline

- `RefinerModel.batch_select` no longer overwrites the selection with `[1, 2, 3, 4]`
  (a leftover debug line made the reranker a no-op) and accepts positions 1..k instead of
  1..4; empty selections still fall back to all candidates.
- The refiner prompt uses the actual candidate count (`{k}`); for k = 4 the text is unchanged.
- `RefinerModel` pads on the left for batched generation, like `AnswerModel`, and uses the
  configured `max_new_tokens`.
- `MyModel.predict_dataset` keeps sample indices explicitly (resumed runs no longer write to
  shifted samples or drop a final partial batch) and stores the refiner output and full response.
- Optional `backend: vllm` for refiner/answer sends the same messages to a local server
  (greedy decoding with repetition penalty 1.05 as the model's generation config). Images are
  first passed through the same `qwen_vl_utils.process_vision_info` call as the transformers
  path and cached as lossless PNGs (`models/local_llm.py::hf_preprocessed`). Sending the raw
  page instead lets the server resize once rather than twice, which changed the visual-token
  grid of landscape pages (38x50 vs 38x52) and made 16 of 48 reranker selections differ from the
  transformers path; with the shared preprocessing the processor inputs agree to 1e-5 on average.
- `answer.eval_base_url` makes the judge endpoint configurable (local judge instead of the API).

With top-10 candidates and the transformers backend, use `decode_batch_size=1` (or install
flash-attn): without flash-attn the Qwen2.5-VL vision tower builds one attention mask over all
images in the batch, and 8 x 10 pages needs a single 43 GB allocation.
