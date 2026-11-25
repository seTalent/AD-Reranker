#!/usr/bin/env bash
# Local model servers (vLLM, OpenAI-compatible) for backend=vllm, the recovery verifier and the judge.
#   bash scripts/serve_vllm.sh vlm|verifier|verifier_mm|verifier_mm_lr|verifier_mm27|verifier_internvl|verifier_gemma|judge|judge_gemma|reader3|reader32|mmr5 <gpu> [gpu_memory_utilization] [kv_cache_bytes]
# vlm      : Qwen2.5-VL-7B-Instruct for refiner + answer model (512 visual tokens per page, as all_pixels)
# verifier : Qwen3.5-9B, text-only evidence-recovery verifier
# judge    : Qwen3.5-27B, MMLongBench-Doc-V2 judge prompt
# judge_gemma: gemma-3-27b-it, cross-family check of the judge
# SNAME / PORT rename the verifier_mm / verifier_mm27 servers (a second copy next to a running chain).
# MAXLEN=n overrides the context length of the vlm / reader3 / reader32 readers (e.g. 32768 when page text is added).
# On shared GPUs pass kv_cache_bytes (e.g. 20G): vLLM's profiler otherwise attributes other
# users' concurrent allocations to this process.
set -euo pipefail
ROLE=$1; GPU=$2; UTIL=${3:-0.40}; KV=${4:-}
MODELS=${MODELS_DIR:?set MODELS_DIR to the directory holding the model checkpoints}
VLLM=${VLLM_BIN:-vllm}
MEDIA_ROOT=${MEDIA_ROOT:-$(pwd)}           # page images must live under this path (file:// URLs)
export CUDA_VISIBLE_DEVICES=$GPU VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1
export PATH=$(dirname "$(command -v "$VLLM")"):$PATH
EXTRA=()
[ -n "$KV" ] && EXTRA=(--kv-cache-memory-bytes "$KV")
# DP=n with <gpu> = "a,b,...": one endpoint, n data-parallel replicas (e.g. the judge on two GPUs)
[ -n "${DP:-}" ] && EXTRA+=(--data-parallel-size "$DP")
# EAGER=1 skips torch.compile / CUDA graphs (a second Qwen3.5 instance on one GPU can fail to compile)
[ -n "${EAGER:-}" ] && EXTRA+=(--enforce-eager)
# LORA=name=path also serves a LoRA adapter (e.g. the trained AD-Reranker: LORA=adr=ckpt); request it by name
[ -n "${LORA:-}" ] && EXTRA+=(--enable-lora --max-lora-rank 32 --lora-modules "$LORA")
# NO_MM_CACHE=1 disables the multimodal preprocessing cache (vLLM 0.19 can lose a cached item after an aborted
# client and then fail or hang requests with "Expected a cached item for mm_hash")
[ -n "${NO_MM_CACHE:-}" ] && EXTRA+=(--mm-processor-cache-gb 0)
# NO_PREFIX=1 disables automatic prefix caching (latency probes: every request pays its full prefill)
[ -n "${NO_PREFIX:-}" ] && EXTRA+=(--no-enable-prefix-caching)
case $ROLE in
  vlm)
    exec "$VLLM" serve "$MODELS/Qwen2.5-VL-7B-Instruct" --served-model-name qwen25vl7b --port 8101 \
      --dtype bfloat16 --max-model-len ${MAXLEN:-16384} --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 12}' \
      --mm-processor-kwargs '{"min_pixels": 401408, "max_pixels": 401408}' \
      --allowed-local-media-path "$MEDIA_ROOT" --seed 0 "${EXTRA[@]}" ;;
  verifier)
    exec "$VLLM" serve "$MODELS/Qwen3.5-9B" --served-model-name qwen35-9b --port 8102 \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 0, "video": 0}' --reasoning-parser qwen3 \
      --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  verifier_mm)
    exec "$VLLM" serve "$MODELS/Qwen3.5-9B" --served-model-name ${SNAME:-qwen35-9b-mm} --port ${PORT:-8104} \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 16, "video": 0}' --allowed-local-media-path "$MEDIA_ROOT" \
      --reasoning-parser qwen3 --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  judge)
    exec "$VLLM" serve "$MODELS/Qwen3.5-27B" --served-model-name qwen35-27b --port 8103 \
      --dtype bfloat16 --max-model-len 20480 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 0, "video": 0}' --reasoning-parser qwen3 \
      --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  reader3)   # second reader family (models.answer.vllm_base_url=http://127.0.0.1:8105/v1 models.answer.vllm_model=qwen3vl8b)
    exec "$VLLM" serve "$MODELS/Qwen3-VL-8B-Instruct" --served-model-name qwen3vl8b --port 8105 \
      --dtype bfloat16 --max-model-len ${MAXLEN:-20480} --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 12, "video": 0}' \
      --allowed-local-media-path "$MEDIA_ROOT" --seed 0 "${EXTRA[@]}" ;;
  reader32)  # larger reader of the same family (port 8106, models.answer.vllm_model=qwen25vl32b)
    exec "$VLLM" serve "$MODELS/Qwen2.5-VL-32B-Instruct" --served-model-name qwen25vl32b --port 8106 \
      --dtype bfloat16 --max-model-len ${MAXLEN:-20480} --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 12, "video": 0}' \
      --allowed-local-media-path "$MEDIA_ROOT" --seed 0 "${EXTRA[@]}" ;;
  verifier_mm_lr)   # the default verifier on its own port, for the 256-tokens-per-page run (pixels set per request)
    exec "$VLLM" serve "$MODELS/Qwen3.5-9B" --served-model-name qwen35-9b-mm-lr --port 8111 \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 16, "video": 0}' --allowed-local-media-path "$MEDIA_ROOT" \
      --reasoning-parser qwen3 --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  verifier_mm27)   # larger verifier of the same family (recovery.verifier_mm.base_url=http://127.0.0.1:8110/v1)
    exec "$VLLM" serve "$MODELS/Qwen3.5-27B" --served-model-name ${SNAME:-qwen35-27b-mm} --port ${PORT:-8110} \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 16, "video": 0}' --allowed-local-media-path "$MEDIA_ROOT" \
      --reasoning-parser qwen3 --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  verifier_internvl)   # other-family verifier, 256 tokens per page (one 448 x 448 tile)
    exec "$VLLM" serve "$MODELS/InternVL3_5-8B-HF" --served-model-name internvl35-8b --port 8108 \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 16, "video": 0}' --allowed-local-media-path "$MEDIA_ROOT" \
      --seed 0 "${EXTRA[@]}" ;;
  verifier_gemma)   # other-family verifier, 256 tokens per page (896 x 896)
    exec "$VLLM" serve "$MODELS/gemma-3-12b-it" --served-model-name gemma3-12b --port 8109 \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 16}' --allowed-local-media-path "$MEDIA_ROOT" \
      --seed 0 "${EXTRA[@]}" ;;
  mmr5)   # MM-R5 reasoning reranker (Qwen2.5-VL-7B), recovery/mmr5.py; pixels set per request
    exec "$VLLM" serve "$MODELS/MM-R5" --served-model-name mmr5 --port 8112 \
      --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 12}' --allowed-local-media-path "$MEDIA_ROOT" --seed 0 "${EXTRA[@]}" ;;
  judge_gemma)   # cross-family judge, text only (judge.base_url=http://127.0.0.1:8107/v1 judge.model=gemma3-27b)
    exec "$VLLM" serve "$MODELS/gemma-3-27b-it" --served-model-name gemma3-27b --port 8107 \
      --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization "$UTIL" --max-num-seqs 128 \
      --limit-mm-per-prompt '{"image": 0}' --seed 0 "${EXTRA[@]}" ;;
  *) echo "unknown role $ROLE"; exit 1 ;;
esac
