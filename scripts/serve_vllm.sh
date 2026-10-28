#!/usr/bin/env bash
# Local model servers (vLLM, OpenAI-compatible) for backend=vllm, the recovery verifier and the judge.
#   bash scripts/serve_vllm.sh vlm|verifier|judge <gpu> [gpu_memory_utilization] [kv_cache_bytes]
# vlm      : Qwen2.5-VL-7B-Instruct for refiner + answer model (512 visual tokens per page, as all_pixels)
# verifier : Qwen3.5-9B, text-only evidence-recovery verifier
# judge    : Qwen3.5-27B, MMLongBench-Doc-V2 judge prompt
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
# LORA=name=path also serves a LoRA adapter (e.g. the trained AD-Reranker: LORA=adr=ckpt); request it by name
[ -n "${LORA:-}" ] && EXTRA+=(--enable-lora --max-lora-rank 32 --lora-modules "$LORA")
case $ROLE in
  vlm)
    exec "$VLLM" serve "$MODELS/Qwen2.5-VL-7B-Instruct" --served-model-name qwen25vl7b --port 8101 \
      --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 12}' \
      --mm-processor-kwargs '{"min_pixels": 401408, "max_pixels": 401408}' \
      --allowed-local-media-path "$MEDIA_ROOT" --seed 0 "${EXTRA[@]}" ;;
  verifier)
    exec "$VLLM" serve "$MODELS/Qwen3.5-9B" --served-model-name qwen35-9b --port 8102 \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 0, "video": 0}' --reasoning-parser qwen3 \
      --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  verifier_mm)
    exec "$VLLM" serve "$MODELS/Qwen3.5-9B" --served-model-name qwen35-9b-mm --port 8104 \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 16, "video": 0}' --allowed-local-media-path "$MEDIA_ROOT" \
      --reasoning-parser qwen3 --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  judge)
    exec "$VLLM" serve "$MODELS/Qwen3.5-27B" --served-model-name qwen35-27b --port 8103 \
      --dtype bfloat16 --max-model-len 20480 --gpu-memory-utilization "$UTIL" --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 0, "video": 0}' --reasoning-parser qwen3 \
      --gdn-prefill-backend triton --seed 0 "${EXTRA[@]}" ;;
  *) echo "unknown role $ROLE"; exit 1 ;;
esac
