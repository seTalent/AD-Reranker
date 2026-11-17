#!/usr/bin/env bash
# Full research-point-2 run on H200 once GPUs are free.
#   VLM_GPU=5 VER_GPU=1 JUDGE_GPU=0 CHECK_GPU=0 bash run_all_h200.sh
# Every stage resumes from its cached outputs, so the script can be rerun after an interruption.
set -euo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
A=$R/AD-Reranker
export MODELS_DIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
export VLLM_BIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm
export MEDIA_ROOT=$R
L=$R/logs
wait_up() {  # port, log
  until curl -s -m 5 "http://127.0.0.1:$1/v1/models" > /dev/null; do
    grep -a -q "Engine core initialization failed" "$2" && { echo "server on $1 failed, see $2"; exit 1; }
    sleep 10
  done
}
stop() { pkill -u "$USER" -f -- "--port $1" || true; sleep 10; }

cd $A
setsid -f bash scripts/serve_vllm.sh vlm "${VLM_GPU:?}" 0.40 > $L/serve_vlm.log 2>&1 < /dev/null
setsid -f bash scripts/serve_vllm.sh verifier "${VER_GPU:?}" 0.35 20G > $L/serve_verifier.log 2>&1 < /dev/null
wait_up 8101 $L/serve_vlm.log; wait_up 8102 $L/serve_verifier.log

cd $R
GPU=${CHECK_GPU:?} N=48 bash run_rp2_h200.sh check models.refiner.decode_batch_size=1 \
  models.answer.decode_batch_size=1 run-name=rp2-backend-check-aligned > $L/adr_backend_check_aligned.log 2>&1
bash run_rp2_h200.sh rerank  > $L/rp2_rerank.log 2>&1
bash run_rp2_h200.sh recover > $L/rp2_recover.log 2>&1
bash run_rp2_h200.sh answer  > $L/rp2_answer.log 2>&1
bash run_rp2_h200.sh probe +probe_n=100 > $L/rp2_probe.log 2>&1
stop 8101; stop 8102

cd $A
setsid -f bash scripts/serve_vllm.sh judge "${JUDGE_GPU:?}" 0.55 16G > $L/serve_judge.log 2>&1 < /dev/null
wait_up 8103 $L/serve_judge.log
cd $R
bash run_rp2_h200.sh eval   > $L/rp2_eval.log 2>&1
bash run_rp2_h200.sh report > $L/rp2_report.md 2>&1
stop 8103
echo ALL_DONE
