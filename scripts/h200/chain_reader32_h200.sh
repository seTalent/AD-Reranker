#!/usr/bin/env bash
# Stronger reader of the same family: Qwen2.5-VL-32B on the same page lists (results doc sections 13, 19, 20).
#   step 1: answers. plain prompt (with log-probabilities, for the answer-level fallback) on the main variants;
#           tag / text / tag+text / text_all on rerank / score20@3 / oracle@2. Served data-parallel on 2 GPUs.
#   step 2: Qwen3.5-27B judge (judge_adr.jsonl) and gemma-3-27b-it (judge_gemma.jsonl) on all of them.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
RUNS="mmlb:rp2-top10-mmlb-abl mmlb:rp2-top10-mmlb-adr ldu:rp2-top10-ldu ldu:rp2-top10-ldu-adr"
NAME=qwen25vl32b
MAIN='+variants=[rerank,top10,ret@2,oracle@2,scoreni20@3,score20@3]'
VAR='+variants=[rerank,score20@3,oracle@2]'
MODES="plain tag text tag+text text_all"
FROM=${FROM:-1}
RD=(models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME)
MDIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
VBIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm

free_gpus() {  # <n> <GiB>: the n GPUs with the most free memory, comma-separated, once all have enough
  while true; do
    local sel; sel=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',' \
                     | sort -k2 -n -r | head -$1 | awk -v need=$(( $2 * 1024 )) '$2 >= need {print $1}' | paste -sd,)
    [ $(echo "$sel" | tr ',' '\n' | grep -c .) -eq $1 ] && { echo $sel; return; }
    sleep 60
  done
}
wait_up() { until curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q '"id"'; do
  grep -q -E "Traceback|CUDA out of memory" $2 && { echo "$(date) FAILED $2"; return 1; }; sleep 10; done; echo "$(date) up $1"; }
serve() {  # <role> <port> <n gpus> <GiB each> <util> <kv> [env...]
  local role=$1 port=$2 n=$3 gib=$4 util=$5 kv=$6; shift 6
  curl -s -m 5 http://127.0.0.1:$port/v1/models | grep -q '"id"' && { echo "$(date) reusing server on $port"; return 0; }
  local g; g=$(free_gpus $n $gib); echo "$(date) starting $role on GPU $g"
  local dp=(); [ $n -gt 1 ] && dp=(DP=$n)
  (cd $R/AD-Reranker && env "$@" "${dp[@]}" MODELS_DIR=$MDIR VLLM_BIN=$VBIN MEDIA_ROOT=$R \
     setsid -f bash scripts/serve_vllm.sh $role $g $util $kv > $L/serve_${role}_r32.log 2>&1 < /dev/null)
  wait_up $port $L/serve_${role}_r32.log
}
stop() { pkill -u "$USER" -f "served-model-name $1" && echo "$(date) $1 stopped"; sleep 20; }

if [ $FROM -le 1 ]; then
  serve reader32 8106 2 110 0.85 40G MAXLEN=32768 NO_MM_CACHE=1 || exit 1
  A=("${RD[@]}" models.answer.vllm_base_url=http://127.0.0.1:8106/v1)
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    for mode in $MODES; do
      V=$VAR; [ $mode = plain ] && V=$MAIN
      bash run_rp2_h200.sh answer +reading=$mode "$V" "${A[@]}" > $L/r32_${RUN_NAME}_$mode.log 2>&1 \
        || { echo "$(date) FAILED answer $RUN_NAME $mode"; stop $NAME; exit 1; }
    done
    echo "$(date) $RUN_NAME $NAME answered"
  done
  stop $NAME
  echo "$(date) R32_ANSWERS_DONE"
fi

# the reading-answer Gemma chain stops its server when done; do not share it while that chain is running
for w in ${WAIT_FOR:-}; do until grep -q -E "READING_GEMMA_DONE|FAILED" $w; do sleep 60; done; done
serve judge 8103 1 70 0.5 8G || exit 1
serve judge_gemma 8107 1 70 0.5 10G || exit 1
JQ=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
JG=(judge.base_url=http://127.0.0.1:8107/v1 judge.model=gemma3-27b judge.prompt=adr judge.thinking=false
    judge.max_tokens=64 +judge.out=judge_gemma.jsonl +judge.compare_with=judge_adr.jsonl judge.concurrency=128 +judge.batch=4096)
for run in $RUNS; do
  export CONFIG=${run%%:*} RUN_NAME=${run#*:}
  for mode in $MODES; do
    V='+judge.variants=[rerank,score20@3,oracle@2]'
    [ $mode = plain ] && V='+judge.variants=[rerank,top10,ret@2,oracle@2,scoreni20@3,score20@3]'
    m=${mode/+/-}
    # Qwen first (the Gemma eval compares with its verdicts); the two run on different servers
    bash run_rp2_h200.sh eval "${JQ[@]}" "$V" "${RD[@]}" +reading=$mode +judge.eval_out=eval_ear-${m}_${NAME}_all.json \
      > $L/r32_eval_${RUN_NAME}_$mode.log 2>&1 || echo "$(date) FAILED qwen eval $RUN_NAME $mode"
    bash run_rp2_h200.sh eval "${JG[@]}" "$V" "${RD[@]}" +reading=$mode +judge.eval_out=eval_gemma_ear-${m}_${NAME}_all.json \
      > $L/r32_gemma_${RUN_NAME}_$mode.log 2>&1 || echo "$(date) FAILED gemma eval $RUN_NAME $mode"
  done
  echo "$(date) $RUN_NAME judged"
done
stop qwen35-27b
stop gemma3-27b
echo "$(date) R32_DONE"
