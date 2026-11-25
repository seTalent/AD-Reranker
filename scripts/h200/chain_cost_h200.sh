#!/usr/bin/env bash
# Efficiency and cost (results doc section 23).
#   step 1: cheaper verifier budgets (score20mK@3 / scoreni20mK@3: only the K best-retrieved dropped pages verified;
#           added to recover.json from the cached verification with recovery.extend=true) answered by the 7B and 8B readers
#   step 2: Qwen3.5-27B judge on those answers
#   step 3: latency probe (+stage=probe_v2): sequential per-component wall time on 100 test questions and seconds per
#           question at throughput on 300, both datasets x both rerankers, readers 7B / 8B / 32B. Prefix caching and the
#           multimodal preprocessing cache are off, so every request pays its full cost.
# One GPU at a time; servers are stopped after each step.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
RUNS="mmlb:rp2-top10-mmlb-abl mmlb:rp2-top10-mmlb-adr ldu:rp2-top10-ldu ldu:rp2-top10-ldu-adr"
BUDGET='[score20m1@3,score20m2@3,score20m3@3,score20m4@3,score20m6@3,scoreni20m2@3,scoreni20m4@3]'
JV='+judge.variants=[rerank,scoreni20@3,score20@3,score20m1@3,score20m2@3,score20m3@3,score20m4@3,score20m6@3,scoreni20m2@3,scoreni20m4@3]'
FROM=${FROM:-1}
MDIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
VBIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm

pick_gpu() {  # <GiB needed>
  while true; do
    read -r idx free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',' | sort -k2 -n -r | head -1)
    [ "$free" -ge $(( $1 * 1024 )) ] && { echo $idx; return; }
    sleep 60
  done
}
wait_up() { until curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q '"id"'; do
  grep -q -E "Traceback|CUDA out of memory" $2 && { echo "$(date) FAILED $2"; return 1; }; sleep 10; done; echo "$(date) up $1"; }
serve_on() {  # <gpu> <role> <port> <util> <kv> [env...]
  local g=$1 role=$2 port=$3 util=$4 kv=$5; shift 5
  curl -s -m 5 http://127.0.0.1:$port/v1/models | grep -q '"id"' && { echo "$(date) port $port already up"; return 1; }
  echo "$(date) starting $role on GPU $g"
  (cd $R/AD-Reranker && env "$@" MODELS_DIR=$MDIR VLLM_BIN=$VBIN MEDIA_ROOT=$R \
     setsid -f bash scripts/serve_vllm.sh $role $g $util $kv > $L/serve_${role}_cost.log 2>&1 < /dev/null)
  wait_up $port $L/serve_${role}_cost.log
}
stop() { for n in "$@"; do pkill -u "$USER" -f "served-model-name $n" && echo "$(date) $n stopped"; done; sleep 20; }

if [ $FROM -le 1 ]; then
  g=$(pick_gpu 80)
  serve_on $g vlm 8101 0.2 12G NO_MM_CACHE=1 || exit 1
  serve_on $g reader3 8105 0.2 12G NO_MM_CACHE=1 || { stop qwen25vl7b; exit 1; }
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    for rd in qwen25vl7b:8101 qwen3vl8b:8105; do
      bash run_rp2_h200.sh answer "+variants=$BUDGET" models.answer.all_pixels=1003520 models.answer.vllm_model=${rd%%:*} \
        models.answer.vllm_base_url=http://127.0.0.1:${rd#*:}/v1 > $L/cost_answer_${RUN_NAME}_${rd%%:*}.log 2>&1 \
        || { echo "$(date) FAILED answer $RUN_NAME ${rd%%:*}"; stop qwen25vl7b qwen3vl8b; exit 1; }
    done
    echo "$(date) $RUN_NAME budget answered"
  done
  stop qwen25vl7b qwen3vl8b
fi
if [ $FROM -le 2 ]; then
  g=$(pick_gpu 70)
  serve_on $g judge 8103 0.5 8G || exit 1
  J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048 "$JV")
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    for NAME in qwen25vl7b qwen3vl8b; do
      bash run_rp2_h200.sh eval "${J[@]}" models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME \
        +judge.eval_out=eval_budget_${NAME}_all.json > $L/cost_eval_${RUN_NAME}_$NAME.log 2>&1 || echo "$(date) FAILED eval $RUN_NAME $NAME"
    done
    echo "$(date) $RUN_NAME budget judged"
  done
  stop qwen35-27b
fi
if [ $FROM -le 3 ]; then
  P=(NO_PREFIX=1 NO_MM_CACHE=1)
  g=$(pick_gpu 110)
  serve_on $g vlm 8101 0.15 8G "${P[@]}" LORA=adr=$R/AD-Reranker/ckpt || exit 1
  serve_on $g verifier_mm 8104 0.2 8G "${P[@]}" || { stop qwen25vl7b; exit 1; }
  serve_on $g verifier 8102 0.2 8G "${P[@]}" || { stop qwen25vl7b qwen35-9b-mm; exit 1; }
  serve_on $g reader3 8105 0.2 8G "${P[@]}" || { stop qwen25vl7b qwen35-9b-mm qwen35-9b; exit 1; }
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    RF=(); [[ $RUN_NAME == *adr ]] && RF=(models.refiner.vllm_model=adr)
    bash run_rp2_h200.sh probe_v2 "${RF[@]}" models.answer.all_pixels=1003520 models.answer.vllm_model=qwen25vl7b \
      +probe_tag=qwen25vl7b > $L/probe_${RUN_NAME}_qwen25vl7b.log 2>&1 || echo "$(date) FAILED probe $RUN_NAME 7b"
    bash run_rp2_h200.sh probe_v2 models.answer.all_pixels=1003520 models.answer.vllm_model=qwen3vl8b \
      models.answer.vllm_base_url=http://127.0.0.1:8105/v1 '+probe_parts=[read]' +probe_tag=qwen3vl8b \
      > $L/probe_${RUN_NAME}_qwen3vl8b.log 2>&1 || echo "$(date) FAILED probe $RUN_NAME 8b"
    echo "$(date) $RUN_NAME probed (7B, 8B)"
  done
  stop qwen25vl7b qwen35-9b-mm qwen35-9b qwen3vl8b
  g=$(pick_gpu 100)
  serve_on $g reader32 8106 0.5 16G "${P[@]}" MAXLEN=32768 || exit 1
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    bash run_rp2_h200.sh probe_v2 models.answer.all_pixels=1003520 models.answer.vllm_model=qwen25vl32b \
      models.answer.vllm_base_url=http://127.0.0.1:8106/v1 '+probe_parts=[read]' +probe_tag=qwen25vl32b \
      > $L/probe_${RUN_NAME}_qwen25vl32b.log 2>&1 || echo "$(date) FAILED probe $RUN_NAME 32b"
    echo "$(date) $RUN_NAME probed (32B)"
  done
  stop qwen25vl32b
fi
echo "$(date) COST_DONE"
