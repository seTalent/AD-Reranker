#!/usr/bin/env bash
# Evidence-aware reading (recovery/reading.py): the same recovered page lists, read with different prompts.
#   plain     same prompt as before, plus token log-probabilities (needed for answer-level fallback)
#   tag       recovered pages labelled, with the reason in the prompt's reason slot
#   text      page text after each recovered page
#   tag+text  both
#   text_all  page text after every page (control: is it the recovered pages or text in general?)
# 4 runs x 2 readers x 5 prompts, variants rerank / score20@3 / oracle@2, dev and test; then the Qwen3.5-27B judge.
# Servers are started on the GPU with the most free memory once enough is free, and stopped after use.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
RUNS="mmlb:rp2-top10-mmlb-abl mmlb:rp2-top10-mmlb-adr ldu:rp2-top10-ldu ldu:rp2-top10-ldu-adr"
MODES="plain tag text tag+text text_all"
VAR='+variants=[rerank,score20@3,oracle@2]'
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048
   '+judge.variants=[rerank,score20@3,oracle@2]')
FROM=${FROM:-1}

pick_gpu() {  # <GiB needed>
  while true; do
    read -r idx free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                         | tr -d ',' | sort -k2 -n -r | head -1)
    [ "$free" -ge $(( $1 * 1024 )) ] && { echo $idx; return; }
    sleep 60
  done
}
wait_up() { until curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q '"id"'; do
  grep -q -E "Traceback|CUDA out of memory" $2 && { echo "$(date) FAILED $2"; return 1; }; sleep 10; done; echo "$(date) up $1"; }
serve() {  # <role> <port> <GiB> <util> <kv>
  local g; g=$(pick_gpu $3); echo "$(date) starting $1 on GPU $g"
  (cd $R/AD-Reranker && MODELS_DIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models \
     VLLM_BIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm MEDIA_ROOT=$R \
     setsid -f bash scripts/serve_vllm.sh $1 $g $4 $5 > $L/serve_${1}_reading.log 2>&1 < /dev/null)
  wait_up $2 $L/serve_${1}_reading.log
}
stop() { pkill -u "$USER" -f "served-model-name $1" && echo "$(date) $1 stopped"; sleep 20; }

if [ $FROM -le 1 ]; then
  for reader in qwen25vl7b:8101:vlm:30 qwen3vl8b:8105:reader3:45; do
    IFS=: read -r NAME PORT ROLE GIB <<< "$reader"
    serve $ROLE $PORT $GIB 0.3 12G || exit 1
    RD=(models.answer.all_pixels=1003520 models.answer.vllm_base_url=http://127.0.0.1:$PORT/v1 models.answer.vllm_model=$NAME)
    for run in $RUNS; do
      export CONFIG=${run%%:*} RUN_NAME=${run#*:}
      for mode in $MODES; do
        bash run_rp2_h200.sh answer +reading=$mode "$VAR" "${RD[@]}" > $L/reading_${RUN_NAME}_${NAME}_$mode.log 2>&1 \
          || { echo "$(date) FAILED answer $RUN_NAME $NAME $mode"; exit 1; }
      done
      echo "$(date) $RUN_NAME $NAME answered"
    done
    stop $NAME
  done
fi

serve judge 8103 70 0.5 8G || exit 1
for run in $RUNS; do
  export CONFIG=${run%%:*} RUN_NAME=${run#*:}
  for NAME in qwen25vl7b qwen3vl8b; do
    RD=(models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME)
    for mode in $MODES; do
      for sp in dev test; do
        bash run_rp2_h200.sh eval "${J[@]}" "${RD[@]}" +reading=$mode +split=$sp \
          +judge.eval_out=eval_ear-${mode/+/-}_${NAME}_$sp.json > $L/reading_eval_${RUN_NAME}_${NAME}_${mode}_$sp.log 2>&1 \
          || echo "$(date) FAILED eval $RUN_NAME $NAME $mode $sp"
      done
    done
  done
  echo "$(date) $RUN_NAME judged"
done
stop qwen35-27b
echo "$(date) READING_DONE"
