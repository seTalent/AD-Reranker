#!/usr/bin/env bash
# Single-pass filter at low thresholds (filter1/2/5@10, both verifiers), so that it can be compared at the method's
# page count (results doc section 25). The scores are cached from chain_filter (no verifier server); readers, judge.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
ZS="mmlb:rp2-top10-mmlb-abl ldu:rp2-top10-ldu"
VS='"filter1@10","filter2@10","filter5@10","filter1@10:qwen27b","filter2@10:qwen27b","filter5@10:qwen27b"'
MDIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
VBIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm
for w in ${WAIT_FOR:-}; do until grep -q -E "_DONE" $w; do sleep 120; done; done
echo "$(date) start"
for run in $ZS; do
  export CONFIG=${run%%:*} RUN_NAME=${run#*:}
  bash run_rp2_h200.sh recover 'recovery.plan=[filter1@10,filter2@10,filter5@10]' +recovery.extend=true > $L/filter2_${RUN_NAME}_9b.log 2>&1 || echo "$(date) FAILED $RUN_NAME 9b"
  bash run_rp2_h200.sh recover 'recovery.plan=[filter1@10,filter2@10,filter5@10]' +recovery.extend=true +recovery.verifier_tag=qwen27b \
    > $L/filter2_${RUN_NAME}_27b.log 2>&1 || echo "$(date) FAILED $RUN_NAME 27b"
done
pick_gpu() {
  while true; do
    read -r idx free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',' | sort -k2 -n -r | head -1)
    [ "$free" -ge $(( $1 * 1024 )) ] && { echo $idx; return; }
    sleep 60
  done
}
wait_up() { until curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q '"id"'; do
  grep -q -E "Traceback|CUDA out of memory" $2 && { echo "$(date) FAILED $2"; return 1; }; sleep 10; done; echo "$(date) up $1"; }
serve_on() {
  local g=$1 role=$2 port=$3 util=$4 kv=$5; shift 5
  echo "$(date) starting $role on GPU $g"
  (cd $R/AD-Reranker && env "$@" MODELS_DIR=$MDIR VLLM_BIN=$VBIN MEDIA_ROOT=$R \
     setsid -f bash scripts/serve_vllm.sh $role $g $util $kv > $L/serve_${role}_f2.log 2>&1 < /dev/null)
  wait_up $port $L/serve_${role}_f2.log
}
stop() { for n in "$@"; do pkill -u "$USER" -f "served-model-name $n( |$)" && echo "$(date) $n stopped"; done; sleep 20; }
g=$(pick_gpu 80)
serve_on $g vlm 8101 0.2 12G NO_MM_CACHE=1 || exit 1
serve_on $g reader3 8105 0.2 12G NO_MM_CACHE=1 || { stop qwen25vl7b; exit 1; }
for run in $ZS; do
  export CONFIG=${run%%:*} RUN_NAME=${run#*:}
  for rd in qwen25vl7b:8101 qwen3vl8b:8105; do
    bash run_rp2_h200.sh answer "+variants=[$VS]" models.answer.all_pixels=1003520 models.answer.vllm_model=${rd%%:*} \
      models.answer.vllm_base_url=http://127.0.0.1:${rd#*:}/v1 > $L/filter2_answer_${RUN_NAME}_${rd%%:*}.log 2>&1 \
      || { echo "$(date) FAILED answer $RUN_NAME ${rd%%:*}"; stop qwen25vl7b qwen3vl8b; exit 1; }
  done
done
stop qwen25vl7b qwen3vl8b
g=$(pick_gpu 70)
serve_on $g judge 8103 0.5 8G || exit 1
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
for run in $ZS; do
  export CONFIG=${run%%:*} RUN_NAME=${run#*:}
  for NAME in qwen25vl7b qwen3vl8b; do
    bash run_rp2_h200.sh eval "${J[@]}" "+judge.variants=[$VS]" models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME \
      +judge.eval_out=eval_filter2_${NAME}_all.json > $L/filter2_eval_${RUN_NAME}_$NAME.log 2>&1 || echo "$(date) FAILED eval $RUN_NAME $NAME"
  done
done
stop qwen35-27b
echo "$(date) FILTER2_DONE"
