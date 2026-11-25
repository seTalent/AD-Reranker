#!/usr/bin/env bash
# Related-work baselines in the recovery slot (results doc section 27), all from the same reranked candidates:
#   suffNN@2  CRAG-style sufficiency check of the kept pages, then the top-2 dropped pages by retrieval rank
#   gateNN@3  the same check as a gate in front of the gap-conditioned page scoring (score20)
#   mmrL@b    MMR over page texts and retrieval scores;  rsNN@3  retrieval-score threshold (dynamic top-k)
# (the reader-confidence trigger, Self-RAG / FLARE-like, needs no new run: scripts/rp2_baselines.py)
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
RUNS="mmlb:rp2-top10-mmlb-abl mmlb:rp2-top10-mmlb-adr ldu:rp2-top10-ldu ldu:rp2-top10-ldu-adr"
PLAN='recovery.plan=[suff30@2,suff50@2,suff70@2,gate30@3,gate50@3,gate70@3,mmr5@1,mmr7@1,mmr9@1,mmr7@2,rs50@3,rs70@3,rs90@3]'
VS='"suff30@2","suff50@2","suff70@2","gate30@3","gate50@3","gate70@3","mmr5@1","mmr7@1","mmr9@1","mmr7@2","rs50@3","rs70@3","rs90@3"'
FROM=${FROM:-1}
MDIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
VBIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm
for w in ${WAIT_FOR:-}; do until grep -q -E "FILTER_DONE" $w || ! pgrep -u "$USER" -f "^bash chain_filter_run.sh" > /dev/null; do sleep 120; done; done
echo "$(date) start"
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
  curl -s -m 5 http://127.0.0.1:$port/v1/models | grep -q '"id"' && { echo "$(date) port $port already up"; return 1; }
  echo "$(date) starting $role on GPU $g"
  (cd $R/AD-Reranker && env "$@" MODELS_DIR=$MDIR VLLM_BIN=$VBIN MEDIA_ROOT=$R \
     setsid -f bash scripts/serve_vllm.sh $role $g $util $kv > $L/serve_${role}_bl.log 2>&1 < /dev/null)
  wait_up $port $L/serve_${role}_bl.log
}
stop() { for n in "$@"; do pkill -u "$USER" -f "served-model-name $n( |$)" && echo "$(date) $n stopped"; done; sleep 20; }

if [ $FROM -le 1 ]; then
  g=$(pick_gpu 45)
  serve_on $g verifier_mm 8104 0.5 16G NO_MM_CACHE=1 || exit 1
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    bash run_rp2_h200.sh recover "$PLAN" +recovery.extend=true > $L/bl_recover_$RUN_NAME.log 2>&1 \
      || { echo "$(date) FAILED recover $RUN_NAME"; stop qwen35-9b-mm; exit 1; }
    grep -h "verification" $L/bl_recover_$RUN_NAME.log
  done
  stop qwen35-9b-mm
fi
if [ $FROM -le 2 ]; then
  g=$(pick_gpu 80)
  serve_on $g vlm 8101 0.2 12G NO_MM_CACHE=1 || exit 1
  serve_on $g reader3 8105 0.2 12G NO_MM_CACHE=1 || { stop qwen25vl7b; exit 1; }
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    for rd in qwen25vl7b:8101 qwen3vl8b:8105; do
      bash run_rp2_h200.sh answer "+variants=[$VS]" models.answer.all_pixels=1003520 models.answer.vllm_model=${rd%%:*} \
        models.answer.vllm_base_url=http://127.0.0.1:${rd#*:}/v1 > $L/bl_answer_${RUN_NAME}_${rd%%:*}.log 2>&1 \
        || { echo "$(date) FAILED answer $RUN_NAME ${rd%%:*}"; stop qwen25vl7b qwen3vl8b; exit 1; }
    done
    echo "$(date) $RUN_NAME answered"
  done
  stop qwen25vl7b qwen3vl8b
fi
if [ $FROM -le 3 ]; then
  g=$(pick_gpu 70)
  serve_on $g judge 8103 0.5 8G || exit 1
  J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    for NAME in qwen25vl7b qwen3vl8b; do
      bash run_rp2_h200.sh eval "${J[@]}" "+judge.variants=[rerank,score20@3,score20m2@3,ret@2,top10,$VS]" \
        models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME +judge.eval_out=eval_baselines_${NAME}_all.json \
        > $L/bl_eval_${RUN_NAME}_$NAME.log 2>&1 || echo "$(date) FAILED eval $RUN_NAME $NAME"
    done
  done
  stop qwen35-27b
fi
echo "$(date) BASELINES_DONE"
