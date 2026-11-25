#!/usr/bin/env bash
# MM-R5 (i2vec/MM-R5) as the reranker (results doc section 27): rank the Top-10, keep the first 2 (research point 1),
# then recovery on top; new runs rp2-top10-{mmlb,ldu}-mmr5. Answers / verdicts / verifier caches of the
# Zero-Shot runs are copied first (keys depend only on question, pages and response).
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker
cd $R
RUNS="mmlb:rp2-top10-mmlb-mmr5:rp2-top10-mmlb-abl ldu:rp2-top10-ldu-mmr5:rp2-top10-ldu"
RPLAN='recovery.plan=[none,ret@2,oracle@2,score20@3,scoreni20@3,score20m2@3]'
FROM=${FROM:-1}
MDIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
VBIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm
for w in ${WAIT_FOR:-}; do until grep -q -E "BASELINES_DONE" $w || ! pgrep -u "$USER" -f "^bash chain_baselines_run.sh" > /dev/null; do sleep 120; done; done
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
  (cd $A && env "$@" MODELS_DIR=$MDIR VLLM_BIN=$VBIN MEDIA_ROOT=$R \
     setsid -f bash scripts/serve_vllm.sh $role $g $util $kv > $L/serve_${role}_mmr5.log 2>&1 < /dev/null)
  wait_up $port $L/serve_${role}_mmr5.log
}
stop() { for n in "$@"; do pkill -u "$USER" -f "served-model-name $n( |$)" && echo "$(date) $n stopped"; done; sleep 20; }

for x in $RUNS; do
  IFS=: read -r C N SRC <<< "$x"
  D=$A/results/$([ $C = mmlb ] && echo MMLongBench || echo LongDocURL)
  mkdir -p $D/$N/recovery_cache
  for f in answers_px1003520.jsonl answers_qwen3vl8b_px1003520.jsonl judge_adr.jsonl; do [ -e $D/$N/$f ] || cp $D/$SRC/$f $D/$N/$f; done
  for f in verify_score_mm.jsonl verify_score_noimg.jsonl; do
    [ -e $D/$N/recovery_cache/$f ] || cp $D/$SRC/recovery_cache/$f $D/$N/recovery_cache/$f
  done
done
if [ $FROM -le 1 ]; then
  g=$(pick_gpu 45)
  serve_on $g mmr5 8112 0.5 16G NO_MM_CACHE=1 || exit 1
  for x in $RUNS; do
    IFS=: read -r C N SRC <<< "$x"
    CONFIG=$C RUN_NAME=$N bash run_rp2_h200.sh rerank +models.refiner.ranker=mmr5 models.refiner.vllm_model=mmr5 \
      models.refiner.vllm_base_url=http://127.0.0.1:8112/v1 > $L/mmr5_rerank_$N.log 2>&1 || { echo "$(date) FAILED rerank $N"; stop mmr5; exit 1; }
    echo "$(date) $N reranked"
  done
  stop mmr5
fi
if [ $FROM -le 2 ]; then
  g=$(pick_gpu 45)
  serve_on $g verifier_mm 8104 0.5 16G NO_MM_CACHE=1 || exit 1
  for x in $RUNS; do
    IFS=: read -r C N SRC <<< "$x"
    CONFIG=$C RUN_NAME=$N bash run_rp2_h200.sh recover "$RPLAN" > $L/mmr5_recover_$N.log 2>&1 || { echo "$(date) FAILED recover $N"; stop qwen35-9b-mm; exit 1; }
    echo "$(date) $N recovered"
  done
  stop qwen35-9b-mm
fi
if [ $FROM -le 3 ]; then
  g=$(pick_gpu 80)
  serve_on $g vlm 8101 0.2 12G NO_MM_CACHE=1 || exit 1
  serve_on $g reader3 8105 0.2 12G NO_MM_CACHE=1 || { stop qwen25vl7b; exit 1; }
  for x in $RUNS; do
    IFS=: read -r C N SRC <<< "$x"
    for rd in qwen25vl7b:8101 qwen3vl8b:8105; do
      CONFIG=$C RUN_NAME=$N bash run_rp2_h200.sh answer models.answer.all_pixels=1003520 models.answer.vllm_model=${rd%%:*} \
        models.answer.vllm_base_url=http://127.0.0.1:${rd#*:}/v1 > $L/mmr5_answer_${N}_${rd%%:*}.log 2>&1 \
        || { echo "$(date) FAILED answer $N ${rd%%:*}"; stop qwen25vl7b qwen3vl8b; exit 1; }
    done
    echo "$(date) $N answered"
  done
  stop qwen25vl7b qwen3vl8b
fi
if [ $FROM -le 4 ]; then
  g=$(pick_gpu 70)
  serve_on $g judge 8103 0.5 8G || exit 1
  J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
  for x in $RUNS; do
    IFS=: read -r C N SRC <<< "$x"
    for NAME in qwen25vl7b qwen3vl8b; do
      CONFIG=$C RUN_NAME=$N bash run_rp2_h200.sh eval "${J[@]}" models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME \
        +judge.eval_out=eval_${NAME}_all.json > $L/mmr5_eval_${N}_$NAME.log 2>&1 || echo "$(date) FAILED eval $N $NAME"
    done
  done
  stop qwen35-27b
fi
echo "$(date) MMR5_DONE"
