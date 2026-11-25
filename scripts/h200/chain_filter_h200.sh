#!/usr/bin/env bash
# "Why rerank and recover instead of one careful filter?" (results doc section 26)
#   A  single-pass filter, no reranker: every Top-10 page gets the plain relevance check (score_all, the same
#      model and inputs as the verifier, without the selected pages) and pages with P(Yes) >= τ are read
#      (filterNN@10). Verifiers: Qwen3.5-9B (the default) and Qwen3.5-27B (tag qwen27b). Zero-Shot runs only:
#      the filter does not depend on the reranker.
#   B  a stronger reranker as the filter: Qwen2.5-VL-32B zero-shot rerank (new runs rp2-top10-{mmlb,ldu}-r32),
#      alone and with recovery on top.
# Steps 1-2 use servers under their own names / ports (SNAME / PORT) next to the cost and verifier chains;
# from step 3 on (readers, judge, 32B) the chain waits until both of those chains are done.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker
cd $R
ZS="mmlb:rp2-top10-mmlb-abl ldu:rp2-top10-ldu"
R32="mmlb:rp2-top10-mmlb-r32:rp2-top10-mmlb-abl ldu:rp2-top10-ldu-r32:rp2-top10-ldu"
FPLAN='recovery.plan=[filter10@10,filter20@10,filter30@10,filter50@10,filter70@10,filter90@10]'
FV9='"filter10@10","filter20@10","filter30@10","filter50@10","filter70@10","filter90@10"'
FV27='"filter10@10:qwen27b","filter20@10:qwen27b","filter30@10:qwen27b","filter50@10:qwen27b","filter70@10:qwen27b","filter90@10:qwen27b"'
RPLAN='recovery.plan=[none,ret@2,oracle@2,score20@3,scoreni20@3,score20m2@3]'
FROM=${FROM:-1}
MDIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
VBIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm

pick_gpu() {
  while true; do
    read -r idx free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',' | sort -k2 -n -r | head -1)
    [ "$free" -ge $(( $1 * 1024 )) ] && { echo $idx; return; }
    sleep 60
  done
}
wait_up() { until curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q '"id"'; do
  grep -q -E "Traceback|CUDA out of memory" $2 && { echo "$(date) FAILED $2"; return 1; }; sleep 10; done; echo "$(date) up $1"; }
serve_on() {  # <gpu> <role> <port> <util> <kv> <log tag> [env...]
  local g=$1 role=$2 port=$3 util=$4 kv=$5 t=$6; shift 6
  curl -s -m 5 http://127.0.0.1:$port/v1/models | grep -q '"id"' && { echo "$(date) port $port already up"; return 1; }
  echo "$(date) starting $role ($t) on GPU $g"
  (cd $A && env "$@" MODELS_DIR=$MDIR VLLM_BIN=$VBIN MEDIA_ROOT=$R \
     setsid -f bash scripts/serve_vllm.sh $role $g $util $kv > $L/serve_${t}_filter.log 2>&1 < /dev/null)
  wait_up $port $L/serve_${t}_filter.log
}
stop() { for n in "$@"; do pkill -u "$USER" -f "served-model-name $n( |$)" && echo "$(date) $n stopped"; done; sleep 20; }

if [ $FROM -le 1 ]; then   # A: Qwen3.5-9B filter scores
  g=$(pick_gpu 45)
  serve_on $g verifier_mm 8124 0.5 16G v9f NO_MM_CACHE=1 SNAME=qwen35-9b-mm-f PORT=8124 || exit 1
  for run in $ZS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    bash run_rp2_h200.sh recover "$FPLAN" +recovery.extend=true recovery.verifier_mm.base_url=http://127.0.0.1:8124/v1 \
      recovery.verifier_mm.model=qwen35-9b-mm-f > $L/filter_${RUN_NAME}_qwen9b.log 2>&1 \
      || { echo "$(date) FAILED filter $RUN_NAME qwen9b"; stop qwen35-9b-mm-f; exit 1; }
    grep -h "verification score_all" $L/filter_${RUN_NAME}_qwen9b.log
  done
  stop qwen35-9b-mm-f
  echo "$(date) A qwen9b scored"
fi
if [ $FROM -le 2 ]; then   # A: Qwen3.5-27B filter scores
  g=$(pick_gpu 90)
  serve_on $g verifier_mm27 8130 0.5 16G v27f NO_MM_CACHE=1 SNAME=qwen35-27b-mm-f PORT=8130 || exit 1
  for run in $ZS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    bash run_rp2_h200.sh recover "$FPLAN" +recovery.extend=true +recovery.verifier_tag=qwen27b \
      recovery.verifier_mm.base_url=http://127.0.0.1:8130/v1 recovery.verifier_mm.model=qwen35-27b-mm-f \
      > $L/filter_${RUN_NAME}_qwen27b.log 2>&1 || { echo "$(date) FAILED filter $RUN_NAME qwen27b"; stop qwen35-27b-mm-f; exit 1; }
    grep -h "verification score_all" $L/filter_${RUN_NAME}_qwen27b.log
  done
  stop qwen35-27b-mm-f
  echo "$(date) A qwen27b scored"
fi

# readers / judge / 32B share names and ports with the other chains: wait until they are done
for w in $L/chain_cost.out $L/chain_verifier.out; do
  until grep -q -E "COST_DONE|VERIFIERS_DONE" $w || ! pgrep -u "$USER" -f "^bash chain_(cost3c|verifier)_h200.sh" > /dev/null; do sleep 120; done
done
echo "$(date) other chains done"

if [ $FROM -le 4 ]; then   # B: Qwen2.5-VL-32B rerank, then recovery (verifier caches / answers / verdicts are reused)
  for x in $R32; do
    IFS=: read -r C N SRC <<< "$x"
    D=$A/results/$([ $C = mmlb ] && echo MMLongBench || echo LongDocURL)
    mkdir -p $D/$N/recovery_cache
    for f in answers_px1003520.jsonl answers_qwen3vl8b_px1003520.jsonl judge_adr.jsonl; do
      [ -e $D/$N/$f ] || cp $D/$SRC/$f $D/$N/$f
    done
    for f in verify_score_mm.jsonl verify_score_noimg.jsonl; do
      [ -e $D/$N/recovery_cache/$f ] || cp $D/$SRC/recovery_cache/$f $D/$N/recovery_cache/$f
    done
  done
  g=$(pick_gpu 100)
  serve_on $g reader32 8106 0.5 24G r32f NO_MM_CACHE=1 || exit 1
  for x in $R32; do
    IFS=: read -r C N SRC <<< "$x"
    CONFIG=$C RUN_NAME=$N bash run_rp2_h200.sh rerank models.refiner.vllm_model=qwen25vl32b \
      models.refiner.vllm_base_url=http://127.0.0.1:8106/v1 > $L/r32rerank_$N.log 2>&1 \
      || { echo "$(date) FAILED rerank $N"; stop qwen25vl32b; exit 1; }
    echo "$(date) $N reranked"
  done
  stop qwen25vl32b
  g=$(pick_gpu 45)
  serve_on $g verifier_mm 8104 0.5 16G v9 NO_MM_CACHE=1 || exit 1
  for x in $R32; do
    IFS=: read -r C N SRC <<< "$x"
    CONFIG=$C RUN_NAME=$N bash run_rp2_h200.sh recover "$RPLAN" > $L/r32recover_$N.log 2>&1 \
      || { echo "$(date) FAILED recover $N"; stop qwen35-9b-mm; exit 1; }
    echo "$(date) $N recovered"
  done
  stop qwen35-9b-mm
fi
if [ $FROM -le 5 ]; then   # readers on A's and B's page lists
  g=$(pick_gpu 80)
  serve_on $g vlm 8101 0.2 12G vlm NO_MM_CACHE=1 || exit 1
  serve_on $g reader3 8105 0.2 12G r3 NO_MM_CACHE=1 || { stop qwen25vl7b; exit 1; }
  JOBS="$(for r in $ZS; do echo "$r:[$FV9,$FV27]"; done) $(for x in $R32; do IFS=: read -r C N S <<< "$x"; echo "$C:$N:"; done)"
  for j in $JOBS; do
    IFS=: read -r C N V <<< "$j"
    VA=(); [ -n "$V" ] && VA=("+variants=$V")
    for rd in qwen25vl7b:8101 qwen3vl8b:8105; do
      CONFIG=$C RUN_NAME=$N bash run_rp2_h200.sh answer "${VA[@]}" models.answer.all_pixels=1003520 models.answer.vllm_model=${rd%%:*} \
        models.answer.vllm_base_url=http://127.0.0.1:${rd#*:}/v1 > $L/filter_answer_${N}_${rd%%:*}.log 2>&1 \
        || { echo "$(date) FAILED answer $N ${rd%%:*}"; stop qwen25vl7b qwen3vl8b; exit 1; }
    done
    echo "$(date) $N answered"
  done
  stop qwen25vl7b qwen3vl8b
fi
if [ $FROM -le 6 ]; then
  g=$(pick_gpu 70)
  serve_on $g judge 8103 0.5 8G judge || exit 1
  J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
  for r in $ZS; do
    export CONFIG=${r%%:*} RUN_NAME=${r#*:}
    for NAME in qwen25vl7b qwen3vl8b; do
      bash run_rp2_h200.sh eval "${J[@]}" "+judge.variants=[rerank,score20@3,score20m2@3,top10,$FV9,$FV27]" \
        models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME +judge.eval_out=eval_filter_${NAME}_all.json \
        > $L/filter_eval_${RUN_NAME}_$NAME.log 2>&1 || echo "$(date) FAILED eval $RUN_NAME $NAME"
    done
  done
  for x in $R32; do
    IFS=: read -r C N S <<< "$x"
    for NAME in qwen25vl7b qwen3vl8b; do
      CONFIG=$C RUN_NAME=$N bash run_rp2_h200.sh eval "${J[@]}" models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME \
        +judge.eval_out=eval_${NAME}_all.json > $L/filter_eval_${N}_$NAME.log 2>&1 || echo "$(date) FAILED eval $N $NAME"
    done
  done
  stop qwen35-27b
fi
echo "$(date) FILTER_DONE"
