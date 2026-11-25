#!/usr/bin/env bash
# Verifier models (results doc section 25): the same gap-conditioned check (score_mm) with other verifiers.
#   qwen9b_lr  Qwen3.5-9B at 256 visual tokens per page (the default runs at ~590): resolution / cost control
#   qwen27b    Qwen3.5-27B, same family, larger
#   internvl8b InternVL3.5-8B, other family, 256 tokens per page
#   gemma12b   gemma-3-12b-it, other family, 256 tokens per page
# step 1: verification of every question in the 4 runs (recovery.extend: adds score20@3:TAG to recover.json),
#         plus a sequential / batch latency probe of the verifier on 2 runs
# step 2: τ* per verifier on MMLongBench dev (equal restored pages), added as score{τ*}@3:TAG
# step 3: 7B and 8B readers on the new page lists; step 4: Qwen3.5-27B judge.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
RUNS="mmlb:rp2-top10-mmlb-abl mmlb:rp2-top10-mmlb-adr ldu:rp2-top10-ldu ldu:rp2-top10-ldu-adr"
#      tag        role               port model            pixels  GiB
VERS=${VERS:-"qwen9b_lr:verifier_mm_lr:8111:qwen35-9b-mm-lr:262144:45
qwen27b:verifier_mm27:8110:qwen35-27b-mm:602112:90
internvl8b:verifier_internvl:8108:internvl35-8b:null:45
gemma12b:verifier_gemma:8109:gemma3-12b:null:55"}
# InternVL3.5-8B-HF returns meaningless text under vLLM 0.19 (also text-only); it is replaced by
#   VERS="qwen3vl8b:verifier_qwen3vl:8113:qwen3vl8b-ver:602112:45"   (results doc section 24)
# WAIT_READ="<log>..." makes steps 3-4 (readers, judge: shared names / ports) wait for those chains' *_DONE line.
TAGS=$(echo "$VERS" | cut -d: -f1 | paste -sd,)
FROM=${FROM:-1}
MDIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models
VBIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm
PY=$R/envs/mdocagent/bin/python

for w in ${WAIT_FOR:-}; do until grep -q -E "COST_DONE|FAILED" $w; do sleep 120; done; done
pick_gpu() {
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
     setsid -f bash scripts/serve_vllm.sh $role $g $util $kv > $L/serve_${role}_ver.log 2>&1 < /dev/null)
  wait_up $port $L/serve_${role}_ver.log
}
stop() { for n in "$@"; do pkill -u "$USER" -f "served-model-name $n( |$)" && echo "$(date) $n stopped"; done; sleep 20; }

if [ $FROM -le 1 ]; then
  echo "$VERS" | while IFS=: read -r tag role port model px gib; do
    g=$(pick_gpu $gib)
    serve_on $g $role $port 0.5 16G NO_MM_CACHE=1 < /dev/null || exit 1
    V=(recovery.verifier_mm.base_url=http://127.0.0.1:$port/v1 recovery.verifier_mm.model=$model
       recovery.verifier_mm.pixels=$px +recovery.verifier_tag=$tag)
    for run in $RUNS; do
      export CONFIG=${run%%:*} RUN_NAME=${run#*:}
      bash run_rp2_h200.sh recover 'recovery.plan=[score20@3]' +recovery.extend=true "${V[@]}" \
        > $L/ver_${RUN_NAME}_$tag.log 2>&1 < /dev/null || { echo "$(date) FAILED recover $RUN_NAME $tag"; stop $model; exit 1; }
      grep -h "verification score_mm" $L/ver_${RUN_NAME}_$tag.log
    done
    for run in mmlb:rp2-top10-mmlb-abl ldu:rp2-top10-ldu; do
      export CONFIG=${run%%:*} RUN_NAME=${run#*:}
      bash run_rp2_h200.sh probe_v2 "${V[@]}" '+probe_parts=[context,verify]' '+probe_modes=[score_mm]' \
        +probe_tag=verifier_$tag > $L/ver_probe_${RUN_NAME}_$tag.log 2>&1 < /dev/null || echo "$(date) FAILED probe $RUN_NAME $tag"
    done
    stop $model
    echo "$(date) $tag verified"
  done || exit 1
  # the default verifier's latency in the same probe (it already has its verification)
  [ "${SKIP_DEFAULT_PROBE:-0}" = 1 ] || {
  g=$(pick_gpu 45)
  serve_on $g verifier_mm 8104 0.5 16G NO_MM_CACHE=1 || exit 1
  for run in mmlb:rp2-top10-mmlb-abl ldu:rp2-top10-ldu; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    bash run_rp2_h200.sh probe_v2 '+probe_parts=[context,verify]' '+probe_modes=[score_mm]' +probe_tag=verifier_qwen9b \
      > $L/ver_probe_${RUN_NAME}_qwen9b.log 2>&1 || echo "$(date) FAILED probe $RUN_NAME qwen9b"
  done
  stop qwen35-9b-mm
  }
  echo "$(date) VERIFIED"
fi

cd $R/AD-Reranker
TAU=$($PY scripts/rp2_verifier_compare.py results/MMLongBench/rp2-top10-mmlb-abl --tags $TAGS --pick)
echo "$(date) tau* $TAU"
cd $R
VLIST=()
if [ $FROM -le 2 ]; then
  echo "$VERS" | while IFS=: read -r tag role port model px gib; do
    t=$(echo "$TAU" | $PY -c "import json,sys; print(round(100*json.load(sys.stdin)['$tag']))")
    [ "$t" = 20 ] && continue
    V=(recovery.verifier_mm.model=$model recovery.verifier_mm.pixels=$px +recovery.verifier_tag=$tag)
    for run in $RUNS; do
      export CONFIG=${run%%:*} RUN_NAME=${run#*:}
      bash run_rp2_h200.sh recover "recovery.plan=[score$t@3]" +recovery.extend=true "${V[@]}" \
        > $L/ver_tau_${RUN_NAME}_$tag.log 2>&1 < /dev/null || echo "$(date) FAILED tau $RUN_NAME $tag"
    done
  done
fi
# every page list of the new verifiers
LIST=$(echo "$VERS" | while IFS=: read -r tag rest; do
  t=$(echo "$TAU" | $PY -c "import json,sys; print(round(100*json.load(sys.stdin)['$tag']))")
  echo "\"score20@3:$tag\""; [ "$t" != 20 ] && echo "\"score$t@3:$tag\""; done | paste -sd,)
echo "$(date) variants [$LIST]"
for w in ${WAIT_READ:-}; do until grep -q -E "_DONE" $w; do sleep 120; done; done
if [ $FROM -le 3 ]; then
  g=$(pick_gpu 80)
  serve_on $g vlm 8101 0.2 12G NO_MM_CACHE=1 || exit 1
  serve_on $g reader3 8105 0.2 12G NO_MM_CACHE=1 || { stop qwen25vl7b; exit 1; }
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    for rd in qwen25vl7b:8101 qwen3vl8b:8105; do
      bash run_rp2_h200.sh answer "+variants=[$LIST]" models.answer.all_pixels=1003520 models.answer.vllm_model=${rd%%:*} \
        models.answer.vllm_base_url=http://127.0.0.1:${rd#*:}/v1 > $L/ver_answer_${RUN_NAME}_${rd%%:*}.log 2>&1 \
        || { echo "$(date) FAILED answer $RUN_NAME ${rd%%:*}"; stop qwen25vl7b qwen3vl8b; exit 1; }
    done
    echo "$(date) $RUN_NAME answered"
  done
  stop qwen25vl7b qwen3vl8b
fi
if [ $FROM -le 4 ]; then
  g=$(pick_gpu 70)
  serve_on $g judge 8103 0.5 8G || exit 1
  J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048
     "+judge.variants=[rerank,score20@3,$LIST]")
  for run in $RUNS; do
    export CONFIG=${run%%:*} RUN_NAME=${run#*:}
    for NAME in qwen25vl7b qwen3vl8b; do
      bash run_rp2_h200.sh eval "${J[@]}" models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME \
        +judge.eval_out=eval_verifiers_${NAME}_all.json > $L/ver_eval_${RUN_NAME}_$NAME.log 2>&1 || echo "$(date) FAILED eval $RUN_NAME $NAME"
    done
    echo "$(date) $RUN_NAME judged"
  done
  stop qwen35-27b
fi
echo "$(date) VERIFIERS_DONE"
