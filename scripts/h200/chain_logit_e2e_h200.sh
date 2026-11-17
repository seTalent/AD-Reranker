#!/usr/bin/env bash
# End-to-end check of the zero-sampling signal: the FIRST-style first-token distribution replaces the
# group-sampling frequency in "verifier OR signal >= f". f is the dev-chosen, budget-matched threshold
# from scripts/rp2_logit_analysis.py. Each run is copied to <run>-first so recover.json of the original
# run (and its sampling variants) stays untouched; answer and judge caches carry over by key.
# Servers: VLM 8101 (running, step 1 only); reader3 8105 and judge 8103 are started here on the GPU with the most free
# memory once enough is free, and stopped after use.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
A=$R/AD-Reranker/results
PY=$R/envs/mdocagent/bin/python
cd $R
RUNS="MMLongBench:mmlb:rp2-top10-mmlb-abl:30 MMLongBench:mmlb:rp2-top10-mmlb-adr:30 LongDocURL:ldu:rp2-top10-ldu:15 LongDocURL:ldu:rp2-top10-ldu-adr:40"
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
rd() { echo "models.answer.all_pixels=1003520 models.answer.vllm_base_url=http://127.0.0.1:$2/v1 models.answer.vllm_model=$1"; }

pick_gpu() {  # <GiB needed>: index of the GPU with the most free memory, once it has enough
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
     setsid -f bash scripts/serve_vllm.sh $1 $g $4 $5 > $L/serve_${1}_e2e.log 2>&1 < /dev/null)
  wait_up $2 $L/serve_${1}_e2e.log
}

FROM=${FROM:-1}   # FROM=2 skips step 1 (e.g. after the VLM server was stopped to free memory)

# the latency probe shares the VLM server; do not disturb its timings
until grep -q PROBE_DONE $L/logit_probe_mmlb.log 2>/dev/null; do sleep 30; done

# 1. copy, write the signal file, recover, answer with the 7B reader (VLM server already up)
[ $FROM -le 1 ] && for spec in $RUNS; do
  IFS=: read -r DS C N F <<< "$spec"
  NEW=$N-first
  [ -d $A/$DS/$NEW ] || cp -a $A/$DS/$N $A/$DS/$NEW
  rm -f $A/$DS/$NEW/eval_*.json $A/$DS/$NEW/breakdown_*.json
  $PY AD-Reranker/scripts/rp2_logit_analysis.py $A/$DS/$NEW --write first > $L/e2e_${N}_signal.log 2>&1 \
    || { echo "$N signal failed"; continue; }
  PLAN="recovery.plan=[none,ret@2,scoreni20@3,score20@3,\"scoreni20+f$F@3\",\"score20+f$F@3\",oracle@2]"
  CONFIG=$C RUN_NAME=$NEW bash run_rp2_h200.sh recover "$PLAN" +recovery.resample_file=resample_logit_first.jsonl \
    > $L/e2e_${N}_recover.log 2>&1 || { echo "$N recover failed"; continue; }
  CONFIG=$C RUN_NAME=$NEW bash run_rp2_h200.sh answer $(rd qwen25vl7b 8101) > $L/e2e_${N}_answer_7b.log 2>&1 \
    || echo "$N answer 7b failed"
  echo "$(date) $N recover + 7B answers finished"
done

# 2. second reader
serve reader3 8105 45 0.30 20G || exit 1
for spec in $RUNS; do
  IFS=: read -r DS C N F <<< "$spec"
  CONFIG=$C RUN_NAME=$N-first bash run_rp2_h200.sh answer $(rd qwen3vl8b 8105) > $L/e2e_${N}_answer_q3.log 2>&1 \
    || echo "$N answer qwen3vl8b failed"
done
pkill -u zhukeyu -f "served-model-name qwen3vl8b"
echo "$(date) reader3 answers finished, reader3 stopped"

# 3. judge, both readers, dev / test
# judge prompts are short: 8 GiB of KV cache is enough; util only gates vLLM's free-memory check
serve judge 8103 68 0.46 8G || exit 1
for spec in $RUNS; do
  IFS=: read -r DS C N F <<< "$spec"
  for reader in qwen25vl7b:8101 qwen3vl8b:8105; do
    NAME=${reader%:*}; PORT=${reader#*:}
    for sp in dev test; do
      CONFIG=$C RUN_NAME=$N-first bash run_rp2_h200.sh eval "${J[@]}" $(rd $NAME $PORT) +split=$sp \
        +judge.eval_out=eval_${NAME}_$sp.json > $L/e2e_${N}_eval_${NAME}_$sp.log 2>&1 || echo "$N eval $NAME $sp failed"
    done
  done
  echo "$(date) $N judged"
done
pkill -u zhukeyu -f "served-model-name qwen35-27b"
echo "$(date) judge stopped"
echo "$(date) E2E_DONE"
