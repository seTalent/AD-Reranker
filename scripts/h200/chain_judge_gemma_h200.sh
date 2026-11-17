#!/usr/bin/env bash
# Cross-family judge: re-score the main methods' existing answers with Google gemma-3-27b-it
# (same binary-correctness "adr" prompt as the Qwen3.5-27B judge). Answers are not regenerated.
# Writes judge_gemma.jsonl and eval_gemma_<reader>_<split>.json next to the Qwen judge files;
# eval.json "judge_agreement" compares the two judges on the shared responses.
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
RUNS="mmlb:rp2-top10-mmlb-abl mmlb:rp2-top10-mmlb-adr ldu:rp2-top10-ldu ldu:rp2-top10-ldu-adr"
V='+judge.variants=[rerank,top10,ret@2,oracle@2,scoreni20@3,score20@3,"scoreni20+f50@3","score20+f50@3"]'
J=(judge.base_url=http://127.0.0.1:8107/v1 judge.model=gemma3-27b judge.prompt=adr judge.thinking=false
   judge.max_tokens=64 +judge.out=judge_gemma.jsonl +judge.compare_with=judge_adr.jsonl
   judge.concurrency=128 +judge.batch=4096)

pick_gpu() {  # <GiB needed>: GPU with the most free memory, once it has enough
  while true; do
    read -r idx free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                         | tr -d ',' | sort -k2 -n -r | head -1)
    [ "$free" -ge $(( $1 * 1024 )) ] && { echo $idx; return; }
    sleep 60
  done
}
wait_up() { until curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q '"id"'; do
  grep -q -E "Traceback|CUDA out of memory" $2 && { echo "$(date) FAILED $2"; return 1; }; sleep 10; done; echo "$(date) up $1"; }

g=$(pick_gpu 70); echo "$(date) starting gemma judge on GPU $g"
(cd $R/AD-Reranker && MODELS_DIR=/user_mount/home/zhukeyu/datasets/group-memory-pilot/models \
   VLLM_BIN=/user_mount/home/zhukeyu/spd/envs/infer/bin/vllm MEDIA_ROOT=$R \
   setsid -f bash scripts/serve_vllm.sh judge_gemma $g 0.5 10G > $L/serve_judge_gemma.log 2>&1 < /dev/null)
wait_up 8107 $L/serve_judge_gemma.log || exit 1

for run in $RUNS; do
  export CONFIG=${run%%:*} RUN_NAME=${run#*:}
  for NAME in qwen25vl7b qwen3vl8b; do
    RD=(models.answer.all_pixels=1003520 models.answer.vllm_model=$NAME)
    for sp in all test; do
      S=(); [ $sp != all ] && S=(+split=$sp)
      bash run_rp2_h200.sh eval "${J[@]}" "$V" "${RD[@]}" "${S[@]}" +judge.eval_out=eval_gemma_${NAME}_$sp.json \
        > $L/gemma_${RUN_NAME}_${NAME}_$sp.log 2>&1 || echo "$(date) FAILED $RUN_NAME $NAME $sp"
    done
    echo "$(date) $RUN_NAME $NAME judged"
  done
done
pkill -u "$USER" -f "served-model-name gemma3-27b" && echo "$(date) gemma judge stopped"
echo "$(date) GEMMA_DONE"
