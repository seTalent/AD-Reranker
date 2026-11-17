#!/usr/bin/env bash
# LongDocURL (MDocAgent version, 2,325 questions, retrieval inside each question's page_ids window):
# ColPali top-10 -> zero-shot rerank -> group sampling -> recovery -> answer (1,280 tokens/page) -> judge.
# Thresholds are carried over from MMLongBench dev (tau 0.2, f 0.5, b 3); the sweep is for analysis only.
# Servers: VLM 8101, text verifier 8102, multimodal verifier 8104 (GPU 2); judge 8103 (GPU 0).
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
export CONFIG=ldu RUN_NAME=rp2-top10-ldu
until grep -a -q "\[extract\] 396/396" $L/ldu_extract_par.log; do sleep 30; done
GPU=${RET_GPU:-3} bash run_rp2_h200.sh retrieve > $L/ldu_retrieve.log 2>&1
grep -a -q "Save retrieval results" $L/ldu_retrieve.log || { echo "ldu retrieve failed"; exit 1; }
echo "$(date) retrieve finished"
bash run_rp2_h200.sh rerank > $L/ldu_rerank.log 2>&1 || { echo "ldu rerank failed"; exit 1; }
echo "$(date) rerank finished"
bash run_rp2_h200.sh resample > $L/ldu_resample.log 2>&1 || { echo "ldu resample failed"; exit 1; }
echo "$(date) resample finished"
PLAN="recovery.plan=[none,ret@1,ret@2,ret@3,adj@2,oracle@2,f50@3"
for m in score scoreni; do for t in 10 20 30 50; do PLAN="$PLAN,$m$t@3"; done; done
PLAN="$PLAN,\"scoreni20+f50@3\",\"score20+f50@3\"]"
bash run_rp2_h200.sh recover "$PLAN" > $L/ldu_recover.log 2>&1 || { echo "ldu recover failed"; exit 1; }
echo "$(date) recover finished"
HI=(models.answer.all_pixels=1003520)
bash run_rp2_h200.sh answer "${HI[@]}" > $L/ldu_answer.log 2>&1 || { echo "ldu answer failed"; exit 1; }
echo "$(date) answer finished"
J=(judge.prompt=adr judge.thinking=false judge.max_tokens=256 +judge.out=judge_adr.jsonl judge.concurrency=128 +judge.batch=2048)
for sp in dev test all; do
  S=(); [ $sp != all ] && S=(+split=$sp)
  bash run_rp2_h200.sh eval "${J[@]}" "${HI[@]}" "${S[@]}" +judge.eval_out=eval_$sp.json > $L/ldu_eval_$sp.log 2>&1 \
    || { echo "ldu eval $sp failed"; exit 1; }
done
bash run_rp2_h200.sh breakdown "${HI[@]}" +split=test > $L/ldu_breakdown_test.log 2>&1
echo "$(date) LDU_DONE"
