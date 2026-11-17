#!/usr/bin/env bash
# Runs the remaining stages back to back once rerank (already started) finishes.
# Servers must be up: vlm 8101 + verifier 8102 (GPU 2), judge 8103 (GPU 3).
set -uo pipefail
R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery
L=$R/logs
cd $R
while pgrep -u "$USER" -f "[+]stage=rerank" > /dev/null; do sleep 15; done
echo "$(date) rerank finished"
bash run_rp2_h200.sh recover > $L/rp2_recover.log 2>&1 || { echo "recover failed"; exit 1; }
echo "$(date) recover finished"
bash run_rp2_h200.sh answer > $L/rp2_answer.log 2>&1 || { echo "answer failed"; exit 1; }
echo "$(date) answer finished"
until curl -s -m 5 http://127.0.0.1:8103/v1/models > /dev/null; do sleep 15; done
bash run_rp2_h200.sh probe +probe_n=100 > $L/rp2_probe.log 2>&1 &
PROBE=$!
bash run_rp2_h200.sh eval > $L/rp2_eval.log 2>&1 || { echo "eval failed"; exit 1; }
echo "$(date) eval finished"
wait $PROBE; echo "$(date) probe finished (exit $?)"
bash run_rp2_h200.sh eval > $L/rp2_eval_final.log 2>&1      # all cached; adds the latency probe
bash run_rp2_h200.sh report > $L/rp2_report.md 2>&1
echo "$(date) ALL_DONE"
