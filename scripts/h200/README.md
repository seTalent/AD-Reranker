# H200 experiment scripts (evidence recovery, research point 2)

These are the shell chains used to run the evidence-recovery experiments on the H200 server. They are kept for reproducibility. Paths are hard-coded to the server layout (`R=/user_mount/home/zhukeyu/datasets/thesis-evidence-recovery`, with the repo at `$R/AD-Reranker` and the Python env at `$R/envs/mdocagent`). Edit `R` before running them anywhere else. No data is stored here.

| Script | Purpose |
|---|---|
| `install_mdocagent_env.sh`, `prepare_mdocagent.sh` | Environment and MDocAgent data preparation |
| `serve.sh` | Start vLLM servers (VLM reranker + LoRA `adr`, readers, judge) |
| `run_rp2_h200.sh` | Run one `scripts/rp2.py` stage: `CONFIG=<mmlb\|ldu> RUN_NAME=<run> bash run_rp2_h200.sh <stage> [overrides]` |
| `run_all_h200.sh`, `chain_rp2_h200.sh`, `chain_mmlb_h200.sh` | Main MMLongBench pipeline (retrieve → rerank → recover → answer → judge) |
| `chain_exp_h200.sh`, `chain_extra_h200.sh`, `chain_extra2_h200.sh`, `chain_ablation_h200.sh` | Baselines, ablations and extra metrics |
| `chain_sampling_h200.sh`, `chain_perm_h200.sh`, `chain_adr2_h200.sh` | Group sampling (temperature / permutation), AD-Reranker runs |
| `chain_reader_h200.sh` | Second reader (Qwen3-VL-8B) |
| `chain_ldu_h200.sh`, `chain_ldu_adr_h200.sh`, `redo_breakdown_h200.sh` | LongDocURL runs and breakdowns |
| `chain_logit_h200.sh` | Zero-sampling `first` / `slot` / content-free signals for all four runs (`+stage=logit`) |
| `chain_logit_e2e_h200.sh` | End-to-end check of `first` replacing sampling frequency (recover → both readers → judge) |
| `judge_chain_h200.sh` | Judge only |
| `smoke.py`, `pagerep_stats.py` | Smoke test and page-representation statistics |

Results and their interpretation: `docs/evidence_recovery_results.md`.
