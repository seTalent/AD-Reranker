"""Research point 2 (post-rerank evidence recovery) experiments, stage by stage.

    python scripts/rp2.py --config-name mmlb +stage=rerank  run-name=rp2-top10 dataset.top_k=10
    python scripts/rp2.py --config-name mmlb +stage=recover run-name=rp2-top10 dataset.top_k=10 recovery.mineru_dir=...
    python scripts/rp2.py --config-name mmlb +stage=answer  run-name=rp2-top10 dataset.top_k=10
    python scripts/rp2.py --config-name mmlb +stage=eval    run-name=rp2-top10 dataset.top_k=10
    python scripts/rp2.py --config-name mmlb +stage=report  run-name=rp2-top10

See docs/evidence_recovery.md.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_datasets.base_dataset import BaseDataset
import hydra


@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    from recovery import pipeline
    dataset = BaseDataset(cfg.dataset)
    stage = cfg.stage
    if stage == "rerank":
        pipeline.rerank_stage(cfg, dataset)
    elif stage == "recover":
        pipeline.recover_stage(cfg, dataset)
    elif stage == "answer":
        pipeline.answer_stage(cfg, dataset)
    elif stage == "eval":
        pipeline.eval_stage(cfg, dataset)
    elif stage == "probe":
        from recovery.probe import probe
        probe(cfg, dataset)
    elif stage == "report":
        from recovery.report import report
        print(report(os.path.join(cfg.dataset.result_dir, "eval.json"), cfg))
    else:
        raise ValueError(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
