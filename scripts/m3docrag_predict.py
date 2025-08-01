import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_datasets.base_dataset import BaseDataset

import hydra
from models.m3docrag import M3DocRAG

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    dataset = BaseDataset(cfg.dataset)
    model = M3DocRAG(cfg.models)
    model.predict_dataset(dataset)
    model.eval_dataset(dataset)

if __name__ == "__main__":
    main()