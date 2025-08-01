import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_datasets.base_dataset import BaseDataset
import hydra

from models.mmr5 import MMR5

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):

    dataset = BaseDataset(cfg.dataset)
    model = MMR5(cfg.models)
    model.eval_dataset(dataset)

    
if __name__ == "__main__":
    main()