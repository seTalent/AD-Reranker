import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_datasets.base_dataset import BaseDataset
import hydra

from models.qwen import MyModel

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    # cfg.eval_agent.agent = hydra.compose(config_name="agent/"+cfg.eval_agent.agent, overrides=[]).agent
    # cfg.eval_agent.model = hydra.compose(config_name="model/"+cfg.eval_agent.model, overrides=[]).model
    dataset = BaseDataset(cfg.dataset)
    model = MyModel(config=cfg.models)
    model.eval_dataset(dataset)


    # eval_agent = Agent(cfg.eval_agent)
    # eval_agent.eval_dataset(dataset)
    
if __name__ == "__main__":
    main()