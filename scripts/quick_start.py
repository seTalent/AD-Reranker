import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_datasets.base_dataset import BaseDataset

import hydra
from models.qwen import MyModel

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    model =  MyModel(cfg.models)
    result = model.quick_start(
    query="Which document contains the student's attendance details?",
    image_paths=["img1.png", "img2.png", "img3.png", "img4.png"]
    )
    print(result)


if __name__ == "__main__":
    main()