
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_datasets.qwen_dataset import QwenDataset

import hydra
from models.qwen import MyModel
import json
from PIL import Image
from datasets import DatasetDict, Dataset

import time


def json_to_dataset(data):
    # read json file
    image_paths = [item['image_path'] for item in data]
    problems = [item['problem'] for item in data]
    solutions = [item['solution'] for item in data]

    images = [[Image.open(image).convert('RGBA') for image in image_path] for image_path in image_paths]

    dataset_dict = {
        'image': images,
        'problem': problems,
        'solution': solutions
    }

    dataset = Dataset.from_dict(dataset_dict)
    dataset_dict = DatasetDict({
        'train': dataset
    })
    return dataset_dict

def save_dataset(dataset_dict, save_path):
    dataset_dict.save_to_disk(save_path)

def load_dataset(save_path):
    # load DatasetDict
    return DatasetDict.load_from_disk(save_path)

def save_2_json(datas):
    with open("./data/sft_data.json", 'w') as f:
        json.dump(datas, f,ensure_ascii=True, indent=4)




@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    dataset = QwenDataset(cfg.dataset,cfg.models.refiner)
    model =  MyModel(cfg.models)
    sft_datas = model.build_sft_dataset(dataset=dataset)
    save_2_json(datas=sft_datas)



if __name__ == "__main__":
    main()