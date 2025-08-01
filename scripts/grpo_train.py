
import os

import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_datasets.qwen_dataset import QwenDataset
import hydra

from trl import GRPOTrainer, GRPOConfig

from datasets import Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import re
from peft import LoraConfig, get_peft_model, PeftModel
from vlm_rft_trainer.grpo_trainer import Qwen2VLGRPOTrainer
from models.qwen import AnswerModel
import requests
from transformers.trainer_utils import get_last_checkpoint

import random
import numpy as np
import torch
import transformers

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



answer_model = None

def format_reward_func(completions, **kwargs):
    # print("***************************************")
    # print(f"recerived:\n completions: {completions}\n kwargs: {kwargs}")
    # print("**************************************")
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    for content in completions:
        content_text = content[0]['content'].strip()  
        content_text = re.sub(r'^assistant\s*\n?', '', content_text, flags=re.IGNORECASE)
        content[0]['content'] = content_text
    matches = [re.match(pattern, content[0]['content'], re.DOTALL) for content in completions]
    
    return [1.0 if match else 0.0 for match in matches]



# batch_size * G
def llm_answer_reward(completions, **kwargs):
 


    rewards = answer_model.call(completions, kwargs)

    return rewards



# def llm_answer_reward(completions, **kwargs):
#     try:
#         response = requests.post(
#             "http://localhost:8000/reward", 
#             json={"completions": completions, "kwargs": kwargs},
#             timeout=10
#         )
#         response.raise_for_status()
#         result = response.json()
#         if "rewards" in result:
#             return result["rewards"]
#         else:
#             print("Error from server:", result.get("error", "Unknown error"))
#             return [0.0] * len(completions)
#     except Exception as e:
#         print("HTTP error:", e)
#         return [0.0] * len(completions)


@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    set_seed(cfg.seed)
    # transformers.utils.logging.set_verbosity_info()

    global answer_model
    is_eval = False
    if answer_model == None:
        answer_model = AnswerModel(cfg.models.answer, is_eval)
    
    
    dataset = QwenDataset(cfg.dataset, cfg.models.refiner)
    dataset_train  = dataset.prepare_train_data()

    # batch_size = 
    # model_name = 'Qwen2.5-VL-3B-Instruct'
    model_name = '/data/share_weight/Qwen2.5-VL-3B-Instruct'
    # model_name = '/data/share_weight/Qwen2.5-VL-7B-Instruct'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name,
                                                               torch_dtype='bfloat16',
                                                               device_map='auto',
                                                            #    device_map={"": "cuda:0"},
                                                               trust_remote_code=True)
    # model = Qwen2VLForConditionalGeneration.from_pretrained(model_name,
    #                                                         torch_dtype='bfloat16',
    #                                                         device_map='auto',
    #                                                     #    device_map={"": "cuda:0"},
    #                                                         trust_remote_code=True
    #                                                         )

    # print(dataset_train)
    model.train()

    lora_config = LoraConfig(

        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias='none',
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )

    train_args = GRPOConfig(

        # use_vllm = True, # use vLLM for fast inference!
        learning_rate=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=True,
        # vllm_mode='server',
        
        # fp16=False,
        per_device_train_batch_size=1,  
        gradient_accumulation_steps= cfg.models.answer.G,#cfg.models.answer.G // cfg.models.answer.BS,  #
        num_generations=cfg.models.answer.G,  # Decrease if out of memory
        max_prompt_length=4096,
        max_completion_length=1024,
        # num_train_epochs=1,  # Set to 1 for a full training run
        max_steps=1000,
        save_steps=10,
        max_grad_norm=0.1,
        report_to='tensorboard',  # Can use Weights & Biases
        save_strategy="steps",
        output_dir="outputs-G6-3B-2",
        logging_dir="log",
        save_total_limit=3
    )
   
    tokenizer = AutoProcessor.from_pretrained(model_name,
                                              trust_remote_code=True, min_pixels=cfg.models.refiner.all_pixels, max_pixels=cfg.models.refiner.all_pixels
    )



    
    trainer = Qwen2VLGRPOTrainer(

        model = model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            llm_answer_reward
        ],
        args=train_args,
        train_dataset=dataset_train,
        peft_config=lora_config
     
    )


 
    ckpt_path = get_last_checkpoint(train_args.output_dir)
    if ckpt_path:
        trainer.train(resume_from_checkpoint=ckpt_path)
    else:
        trainer.train()

if __name__ == "__main__":
    main()