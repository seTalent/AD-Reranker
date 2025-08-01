
from models.base_model import BaseModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from scipy.special import softmax
import numpy as np
import torch

from my_datasets.base_dataset import BaseDataset
import os
import json
from tqdm import tqdm
import ast

from peft import PeftModel

import pandas as pd
from openai import OpenAI
from typing import Dict, Union

from accelerate.utils import send_to_device
from models.qwen import extract_evaluation_metrics
from models.MMR5.examples.reranker import QueryReranker
from models.m3docrag import M3DocRAG
class MMR5(M3DocRAG):

    def __init__(self, config):
        super().__init__(config)
        self.reranker = QueryReranker('i2vec/MM-R5')



    def reorder_images(self, question:str, image_paths: list[str]) -> list[str]:
        """
        rerank the pdfs by MM-R5
        """

        if not image_paths or len(image_paths) <= 1:
            return image_paths

        else:
            order, reason = self.reranker.rerank(question, image_paths)
            reorder_images = [image_paths[i] for i in order]

        return reorder_images, reason

    
    @torch.no_grad()
    def predict(self, question, texts=None, page_ids=None ,images=None, history=None):
        reordered_images = images
        if images:
            reordered_images, reason = self.reorder_images(question, images)
        print(images)
        print(reordered_images)
        print(reason)
       
        return super().predict(question, texts, page_ids, reordered_images, history)
    

    @torch.no_grad()
    def batch_predict(self, questions, images, texts):

        reordered_images = [self.reorder_images(q, img) for q, img in zip(questions, images)]
        return super().batch_predict(questions, reordered_images, texts)

    @torch.no_grad()
    def eval_retrieve_dataset(self, dataset: BaseDataset, resume_path=None):
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        else:
            samples = dataset.load_data(use_retrieval=True)

        questions, texts, images = [], [], []
        batch_samples = []

       
        recall_total = 0
        recall_hits = 0
        hit_total = 0
        hit_correct = 0
        mrr_total = 0.0
        mrr_count = 0
        k = 2  # top-k

        for i, sample in enumerate(tqdm(samples)):
            if resume_path and self.ans_key in sample and self.cnt_key in sample:
                continue

            batch_samples.append(sample)
            is_last = (i == len(samples) - 1)

            if len(batch_samples) >= self.batch_size or is_last:
                for sp in batch_samples:
                    question, retrieved_texts, image_paths = dataset.load_sample_retrieval_data(sp)

                    reordered_images = self.reorder_images(question, image_paths)

                    selected_ids = [image_paths.index(img) + 1 for img in reordered_images]  # [1-based]

                    try:
                        evidence_pages = ast.literal_eval(sp["evidence_pages"])
                    except:
                        evidence_pages = []
                    if not evidence_pages:
                        continue

                    # top10 = sp["image-top-10-question"]
                    top10 = [p + 1 for p in sp["image-top-10-question"]]
                    selected_actual_pages = []
                    for idx in selected_ids:
                        if idx <= len(top10):
                            selected_actual_pages.append(top10[idx - 1])  # 1-based -> 0-based

                    recall_total += len(evidence_pages)
                    recall_hits += sum(1 for p in evidence_pages if p in selected_actual_pages[:k])

                    hit_total += 1
                    if any(p in selected_actual_pages[:k] for p in evidence_pages):
                        hit_correct += 1

                    for rank, pid in enumerate(selected_actual_pages):
                        if pid in evidence_pages:
                            mrr_total += 1.0 / (rank + 1)
                            mrr_count += 1
                            break

                self.clean_up()
                questions.clear()
                texts.clear()
                images.clear()
                batch_samples.clear()

        path = dataset.dump_reults(samples)
        print(f"[MMR5] Save final results to {path}")

    
        if recall_total > 0:
            print(f"Recall@{k}: {recall_hits / recall_total:.4f}")
        if hit_total > 0:
            print(f"Hit@{k}: {hit_correct / hit_total:.4f}")
        if mrr_count > 0:
            print(f"MRR: {mrr_total / mrr_count:.4f}")



