
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

from peft import PeftModel

import pandas as pd
from openai import OpenAI
from typing import Dict, Union

from accelerate.utils import send_to_device
from models.qwen import extract_evaluation_metrics

class M3DocRAG(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.ans_key = config.answer.answer_key
        self.config = config.m3docrag
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.config.model_id, torch_dtype='auto', device_map='auto', trust_remote_code=True).eval()
        self.processor = AutoProcessor.from_pretrained(self.config.model_id, trust_remote_code=True, min_pixels=self.config.all_pixels, max_pixels=self.config.all_pixels)
        
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }

        self.eval_client = OpenAI(
            api_key=self.config.api_key,
            base_url="https://chatapi.littlewheat.com/v1"
        )

        self.system_message = self.config.system_prompt

        self.user_message = self.config.user_message

        self.eval_prompt = self.config.eval_prompt

        self.eval_model_name = self.config.eval_model_name

        self.batch_size = self.config.decode_batch_size

        print(f"[M3DOCRAG CONFIG]: {self.config}")



    def create_text_message(self, question, texts):

        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
        
    def create_image_message(self, question: str , images: list[str]):
        '''
        images: image_paths
        create image message, every message is formatted as [page_id]: <img>
        '''

        content = []
        prompt_user = self.user_message.format(question=question)
        content.append({"type": "text", "text": prompt_user})
        #[page_id]: {document}
        for i in range(len(images)):
            content.append({"type": "text", "text": f'[{i+1}] :'}) 
            content.append({"type": "image", "image": images[i]}) # document_path

        message = {"role": "user", "content": content}

        return message

    def process_message(self, question, texts, page_ids, images, history) -> list:
        if history is not None:
            assert(self.is_valid_history(history))
            messages = history
        else:
            messages = []
        #system message
        messages.append({"role" : "system", "content" : self.system_message})
        #add documents and query
        messages.append(self.create_image_message(question, images))

        return messages
    
    @torch.no_grad()
    def predict(self, question, texts=None, page_ids=None ,images=None, history=None, reason=None):
        self.clean_up()

        messages = self.process_message(question, texts, page_ids, images, history)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_space=False
        )
        messages.append(self.create_ans_message(output_text))

        self.clean_up()
        return output_text, messages


    @torch.no_grad()
    def batch_predict(self, questions, images, texts):
        messages = []
        for (question, image_path, text) in zip(questions, images, texts):
            message = self.process_message(question,None,None,image_path,None)
            messages.append(message)

        txts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=txts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt'
        )

        inputs = inputs.to("cuda")


        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answers = output_texts

        return answers

    @torch.no_grad()
    def predict_dataset(self, dataset: BaseDataset, resume_path=None):
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        else:
            samples = dataset.load_data(use_retrieval=True)
        print(f"[CASE STUDY]")

        questions, texts, images = [], [], []
        batch_samples = []
        samples = [sample for sample in samples if sample['question']=='How many percent of Germanwings focused tweets are in English?']
        for i, sample in enumerate(tqdm(samples)):
            if resume_path and self.ans_key in sample and self.cnt_key in sample:
                continue
            
            batch_samples.append(sample)
          
            is_last = (i == len(samples) - 1)
            if len(batch_samples) >= self.batch_size or is_last:
                answers = []
                for sp in batch_samples:
                    question, retrieved_texts, image = dataset.load_sample_retrieval_data(sp)
                    questions.append(question)
                    texts.append(retrieved_texts)
                    images.append(image)
                    answer, _ = self.predict(question,None,None,image,None)
                    print(f"answer= {answer}")
                    answers.append(answer)


                for (j, (final_answer)) in enumerate(zip(answers)):
                    global_idx = i - len(batch_samples) + 1 + j

                    try:
                        samples[global_idx][self.ans_key] = json.loads(final_answer)['Answer']
                    except:
                        samples[global_idx][self.ans_key] = final_answer
                self.clean_up()

                questions.clear()
                texts.clear()
                images.clear()
                batch_samples.clear()

        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}")

    def eval_dataset(self, dataset:BaseDataset, resume_path=None):
        samples, ans_path = dataset.load_latest_results()
        samples_with_answer = []
        print("eval ...")
        for sample in tqdm(samples):
            try:
                question = sample[dataset.config.question_key]
                answer = sample[self.ans_key]
                gt = sample[dataset.config.gt_key]
                result = self.eval(question, answer, gt)
                sample['binary_correctness'] = result.get('binary_correctness', None)
                samples_with_answer.append(sample)
            except Exception as e:
                print(f"Error evaluating sample: {str(e)}")
        ans_file_path_name = ans_path[:-5] + "_results.json"
        with open(ans_file_path_name, 'w') as file:
            json.dump(samples_with_answer, file, indent=4)

        samples_with_answer = pd.DataFrame(samples_with_answer)
        path = os.path.join(dataset.config.result_dir, "results.txt")
        with open(path, "a") as file:
            file.write("\nEvaluation Results Summary:\n")
            file.write(f"Result file: {ans_path}\n")
            file.write(f"Average Binary Correctness: {samples_with_answer['binary_correctness'].mean():.3f}\n")
            # file.write(f"Average Document Count: {samples_with_answer[self.cnt_key].mean():.3f}\n")

        print(f"Save results to {path}.")

    @torch.no_grad()
    def cal_token_dataset(self, dataset: BaseDataset):

        samples = dataset.load_data(use_retrieval=True)

        questions, texts, images = [], [], []
        batch_samples = []

        all_token_stats = []

        for i, sample in enumerate(tqdm(samples)):
            batch_samples.append(sample)

            is_last = (i == len(samples) - 1)
            if len(batch_samples) >= self.batch_size or is_last:
                questions.clear()
                texts.clear()
                images.clear()
                for sp in batch_samples:
                    question, retrieved_texts, image = dataset.load_sample_retrieval_data(sp)
                    questions.append(question)
                    texts.append(retrieved_texts)
                    images.append(image)


                # 🔍 Token统计，无需推理
                token_stats = self.compute_token_count(
                    questions, images
                )
                print(f"[DEBUG] : token_states :{token_stats}")

                for (j, stat) in enumerate(token_stats):
                    global_idx = i - len(batch_samples) + 1 + j
                    samples[global_idx]['token_text'] = stat['text_tokens']
                    samples[global_idx]['token_image'] = stat['image_tokens']
                    samples[global_idx]['token_total'] = stat['total_tokens']
                    all_token_stats.append(stat)

                batch_samples.clear()


        result_path = dataset.dump_reults(samples)
        print(f"[Token Analysis] Save token-annotated results to {result_path}")

   
        token_df = pd.DataFrame(all_token_stats)
        summary_path = os.path.join(dataset.config.result_dir, "token_stats.txt")
        with open(summary_path, "a") as f:
            f.write("\nToken Stats Summary:\n")
            f.write(f"Result file: {result_path}\n")
            f.write(f"Average Total Tokens: {token_df['total_tokens'].mean():.1f}\n")
            f.write(f"Average Text Tokens: {token_df['text_tokens'].mean():.1f}\n")
            f.write(f"Average Image Tokens: {token_df['image_tokens'].mean():.1f}\n")

        print(f"[Token Analysis] Summary written to {summary_path}")

    def eval(self, question, answer, gt):
        prompt = self.eval_prompt.format(question=question, answer=answer, gt=gt)
        try:
            generated_ans, _ = self.eval_call(prompt)
            result = extract_evaluation_metrics(generated_ans)
            return result
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return {"binary_correctness": 0}
        

    def eval_call(self, prompt):
        temp = 0
        messages = [
            {"role": "user",  "content": prompt}
        ]
        while temp <3 :
            temp += 1
            response = self.eval_client.chat.completions.create(
                model=self.eval_model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
            )
            result = response.choices[0].message.content
            messages.append(self.create_ans_message(result))

            return result, messages
        
    def compute_token_count(self, questions, images):
        token_counts = []
        for (question, image) in zip(questions, images):
            message = self.process_message(question,None,None,image, None)
            print(f"[DEBUG] message = {message}")
        
            text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            # text token
            input_ids = self.processor.tokenizer(text, return_tensors='pt')['input_ids']
            text_token_count = input_ids.shape[1]

            # image_token
            image_token_count = 512 * len(images)
            
            total_token_count = text_token_count + image_token_count

            token_counts.append({
                    "text_tokens": text_token_count,
                    "image_tokens": image_token_count,
                    "total_tokens": total_token_count
                })
        return token_counts
        