
from models.base_model import BaseModel

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
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
class MyModel(BaseModel):
    def __init__(self, config):
        self.refiner = RefinerModel(config.refiner)
        self.answer_model = AnswerModel(config.answer)
        self.batch_size = config.answer.decode_batch_size
        self.config = config
        
        self.eval_client = OpenAI(
            api_key=self.config.answer.api_key,
            base_url="https://chatapi.littlewheat.com/v1"
            # base_url="https://api.moonshot.cn/v1"
        )

        self.eval_prompt = config.answer.eval_prompt

        self.eval_model_name = config.answer.eval_model_name
        self.ans_key = self.config.answer.answer_key
        self.cnt_key = self.config.answer.cnt_key


    @torch.no_grad()
    def predict_dataset(self, dataset: BaseDataset, resume_path=None):
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        else:
            samples = dataset.load_data(use_retrieval=True)

        questions, texts, images = [], [], []
        batch_samples = []

        for i, sample in enumerate(tqdm(samples)):
            if resume_path and self.ans_key in sample and self.cnt_key in sample:
                continue
            
            batch_samples.append(sample)
          
            is_last = (i == len(samples) - 1)
            if len(batch_samples) >= self.batch_size or is_last:
            
                for sp in batch_samples:
                    question, retrieved_texts, image = dataset.load_sample_retrieval_data(sp)
                    questions.append(question)
                    texts.append(retrieved_texts)

                    images.append(image)
        
                selected_ids, reasons, doc_cnts = self.refiner.batch_select(questions, images, texts)
            
                final_answers, final_messages = self.answer_model.batch_answer(
                    questions, images, selected_ids, reasons
                )
                print(f"final_answers= {final_answers}")
                for (j, (final_answer, doc_cnt)) in enumerate(zip(final_answers, doc_cnts)):
                    global_idx = i - len(batch_samples) + 1 + j

                    print(final_answer)
                    try:
                        samples[global_idx][self.cnt_key] = doc_cnt
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
    @torch.no_grad()
    def eval_dataset(self, dataset:BaseDataset, resume_path=None):
        samples, ans_path = dataset.load_latest_results()
        samples_with_answer = []
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
        # samples_with_cnt = pd.DataFrame(samples_with_cnt)
        path = os.path.join(dataset.config.result_dir, "results.txt")
        with open(path, "a") as file:
            file.write("\nEvaluation Results Summary:\n")
            file.write(f"Result file: {ans_path}\n")
            file.write(f"Average Binary Correctness: {samples_with_answer['binary_correctness'].mean():.3f}\n")
            file.write(f"Average Document Count: {samples_with_answer[self.cnt_key].mean():.3f}\n")

        print(f"Save results to {path}.")

    @torch.no_grad()
    def cal_token_dataset(self, dataset:BaseDataset, resume_path=None):
        
        if resume_path:
                assert os.path.exists(resume_path)
                with open(resume_path, 'r') as f:
                    samples = json.load(f)
        else:
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

                selected_ids, reasons, doc_cnts = self.refiner.batch_select(questions, images, texts)

                
                token_stats = self.answer_model.compute_token_count(
                    questions, images, selected_ids, reasons
                )
                print(f"[DEBUG] : token_states :{token_stats}")

                for j, (stat, doc_cnt) in enumerate(zip(token_stats, doc_cnts)):
                    global_idx = i - len(batch_samples) + 1 + j
                    samples[global_idx]['token_text'] = stat['text_tokens']
                    samples[global_idx]['token_image'] = stat['image_tokens']
                    samples[global_idx]['token_total'] = stat['total_tokens']
                    samples[global_idx][self.cnt_key] = doc_cnt
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
                temperature=self.config.answer.temperature,
                max_tokens=self.config.answer.max_new_tokens,
            )
            result = response.choices[0].message.content
            # messages.append(self.create_ans_message(result))

            return result, messages
        
    @torch.no_grad()
    def build_sft_dataset(self, dataset):
        print("[Build SFT Dataset]")
        system_message = self.config.refiner.system_prompt
        user_message = self.config.refiner.user_message
        few_shot_message = self.config.refiner.few_shot_message


        samples = dataset.load_data(use_retrieval=True)



        datas = []
        
        for i, sample in enumerate(tqdm(samples)):
                if len(datas) >= 10:
                    break
                question, retrieved_texts, image = dataset.load_sample_retrieval_data(sample)
                messages = []
                messages.append({"role": "system", "content":system_message + '\n' + few_shot_message})
                content = []
                image_paths = []
                content.append({"type":"text", "text": user_message.format(question=question)})
                for i in range(0, len(image)):
                    content.append({"type": "text", "text": f'[{i+1}]:'}) #[page_id] :
                    content.append({"type": "image", "image": image[i]}) # document_path

                    image_paths.append(image[i])
                messages.append({"role": "user", "content": content})
    
                selected_ids, reasons, _ = self.refiner.batch_select([question], [image], [""])

                if isinstance(selected_ids, list) and reasons!='':
                    gt = f"<think>{reasons[0]}</think> <answer>{selected_ids[0]}</aswer>"
        

                    data = {"problem": messages, "image_path":image_paths, "solution": gt}
                    datas.append(data)
        print(f"[SFT DATA BUILDED]")
        return datas
        # print(f"SFT Dataset[0]:{datas[0]}")
        
class RefinerModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.config.model_id, torch_dtype='auto', device_map='auto', trust_remote_code=True).eval()
        # self.processor = AutoProcessor.from_pretrained(self.config.model_id, trust_remote_code=True, min_pixels=self.config.min_pixels, max_pixels=self.config.max_pixels)

        self.processor = AutoProcessor.from_pretrained(self.config.model_id, trust_remote_code=True, min_pixels=self.config.all_pixels, max_pixels=self.config.all_pixels)
        print("[Refiner]:")
        if getattr(self.config, "use_grpo_model", True):
            print(f"[LORA]Load lora model from {self.config.lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.lora_path,
                torch_dtype='auto'
            ).eval()

        
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

        self.system_message = self.config.system_prompt

        self.user_message = self.config.user_message

        self.few_shot_message= self.config.few_shot_message

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
        #output_example <think></think>, <answer></answer>
        content.append({"type": "text", "text": self.few_shot_message})

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
    
    def predict(self, question, texts=None, page_ids=None ,images=None, history=None):
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


    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True

    @torch.no_grad()
    def batch_select(self, questions, images, texts):
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
        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answers = []
        thinks = []
        page_cnt = []
        for output_text in output_texts:
            think, answer = self._extract_info(output_text)
            think=''    
            all_numbers = re.findall(r'\d+', answer)
            result = [int(n) for n in all_numbers if n in {'1', '2', '3', '4'}]
            
            if result == []: #debug
                result = [1, 2, 3, 4]
            result = [1, 2, 3, 4]
            page_cnt.append(len(result))
            answer = result
            answers.append(answer)
            thinks.append(think)
        return (answers, thinks, page_cnt)

    @torch.no_grad()
    def build_sft_data(self, question, images, texts):
        messages = []
        message = self.process_message(question,None,None,images,None)
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
        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answers = []
        thinks = []
        page_cnt = []
        for output_text in output_texts:
            think, answer = self._extract_info(output_text)
            all_numbers = re.findall(r'\d+', answer)
            result = [int(n) for n in all_numbers if n in {'1', '2', '3', '4'}]
            
            if result == []: #DEBUG
                result = [1, 2, 3, 4]
            page_cnt.append(len(result))
            answer = result
            answers.append(answer)
            thinks.append(think)
        return (answers, thinks, page_cnt)
    
    
        
class AnswerModel(BaseModel):
    def __init__(self, config, is_eval=True):
        self.config = config
        print(f"[Answer] :\n {self.config}")
        # last_gpu = f"cuda:{torch.cuda.device_count() - 1}"
        if not is_eval:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype
            )
            quant_config = bnb_config
        else:
            quant_config = None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.config.model_id, torch_dtype='bfloat16', 
                                                                        device_map='auto',
                                                                        # device_map={"":last_gpu},
                                                                        trust_remote_code=True,
                                                                          quantization_config=quant_config
                                                                        ).eval()

        # self.processor = AutoProcessor.from_pretrained(self.config.model_id, trust_remote_code=True, min_pixels=self.config.min_pixels, max_pixels=self.config.max_pixels)

        self.processor = AutoProcessor.from_pretrained(self.config.model_id, trust_remote_code=True, min_pixels=self.config.all_pixels, max_pixels=self.config.all_pixels, padding_side='left')

        self.system_prompt = self.config.system_prompt
        self.user_message = self.config.user_message
        # self.few_shot_message = self.config.few_shot_message

        
        self.train_client = OpenAI(
            api_key=self.config.api_key,
            base_url="https://chatapi.littlewheat.com/v1"
            # base_url="https://api.moonshot.cn/v1"
        )

        self.train_prompt = self.config.train_prompt

        self.train_model_name = self.config.train_model_name
   
    
        self.batch_size = self.config.BS
        self.num_generation = self.config.G
        
        self.judge = None

   
        


    @torch.no_grad()
    def call(self, completions, kwargs)-> list[int]:

        prompts = []
        querys = kwargs['query'] #self.num_generation * batch_size : 
        answers = kwargs['answer'] # 


        image_paths = kwargs['image'] #  8* list
        doc_counts = []

        for (query, content, image_path) in zip(querys, completions, image_paths):

            text = content[0]['content'].strip()
            text = re.sub(r'^assistant\s*\n?', '', text, flags=re.IGNORECASE)

            # print(f"text = {text}")
            think, answer = self._extract_info(text)
            page_list = None

            all_numbers = re.findall(r'\d+', answer)
            result = [int(n) for n in all_numbers if n in {'1', '2', '3', '4'}]
            doc_counts.append(len(result))
            if result == []:
                result = [1, 2, 3, 4]
            page_list = result

            prompt = self.create_image_message(query, think, page_list, image_path)
            prompts.append(prompt)
        
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in prompts
        ]

        image_inputs, video_inputs = process_vision_info(prompts)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )



        inputs = send_to_device(inputs, self.model.device)


        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.device}, shape: {v.shape}")
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
           generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        rewards = []

        for i in range(self.batch_size):
            text = output_texts[i*self.num_generation: (i+1)*self.num_generation]
            gt = answers[i*self.num_generation]
            question = querys[i*self.num_generation]
            doc_count = doc_counts[i*self.num_generation: (i+1)*self.num_generation]
            reward = self.get_reward(question, text, gt, doc_count)
            rewards.extend(reward)
        return rewards 
    



    def get_reward(self, question:str ,answers:list[str], gt:str, doc_counts: list[int]) -> list[float]:
        rewards_1 = []
        
        # for answer in answers:
        prompt = self.train_prompt.format(question=question, gt=gt, answer=answers)
        messages = [
            {"role": "user", "content": prompt}
        ]
        temp = 0
        while temp < 3:
            temp += 1
            try:
                result = None
                response = self.train_client.chat.completions.create(
                    model=self.train_model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_new_tokens
                )
                result = response.choices[0].message.content
                print(f"result = {result}")
                try:
                   
                    reward_dict = json.loads(result)
                except json.JSONDecodeError:
                    
                    result_fixed = result.replace("'", '"')
                    reward_dict = json.loads(result_fixed)
                rewards_1.extend(reward_dict['score'])

                if len(rewards_1) != len(answers):
                    rewards_1 = []
                    raise ConnectionError
                # rewards_1.append(float(reward_dict['score']))
                break  

            except ConnectionError:
                if temp < 3:
                    continue
                else:
                    rewards_1 = []
                    rewards_1.extend([0.0] * len(answers))
                    break
            except Exception as e:
                print(f"***********\nerror: {e} when getting reward. Raw result:{result}\n*************")
                rewards_1 = []
                rewards_1.extend([0.0]*len(answers))
                break

        rewards_2 = []
        beta = 0.5   # Weight for zero-document penalty
        gamma = 0.5  # Penalty value for zero documents (e.g., 0.3 to 0.5 is good for 0-1 reward_1 scale)

        # New parameters for more aggressive decay
        max_positive_reward_val = 0.6 
        min_positive_reward_val = 0.1 
        
       
        decay_exponent = 1.3 

        for doc_count in doc_counts:
            doc_count_reward = 0.0
            zero_penalty = 0.0
            if doc_count == 0:
                zero_penalty = -gamma
            else:

                if doc_count == 1:
                    doc_count_reward = max_positive_reward_val
                else:

                    doc_count_reward = max_positive_reward_val / (doc_count ** decay_exponent)
                doc_count_reward = max(doc_count_reward, min_positive_reward_val)

            combined_reward_penalty = (beta * zero_penalty) + doc_count_reward
            rewards_2.append(combined_reward_penalty)
            

        return [0.5*r1 + 0.5*r2 for r1,r2 in zip(rewards_1, rewards_2)]

    

        
    def create_image_message(self, question: str, think: str, selected_ids: list[int], images: list[str]):
        if selected_ids == None or selected_ids == [None]:
            selected_ids = []

        if isinstance(selected_ids, int):
            selected_ids = [selected_ids]

        selected_ids = list(dict.fromkeys(selected_ids))
        messages = []
        messages.append({"role" : "system", "content" : self.system_prompt})
        
        content = []
        prompt_user = self.user_message.format(question=question, reason=think)
        content.append({"type": "text", "text": prompt_user})
        #[page_id]: {document}
        for i in range(len(selected_ids)):
            try:
                image_path = images[selected_ids[i]-1] #list out of range
            except Exception as e:
                continue
            content.append({"type": "text", "text": f'[{selected_ids[i]}] :'}) #[page_id] :
            # content.append({"type": "text", "text": f'[{page_ids[i]}] : '}) #[page_id] :
            content.append({"type": "image", "image": image_path}) # document_path
        #output_example <think></think>, <answer></answer>
        # content.append({"type": "text", "text": self.few_shot_message})

        message = {"role": "user", "content": content}
        messages.append(message)
        return messages
    
    @torch.no_grad()
    def batch_answer(self, questions, images, selected_ids, reasons):
        prompts = []

        for (question, image_path, selected_id, reason) in zip(questions, images, selected_ids, reasons):
            message = self.create_image_message(question, reason, selected_id, image_path)
            prompts.append(message)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in prompts
        ]


        image_inputs, video_inputs = process_vision_info(prompts)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = send_to_device(inputs, self.model.device)
        # inputs = inputs.to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answers = self.processor.batch_decode(
           generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return answers, []
    def extract_evaluation_metrics(eval_str: str) -> Dict[str, Union[float, int]]:
        try:
            start_index = eval_str.find('{') 
            end_index = eval_str.rfind('}') + 1 
            eval_str = eval_str[start_index:end_index]
            metrics = json.loads(eval_str)
            return {
                'score': float(metrics.get('score', 0))
            }
        except json.JSONDecodeError as e:
            return {
                'score': 0.0
            }
        except Exception as e:
            return {
                'score': 0.0
            }
        
    @torch.no_grad()
    def compute_token_count(self, questions, images, selected_ids, reasons):
        token_counts = []
        for (question, image_paths, selected_id, reason) in zip(questions, images, selected_ids, reasons):
          
            message = self.create_image_message(question, reason, selected_id, image_paths)
            text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

      
            input_ids = self.processor.tokenizer(text, return_tensors='pt')['input_ids']
            text_token_count = input_ids.shape[1]

       
            image_token_count = 512 * len(selected_id)

    
            total_token_count = text_token_count + image_token_count

            token_counts.append({
                "text_tokens": text_token_count,
                "image_tokens": image_token_count,
                "total_tokens": total_token_count
            })
        return token_counts
        pass



def extract_evaluation_metrics(eval_str: str) -> Dict[str, Union[float, int]]:
    try:
        start_index = eval_str.find('{') 
        end_index = eval_str.rfind('}') + 1 
        eval_str = eval_str[start_index:end_index]
        metrics = json.loads(eval_str)
        return {
            'binary_correctness': int(metrics.get('binary_correctness', 0))
        }
    except json.JSONDecodeError as e:
        return {
            'binary_correctness': 0
        }
    except Exception as e:
        return {
            'binary_correctness': 0
        }
    




class QwenVLAnswerModel:
    def __init__(self, model_id: str, max_new_tokens: int = 1024):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        
        #Qwen2.5 or Qwen 2 -VL-7B-Instruct
        if "2.5" in model_id:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", trust_remote_code=True, torch_dtype='auto'
            ).eval()
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", trust_remote_code=True, torch_dtype='auto'
            ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )

    def create_prompt(self, question: str, images: list[str]) -> list[dict]:
        """build multi-modal prompt：question + [1]: <image> + ..."""
        content = [{"type": "text", "text": question}]
        for i, img_path in enumerate(images):
            content.append({"type": "text", "text": f"[{i+1}]:"})
            content.append({"type": "image", "image": img_path})
        return [{"role": "user", "content": content}]

    @torch.no_grad()
    def answer(self, question: str, images: list[str]) -> str:
        messages = self.create_prompt(question, images)
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = send_to_device(inputs, self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        output_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        decoded = self.processor.batch_decode(
            output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0]


    
    