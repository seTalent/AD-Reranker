from my_datasets.base_dataset import BaseDataset
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple


@dataclass
class QwenContent():
    image: Image
    image_path: str
    page_id: int
    txt: str


class QwenDataset(BaseDataset):

    def __init__(self, config, model_config):
        super().__init__(config)
        self.config = config
        self.samples = None

        self.page_id_key = self.config.r_image_key
        self.query_key = self.config.question_key
        # self.answer_key = self.config.
        self.max_pages = self.config.top_k

        self.model_config = model_config

        self.system_message = self.model_config.system_prompt
        self.user_message = self.model_config.user_message
        self.few_shot_message = self.model_config.few_shot_message

    def load_qwen_data(self, use_retrieval):
        self.samples = self.load_data(use_retrieval=use_retrieval)
        return self.samples

    def prepare_qwen_data(self, use_retrieval=True) -> List[Tuple[str, str, List[Image], List[int]]]:
        '''
        @return: List[(query, answer, image_paths, page_ids)]
        '''
        if self.samples is None:
            self.load_qwen_data(use_retrieval=use_retrieval)

        prepared_data = []
        for sample in tqdm(self.samples):
            query, answer, qwen_content_list = self.load_processed_content_sample(sample, disabled_load_image=False)
            images = [content.image_path for content in qwen_content_list if content.image_path is not None]
            page_ids = [content.page_id for content in qwen_content_list if content.page_id is not None]
        
            prepared_data.append((query, answer, page_ids, images))

        return prepared_data

    def load_processed_content_sample(self, sample, disabled_load_image=False) -> Tuple[str,str, List[QwenContent]]:
        assert self.page_id_key in sample, f"{self.page_id_key} not in sample keys!"

        doc_id = self.EXTRACT_DOCUMENT_ID(sample)
        page_ids = sample[self.page_id_key]
        query = sample[self.query_key]

        answer = sample['answer']
        content_list = []
        if not disabled_load_image:
            for id in page_ids:
                img_file = self.IM_FILE(doc_id, id)
                img = self.load_image(img_file)
                txt_file = self.TEXT_FILE(doc_id, id)
                # TODO: append txt
                txt = self.load_txt(txt_file)
                txt = ""
                content_list.append(QwenContent(img, img_file, id, txt))

                if len(content_list) >= self.max_pages:
                    break
        return (query, answer, content_list)
    def create_image_message(self, question: str , page_ids: list[int], images: list[str]):
        '''
        images: image_paths
        create image message, every message is formatted as [page_id]: <img>
        '''
        assert len(page_ids) == len(images), f"the len of page_ids is {len(page_ids)} mismatch the len of images(path) {len(images)}"
        content = []
        prompt_user = self.user_message.format(question=question)
        content.append({"type": "text", "text": prompt_user})
        image_paths = []
        #[page_id]: {document}
        for i in range(len(page_ids)):
            content.append({"type": "text", "text": f'[{i+1}]:'}) #[page_id] :
            content.append({"type": "image", "image": images[i]}) # document_path

            image_paths.append(images[i])
            # content.append({"type": "text", "text": '\n'})

        # content.append({"type": "text", "text": self.few_shot_message})

        message = {"role": "user", "content": content}

        return message, image_paths

    def process_message(self, question, texts, page_ids, images, history) -> list:
        messages = []
        image_paths = []
        #system message
        messages.append({"role" : "system", "content" : self.system_message + '\n' + self.few_shot_message})
        #add documents and query
        message, image_path  = self.create_image_message(question, page_ids, images)
        # messages.append(self.create_image_message(question, page_ids, images))

        messages.append(message)
        image_paths.extend(image_path) #list
        
        return messages, image_paths, page_ids


    def prepare_train_data(self):
        results = []
        prepared_data = self.prepare_qwen_data(True) #[query, page_id, image_path] -> {'prompt':[{'role': 'system', 'content':...},{{'role': 'user', 'content':...}],         'image':img_pil, 'solution':answer_sample,}
        for (query, answer,  page_ids, images) in prepared_data:
            # if query == 'Which creation has more steps, To remove the drop-in trim piece or to remove the crisper?': #[debug]
            #     continue
            message, image_paths, page_nums = self.process_message(query,"", page_ids, images, None)

            results.append({'prompt': message , 'image': image_paths, 'answer': answer, 'page_ids': page_ids, 'query' : query})
        
        return results

    def prepare_test_data(self):
        results = []
        prepared_data = self.prepare_qwen_data(True) #[query, page_id, image_path] -> {'prompt':[{'role': 'system', 'content':...},{{'role': 'user', 'content':...}],         'image':img_pil, 'solution':answer_sample,}
        for (query, answer,  page_ids, images) in prepared_data:
            message, image_paths, page_nums = self.process_message(query,"", page_ids, images, None)

            results.append({'prompt': message , 'image': image_paths, 'answer': answer, 'page_ids': page_ids, 'query' : query})
            
        return results



    def load_sample_retrieval_data(self, sample):
        content_list = self.load_processed_content(sample, disable_load_image=True)
        question: str = sample[self.config.question_key]
        texts = []
        images = []
        if self.config.use_mix:
            if self.config.r_mix_key in sample:
                for page in sample[self.config.r_mix_key][:self.config.top_k]:
                    if page in sample[self.config.r_image_key]:
                        origin_image_path = ""
                        origin_image_path = content_list[page].image_path
                        images.append(origin_image_path)
                    if page in sample[self.config.r_text_key]:
                        texts.append(content_list[page].txt.replace("\n", ""))
        else:
            if self.config.r_text_key in sample:
                for page in sample[self.config.r_text_key][:self.config.top_k]:
                    texts.append(content_list[page].txt.replace("\n", ""))
            if self.config.r_image_key in sample:
                for page in sample[self.config.r_image_key][:self.config.top_k]:
                    origin_image_path = ""
                    origin_image_path = content_list[page].image_path
                    images.append(origin_image_path)

        return question, texts, images
        
    def load_sample_retrieval_data_qwen(self, sample):
        query, answer, qwen_content_list =  self.load_processed_content_sample(sample, False)
        images = [content.image_path for content in qwen_content_list if content.image_path is not None]
        page_ids = [content.page_id for content in qwen_content_list if content.page_id is not None]
        texts = [content.txt for content in qwen_content_list if content.txt is not None]
        return (query, answer, images,texts, page_ids)
    