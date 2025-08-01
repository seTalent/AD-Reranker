import torch
import re
class BaseModel():
    def __init__(self, config):
        """
        Base model constructor to initialize common attributes.
        :param config: A dictionary containing model configuration parameters.
        """
        self.config = config
        
    def predict(self, question, texts = None, images = None, history = None):
        pass
    
    def clean_up(self):
        torch.cuda.empty_cache()
        
    def process_message(self, question, texts, images, history):
        if history is not None:
            assert(self.is_valid_history(history))
            messages = history
        else:
            messages = []
        
        if texts is not None:
            messages.append(self.create_text_message(texts, question))
        if images is not None:
            messages.append(self.create_image_message(images, question))
        
        if (texts is None or len(texts) == 0) and (images is None or len(images) == 0):
            messages.append(self.create_ask_message(question))
        
        return messages
    
    def is_valid_history(self, history):
        return True
    
    def _extract_info(self, text):
        all_thinks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        think = all_thinks[0].strip() if len(all_thinks) >= 1 else ''

        all_answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
        answer = all_answers[0].strip() if len(all_answers) >= 1 else ''

        return think, answer