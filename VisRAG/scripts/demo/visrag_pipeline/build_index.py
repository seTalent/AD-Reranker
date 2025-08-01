import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils import encode

BASE_DATA_ROOT = "/data/zky_1/codes/agent/MDocAgent/data"

def get_samples_path(dataset_name):
    return os.path.join(BASE_DATA_ROOT, dataset_name, "samples.json")

def get_output_path(dataset_name):
    return os.path.join(BASE_DATA_ROOT, dataset_name, "samples_visrag.json")

def get_image_root(dataset_name):
    # 假设图像目录也在 dataset_name 文件夹下，名为 images 或 img，可根据实际调整
    return os.path.join(BASE_DATA_ROOT, dataset_name, "images")

def get_all_images_for_pdf(doc_id, image_root):
    base_name = os.path.splitext(doc_id)[0]
    image_files = [f for f in os.listdir(image_root) if f.startswith(base_name + "_") and f.endswith(('.jpg', '.png'))]
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
    return image_files

def rerank_sample_by_images(sample, image_root, doc_key, query_key, topk=10):
    global model, tokenizer

    doc_id = sample[doc_key]
    query = sample[query_key]

    image_files = get_all_images_for_pdf(doc_id, image_root)
    if not image_files:
        print(f"[Warning] No images found for: {doc_id}")
        return sample

    image_paths = [os.path.join(image_root, f) for f in image_files]
    images = [Image.open(p).convert("RGB") for p in image_paths]

    query_instruction = "Represent this query for retrieving relevant document: " + query
    with torch.no_grad():
        query_vec = torch.from_numpy(encode(model, tokenizer, [query_instruction])).cuda()

    scores = []
    for idx, img in enumerate(images):
        with torch.no_grad():
            img_vec = torch.from_numpy(encode(model, tokenizer, [img])).cuda()
            sim = torch.matmul(query_vec, img_vec.T).item()
            scores.append((idx, sim))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:topk]
    topk_indices = [i for i, _ in sorted_scores]
    topk_scores = [s for _, s in sorted_scores]

    sample["image-top-10-question"] = topk_indices
    sample["image-top-10-question_score"] = topk_scores

    return sample

def process_all_samples(samples_path, image_root, output_path, doc_key, query_key, topk=10):
    with open(samples_path, 'r') as f:
        samples = json.load(f)

    updated_samples = []
    for sample in tqdm(samples):
        if doc_key not in sample or query_key not in sample:
            print(f"[Skipping] Missing key in sample: {sample}")
            continue
        updated = rerank_sample_by_images(sample, image_root, doc_key, query_key, topk=topk)
        updated_samples.append(updated)

    with open(output_path, 'w') as f:
        json.dump(updated_samples, f, indent=4, ensure_ascii=False)

    print(f"All samples processed. Output saved to {output_path}")

if __name__ == "__main__":
    model_path = 'openbmb/VisRAG-Ret'
    device = 'cuda'

    dataset_name = input("Enter dataset name (e.g., FetaTab, MMLongBench): ").strip()

    samples_path = get_samples_path(dataset_name)
    output_path = get_output_path(dataset_name)

    print(f"Using samples path: {samples_path}")
    print(f"Using output path: {output_path}")

    # 手动输入图片目录，提示默认目录（方便直接回车使用）
    default_image_root = os.path.join(BASE_DATA_ROOT, dataset_name, "images")
    image_root = input(f"Enter path to image directory (default: {default_image_root}): ").strip()
    if image_root == "":
        image_root = default_image_root

    # 校验图片目录是否存在
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")

    # doc_key和query_key依然手动输入
    doc_key = input("Enter the key name for the PDF file (e.g., 'doc_id'): ").strip()
    query_key = input("Enter the key name for the query (e.g., 'question'): ").strip()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model.eval().to(device)
    print("Model loaded.")

    process_all_samples(samples_path, image_root, output_path, doc_key, query_key, topk=10)


# def add_pdfs(pdf_dir):
#     global model, tokenizer, knowledge_base_path
#     model.eval()

#     pdf_file_list = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
#     pdf_file_list = [os.path.join(pdf_dir, f) for f in pdf_file_list]

#     reps_list = []
#     index2img_filename = []

#     for pdf_file_path in pdf_file_list:
#         print(f"Processing {pdf_file_path}")
#         pdf_name = os.path.basename(pdf_file_path)

#         with open(os.path.join(knowledge_base_path, pdf_name), 'wb') as file1:
#             with open(pdf_file_path, "rb") as file2:
#                 file1.write(file2.read())

#         dpi = 200
#         doc = fitz.open(pdf_file_path)
        
#         images = []

#         for page in tqdm.tqdm(doc, desc=f"Processing {pdf_name}"):
#             # with self.lock: # because we hope one 16G gpu only process one image at the same time
#             pix = page.get_pixmap(dpi=dpi)
#             image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             with torch.no_grad():
#                 reps = encode(model, tokenizer, [image])
#             reps_list.append(reps)
#             images.append(image)

#         for idx in range(len(images)):
#             image = images[idx]
#             cache_image_path = os.path.join(knowledge_base_path, f"{pdf_name}_{idx}.png")
#             image.save(cache_image_path)
#             index2img_filename.append(os.path.basename(cache_image_path))

#     reps_list = [torch.from_numpy(reps) for reps in reps_list]
#     final_reps = torch.cat(reps_list, dim=0)

#     np.save(os.path.join(knowledge_base_path, f"reps.npy"), final_reps.cpu().numpy())

#     with open(os.path.join(knowledge_base_path, 'index2img_filename.txt'), 'w') as f:
#         f.write('\n'.join(index2img_filename))
        
#     print(f"Knowledge base built at {knowledge_base_path}")

# model_path = 'openbmb/VisRAG-Ret'

# device = 'cuda'

# while(True):
#     knowledge_base_path = input("Please enter the knowledge base path in which we build the index: ")
#     if os.path.isabs(knowledge_base_path):
#         break
#     else:
#         print("Invalid knowledge base path, please try again.")
# os.makedirs(knowledge_base_path, exist_ok=True)

# while(True):
#     pdf_dir = input("Enter the directory that contains the pdf files: ")
#     if os.path.isdir(pdf_dir):
#         break
#     else:
#         print("Invalid pdf directory, please try again.") 

# print("emb model load begin...")
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
#     attn_implementation='sdpa', torch_dtype=torch.bfloat16)
# model.eval()
# model.to(device)
# print("emb model load success!")

# add_pdfs(pdf_dir)




