"""Parallel version of BaseDataset.extract_content: page images (144 dpi) and PyMuPDF text, one process
per document, written atomically so an interrupted run never leaves a truncated page behind.

    python scripts/extract_parallel.py --config-name ldu +workers=32
"""
import os
import sys
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
import pymupdf
from my_datasets.base_dataset import BaseDataset


def one(args):
    pdf_path, doc_name, extract_path, max_pages, dpi = args
    with pymupdf.open(pdf_path) as pdf:
        for i, page in enumerate(pdf[:max_pages]):
            im = os.path.join(extract_path, f"{doc_name}_{i}.png")
            tmp = f"{im}.{os.getpid()}.tmp.png"
            page.get_pixmap(dpi=dpi).save(tmp)
            os.replace(tmp, im)
            txt = os.path.join(extract_path, f"{doc_name}_{i}.txt")
            tmp = f"{txt}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(page.get_text("text"))
            os.replace(tmp, txt)
    return doc_name


@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    ds = BaseDataset(cfg.dataset)
    c = ds.config
    os.makedirs(c.extract_path, exist_ok=True)
    jobs = {}
    for s in ds.load_data():   # one job per output name: several doc_id spellings can share a document
        jobs[ds.EXTRACT_DOCUMENT_ID(s)] = (os.path.join(c.document_path, s["doc_id"]), ds.EXTRACT_DOCUMENT_ID(s),
                                           c.extract_path, c.max_page, 144)
    print(f"[extract] {len(jobs)} documents")
    with Pool(int(cfg.get("workers", 16))) as pool:
        for i, d in enumerate(pool.imap_unordered(one, jobs.values()), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f"[extract] {i}/{len(jobs)} {d}", flush=True)


if __name__ == "__main__":
    main()
