import os
from huggingface_hub import hf_hub_download

HF_REPO_ID = "princedk/rag-faiss-index"

def download_index():
    os.makedirs("faiss_index", exist_ok=True)

    for file in ["index.faiss", "index.pkl"]:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=file,
            repo_type="dataset",
            local_dir="faiss_index",
        )

def hf_index_exists():
    return True