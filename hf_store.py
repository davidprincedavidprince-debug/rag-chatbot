"""
hf_store.py

Stores and retrieves the faiss_index/ folder on Hugging Face Hub
(private dataset repo). Replaces drive_sync.py entirely.

Repo: https://huggingface.co/datasets/princedk/rag-faiss-index
"""

import os
from huggingface_hub import HfApi, hf_hub_download

HF_REPO_ID   = "princedk/rag-faiss-index"
HF_REPO_TYPE = "dataset"
INDEX_PATH   = "faiss_index"
INDEX_FILES  = ["index.faiss", "index.pkl", "manifest.json"]


def _get_token() -> str:
    """Get HF token from Streamlit secrets or environment."""
    try:
        import streamlit as st
        return st.secrets["HF_TOKEN"]
    except Exception:
        token = os.environ.get("HF_TOKEN", "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")
        return token


def upload_index() -> bool:
    """Upload faiss_index/ files to Hugging Face dataset repo."""
    if not os.path.exists(INDEX_PATH):
        print("No local index to upload.")
        return False

    try:
        api   = HfApi()
        token = _get_token()

        for filename in INDEX_FILES:
            local_path = os.path.join(INDEX_PATH, filename)
            if not os.path.exists(local_path):
                continue

            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                token=token,
            )
            print(f"  Uploaded {filename} to HuggingFace")

        print("✅  Index uploaded to Hugging Face")
        return True

    except Exception as e:
        print(f"⚠️  HF upload failed: {e}")
        return False


def download_index() -> bool:
    """Download faiss_index/ files from Hugging Face dataset repo."""
    try:
        token = _get_token()
        os.makedirs(INDEX_PATH, exist_ok=True)

        for filename in INDEX_FILES:
            try:
                local_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    repo_type=HF_REPO_TYPE,
                    token=token,
                    local_dir=INDEX_PATH,
                )
                print(f"  Downloaded {filename} from HuggingFace")
            except Exception as e:
                if "manifest.json" in filename:
                    continue  # manifest is optional
                print(f"  ⚠️  Could not download {filename}: {e}")
                return False

        print("✅  Index downloaded from Hugging Face")
        return True

    except Exception as e:
        print(f"⚠️  HF download failed: {e}")
        return False


def hf_index_exists() -> bool:
    """Check if a complete index exists on Hugging Face."""
    try:
        from huggingface_hub import list_repo_files
        token = _get_token()
        files = list(list_repo_files(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=token,
        ))
        return "index.faiss" in files and "index.pkl" in files
    except Exception:
        return False
