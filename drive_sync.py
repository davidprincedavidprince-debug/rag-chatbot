"""
drive_sync.py

Persists the faiss_index/ folder to Google Drive so it survives
Streamlit Cloud redeploys. Uses a service account for auth — no
OAuth browser popup needed.

Folder structure on Drive:
    rag_faiss_index/
        index.faiss
        index.pkl
        manifest.json
"""

import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2 import service_account

INDEX_PATH      = "faiss_index"
DRIVE_FOLDER    = "rag_faiss_index"
INDEX_FILES     = ["index.faiss", "index.pkl", "manifest.json"]
SCOPES          = ["https://www.googleapis.com/auth/drive"]


# ── Auth ──────────────────────────────────────────────────────────────

def _get_service():
    """
    Build a Drive service client.
    Credentials come from Streamlit secrets (recommended) or a local
    service_account.json file (local dev only — never commit this file).
    """
    try:
        # Streamlit Cloud: secrets stored as a TOML dict
        import streamlit as st
        info = dict(st.secrets["gcp_service_account"])
        # secrets.toml stores private_key with literal \n — expand them
        info["private_key"] = info["private_key"].replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )
    except Exception:
        # Local dev fallback
        creds = service_account.Credentials.from_service_account_file(
            "service_account.json", scopes=SCOPES
        )

    return build("drive", "v3", credentials=creds)


# ── Folder helpers ────────────────────────────────────────────────────

def _get_or_create_folder(service) -> str:
    """Return the Drive folder ID for DRIVE_FOLDER, creating it if needed."""
    query = (
        f"name='{DRIVE_FOLDER}' "
        f"and mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files   = results.get("files", [])

    if files:
        return files[0]["id"]

    # Create the folder
    meta = {
        "name":     DRIVE_FOLDER,
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = service.files().create(body=meta, fields="id").execute()
    print(f"Created Drive folder '{DRIVE_FOLDER}' (id={folder['id']})")
    return folder["id"]


def _list_folder_files(service, folder_id: str) -> dict:
    """Return {filename: file_id} for all files in the folder."""
    query   = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return {f["name"]: f["id"] for f in results.get("files", [])}


# ── Upload ────────────────────────────────────────────────────────────

def upload_index() -> bool:
    """
    Upload all files in faiss_index/ to Google Drive.
    Overwrites existing files if present.
    Returns True on success, False on failure.
    """
    if not os.path.exists(INDEX_PATH):
        print("No local index to upload.")
        return False

    try:
        service   = _get_service()
        folder_id = _get_or_create_folder(service)
        existing  = _list_folder_files(service, folder_id)

        for filename in INDEX_FILES:
            local_path = os.path.join(INDEX_PATH, filename)
            if not os.path.exists(local_path):
                continue

            media = MediaFileUpload(local_path, resumable=True)

            if filename in existing:
                # Update existing file
                service.files().update(
                    fileId=existing[filename],
                    media_body=media,
                ).execute()
                print(f"  Updated {filename} on Drive")
            else:
                # Create new file
                meta = {"name": filename, "parents": [folder_id]}
                service.files().create(
                    body=meta,
                    media_body=media,
                    fields="id",
                ).execute()
                print(f"  Uploaded {filename} to Drive")

        print("✅  Index uploaded to Google Drive")
        return True

    except Exception as e:
        print(f"⚠️  Drive upload failed: {e}")
        return False


# ── Download ──────────────────────────────────────────────────────────

def download_index() -> bool:
    """
    Download faiss_index/ files from Google Drive to local disk.
    Returns True if all files downloaded successfully, False otherwise.
    """
    try:
        service   = _get_service()
        folder_id = _get_or_create_folder(service)
        existing  = _list_folder_files(service, folder_id)

        # Need at least index.faiss and index.pkl to be usable
        required = {"index.faiss", "index.pkl"}
        if not required.issubset(existing.keys()):
            print("Drive folder doesn't have a complete index yet.")
            return False

        os.makedirs(INDEX_PATH, exist_ok=True)

        for filename, file_id in existing.items():
            if filename not in INDEX_FILES:
                continue

            local_path = os.path.join(INDEX_PATH, filename)
            request    = service.files().get_media(fileId=file_id)
            buf        = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            with open(local_path, "wb") as f:
                f.write(buf.getvalue())

            print(f"  Downloaded {filename} from Drive")

        print("✅  Index downloaded from Google Drive")
        return True

    except Exception as e:
        print(f"⚠️  Drive download failed: {e}")
        return False


# ── Index exists on Drive? ────────────────────────────────────────────

def drive_index_exists() -> bool:
    """Quick check — does Drive have a complete index?"""
    try:
        service   = _get_service()
        folder_id = _get_or_create_folder(service)
        existing  = _list_folder_files(service, folder_id)
        return {"index.faiss", "index.pkl"}.issubset(existing.keys())
    except Exception:
        return False
