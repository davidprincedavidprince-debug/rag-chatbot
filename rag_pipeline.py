# rag_pipeline.py

import os
import glob
import pandas as pd

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ---------- TEXT CLEANING ----------

def clean_text(text):
    return str(text).replace("\n", " ").strip()


# ---------- LOAD DOCUMENTS WITH METADATA ----------

def load_documents(data_path="data"):
    docs = []

    for file_path in glob.glob(f"{data_path}/**/*", recursive=True):
        if os.path.isfile(file_path):
            file = os.path.basename(file_path)

            try:
                # ---------- TEXT / MARKDOWN ----------
                if file.endswith((".txt", ".md")):
                    loaded = TextLoader(file_path).load()
                    docs.extend([
                        Document(
                            page_content=clean_text(d.page_content),
                            metadata={
                                "source": file_path,
                                "filename": file
                            }
                        )
                        for d in loaded
                    ])

                # ---------- PDF ----------
                elif file.endswith(".pdf"):
                    loaded = PyPDFLoader(file_path).load()
                    docs.extend([
                        Document(
                            page_content=clean_text(d.page_content),
                            metadata={
                                "source": file_path,
                                "filename": file
                            }
                        )
                        for d in loaded
                    ])

                # ---------- CSV (ROW-WISE) ----------
                elif file.endswith(".csv"):
                    df = pd.read_csv(file_path)

                    for _, row in df.iterrows():
                        text = " | ".join([
                            f"{col}: {row[col]}" for col in df.columns
                        ])

                        docs.append(
                            Document(
                                page_content=clean_text(text),
                                metadata={
                                    "source": file_path,
                                    "filename": file
                                }
                            )
                        )

                # ---------- EXCEL (ROW-WISE) ----------
                elif file.endswith(".xlsx"):
                    df = pd.read_excel(file_path)

                    for _, row in df.iterrows():
                        text = " | ".join([
                            f"{col}: {row[col]}" for col in df.columns
                        ])

                        docs.append(
                            Document(
                                page_content=clean_text(text),
                                metadata={
                                    "source": file_path,
                                    "filename": file
                                }
                            )
                        )

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return docs


# ---------- SPLIT DOCUMENTS ----------

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,      # optimized
        chunk_overlap=80     # optimized
    )
    return splitter.split_documents(docs)


# ---------- EMBEDDINGS ----------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------- VECTOR STORE ----------

def create_vectorstore(chunks):
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


# ---------- BUILD INDEX ----------

def build_index():
    docs = load_documents()
    chunks = split_documents(docs)

    print(f"Indexing {len(chunks)} chunks...")

    vectorstore = create_vectorstore(chunks)
    vectorstore.save_local("faiss_index")

    return vectorstore


# ---------- LOAD INDEX ----------

def load_index():
    if not os.path.exists("faiss_index"):
        print("Index not found. Building new index...")
        return build_index()

    embeddings = get_embeddings()

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


# ---------- FILTERED RETRIEVAL BY FILE ----------

def get_chunks_from_file(vectorstore, query, filename, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents(query)

    filtered = [
        d for d in docs
        if filename in d.metadata.get("source", "")
    ]

    return filtered[:k]