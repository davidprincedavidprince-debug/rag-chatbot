import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        embeddings.embed_query("test")
        return embeddings
    except Exception:
        raise RuntimeError("Embedding failed — run locally to build index")

def load_index():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

def get_chunks_from_file(vectorstore, query, filename, k=5):
    docs = vectorstore.as_retriever(search_kwargs={"k": 20}).invoke(query)

    filtered = [
        d for d in docs
        if filename.lower() in d.metadata.get("source", "").lower()
    ]

    return filtered[:k]