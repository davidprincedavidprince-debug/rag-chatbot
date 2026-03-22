from langchain_community.vectorstores import FAISS
from rag_pipeline import get_embeddings
from langchain_core.documents import Document

def build_dummy_index():
    embeddings = get_embeddings()

    docs = [
        Document(page_content="Sample data for testing", metadata={"source": "test"})
    ]

    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local("faiss_index")

    print("Index built successfully")