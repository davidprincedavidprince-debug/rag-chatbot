import streamlit as st
import os
from google import genai
from google.genai import types

from rag_pipeline import get_chunks_from_file, load_index

# ---------- CONFIG ----------
st.set_page_config(page_title="Prince David's RAG Chatbot", layout="wide")
st.title("🧠 Prince David's RAG Chatbot")

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# ---------- LOAD VECTORSTORE ----------
@st.cache_resource
def get_vectorstore():
    try:
        from hf_store import download_index, hf_index_exists

        if hf_index_exists():
            st.info("Downloading index...")
            download_index()
    except Exception as e:
        print(f"HF skipped: {e}")

    if os.path.exists("faiss_index/index.faiss"):
        return load_index()

    raise RuntimeError("No FAISS index found. Upload index first.")

with st.spinner("Loading knowledge base..."):
    try:
        vectorstore = get_vectorstore()
        st.success("Knowledge base ready!")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ---------- CHAT ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(prompt)

        context = "\n\n".join([d.page_content[:500] for d in docs])

        full_prompt = f"""
Context:
{context}

Question:
{prompt}
"""

        response = ""
        stream = client.models.generate_content_stream(
            model="models/gemini-2.5-flash",
            contents=full_prompt,
        )

        for chunk in stream:
            if chunk.text:
                response += chunk.text
                st.markdown(response + "▌")

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})