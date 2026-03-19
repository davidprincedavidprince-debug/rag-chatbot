# app.py

import streamlit as st
from google import genai
import os

from rag_pipeline import build_index, load_index, get_chunks_from_file

# ---------- GEMINI CLIENT ----------
client = genai.Client(api_key=os.getenv("AIzaSyB2_wj3KSNsRkaBtw2tGIxbSzjKBY6DgQA"))

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Prince David's Local RAG Chatbot", layout="wide")
st.title("Prince David's Local RAG Chatbot")

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")

if st.sidebar.button("🔄 Rebuild Index"):
    with st.spinner("Indexing documents..."):
        vectorstore = build_index()
    st.sidebar.success("Index rebuilt!")

# ---------- LOAD VECTOR DB ----------
vectorstore = load_index()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ---------- CHAT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- SIDEBAR FILTER ----------
st.sidebar.subheader("Optional: Filter by file")
filename_filter = st.sidebar.text_input(
    "Enter file path to restrict answers (leave empty for all files)"
)

# ---------- USER INPUT ----------
if prompt := st.chat_input("Ask something about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # ---------- FILE FILTER MODE ----------
            if filename_filter.strip():
                chunks = get_chunks_from_file(
                    vectorstore,
                    prompt,
                    filename_filter.strip(),
                    k=3
                )

                if chunks:
                    context = "\n\n".join([
                        f"Source: {c.metadata.get('source', '')}\n{c.page_content}"
                        for c in chunks
                    ])

                    full_prompt = f"""
You are an intelligent assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{prompt}

Answer:
"""

                    response = client.models.generate_content(
                        model="models/gemini-2.5-flash",
                        contents=full_prompt
                    ).text

                else:
                    response = f"No relevant content found in '{filename_filter.strip()}'"

            # ---------- NORMAL RAG ----------
            else:
                docs = retriever.invoke(prompt)

                context = "\n\n".join([
                    f"Source: {d.metadata.get('source', '')}\n{d.page_content}"
                    for d in docs
                ])

                full_prompt = f"""
You are an intelligent assistant.

Use the context below to answer the question.
- If answer is in context → use it
- If not → answer normally

Context:
{context}

Question:
{prompt}

Answer:
"""

                response = client.models.generate_content(
                    model="models/gemini-2.5-flash",
                    contents=full_prompt
                ).text

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})