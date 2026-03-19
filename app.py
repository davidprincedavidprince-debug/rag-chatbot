# app.py

import streamlit as st
from google import genai

from rag_pipeline import build_index, load_index, get_chunks_from_file

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Prince David's RAG Chatbot", layout="wide")
st.title("🧠 Prince David's RAG Chatbot")

# ---------- GEMINI CLIENT ----------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# ---------- CACHE VECTOR DB ----------
@st.cache_resource
def get_vectorstore():
    return load_index()

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")

if st.sidebar.button("🔄 Rebuild Index"):
    with st.spinner("Rebuilding index..."):
        vectorstore = build_index()
    st.sidebar.success("Index rebuilt!")
    st.cache_resource.clear()

# ---------- CHAT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- FILE FILTER ----------
st.sidebar.subheader("📂 Filter by file")
filename_filter = st.sidebar.text_input(
    "Enter file name or path (optional)"
)

# ---------- USER INPUT ----------
if prompt := st.chat_input("Ask something about your data..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
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
                            c.page_content for c in chunks
                        ])
                    else:
                        context = ""

                # ---------- NORMAL RAG ----------
                else:
                    docs = retriever.invoke(prompt)

                    context = "\n\n".join([
                        d.page_content for d in docs
                    ])

                # ---------- PROMPT ----------
                full_prompt = f"""
You are an intelligent assistant.

Answer the question using the context below.

STRICT RULES:
- Do NOT mention sources
- Do NOT mention file names
- Do NOT copy raw text
- Give a clean, professional answer

Context:
{context}

Question:
{prompt}

Answer:
"""

                # ---------- GEMINI CALL ----------
                response = client.models.generate_content(
                    model="models/gemini-2.5-flash",
                    contents=full_prompt
                ).text

            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

            # ---------- DISPLAY ----------
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})