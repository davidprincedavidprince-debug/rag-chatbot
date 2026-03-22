import streamlit as st
from google import genai
from google.genai import types

from rag_pipeline import get_chunks_from_file
from index_manager import incremental_update

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Prince David's RAG Chatbot",
    page_icon="🧠",
    layout="wide",
)
st.title("🧠 Prince David's RAG Chatbot")

# ---------- GEMINI CLIENT ----------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

SYSTEM_INSTRUCTION = """You are an expert technical assistant with deep knowledge of
engineering, reverse engineering, and document analysis.

When answering:
- Be thorough and precise — use exact technical terminology from the context
- Structure your answer with clear headings or numbered steps when appropriate
- If the context contains tables, formulas, or structured data, reflect that structure
- If the context is insufficient to answer fully, say so clearly rather than guessing
- Never fabricate information not present in the context
- Keep answers focused on what was asked; avoid padding"""

# ---------- CACHE VECTOR DB ----------
@st.cache_resource
def get_vectorstore():
    # index_manager handles HuggingFace restore + rebuild automatically
    vs, stats = incremental_update()
    return vs

with st.spinner("Loading knowledge base... (first load may take a few minutes)"):
    try:
        vectorstore = get_vectorstore()
        st.success("Knowledge base ready!", icon="✅")
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        st.info("Please click 'Force full rebuild' in the sidebar Advanced section.")
        st.stop()

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️  Settings")
show_sources    = st.sidebar.toggle("📎 Show source chunks", value=False)

st.sidebar.subheader("📂 Filter by file")
filename_filter = st.sidebar.text_input(
    "Enter filename or partial path (optional)",
    placeholder="e.g. transformation_output.xlsx",
)

st.sidebar.divider()
st.sidebar.subheader("🗂️  Index management")

if st.sidebar.button("🔄 Smart sync"):
    with st.spinner("Checking for changes …"):
        vs, stats = incremental_update()
    st.cache_resource.clear()

    if stats.get("rebuild"):
        st.sidebar.success(f"Full index built — {stats['total_files']} files indexed.")
    elif not any([stats["added"], stats["modified"], stats["deleted"]]):
        st.sidebar.info("Already up to date — no changes found.")
    else:
        parts = []
        if stats["added"]:    parts.append(f"+{stats['added']} added")
        if stats["modified"]: parts.append(f"~{stats['modified']} updated")
        if stats["deleted"]:  parts.append(f"-{stats['deleted']} removed")
        st.sidebar.success("Index updated: " + ", ".join(parts))
    st.rerun()

with st.sidebar.expander("Advanced"):
    if st.button("🔨 Force full rebuild", type="secondary"):
        with st.spinner("Rebuilding entire index from scratch …"):
            vs, stats = incremental_update(force_rebuild=True)
        st.cache_resource.clear()
        st.sidebar.success(f"Rebuilt — {stats['total_files']} files.")
        st.rerun()

st.sidebar.divider()
if st.sidebar.button("🗑️  Clear chat"):
    st.session_state.messages = []
    st.rerun()

# ---------- CHAT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander("📎 Source chunks used"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"**Chunk {i}** — `{src['meta']}`")
                    st.text(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))

# ---------- HELPERS ----------

def get_adaptive_k(question: str) -> int:
    words = len(question.split())
    if words <= 8:    return 3
    elif words <= 20: return 5
    else:             return 8

def trim_chunks(chunks, max_chars: int = 600):
    for c in chunks:
        c.page_content = c.page_content[:max_chars]
    return chunks

def compress_history(messages: list, keep_recent: int = 4) -> str:
    if len(messages) <= keep_recent:
        return "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )
    old    = messages[:-keep_recent]
    recent = messages[-keep_recent:]
    summary = (
        f"[Earlier: {len(old)//2} exchanges about "
        + ", ".join(m["content"][:40] for m in old[::2][:2])
        + "...]"
    )
    return summary + "\n" + "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in recent
    )

# ---------- USER INPUT ----------
if prompt := st.chat_input("Ask something about your documents …"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context …"):
            k = get_adaptive_k(prompt)
            try:
                if filename_filter.strip():
                    chunks = get_chunks_from_file(
                        vectorstore, prompt, filename_filter.strip(), k=k
                    )
                    if not chunks:
                        st.warning(
                            f"No chunks found for **{filename_filter}** — "
                            "falling back to full index."
                        )
                        chunks = vectorstore.as_retriever(
                            search_kwargs={"k": k}
                        ).invoke(prompt)
                else:
                    chunks = vectorstore.as_retriever(
                        search_kwargs={"k": k}
                    ).invoke(prompt)

                chunks = trim_chunks(chunks)

            except Exception as e:
                st.error(f"Retrieval error: {e}")
                chunks = []

            context_parts, source_meta = [], []
            for i, chunk in enumerate(chunks, 1):
                meta  = chunk.metadata
                label = meta.get("filename", meta.get("source", "unknown"))
                if meta.get("page"):  label += f" (p.{meta['page']})"
                if meta.get("sheet"): label += f" [{meta['sheet']}]"
                context_parts.append(f"[Chunk {i} — {label}]\n{chunk.page_content}")
                source_meta.append({"meta": label, "text": chunk.page_content})

            context      = "\n\n---\n\n".join(context_parts) or "No relevant context found."
            history_text = compress_history(st.session_state.messages[:-1], keep_recent=4)

            full_prompt = f"""CONTEXT FROM DOCUMENTS:
{context}

{"CONVERSATION SO FAR:" + chr(10) + history_text + chr(10) if history_text else ""}CURRENT QUESTION:
{prompt}

Answer the question thoroughly using the context. Use technical language where appropriate.
If the context spans multiple sources, synthesise them into one coherent answer.
"""

        response_text = ""
        placeholder   = st.empty()

        try:
            stream = client.models.generate_content_stream(
                model="models/gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            )
            for chunk_resp in stream:
                if chunk_resp.text:
                    response_text += chunk_resp.text
                    placeholder.markdown(response_text + "▌")
            placeholder.markdown(response_text)

        except Exception as e:
            response_text = (
                "Sorry, I encountered an error. Please try again."
            )
            placeholder.error(response_text)
            print(f"Gemini error: {e}")

        if source_meta and show_sources:
            with st.expander("📎 Source chunks used"):
                for i, src in enumerate(source_meta, 1):
                    st.caption(f"**Chunk {i}** — `{src['meta']}`")
                    st.text(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response_text,
        "sources": source_meta,
    })
