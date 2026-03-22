import streamlit as st
from google import genai
from google.genai import types
import os
import glob
import pandas as pd

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

# ==========================================================================
# PANDAS AGENT — Excel / CSV handler
# ==========================================================================

@st.cache_resource
def load_excel_dataframes() -> dict:
    """
    Load all Excel and CSV files from data/ into DataFrames.
    Returns: { "filename.xlsx": { "SheetName": DataFrame, ... }, ... }
    """
    dfs = {}
    for pattern in ["data/**/*.xlsx", "data/**/*.xls", "data/**/*.csv"]:
        for path in glob.glob(pattern, recursive=True):
            try:
                fname = os.path.basename(path)
                if path.endswith(".csv"):
                    dfs[fname] = {"Sheet1": pd.read_csv(path)}
                else:
                    xl = pd.ExcelFile(path)
                    dfs[fname] = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
                print(f"✅  Loaded {fname} for Pandas Agent")
            except Exception as e:
                print(f"⚠️  Could not load {path}: {e}")
    return dfs


def query_excel_with_gemini(question: str, dfs: dict) -> str | None:
    """
    Two-step Pandas Agent:
      Step 1 — Gemini writes a pandas expression from the schema
      Step 2 — Python executes it, Gemini formats the result

    Returns answer string, or None if question is not about tabular data.
    """
    if not dfs:
        return None

    # ── Build schema so Gemini knows what columns/data exist ──────────
    schema_parts = []
    for fname, sheets in dfs.items():
        for sheet_name, df in sheets.items():
            sample = df.head(3).to_string(index=False)
            schema_parts.append(
                f"File: {fname} | Sheet: {sheet_name}\n"
                f"Columns: {', '.join(str(c) for c in df.columns)}\n"
                f"Row count: {len(df)}\n"
                f"Sample (first 3 rows):\n{sample}"
            )
    schema_text = "\n\n---\n\n".join(schema_parts)

    # ── Step 1: Gemini writes the pandas expression ────────────────────
    planning_prompt = f"""You have access to these Excel/CSV files as pandas DataFrames:

{schema_text}

DataFrames are in a dict called `dfs`:
  dfs['filename.xlsx']['SheetName'] -> DataFrame

User question: {question}

Task:
1. If this question is about the tabular data — write ONE pandas expression that answers it.
   - Use only: dfs, pd, standard Python
   - Must be executable with eval() — no imports, no print(), no multi-line
   - Example: dfs['scores.xlsx']['L4_Tool'][dfs['scores.xlsx']['L4_Tool']['project_code']=='P001']['Relevance_overall'].values[0]
2. If NOT about tabular data — respond with exactly: NOT_TABULAR

Respond with ONLY the pandas expression or NOT_TABULAR."""

    try:
        plan_resp = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=planning_prompt,
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=512),
        )
        pandas_expr = plan_resp.text.strip().strip("`").strip()

        # Strip markdown code fences if Gemini wrapped the expression
        if pandas_expr.startswith("python"):
            pandas_expr = pandas_expr[6:].strip()

        if "NOT_TABULAR" in pandas_expr:
            return None

        # ── Step 2: Execute the expression ────────────────────────────
        try:
            result = eval(pandas_expr, {"dfs": dfs, "pd": pd})
        except Exception as exec_err:
            print(f"Pandas exec error: {exec_err} | expr: {pandas_expr}")
            return None

        # Convert result to a readable string
        if isinstance(result, pd.DataFrame):
            result_str = result.to_string(index=False)
        elif isinstance(result, pd.Series):
            result_str = result.to_string()
        else:
            result_str = str(result)

        # ── Step 3: Gemini formats a clean human answer ────────────────
        answer_prompt = f"""The user asked: "{question}"

Pandas query returned this exact data:
{result_str}

Write a clear, accurate answer:
- Use exact numbers — do not round or approximate
- Format tables neatly in markdown if the result is tabular
- Be concise but complete"""

        answer_resp = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=answer_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        return answer_resp.text

    except Exception as e:
        print(f"Pandas agent error: {e}")
        return None  # Fall back to RAG silently


# ---------- CACHE VECTOR DB ----------
@st.cache_resource
def get_vectorstore():
    # index_manager handles HuggingFace restore + rebuild automatically
    vs, stats = incremental_update()
    return vs

with st.spinner("Loading knowledge base... (first load may take a few minutes)"):
    try:
        vectorstore = get_vectorstore()
        excel_data  = load_excel_dataframes()
        n_sheets    = sum(len(sheets) for sheets in excel_data.values())
        st.success(
            f"Knowledge base ready!"
            + (f" ({n_sheets} Excel sheet(s) loaded)" if n_sheets else ""),
            icon="✅"
        )
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

# Show loaded Excel files in sidebar
if excel_data:
    st.sidebar.divider()
    st.sidebar.subheader("📊 Loaded Excel files")
    for fname, sheets in excel_data.items():
        for sheet_name, df in sheets.items():
            st.sidebar.caption(f"📄 {fname} › {sheet_name} ({len(df)} rows)")

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
if prompt := st.chat_input("Ask about your documents or Excel data …"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = ""
        placeholder   = st.empty()
        source_meta   = []
        answered_by   = "rag"

        # ── Route 1: Pandas Agent for Excel/numerical questions ────────
        with st.spinner("Thinking …"):
            excel_answer = query_excel_with_gemini(prompt, excel_data)

        if excel_answer:
            answered_by   = "excel"
            response_text = excel_answer
            placeholder.markdown(response_text)

        else:
            # ── Route 2: RAG for document questions ───────────────────
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

                context_parts = []
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
                response_text = "Sorry, I encountered an error. Please try again."
                placeholder.error(response_text)
                print(f"Gemini error: {e}")

        # ── Source label ───────────────────────────────────────────────
        if answered_by == "excel":
            st.caption("📊 Answered from Excel data (Pandas Agent)")
        elif source_meta and show_sources:
            with st.expander("📎 Source chunks used"):
                for i, src in enumerate(source_meta, 1):
                    st.caption(f"**Chunk {i}** — `{src['meta']}`")
                    st.text(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response_text,
        "sources": source_meta,
    })
