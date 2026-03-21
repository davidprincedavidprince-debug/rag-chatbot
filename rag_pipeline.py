import os
import glob
import pandas as pd

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------- OPTIONAL IMPORTS (graceful fallback) ----------
# OCR is disabled on Streamlit Cloud — too slow for free tier CPU.
# To re-enable locally, change OCR_AVAILABLE to True below.
OCR_AVAILABLE = False

try:
    import fitz
    import pytesseract
    from PIL import Image
    import io
except ImportError:
    pass

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️  python-docx unavailable. Install: pip install python-docx")


# ---------- TEXT CLEANING ----------

def clean_text(text: str) -> str:
    """Normalise whitespace while preserving paragraph breaks."""
    lines = [line.strip() for line in str(text).splitlines()]
    # collapse runs of blank lines to a single blank line
    cleaned, prev_blank = [], False
    for line in lines:
        if line == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return " ".join(cleaned).strip()


# ---------- OCR FALLBACK FOR IMAGE-BASED PDFs ----------

def ocr_pdf(file_path: str) -> list[Document]:
    """
    Rasterise every page of a PDF and run Tesseract OCR on it.
    Used when PyPDFLoader extracts fewer than 50 characters per page on average.
    """
    if not OCR_AVAILABLE:
        return []

    docs = []
    pdf = fitz.open(file_path)

    for page_num, page in enumerate(pdf):
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)

        if text.strip():
            docs.append(Document(
                page_content=clean_text(text),
                metadata={
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    "page": page_num + 1,
                    "loader": "ocr",
                }
            ))

    pdf.close()
    return docs


# ---------- LOAD DOCUMENTS WITH METADATA ----------

def load_documents(data_path: str = "data") -> list[Document]:
    docs = []

    for file_path in glob.glob(f"{data_path}/**/*", recursive=True):
        if not os.path.isfile(file_path):
            continue

        file = os.path.basename(file_path)
        loader_tag = "text"

        try:

            # ── TXT / MARKDOWN ──────────────────────────────────────
            if file.endswith((".txt", ".md")):
                loaded = TextLoader(file_path, encoding="utf-8").load()
                for d in loaded:
                    docs.append(Document(
                        page_content=clean_text(d.page_content),
                        metadata={"source": file_path, "filename": file, "loader": loader_tag}
                    ))

            # ── PDF ─────────────────────────────────────────────────
            elif file.endswith(".pdf"):
                loaded = PyPDFLoader(file_path).load()

                # Check whether the PDF actually has selectable text
                total_chars = sum(len(d.page_content) for d in loaded)
                avg_chars   = total_chars / max(len(loaded), 1)

                if avg_chars >= 50:
                    # Normal text-layer PDF
                    for d in loaded:
                        docs.append(Document(
                            page_content=clean_text(d.page_content),
                            metadata={
                                "source": file_path,
                                "filename": file,
                                "page": d.metadata.get("page", "?"),
                                "loader": "pypdf",
                            }
                        ))
                else:
                    # Image-heavy / scanned PDF → OCR
                    print(f"  ↳ Low text yield in {file} — switching to OCR")
                    ocr_docs = ocr_pdf(file_path)
                    if ocr_docs:
                        docs.extend(ocr_docs)
                    else:
                        # Last resort: keep whatever PyPDF extracted
                        for d in loaded:
                            docs.append(Document(
                                page_content=clean_text(d.page_content),
                                metadata={"source": file_path, "filename": file, "loader": "pypdf-fallback"}
                            ))

            # ── WORD DOCUMENTS (.docx) ───────────────────────────────
            elif file.endswith(".docx") and DOCX_AVAILABLE:
                word_doc  = DocxDocument(file_path)
                full_text = "\n".join(
                    para.text for para in word_doc.paragraphs if para.text.strip()
                )

                # Also extract text from tables
                for table in word_doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        )
                        if row_text:
                            full_text += "\n" + row_text

                if full_text.strip():
                    docs.append(Document(
                        page_content=clean_text(full_text),
                        metadata={"source": file_path, "filename": file, "loader": "docx"}
                    ))

            # ── CSV ─────────────────────────────────────────────────
            elif file.endswith(".csv"):
                df = pd.read_csv(file_path)
                _ingest_dataframe(df, file_path, file, docs)

            # ── EXCEL (all sheets) ───────────────────────────────────
            elif file.endswith((".xlsx", ".xls")):
                xl = pd.ExcelFile(file_path)

                for sheet_name in xl.sheet_names:
                    df = xl.parse(sheet_name)
                    _ingest_dataframe(df, file_path, file, docs, sheet=sheet_name)

        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")

    print(f"✅  Loaded {len(docs)} raw document sections from {data_path}/")
    return docs


def _ingest_dataframe(
    df: pd.DataFrame,
    file_path: str,
    filename: str,
    docs: list,
    sheet: str = "",
    rows_per_chunk: int = 10,
    max_chunks: int = 300,
) -> None:
    """
    Batch rows into chunks instead of one Document per row.
    rows_per_chunk=10 means a 1,000-row sheet becomes ~100 chunks.
    max_chunks=300 is a hard cap per sheet.
    """
    df = df.dropna(how="all").fillna("")
    if df.empty:
        return

    print(f"  Ingesting {filename}" + (f"[{sheet}]" if sheet else "") + f" — {len(df)} rows")

    chunk_count = 0
    for start in range(0, len(df), rows_per_chunk):
        if chunk_count >= max_chunks:
            print(f"  Reached max_chunks={max_chunks} for {filename} — truncating")
            break
        batch = df.iloc[start : start + rows_per_chunk]
        lines = []
        for _, row in batch.iterrows():
            line = " | ".join(f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip())
            if line.strip():
                lines.append(line)
        if lines:
            docs.append(Document(
                page_content=clean_text("\n".join(lines)),
                metadata={
                    "source":   file_path,
                    "filename": filename,
                    "sheet":    sheet,
                    "rows":     f"{start+1}-{min(start+rows_per_chunk, len(df))}",
                    "loader":   "tabular",
                }
            ))
            chunk_count += 1

    preview_rows = df.head(5).to_string(index=False)
    summary = (
        f"File: {filename}"
        + (f" | Sheet: {sheet}" if sheet else "")
        + f"\nColumns: {', '.join(df.columns)}\n"
        + f"Row count: {len(df)}\n"
        + f"Preview:\n{preview_rows}"
    )
    docs.append(Document(
        page_content=clean_text(summary),
        metadata={
            "source":   file_path,
            "filename": filename,
            "sheet":    sheet,
            "loader":   "tabular-summary",
        }
    ))


# ---------- SPLIT DOCUMENTS ----------

def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Larger chunks retain more technical context
        chunk_overlap=150,    # Overlap prevents facts from being split across chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Deduplicate by content hash
    seen, unique = set(), []
    for chunk in chunks:
        h = hash(chunk.page_content.strip())
        if h not in seen:
            seen.add(h)
            unique.append(chunk)

    print(f"✅  {len(unique)} unique chunks after splitting & deduplication")
    return unique


# ---------- EMBEDDINGS ----------

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    all-mpnet-base-v2 produces much richer semantic embeddings than MiniLM,
    especially on technical / domain-specific text.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------- VECTOR STORE ----------

def create_vectorstore(chunks: list[Document]) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


# ---------- BUILD INDEX ----------

def build_index(data_path: str = "data") -> FAISS:
    docs   = load_documents(data_path)
    chunks = split_documents(docs)

    print(f"🔨  Building FAISS index over {len(chunks)} chunks …")
    vectorstore = create_vectorstore(chunks)
    vectorstore.save_local("faiss_index")
    print("✅  Index saved to faiss_index/")

    return vectorstore


# ---------- LOAD INDEX ----------

def load_index() -> FAISS:
    if not os.path.exists("faiss_index"):
        print("Index not found. Building new index …")
        return build_index()

    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ---------- PUBLIC EXPORTS (used by index_manager.py) ----------
# These names are imported directly:
#   from rag_pipeline import clean_text, ocr_pdf, _ingest_dataframe,
#                            OCR_AVAILABLE, DOCX_AVAILABLE
# All are already defined above — this comment just makes the contract explicit.


# ---------- FILTERED RETRIEVAL BY FILE ----------

def get_chunks_from_file(
    vectorstore: FAISS,
    query: str,
    filename: str,
    k: int = 5,
) -> list[Document]:
    """Return the top-k chunks from a specific file matching the query."""
    all_docs = vectorstore.as_retriever(search_kwargs={"k": 20}).invoke(query)

    filtered = [
        d for d in all_docs
        if filename.lower() in d.metadata.get("source", "").lower()
        or filename.lower() in d.metadata.get("filename", "").lower()
    ]

    return filtered[:k]