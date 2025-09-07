from pathlib import Path
import os
import importlib.util
import shutil

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.agent.rag_agent import RAGAgent
from src.chunker.markdown_section_chunker import MarkdownSectionChunker
from src.converter.pymu import PymuConverter
from src.converter.pymu_hdr_converter import PymuHeaderConverter
from src.converter.markitdown_converter import MarkItDownConverter
from src.converter.marker_converter import MarkerConverter
from src.converter.unstructured_converter import UnstructuredConverter
from src.loader.pdf_loader import DirectoryPDFLoader
from src.vector_store.in_memory import InMemoryVectorStore

# Optional Docling converter (enable side-by-side compare when available)
DOCLING_AVAILABLE = False
try:
    from src.converter.docling_converter import DoclingConverter  # type: ignore

    DOCLING_AVAILABLE = True
except Exception:
    DOCLING_AVAILABLE = False

# Optional MarkItDown (detect via module availability)
MARKITDOWN_AVAILABLE = importlib.util.find_spec("markitdown") is not None

# Optional Marker (detect via CLI availability)
MARKER_AVAILABLE = (shutil.which("marker") is not None) or (shutil.which("marker_single") is not None)
UNSTRUCTURED_AVAILABLE = importlib.util.find_spec("unstructured") is not None

# Control which converters are shown via env var (comma-separated keys)
# Keys: pymu, pymu_hdr, docling, markitdown, marker, unstructured
DEFAULT_ENABLED = "pymu,pymu_hdr,docling,markitdown,marker,unstructured"
_enabled_csv = os.getenv("KAPA_ENABLED_CONVERTERS", DEFAULT_ENABLED)
ENABLED_SET = {x.strip().lower() for x in _enabled_csv.split(",") if x.strip()}

SHOW_PYMU = "pymu" in ENABLED_SET
SHOW_PYMU_HDR = "pymu_hdr" in ENABLED_SET
SHOW_DOCLING = ("docling" in ENABLED_SET) and DOCLING_AVAILABLE
SHOW_MARKITDOWN = ("markitdown" in ENABLED_SET) and MARKITDOWN_AVAILABLE
SHOW_MARKER = ("marker" in ENABLED_SET) and MARKER_AVAILABLE
SHOW_UNSTRUCTURED = ("unstructured" in ENABLED_SET) and UNSTRUCTURED_AVAILABLE

DATA_DIR = Path(__file__).parent.parent / "data" / "pdfs"

# ────────────────────────────────────────────────────────────────
# Page config (browser-tab title stays constant)
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chat over PDFs", layout="wide")


# ────────────────────────────────────────────────────────────────
# Session-state defaults
# ────────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "Browse"

# Always create PyMuPDF agent
if "agent_pymu" not in st.session_state:
    st.session_state.agent_pymu = RAGAgent(
        loader=DirectoryPDFLoader(DATA_DIR),
        converter=PymuConverter(),
        chunker=MarkdownSectionChunker(),
        store=InMemoryVectorStore(table_name="chunks_pymu"),
    )
# Always create PyMuPDF (hdr) agent
if "agent_pymu_hdr" not in st.session_state:
    hdr_mode = os.getenv("KAPA_PYMU_HDR_MODE", "auto")
    size_h1 = float(os.getenv("KAPA_PYMU_HDR_H1", "14"))
    size_h2 = float(os.getenv("KAPA_PYMU_HDR_H2", "10"))
    debug_flag = os.getenv("KAPA_PYMU_HDR_DEBUG", "1") == "1"
    debug_dir = Path(__file__).parent.parent / "data" / "md_logs"
    st.session_state.agent_pymu_hdr = RAGAgent(
        loader=DirectoryPDFLoader(DATA_DIR),
        converter=PymuHeaderConverter(mode=hdr_mode, size_h1=size_h1, size_h2=size_h2, debug=debug_flag, debug_output_dir=debug_dir),
        chunker=MarkdownSectionChunker(),
        store=InMemoryVectorStore(table_name="chunks_pymu_hdr"),
    )

# Optionally create Docling agent
if DOCLING_AVAILABLE and "agent_docling" not in st.session_state:
    st.session_state.agent_docling = RAGAgent(
        loader=DirectoryPDFLoader(DATA_DIR),
        converter=DoclingConverter(),
        chunker=MarkdownSectionChunker(),
        store=InMemoryVectorStore(table_name="chunks_docling"),
    )

# Optionally create MarkItDown agent
if MARKITDOWN_AVAILABLE and "agent_markitdown" not in st.session_state:
    st.session_state.agent_markitdown = RAGAgent(
        loader=DirectoryPDFLoader(DATA_DIR),
        converter=MarkItDownConverter(),
        chunker=MarkdownSectionChunker(),
        store=InMemoryVectorStore(table_name="chunks_markitdown"),
    )

# Optionally create Marker agent
if MARKER_AVAILABLE and "agent_marker" not in st.session_state:
    st.session_state.agent_marker = RAGAgent(
        loader=DirectoryPDFLoader(DATA_DIR),
        converter=MarkerConverter(),
        chunker=MarkdownSectionChunker(),
        store=InMemoryVectorStore(table_name="chunks_marker"),
    )

# Optionally create Unstructured agent
if UNSTRUCTURED_AVAILABLE and "agent_unstructured" not in st.session_state:
    st.session_state.agent_unstructured = RAGAgent(
        loader=DirectoryPDFLoader(DATA_DIR),
        converter=UnstructuredConverter(),
        chunker=MarkdownSectionChunker(),
        store=InMemoryVectorStore(table_name="chunks_unstructured"),
    )

# ────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    # One-off toast after indexing
    if "index_msg" in st.session_state:
        st.success(st.session_state.index_msg)
        del st.session_state.index_msg

    # Index / reset actions for each agent
    st.subheader("Indexing")

    # PyMuPDF controls
    pymu_agent = st.session_state.agent_pymu
    cols1 = st.columns(2)
    with cols1[0]:
        st.text("PyMuPDF")
    with cols1[1]:
        if pymu_agent.docs:
            if st.button("Reset (PyMuPDF)", use_container_width=True):
                st.session_state.agent_pymu = RAGAgent(
                    loader=DirectoryPDFLoader(DATA_DIR),
                    converter=PymuConverter(),
                    chunker=MarkdownSectionChunker(),
                    store=InMemoryVectorStore(table_name="chunks_pymu"),
                )
                st.session_state.mode = "Browse"
                st.rerun()
        else:
            if st.button("Load & index (PyMuPDF)", use_container_width=True):
                pymu_agent.index()
                st.session_state.index_msg = f"Indexed {len(pymu_agent.docs)} PDF(s) with PyMuPDF"
                st.session_state.mode = "Browse"
                st.rerun()

    # PyMuPDF (hdr) controls
    pymu_hdr_agent = st.session_state.agent_pymu_hdr
    cols1b = st.columns(2)
    with cols1b[0]:
        st.text("PyMuPDF (hdr)")
    with cols1b[1]:
        if pymu_hdr_agent.docs:
            if st.button("Reset (PyMuPDF hdr)", use_container_width=True):
                st.session_state.agent_pymu_hdr = RAGAgent(
                    loader=DirectoryPDFLoader(DATA_DIR),
                    converter=PymuHeaderConverter(mode=hdr_mode, size_h1=size_h1, size_h2=size_h2, debug=debug_flag, debug_output_dir=debug_dir),
                    chunker=MarkdownSectionChunker(),
                    store=InMemoryVectorStore(table_name="chunks_pymu_hdr"),
                )
                st.session_state.mode = "Browse"
                st.rerun()
        else:
            if st.button("Load & index (PyMuPDF hdr)", use_container_width=True):
                pymu_hdr_agent.index()
                st.session_state.index_msg = f"Indexed {len(pymu_hdr_agent.docs)} PDF(s) with PyMuPDF (hdr)"
                st.session_state.mode = "Browse"
                st.rerun()

    # Docling controls (if available)
    if DOCLING_AVAILABLE:
        docling_agent = st.session_state.agent_docling
        cols2 = st.columns(2)
        with cols2[0]:
            st.text("Docling")
        with cols2[1]:
            if docling_agent.docs:
                if st.button("Reset (Docling)", use_container_width=True):
                    st.session_state.agent_docling = RAGAgent(
                        loader=DirectoryPDFLoader(DATA_DIR),
                        converter=DoclingConverter(),
                        chunker=MarkdownSectionChunker(),
                        store=InMemoryVectorStore(table_name="chunks_docling"),
                    )
                    st.session_state.mode = "Browse"
                    st.rerun()
            else:
                if st.button("Load & index (Docling)", use_container_width=True):
                    docling_agent.index()
                    st.session_state.index_msg = f"Indexed {len(docling_agent.docs)} PDF(s) with Docling"
                    st.session_state.mode = "Browse"
                    st.rerun()

    # MarkItDown controls (if available)
    if MARKITDOWN_AVAILABLE:
        markit_agent = st.session_state.agent_markitdown
        cols3 = st.columns(2)
        with cols3[0]:
            st.text("MarkItDown")
        with cols3[1]:
            if markit_agent.docs:
                if st.button("Reset (MarkItDown)", use_container_width=True):
                    st.session_state.agent_markitdown = RAGAgent(
                        loader=DirectoryPDFLoader(DATA_DIR),
                        converter=MarkItDownConverter(),
                        chunker=MarkdownSectionChunker(),
                        store=InMemoryVectorStore(table_name="chunks_markitdown"),
                    )
                    st.session_state.mode = "Browse"
                    st.rerun()
            else:
                if st.button("Load & index (MarkItDown)", use_container_width=True):
                    markit_agent.index()
                    st.session_state.index_msg = f"Indexed {len(markit_agent.docs)} PDF(s) with MarkItDown"
                    st.session_state.mode = "Browse"
                    st.rerun()

    # Marker controls (if available)
    if MARKER_AVAILABLE:
        marker_agent = st.session_state.agent_marker
        cols4 = st.columns(2)
        with cols4[0]:
            st.text("Marker")
        with cols4[1]:
            if marker_agent.docs:
                if st.button("Reset (Marker)", use_container_width=True):
                    st.session_state.agent_marker = RAGAgent(
                        loader=DirectoryPDFLoader(DATA_DIR),
                        converter=MarkerConverter(),
                        chunker=MarkdownSectionChunker(),
                        store=InMemoryVectorStore(table_name="chunks_marker"),
                    )
                    st.session_state.mode = "Browse"
                    st.rerun()
            else:
                if st.button("Load & index (Marker)", use_container_width=True):
                    marker_agent.index()
                    st.session_state.index_msg = f"Indexed {len(marker_agent.docs)} PDF(s) with Marker"
                    st.session_state.mode = "Browse"
                    st.rerun()

    # Unstructured controls (always visible)
    cols5 = st.columns(2)
    with cols5[0]:
        st.text("Unstructured")
    with cols5[1]:
        if UNSTRUCTURED_AVAILABLE:
            un_agent = st.session_state.agent_unstructured
            if un_agent.docs:
                if st.button("Reset (Unstructured)", use_container_width=True):
                    st.session_state.agent_unstructured = RAGAgent(
                        loader=DirectoryPDFLoader(DATA_DIR),
                        converter=UnstructuredConverter(),
                        chunker=MarkdownSectionChunker(),
                        store=InMemoryVectorStore(table_name="chunks_unstructured"),
                    )
                    st.session_state.mode = "Browse"
                    st.rerun()
            else:
                if st.button("Load & index (Unstructured)", use_container_width=True):
                    un_agent.index()
                    st.session_state.index_msg = f"Indexed {len(un_agent.docs)} PDF(s) with Unstructured"
                    st.session_state.mode = "Browse"
                    st.rerun()
        else:
            st.error(
                "Unstructured is not installed. Install dependencies then restart:\n\n"
                "pip install \"unstructured[pdf]\"\n"
                "brew install poppler tesseract\n"
                "See docs: https://github.com/Unstructured-IO/unstructured"
            )

    st.markdown("---")

    # Navigation – stacked buttons (Browse first)
    if st.button("📚 Browse", key="nav_browse", use_container_width=True):
        st.session_state.mode = "Browse"
    if st.button("💬 Chat", key="nav_chat", use_container_width=True):
        st.session_state.mode = "Chat"

# ────────────────────────────────────────────────────────────────
# Dynamic page title (rendered *after* sidebar)
# ────────────────────────────────────────────────────────────────
if st.session_state.mode == "Chat":
    st.title("💬 Chat over indexed PDFs")
else:
    st.title("📚 Browse indexed PDFs")

# Show availability/caption
available_names = [
    *( ["PyMuPDF"] if SHOW_PYMU else [] ),
    *( ["PyMuPDF (hdr)"] if SHOW_PYMU_HDR else [] ),
    *( ["Docling"] if SHOW_DOCLING else [] ),
    *( ["MarkItDown"] if SHOW_MARKITDOWN else [] ),
    *( ["Marker"] if SHOW_MARKER else [] ),
    *( ["Unstructured"] if SHOW_UNSTRUCTURED else [] ),
]
if available_names:
    st.caption("Converters: " + ", ".join(available_names) + " (side-by-side)")

# ────────────────────────────────────────────────────────────────
# Mode A: Chat
# ────────────────────────────────────────────────────────────────
if st.session_state.mode == "Chat":

    # Explanation of how Chat works
    st.info(
        "When you ask a question, the app **retrieves** top chunks and **generates** an answer from those chunks.\n\n"
        "Compare answers side-by-side for the available converters."
    )

    query = st.chat_input("Ask a question …")

    if query:
        st.chat_message("user").markdown(query)

        # Dynamically render columns for available converters
        conv_defs = [
            *( [("PyMuPDF", "agent_pymu", True)] if SHOW_PYMU else [] ),
            *( [("PyMuPDF (hdr)", "agent_pymu_hdr", True)] if SHOW_PYMU_HDR else [] ),
            *( [("Docling", "agent_docling", DOCLING_AVAILABLE)] if SHOW_DOCLING else [] ),
            *( [("MarkItDown", "agent_markitdown", MARKITDOWN_AVAILABLE)] if SHOW_MARKITDOWN else [] ),
            *( [("Marker", "agent_marker", MARKER_AVAILABLE)] if SHOW_MARKER else [] ),
            *( [("Unstructured", "agent_unstructured", UNSTRUCTURED_AVAILABLE)] if SHOW_UNSTRUCTURED else [] ),
        ]
        active = [(n, k) for (n, k, ok) in conv_defs if ok]
        cols = st.columns(len(active))
        for col, (name, key) in zip(cols, active):
            with col:
                st.subheader(name)
                if key in st.session_state and st.session_state[key].docs:
                    answer, chunks = st.session_state[key].answer(query)
                    st.markdown(answer)
                    if chunks:
                        sources_md = "\n\n---\n\n".join(
                            f"**Chunk {i+1} (score ≈ {score:.2f})**\n\n{chunk}"
                            for i, (chunk, score) in enumerate(chunks)
                        )
                        st.markdown(f"---\n\n### Source chunks\n\n{sources_md}")
                else:
                    st.info(f"Index PDFs for {name} first.")

# ────────────────────────────────────────────────────────────────
# Mode B: Browse converted Markdown
# ────────────────────────────────────────────────────────────────
else:
    st.info(
        "Inspect the converted Markdown and the chunks used for retrieval."
    )

    conv_defs = [
        *( [("PyMuPDF", "agent_pymu", True, "doc_select_pymu")] if SHOW_PYMU else [] ),
        *( [("PyMuPDF (hdr)", "agent_pymu_hdr", True, "doc_select_pymu_hdr")] if SHOW_PYMU_HDR else [] ),
        *( [("Docling", "agent_docling", DOCLING_AVAILABLE, "doc_select_docling")] if SHOW_DOCLING else [] ),
        *( [("MarkItDown", "agent_markitdown", MARKITDOWN_AVAILABLE, "doc_select_markitdown")] if SHOW_MARKITDOWN else [] ),
        *( [("Marker", "agent_marker", MARKER_AVAILABLE, "doc_select_marker")] if SHOW_MARKER else [] ),
        *( [("Unstructured", "agent_unstructured", UNSTRUCTURED_AVAILABLE, "doc_select_unstructured")] if SHOW_UNSTRUCTURED else [] ),
    ]
    active = [(n, k, selkey) for (n, k, ok, selkey) in conv_defs if ok]
    cols = st.columns(len(active))
    for col, (name, key, selkey) in zip(cols, active):
        with col:
            st.subheader(name)
            if key in st.session_state and st.session_state[key].docs:
                fnames = list(st.session_state[key].docs.keys())
                fname = st.selectbox(
                    f"Select a document ({name})",
                    fnames,
                    key=selkey,
                )
                doc = st.session_state[key].docs[fname]
                with st.expander("View Markdown", expanded=True):
                    st.markdown(doc.markdown)
                with st.expander("View Chunks", expanded=True):
                    chunk_string = "\n\n---\n\n".join([c.content for c in doc.chunks])
                    st.markdown(chunk_string)
            else:
                st.info(f"Index some PDFs with {name}.")
