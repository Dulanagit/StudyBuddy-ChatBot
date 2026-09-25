"""
app.py
======
StudyBuddy — Streamlit UI Entry Point

Layout:
  SIDEBAR  — API key input, PDF uploader, mode toggle, file status, clear button
  MAIN     — Branded header, chat window (st.chat_message), streaming chat input

Run with:
    streamlit run app.py
"""

import os
import time
import logging

import streamlit as st
from dotenv import load_dotenv

from document_processor import (
    get_or_create_vector_store,
    load_and_split_pdf,
    ingest_documents,
    compute_file_hash,
    is_file_processed,
    mark_file_as_processed,
    get_processed_files,
    delete_file_from_store,
    remove_file_from_registry,
)
from chat_engine import (
    build_chain,
    clear_session_history,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Load GROQ_API_KEY from .env file if present (optional — can also be entered in UI)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed session ID for this single-user Streamlit app
SESSION_ID = "studybuddy_main_session"

# ---------------------------------------------------------------------------
# Page configuration (must be the FIRST Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="StudyBuddy — AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode design
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  /* ---- Google Font ---- */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ---- Global ---- */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ---- App background ---- */
  .stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a3e, #0f0c29);
    min-height: 100vh;
  }

  /* ---- Sidebar ---- */
  [data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.04);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
  }

  [data-testid="stSidebar"] * {
    color: #e0e0f0 !important;
  }

  /* ---- Hero header ---- */
  .hero-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
  }
  .hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
  }
  .hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
  }

  /* ---- Status badges ---- */
  .status-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 0.4rem;
  }
  .badge-success  { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid #34d399; }
  .badge-warning  { background: rgba(251, 191, 36,  0.15); color: #fbbf24; border: 1px solid #fbbf24; }
  .badge-info     { background: rgba(96, 165, 250,  0.15); color: #60a5fa; border: 1px solid #60a5fa; }

  /* ---- Chat bubbles ---- */
  [data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 14px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
    backdrop-filter: blur(8px);
  }

  /* ---- Source expander ---- */
  [data-testid="stExpander"] {
    background: rgba(167, 139, 250, 0.06) !important;
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    border-radius: 10px !important;
  }

  /* ---- Divider ---- */
  hr { border-color: rgba(255,255,255,0.08) !important; }

  /* ---- Sidebar section headers ---- */
  .sidebar-section {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b !important;
    margin: 1.2rem 0 0.4rem;
  }

  /* ---- Warning box ---- */
  .outside-knowledge-banner {
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid rgba(251, 191, 36, 0.4);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    color: #fbbf24;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state():
    """
    Ensures all required session state keys are initialised on first load.
    Streamlit re-runs this file on every user interaction, so we guard
    each key with `if key not in st.session_state`.
    """
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None       # Chroma instance

    if "chain" not in st.session_state:
        st.session_state.chain = None              # RunnableWithMessageHistory

    if "messages" not in st.session_state:
        # List of dicts: {"role": "user"|"assistant", "content": str, "sources": list|None}
        st.session_state.messages = []

    if "use_outside_knowledge" not in st.session_state:
        st.session_state.use_outside_knowledge = False

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL

    if "selected_sources" not in st.session_state:
        # Default: all already-processed files are selected
        st.session_state.selected_sources = set(
            info["filename"] for info in get_processed_files().values()
        )

    if "groq_api_key" not in st.session_state:
        # Pre-fill from .env if available
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")


init_session_state()

# ---------------------------------------------------------------------------
# Auto-initialize on startup
# ---------------------------------------------------------------------------
# If the app restarts (or the page is refreshed), session state is cleared but
# ChromaDB and the processed-files registry persist on disk. This block
# automatically reloads the vector store and builds the chain so the user
# doesn't have to re-upload files just to enable the chat bar.

if (
    st.session_state.vector_store is None          # not yet loaded this session
    and st.session_state.groq_api_key              # API key is available
    and get_processed_files()                       # at least one file is embedded
):
    try:
        st.session_state.vector_store = get_or_create_vector_store()
        rebuild_chain()
        logger.info("Auto-initialized vector store and chain from existing ChromaDB.")
    except Exception as e:
        logger.error(f"Auto-init failed: {e}")
        # Non-fatal — user can still upload files to trigger manual init


# ---------------------------------------------------------------------------
# Helper — rebuild the chain (called when mode or key changes)
# ---------------------------------------------------------------------------

def rebuild_chain():
    """
    (Re)builds the LangChain LCEL retrieval chain.
    Called whenever the vector store, API key, model, mode, or
    source selection changes.
    """
    if st.session_state.vector_store and st.session_state.groq_api_key:
        # Determine active source filter
        selected = list(st.session_state.selected_sources)
        all_files = [info["filename"] for info in get_processed_files().values()]
        # If all files are selected, pass None (no filter = search everything)
        use_filter = 0 < len(selected) < len(all_files)
        st.session_state.chain = build_chain(
            vector_store=st.session_state.vector_store,
            groq_api_key=st.session_state.groq_api_key,
            use_outside_knowledge=st.session_state.use_outside_knowledge,
            model_name=st.session_state.selected_model,
            selected_sources=selected if use_filter else None,
        )


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    # Brand logo / title in sidebar
    st.markdown("## 📚 StudyBuddy")
    st.markdown("*Your AI-powered study companion*")
    st.divider()

    # ---- API Key section ----
    st.markdown('<p class="sidebar-section">🔑 Groq API Key</p>', unsafe_allow_html=True)

    groq_key_input = st.text_input(
        label="Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com",
        label_visibility="collapsed",
    )

    # Update session state and rebuild chain if key changed
    if groq_key_input != st.session_state.groq_api_key:
        st.session_state.groq_api_key = groq_key_input
        rebuild_chain()

    if not st.session_state.groq_api_key:
        st.warning("⚠️ Enter your Groq API key to start chatting.", icon="🔑")
    else:
        st.success("API key loaded ✅", icon="🔑")

    st.divider()

    # ---- Model selector ----
    st.markdown('<p class="sidebar-section">🤖 Model</p>', unsafe_allow_html=True)

    model_labels = list(AVAILABLE_MODELS.keys())
    model_ids    = list(AVAILABLE_MODELS.values())
    current_label = next(
        (k for k, v in AVAILABLE_MODELS.items() if v == st.session_state.selected_model),
        model_labels[0]
    )
    selected_label = st.selectbox(
        label="Groq model",
        options=model_labels,
        index=model_labels.index(current_label),
        label_visibility="collapsed",
        help="Larger models give better answers but may be slightly slower.",
    )
    new_model_id = AVAILABLE_MODELS[selected_label]
    if new_model_id != st.session_state.selected_model:
        st.session_state.selected_model = new_model_id
        rebuild_chain()

    st.divider()

    # ---- Mode toggle ----
    st.markdown('<p class="sidebar-section">🧠 Answer Mode</p>', unsafe_allow_html=True)

    new_outside_knowledge = st.toggle(
        label="Allow outside knowledge",
        value=st.session_state.use_outside_knowledge,
        help=(
            "OFF — Answers ONLY from your uploaded materials.\n\n"
            "ON — May use general knowledge when your materials don't cover the topic. "
            "All general-knowledge answers are clearly labelled ⚠️."
        ),
    )

    # Rebuild chain if mode changed
    if new_outside_knowledge != st.session_state.use_outside_knowledge:
        st.session_state.use_outside_knowledge = new_outside_knowledge
        rebuild_chain()

    if st.session_state.use_outside_knowledge:
        st.markdown(
            '<div class="outside-knowledge-banner">'
            "⚠️ Extended mode active. General knowledge answers will be clearly labelled."
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ---- File upload section ----
    st.markdown('<p class="sidebar-section">📄 Upload Materials</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Upload lecture PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload your lecture slides, syllabi, or textbook chapters (PDF only).",
        label_visibility="collapsed",
    )

    # ---- Process uploaded files ----
    if uploaded_files:
        # Initialise the vector store on first upload (lazy init)
        if st.session_state.vector_store is None:
            with st.spinner("Initialising vector store..."):
                try:
                    st.session_state.vector_store = get_or_create_vector_store()
                    rebuild_chain()
                except Exception as e:
                    st.error(f"Failed to initialise vector store: {e}")
                    logger.error(f"Vector store init error: {e}")

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()

            # Guard against empty files
            if len(file_bytes) == 0:
                st.warning(f"⚠️ '{uploaded_file.name}' is empty — skipping.", icon="📄")
                continue

            file_hash = compute_file_hash(file_bytes)

            # Skip files that have already been embedded
            if is_file_processed(file_hash):
                st.markdown(
                    f"✅ **{uploaded_file.name}** — already in knowledge base",
                )
                continue

            # Process the new file
            with st.spinner(f"Processing **{uploaded_file.name}**..."):
                try:
                    chunks, total_pages = load_and_split_pdf(file_bytes, uploaded_file.name)
                    ingest_documents(chunks, st.session_state.vector_store)
                    mark_file_as_processed(file_hash, uploaded_file.name, len(chunks))
                    # Auto-select the newly uploaded file
                    st.session_state.selected_sources.add(uploaded_file.name)
                    rebuild_chain()

                    st.success(
                        f"✅ **{uploaded_file.name}** added!  \n"
                        f"*{total_pages} pages → {len(chunks)} chunks*"
                    )

                except ValueError as e:
                    # E.g. image-only PDF with no text
                    st.error(f"❌ **{uploaded_file.name}**: {e}", icon="📄")
                    logger.warning(f"ValueError for '{uploaded_file.name}': {e}")

                except RuntimeError as e:
                    # E.g. corrupt PDF or ChromaDB failure
                    st.error(f"❌ **{uploaded_file.name}**: {e}", icon="⚠️")
                    logger.error(f"RuntimeError for '{uploaded_file.name}': {e}")

                except Exception as e:
                    st.error(f"❌ Unexpected error processing '{uploaded_file.name}': {e}")
                    logger.exception(f"Unexpected error: {e}")

    st.divider()

    # ---- Knowledge Base: select / unselect / delete ----
    processed = get_processed_files()
    if processed:
        st.markdown('<p class="sidebar-section">🗂️ Knowledge Base</p>', unsafe_allow_html=True)

        # Sync: remove session state entries for files that no longer exist
        all_filenames = {info["filename"] for info in processed.values()}
        st.session_state.selected_sources &= all_filenames

        selection_changed = False

        for file_hash, info in list(processed.items()):
            filename  = info["filename"]
            chunks    = info["chunks"]
            is_active = filename in st.session_state.selected_sources

            col_check, col_del = st.columns([0.80, 0.20])

            with col_check:
                new_active = st.checkbox(
                    f"📄 {filename}",
                    value=is_active,
                    key=f"kb_check_{file_hash}",
                    help=f"{chunks} chunks · Check to include in retrieval, uncheck to exclude.",
                )
                if new_active != is_active:
                    if new_active:
                        st.session_state.selected_sources.add(filename)
                    else:
                        st.session_state.selected_sources.discard(filename)
                    selection_changed = True

            with col_del:
                st.markdown("<div style='margin-top:0.35rem'></div>", unsafe_allow_html=True)
                if st.button(
                    "🗑️",
                    key=f"kb_del_{file_hash}",
                    help=f"Permanently delete '{filename}' from the knowledge base",
                ):
                    with st.spinner(f"Deleting {filename}..."):
                        try:
                            if st.session_state.vector_store:
                                delete_file_from_store(filename, st.session_state.vector_store)
                            remove_file_from_registry(file_hash)
                            st.session_state.selected_sources.discard(filename)
                            st.toast(f"✅ '{filename}' deleted.")
                        except Exception as e:
                            st.error(f"❌ Delete failed: {e}")
                    st.rerun()

        if selection_changed:
            rebuild_chain()

        # Status hint when files are partially selected
        n_active = len(st.session_state.selected_sources & all_filenames)
        n_total  = len(processed)
        if n_active == 0:
            st.warning("⚠️ No files selected — the bot cannot answer questions.", icon="📂")
        elif n_active < n_total:
            st.info(f"🔍 Searching **{n_active}** of **{n_total}** files.")

        st.divider()

    # ---- Clear chat button ----
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        clear_session_history(SESSION_ID)
        st.rerun()

    # ---- Footer ----
    st.markdown(
        "<br><center style='color:#374151;font-size:0.75rem'>"
        "StudyBuddy · Powered by Groq + LangChain"
        "</center>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MAIN PANEL — Hero header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-header">
  <div class="hero-title">📚 StudyBuddy</div>
  <div class="hero-subtitle">Upload your course materials · Ask questions · Ace your exams</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# MAIN PANEL — Status bar
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

processed = get_processed_files()
doc_count = len(processed)
mode_label = "Extended 🌐" if st.session_state.use_outside_knowledge else "Strict 🔒"
chain_status = "Ready ✅" if st.session_state.chain else "Not initialised ⏳"

with col1:
    st.metric("📂 Documents in KB", doc_count)
with col2:
    st.metric("🧠 Answer Mode", mode_label)
with col3:
    st.metric("⚡ Chain Status", chain_status)

st.divider()

# ---------------------------------------------------------------------------
# MAIN PANEL — Chat window
# ---------------------------------------------------------------------------

# Welcome message when no conversation has started yet
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem; color: #4b5563;">
      <div style="font-size: 3rem; margin-bottom: 1rem;">🎓</div>
      <h3 style="color: #9ca3af; font-weight: 500;">Ready to study?</h3>
      <p style="color: #6b7280; max-width: 400px; margin: 0 auto;">
        Upload your lecture PDFs in the sidebar, then ask me anything about your course materials.
        I'll find the exact passages and explain them clearly.
      </p>
    </div>
    """, unsafe_allow_html=True)

# Render existing conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🎓" if message["role"] == "assistant" else "🧑‍🎓"):
        st.markdown(message["content"])

        # Show source passages in a collapsible expander (assistant messages only)
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📎 View source passages", expanded=False):
                for i, source in enumerate(message["sources"], start=1):
                    src_name = source.metadata.get("source", "Unknown")
                    src_page = source.metadata.get("page", "?")
                    st.markdown(
                        f"**Passage {i}** · `{src_name}` · Page {src_page}\n\n"
                        f"> {source.page_content[:400]}{'...' if len(source.page_content) > 400 else ''}"
                    )
                    if i < len(message["sources"]):
                        st.divider()

# ---------------------------------------------------------------------------
# MAIN PANEL — Chat input
# ---------------------------------------------------------------------------

user_input = st.chat_input(
    placeholder="Ask a question about your uploaded materials...",
    disabled=(st.session_state.chain is None),
)

if user_input:
    # Guard: must have a chain (i.e. vector store ready + API key set)
    if not st.session_state.chain:
        st.error(
            "Please upload at least one PDF and enter your Groq API key before asking questions.",
            icon="⚠️",
        )
        st.stop()

    # Display the user's message immediately
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)

    # Append user message to display history
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": None})

    # Call the chain and stream the response
    with st.chat_message("assistant", avatar="🎓"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("Thinking..."):
                # Use stream() for real-time token display
                stream = st.session_state.chain.stream(
                    {"question": user_input},
                    config={"configurable": {"session_id": SESSION_ID}},
                )

                for token in stream:
                    full_response += token
                    # Update the displayed text as tokens arrive
                    response_placeholder.markdown(full_response + "▌")

            # Remove the cursor indicator and show the final response
            response_placeholder.markdown(full_response)

            # Retrieve source documents separately for attribution display
            # (We re-run similarity search for display purposes only — very fast)
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )
            source_docs = retriever.invoke(user_input)

            # Show source passages in expander
            if source_docs:
                with st.expander("📎 View source passages", expanded=False):
                    for i, doc in enumerate(source_docs, start=1):
                        src_name = doc.metadata.get("source", "Unknown")
                        src_page = doc.metadata.get("page", "?")
                        st.markdown(
                            f"**Passage {i}** · `{src_name}` · Page {src_page}\n\n"
                            f"> {doc.page_content[:400]}{'...' if len(doc.page_content) > 400 else ''}"
                        )
                        if i < len(source_docs):
                            st.divider()

            # Append assistant response + sources to display history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_docs,
            })

        except Exception as e:
            error_msg = f"❌ An error occurred while generating the response: {e}"

            # Common Groq API error hints
            if "401" in str(e) or "authentication" in str(e).lower():
                error_msg = "❌ Invalid Groq API key. Please check the key in the sidebar."
            elif "429" in str(e) or "rate limit" in str(e).lower():
                error_msg = "❌ Rate limit reached. Please wait a moment before asking another question."
            elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                error_msg = "❌ Network error — could not reach Groq API. Check your internet connection."

            response_placeholder.error(error_msg)
            logger.error(f"Chain invocation error: {e}")

            # Append the error as an assistant message so the history is consistent
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": None,
            })
