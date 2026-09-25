"""
document_processor.py
=====================
Handles all document ingestion logic for StudyBuddy:
  - Loading and parsing PDF files using PyMuPDF (fitz)
  - Splitting text into overlapping chunks via RecursiveCharacterTextSplitter
  - Initialising (or reloading) the persistent ChromaDB vector store
  - Tracking which files have already been embedded to avoid duplicate work
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

import pymupdf as fitz  # PyMuPDF 1.25+ renamed module from fitz → pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Where ChromaDB stores its files on disk (relative to project root)
CHROMA_PERSIST_DIR = "./chroma_db"

# JSON file that maps file MD5 hashes → filenames of already-processed PDFs
PROCESSED_FILES_REGISTRY = os.path.join(CHROMA_PERSIST_DIR, "processed_files.json")

# The ChromaDB collection name for all study materials
COLLECTION_NAME = "study_materials"

# Sentence-Transformers model for generating embeddings.
# all-MiniLM-L6-v2 is fast, lightweight (~90 MB), and works well for academic text.
# Downloaded automatically on first run; cached locally afterward.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Text chunking parameters:
# 2000 chars ≈ 500 tokens (GPT-style tokeniser), 200-char overlap ≈ 10%
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding initialisation
# ---------------------------------------------------------------------------

def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFace sentence-transformers embedding function.

    The model is downloaded from HuggingFace Hub on first call and then
    cached locally, so subsequent calls are instantaneous.

    Returns:
        HuggingFaceEmbeddings: The configured embedding function.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},       # Use CPU — works on all machines
        encode_kwargs={"normalize_embeddings": True},  # Normalise for cosine similarity
    )


# ---------------------------------------------------------------------------
# Vector store initialisation
# ---------------------------------------------------------------------------

def get_or_create_vector_store() -> Chroma:
    """
    Initialises (or reloads) the persistent ChromaDB vector store.

    If the chroma_db/ directory already exists, the existing collection is
    loaded from disk. Otherwise, a new one is created. This means embeddings
    persist across application restarts.

    Returns:
        Chroma: The vector store client, ready for adding documents or querying.
    """
    # Ensure the persistence directory exists
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_function()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready at '{CHROMA_PERSIST_DIR}'")
    return vector_store


# ---------------------------------------------------------------------------
# Duplicate-detection helpers (MD5 hash registry)
# ---------------------------------------------------------------------------

def _load_processed_registry() -> dict:
    """
    Loads the JSON registry of already-processed file hashes from disk.

    Returns:
        dict: Maps MD5 hash strings → {"filename": str, "chunks": int}
              Returns an empty dict if the file doesn't exist yet.
    """
    if os.path.exists(PROCESSED_FILES_REGISTRY):
        try:
            with open(PROCESSED_FILES_REGISTRY, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load processed files registry: {e}. Starting fresh.")
    return {}


def _save_processed_registry(registry: dict) -> None:
    """
    Persists the processed-files registry to disk.

    Args:
        registry (dict): The registry to save.
    """
    try:
        with open(PROCESSED_FILES_REGISTRY, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save processed files registry: {e}")


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Computes an MD5 hash of raw file bytes for duplicate detection.

    Using content hash (not filename) ensures the same file uploaded under
    a different name is correctly identified as a duplicate.

    Args:
        file_bytes (bytes): Raw bytes of the uploaded file.

    Returns:
        str: Hex-encoded MD5 digest.
    """
    return hashlib.md5(file_bytes).hexdigest()


def is_file_processed(file_hash: str) -> bool:
    """
    Checks whether a file with the given MD5 hash has already been embedded.

    Args:
        file_hash (str): MD5 hash of the file to check.

    Returns:
        bool: True if the file has already been processed, False otherwise.
    """
    registry = _load_processed_registry()
    return file_hash in registry


def mark_file_as_processed(file_hash: str, filename: str, chunk_count: int) -> None:
    """
    Records a file as processed in the persistent registry.

    Args:
        file_hash (str): MD5 hash of the processed file.
        filename (str): Original filename (for informational purposes).
        chunk_count (int): Number of chunks that were embedded.
    """
    registry = _load_processed_registry()
    registry[file_hash] = {
        "filename": filename,
        "chunks": chunk_count,
    }
    _save_processed_registry(registry)
    logger.info(f"Marked '{filename}' as processed ({chunk_count} chunks, hash={file_hash[:8]}...)")


def get_processed_files() -> dict:
    """
    Returns the full registry of processed files.

    Returns:
        dict: Maps MD5 hash → {"filename": str, "chunks": int}
    """
    return _load_processed_registry()


# ---------------------------------------------------------------------------
# PDF loading and chunking
# ---------------------------------------------------------------------------

def load_and_split_pdf(file_bytes: bytes, filename: str) -> Tuple[List[Document], int]:
    """
    Loads a PDF from raw bytes, extracts text page-by-page, and splits
    the text into overlapping chunks suitable for embedding.

    Each chunk carries metadata including the source filename and page number,
    so the chat engine can cite the origin of each retrieved passage.

    Args:
        file_bytes (bytes): Raw bytes of the uploaded PDF file.
        filename (str): Original filename, used for metadata and logging.

    Returns:
        Tuple[List[Document], int]:
            - List of Document chunks ready for embedding
            - Total number of pages in the PDF

    Raises:
        ValueError: If the PDF contains no extractable text.
        RuntimeError: If PyMuPDF fails to open the file.
    """
    logger.info(f"Loading PDF: {filename}")

    try:
        # Open PDF from in-memory bytes (no temp file needed)
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{filename}': {e}") from e

    total_pages = len(pdf_doc)
    raw_documents: List[Document] = []

    for page_num, page in enumerate(pdf_doc, start=1):
        page_text = page.get_text("text")  # Extract plain text

        # Skip pages that have no readable text (e.g. image-only pages)
        if not page_text.strip():
            logger.warning(f"  Page {page_num}/{total_pages} in '{filename}' has no text — skipping.")
            continue

        raw_documents.append(
            Document(
                page_content=page_text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "total_pages": total_pages,
                },
            )
        )

    pdf_doc.close()

    if not raw_documents:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "The PDF may be image-based (scanned). Please use a text-based PDF."
        )

    logger.info(f"  Extracted text from {len(raw_documents)}/{total_pages} pages.")

    # Split each page's Document into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # These separators are tried in order — paragraph > sentence > word > char
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(raw_documents)
    logger.info(f"  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    return chunks, total_pages


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_documents(chunks: List[Document], vector_store: Chroma) -> None:
    """
    Embeds a list of Document chunks and adds them to the ChromaDB vector store.

    This is the final step in the ingestion pipeline. Each chunk is embedded
    using the HuggingFace model and stored persistently in ChromaDB.

    Args:
        chunks (List[Document]): The chunks to embed and store.
        vector_store (Chroma): The target ChromaDB vector store.

    Raises:
        RuntimeError: If the embedding or storage operation fails.
    """
    if not chunks:
        raise ValueError("No chunks to ingest — the document list is empty.")

    try:
        logger.info(f"Embedding and storing {len(chunks)} chunks into ChromaDB...")
        vector_store.add_documents(chunks)
        logger.info("Ingestion complete.")
    except Exception as e:
        raise RuntimeError(f"Failed to ingest documents into ChromaDB: {e}") from e


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_file_from_store(filename: str, vector_store: Chroma) -> int:
    """
    Deletes all embedded chunks for the given filename from ChromaDB.

    Uses ChromaDB's metadata filter to find all chunks whose 'source'
    metadata field matches the filename, then deletes them by ID.

    Args:
        filename (str): The original filename as stored in chunk metadata.
        vector_store (Chroma): The ChromaDB vector store to delete from.

    Returns:
        int: Number of chunks deleted.

    Raises:
        RuntimeError: If the deletion fails.
    """
    try:
        # Retrieve the IDs of all chunks belonging to this file
        result = vector_store._collection.get(where={"source": filename})
        chunk_ids = result.get("ids", [])

        if not chunk_ids:
            logger.warning(f"No chunks found for '{filename}' — nothing to delete.")
            return 0

        vector_store._collection.delete(ids=chunk_ids)
        logger.info(f"Deleted {len(chunk_ids)} chunks for '{filename}' from ChromaDB.")
        return len(chunk_ids)

    except Exception as e:
        raise RuntimeError(f"Failed to delete '{filename}' from ChromaDB: {e}") from e


def remove_file_from_registry(file_hash: str) -> None:
    """
    Removes a file entry from the processed-files JSON registry.

    After calling this, the file will no longer be recognised as
    already-processed, so it can be re-uploaded and re-embedded.

    Args:
        file_hash (str): MD5 hash key of the file to remove.
    """
    registry = _load_processed_registry()
    if file_hash in registry:
        removed = registry.pop(file_hash)
        _save_processed_registry(registry)
        logger.info(f"Removed '{removed['filename']}' from processed files registry.")
    else:
        logger.warning(f"Hash {file_hash[:8]}... not found in registry — nothing removed.")

