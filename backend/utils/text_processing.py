"""
Text extraction and chunking utilities.

Handles PDF and TXT file processing, and splits raw text into
overlapping chunks suitable for embedding and vector storage.
"""

import logging
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ─── Text Extraction ─────────────────────────────────────────────────────────


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text content from a PDF file.

    Writes the bytes to a temporary file so PyPDFLoader can process it,
    then concatenates all page texts.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        Combined text from all pages of the PDF.
    """
    text = ""
    tmp_path = None
    try:
        # Write bytes to a temp file (PyPDFLoader requires a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        text = "\n".join(page.page_content for page in pages)
        logger.info("Extracted text from PDF (%d pages, %d chars)", len(pages), len(text))

    except Exception as e:
        logger.error("Failed to extract text from PDF: %s", e)
        raise

    finally:
        # Clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return text


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decode raw bytes from a plain-text file.

    Args:
        file_bytes: Raw bytes of the TXT file.

    Returns:
        Decoded string content.
    """
    try:
        text = file_bytes.decode("utf-8")
        logger.info("Extracted text from TXT file (%d chars)", len(text))
        return text
    except UnicodeDecodeError as e:
        logger.error("UTF-8 decode failed: %s", e)
        raise ValueError("The uploaded text file is not valid UTF-8.") from e


# ─── Text Chunking ───────────────────────────────────────────────────────────


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split raw text into overlapping chunks for embedding.

    Uses LangChain's RecursiveCharacterTextSplitter which tries to split
    on natural boundaries (paragraphs, sentences, words) before falling
    back to character-level splits.

    Args:
        text:          The full document text.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    logger.info("Split text into %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)
    return chunks
