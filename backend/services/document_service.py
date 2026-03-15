"""
Document Service.

Handles the end-to-end flow of a document upload:
    receive file → extract text → chunk → embed → store in FAISS.
"""

import logging

from fastapi import UploadFile

from utils.text_processing import (
    extract_text_from_pdf,
    extract_text_from_txt,
    split_text_into_chunks,
)
from vectorstore.vector_db import VectorStoreManager

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


async def process_and_index_document(
    file: UploadFile,
    vector_store_manager: VectorStoreManager,
) -> dict:
    """
    Process an uploaded document and index its content.

    Steps:
        1. Validate file type
        2. Read file bytes
        3. Extract text (PDF or TXT)
        4. Split into chunks
        5. Embed and store in FAISS

    Args:
        file:                 The uploaded file from the API endpoint.
        vector_store_manager: Initialised VectorStoreManager instance.

    Returns:
        Dict with ``filename`` and ``chunks_created``.

    Raises:
        ValueError: If the file type is not supported.
    """
    filename = file.filename or "unknown"
    extension = "." + filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    logger.info("Processing document: %s", filename)

    # 1. Read raw bytes
    file_bytes = await file.read()

    # 2. Extract text
    if extension == ".pdf":
        text = extract_text_from_pdf(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text.strip():
        raise ValueError(f"No text content could be extracted from '{filename}'.")

    # 3. Chunk
    chunks = split_text_into_chunks(text)

    # 4. Embed & store
    metadata = {"filename": filename}
    vector_store_manager.add_documents(chunks, metadata=metadata)

    logger.info("Indexed '%s' — %d chunks created.", filename, len(chunks))
    return {"filename": filename, "chunks_created": len(chunks)}
