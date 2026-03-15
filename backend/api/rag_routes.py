"""
FastAPI Route Definitions for the Adaptive-RAG Backend.

Endpoints:
    GET  /health                 — service health check
    POST /rag/query              — submit a query to the adaptive pipeline
    POST /rag/documents/upload   — upload a document for indexing
"""

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from schemas.request_models import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from services.document_service import process_and_index_document
from services.rag_service import process_query

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Health Check ─────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verify that the backend service is running.",
)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", service="adaptive-rag-backend")


# ─── Query Endpoint ──────────────────────────────────────────────────────────


@router.post(
    "/rag/query",
    response_model=QueryResponse,
    summary="Submit Query",
    description="Send a user query to the Adaptive-RAG pipeline for processing.",
)
async def query_rag(request: QueryRequest, req: Request) -> QueryResponse:
    """
    Accept a user query and run it through the LangGraph pipeline.

    The ``model`` field in the request body selects the LLM provider
    (gpt / gemini / claude).
    """
    logger.info("POST /rag/query — query='%s', model='%s'", request.query[:80], request.model)

    try:
        # Retrieve the VectorStoreManager from app state
        vsm = req.app.state.vector_store_manager
        response = await process_query(request.query, request.model, vsm)
        return response

    except ValueError as e:
        logger.warning("Validation error in /rag/query: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("Unexpected error in /rag/query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ─── Document Upload Endpoint ────────────────────────────────────────────────


@router.post(
    "/rag/documents/upload",
    response_model=UploadResponse,
    summary="Upload Document",
    description="Upload a PDF or TXT file for indexing in the vector database.",
)
async def upload_document(req: Request, file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a file upload, extract and chunk the text, generate
    embeddings, and store them in FAISS.
    """
    logger.info("POST /rag/documents/upload — filename='%s'", file.filename)

    try:
        vsm = req.app.state.vector_store_manager
        result = await process_and_index_document(file, vsm)

        return UploadResponse(
            message="Document uploaded and indexed successfully",
            filename=result["filename"],
            chunks_created=result["chunks_created"],
        )

    except ValueError as e:
        logger.warning("Validation error in /rag/documents/upload: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("Unexpected error in /rag/documents/upload: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
