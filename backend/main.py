"""
Adaptive-RAG Backend — FastAPI Application Entry Point.

Initialises the FastAPI application with:
    - CORS middleware (for Streamlit frontend)
    - Lifespan-managed VectorStoreManager
    - All API routes
    - Structured logging

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.rag_routes import router
from vectorstore.vector_db import VectorStoreManager

# ── Load environment variables (.env file in backend/) ────────────────────────
load_dotenv()

# ── Logging Configuration ────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application-scoped resources.

    On startup: initialise the VectorStoreManager and attach it to
    ``app.state`` so it is available to route handlers via dependency
    injection.

    On shutdown: perform any necessary cleanup.
    """
    logger.info("Starting Adaptive-RAG Backend …")

    # Initialise the vector store (loads existing FAISS index if present)
    vsm = VectorStoreManager()
    app.state.vector_store_manager = vsm
    logger.info("VectorStoreManager initialised.")

    yield  # ← application is running

    logger.info("Shutting down Adaptive-RAG Backend …")


# ── FastAPI Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="Adaptive-RAG Backend",
    description=(
        "REST API for the Adaptive-RAG system.  Bridges the Streamlit "
        "frontend to the LangGraph orchestration pipeline with multi-LLM "
        "support (GPT · Gemini · Claude)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allow the Streamlit frontend (default port 8501) and local dev origins.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "*",  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routes ──────────────────────────────────────────────────────────
app.include_router(router)


# ── Uvicorn Runner (for direct execution: python main.py) ────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
