"""
Document Retriever Module.

Provides the complete document retrieval pipeline for the Adaptive-RAG
system, including:

    1. Text embedding generation
    2. Vector store ingestion (FAISS)
    3. Scored similarity retrieval

Supports two embedding backends:
    - **OpenAI** embeddings (default, requires OPENAI_API_KEY)
    - **SentenceTransformers** embeddings (local, no API key needed)

Usage Example::

    from retriever.document_retriever import DocumentRetriever

    retriever = DocumentRetriever(embedding_provider="sentence_transformers")

    # Ingest documents
    chunks = ["Adaptive RAG selects retrieval strategies...", ...]
    retriever.embed_and_store_documents(chunks, metadata={"source": "paper.pdf"})

    # Retrieve relevant context
    results = retriever.retrieve_documents("What is Adaptive RAG?", top_k=4)
    # → [{"content": "...", "source": "paper.pdf", "score": 0.92}, ...]
"""

import logging
import os
from typing import Literal

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "faiss_index",
)


# ─── Embedding Factory ───────────────────────────────────────────────────────


def _create_embeddings(provider: str):
    """
    Instantiate an embedding model based on the chosen provider.

    Args:
        provider: ``'openai'`` or ``'sentence_transformers'``.

    Returns:
        A LangChain-compatible embeddings instance.
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info("Using OpenAI embeddings.")
        return OpenAIEmbeddings()

    if provider == "sentence_transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = "all-MiniLM-L6-v2"
        logger.info("Using SentenceTransformers embeddings: %s", model_name)
        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        "Choose 'openai' or 'sentence_transformers'."
    )


# ─── Core Functions ──────────────────────────────────────────────────────────


def embed_documents(texts: list[str], provider: str = "openai") -> list[list[float]]:
    """
    Generate embedding vectors for a list of text chunks.

    This is a standalone utility when you only need the raw vectors.
    For end-to-end ingestion use :meth:`DocumentRetriever.embed_and_store_documents`.

    Args:
        texts:    List of text strings to embed.
        provider: Embedding backend (``'openai'`` | ``'sentence_transformers'``).

    Returns:
        List of embedding vectors (each a list of floats).
    """
    embeddings_model = _create_embeddings(provider)
    vectors = embeddings_model.embed_documents(texts)
    logger.info("Embedded %d documents using '%s'.", len(vectors), provider)
    return vectors


def store_documents(
    texts: list[str],
    metadatas: list[dict] | None = None,
    provider: str = "openai",
    index_dir: str = DEFAULT_INDEX_DIR,
) -> FAISS:
    """
    Create a FAISS vector store from raw texts and persist it to disk.

    Args:
        texts:     List of text chunks.
        metadatas: Optional per-chunk metadata dicts.
        provider:  Embedding backend.
        index_dir: Directory to persist the FAISS index.

    Returns:
        The populated FAISS vector store instance.
    """
    embeddings_model = _create_embeddings(provider)

    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(texts, metadatas or [{}] * len(texts))
    ]

    vector_store = FAISS.from_documents(docs, embeddings_model)

    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    logger.info("Stored %d documents in FAISS index at %s.", len(docs), index_dir)

    return vector_store


def retrieve_documents(
    query: str,
    top_k: int = 4,
    provider: str = "openai",
    index_dir: str = DEFAULT_INDEX_DIR,
) -> list[dict]:
    """
    Retrieve the top-k most relevant documents for a query (with scores).

    Args:
        query:     Natural-language search query.
        top_k:     Number of results to return.
        provider:  Embedding backend (must match what was used for storage).
        index_dir: Directory where the FAISS index is persisted.

    Returns:
        List of dicts with keys ``content``, ``source``, ``score``.
        Returns an empty list if no index exists.
    """
    index_file = os.path.join(index_dir, "index.faiss")
    if not os.path.exists(index_file):
        logger.warning("No FAISS index found at %s — returning empty results.", index_dir)
        return []

    embeddings_model = _create_embeddings(provider)
    vector_store = FAISS.load_local(
        index_dir, embeddings_model, allow_dangerous_deserialization=True,
    )

    # similarity_search_with_score returns (Document, score) tuples
    results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)

    output = []
    for doc, score in results_with_scores:
        output.append({
            "content": doc.page_content,
            "source": doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
            "score": round(1.0 / (1.0 + score), 4),  # Convert L2 distance → similarity [0, 1]
            "metadata": doc.metadata,
        })

    logger.info(
        "Retrieved %d documents for query '%s' (top score: %.4f).",
        len(output),
        query[:80],
        output[0]["score"] if output else 0.0,
    )
    return output


# ─── DocumentRetriever Class ─────────────────────────────────────────────────


class DocumentRetriever:
    """
    High-level retriever that manages the full lifecycle:
    embedding → storage → retrieval.

    Wraps the module-level functions into a stateful object that can
    be injected into the LangGraph pipeline.

    Args:
        embedding_provider: ``'openai'`` or ``'sentence_transformers'``.
        index_dir:          FAISS index persistence directory.

    Example::

        retriever = DocumentRetriever(embedding_provider="sentence_transformers")

        # Ingest
        retriever.embed_and_store_documents(
            ["chunk 1", "chunk 2"],
            metadata={"source": "paper.pdf"},
        )

        # Retrieve
        results = retriever.retrieve_documents("What is RAG?", top_k=3)
    """

    def __init__(
        self,
        embedding_provider: Literal["openai", "sentence_transformers"] = "openai",
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        self.provider = embedding_provider
        self.index_dir = os.path.abspath(index_dir)
        self._embeddings = _create_embeddings(self.provider)
        self._vector_store: FAISS | None = None
        self._load_existing_index()

    # ── Ingestion ─────────────────────────────────────────────────────────

    def embed_and_store_documents(
        self,
        chunks: list[str],
        metadata: dict | None = None,
    ) -> int:
        """
        Embed text chunks and store them in the FAISS index.

        Args:
            chunks:   List of text strings.
            metadata: Metadata dict applied to every chunk.

        Returns:
            Number of chunks stored.
        """
        docs = [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in chunks
        ]

        if self._vector_store is None:
            self._vector_store = FAISS.from_documents(docs, self._embeddings)
            logger.info("Created new FAISS index with %d documents.", len(docs))
        else:
            self._vector_store.add_documents(docs)
            logger.info("Added %d documents to existing FAISS index.", len(docs))

        self._save_index()
        return len(docs)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve_documents(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Retrieve the most relevant documents with similarity scores.

        Args:
            query: Natural-language search query.
            top_k: Number of results.

        Returns:
            List of dicts with ``content``, ``source``, ``score``.
        """
        if self._vector_store is None:
            logger.warning("No documents indexed — returning empty results.")
            return []

        results_with_scores = self._vector_store.similarity_search_with_score(query, k=top_k)

        output = []
        for doc, score in results_with_scores:
            output.append({
                "content": doc.page_content,
                "source": doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
                "score": round(1.0 / (1.0 + score), 4),
                "metadata": doc.metadata,
            })

        logger.info(
            "Retrieved %d documents (top score: %.4f).",
            len(output),
            output[0]["score"] if output else 0.0,
        )
        return output

    def get_langchain_retriever(self, top_k: int = 4):
        """
        Return a LangChain-compatible retriever interface.

        Useful when integrating with LangChain chains that expect
        a ``BaseRetriever``.

        Args:
            top_k: Number of documents to retrieve.

        Returns:
            A LangChain VectorStoreRetriever, or None if no index exists.
        """
        if self._vector_store is None:
            return None
        return self._vector_store.as_retriever(search_kwargs={"k": top_k})

    def has_documents(self) -> bool:
        """Check whether the index contains any documents."""
        return self._vector_store is not None

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_index(self) -> None:
        if self._vector_store is None:
            return
        os.makedirs(self.index_dir, exist_ok=True)
        self._vector_store.save_local(self.index_dir)
        logger.info("FAISS index saved to %s.", self.index_dir)

    def _load_existing_index(self) -> None:
        index_file = os.path.join(self.index_dir, "index.faiss")
        if os.path.exists(index_file):
            try:
                self._vector_store = FAISS.load_local(
                    self.index_dir,
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded existing FAISS index from %s.", self.index_dir)
            except Exception as e:
                logger.error("Failed to load FAISS index: %s", e)
                self._vector_store = None
        else:
            logger.info("No existing index at %s — starting fresh.", self.index_dir)
