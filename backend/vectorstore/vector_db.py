"""
FAISS Vector Store Manager.

Manages document embedding storage and similarity search using FAISS,
with on-disk persistence so the index survives server restarts.
"""

import logging
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Default persistence directory (relative to backend/)
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")


class VectorStoreManager:
    """
    Manages the lifecycle of a FAISS vector store:
    initialisation, document ingestion, search, and persistence.
    """

    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """
        Initialise the manager and load an existing index if available.

        Args:
            index_dir: Directory where the FAISS index is persisted.
        """
        self.index_dir = os.path.abspath(index_dir)
        self.vector_store: FAISS | None = None
        self.embeddings = None
        try:
            self.embeddings = OpenAIEmbeddings()
            self._load()
        except Exception as e:
            logger.warning("OpenAIEmbeddings failed to initialize (missing API key?): %s", e)
            logger.warning("VectorStoreManager will remain empty until an API key is provided.")

    # ── Public API ────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[str], metadata: dict | None = None) -> int:
        """
        Embed text chunks and add them to the FAISS index.

        Args:
            chunks:   List of text strings to embed.
            metadata: Optional metadata dict attached to every chunk.

        Returns:
            Number of chunks added.
        """
        docs = [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in chunks
        ]

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            logger.info("Created new FAISS index with %d documents.", len(docs))
        else:
            self.vector_store.add_documents(docs)
            logger.info("Added %d documents to existing FAISS index.", len(docs))

        self._save()
        return len(docs)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """
        Retrieve the top-k most relevant documents for a query.

        Args:
            query: Natural-language search query.
            k:     Number of results to return.

        Returns:
            List of LangChain Document objects (may be empty if no index exists).
        """
        if self.vector_store is None:
            logger.warning("No FAISS index loaded; returning empty results.")
            return []

        results = self.vector_store.similarity_search(query, k=k)
        logger.info("Similarity search returned %d results for query: '%s'", len(results), query[:80])
        return results

    def has_documents(self) -> bool:
        """Check whether the vector store contains any documents."""
        return self.vector_store is not None

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist the current FAISS index to disk."""
        if self.vector_store is None:
            return
        os.makedirs(self.index_dir, exist_ok=True)
        self.vector_store.save_local(self.index_dir)
        logger.info("FAISS index saved to %s", self.index_dir)

    def _load(self) -> None:
        """Load an existing FAISS index from disk, if present."""
        index_file = os.path.join(self.index_dir, "index.faiss")
        if os.path.exists(index_file):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded existing FAISS index from %s", self.index_dir)
            except Exception as e:
                logger.error("Failed to load FAISS index: %s", e)
                self.vector_store = None
        else:
            logger.info("No existing FAISS index found at %s — starting fresh.", self.index_dir)
