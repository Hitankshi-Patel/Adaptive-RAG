"""
LangGraph Adaptive-RAG Orchestration Pipeline.

Implements a 7-node StateGraph that dynamically routes queries
through the appropriate processing pipeline:

    START
      → analyze_query_node
      → classify_query_node
      → route_query_node  ──(conditional)──┐
          ├── retriever_pipeline           │
          ├── web_search_pipeline          │
          └── direct_llm_pipeline          │
                       └───────────────────┘
      → generate_response_node
    END

The ``run_adaptive_rag`` function is the single entry point used
by the service layer.
"""

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from analysis.query_analysis import analyze_query
from analysis.query_classification import classify_query
from analysis.query_router import route_from_state
from pipelines.model_loader import load_model
from vectorstore.vector_db import VectorStoreManager

logger = logging.getLogger(__name__)


# ─── State Schema ─────────────────────────────────────────────────────────────


class AdaptiveRAGState(TypedDict, total=False):
    """Shared state object flowing through every node in the graph."""

    query: str
    model: str
    analysis: dict
    classification: str
    documents: list[dict]
    response: str
    sources: list[str]
    # Injected dependency (not serialisable — used within the run only)
    vector_store_manager: object


# ─── Node Functions ───────────────────────────────────────────────────────────


def analyze_query_node(state: AdaptiveRAGState) -> dict:
    """
    Node 1 — Query Analysis.

    Calls the analysis module to determine intent, retrieval needs,
    and web-search needs. Stores the result in ``state['analysis']``.
    """
    llm = load_model(state["model"])
    analysis = analyze_query(state["query"], llm)
    logger.info("Node [analyze_query] complete — intent=%s", analysis.get("intent"))
    return {"analysis": analysis}


def classify_query_node(state: AdaptiveRAGState) -> dict:
    """
    Node 2 — Query Classification.

    Uses the analysis metadata to classify the query into one of:
    ``retriever``, ``web_search``, or ``general_llm``.
    """
    llm = load_model(state["model"])
    classification = classify_query(state["query"], state["analysis"], llm)
    logger.info("Node [classify_query] complete — classification=%s", classification)
    return {"classification": classification}


# Node 3 — Router: delegated to analysis.query_router.route_from_state
# See backend/analysis/query_router.py for the full implementation.


def retriever_pipeline(state: AdaptiveRAGState) -> dict:
    """
    Node 4a — Retriever Pipeline.

    Uses the DocumentRetriever to perform scored similarity search,
    augments the query with retrieved context, and generates an
    answer with the selected LLM.
    """
    from retriever.document_retriever import DocumentRetriever

    llm = load_model(state["model"])
    vsm: VectorStoreManager = state.get("vector_store_manager")  # type: ignore[assignment]

    scored_docs: list[dict] = []
    context = ""
    sources: list[str] = []

    # Try the DocumentRetriever first (provides scored results)
    if vsm and vsm.has_documents():
        retriever = DocumentRetriever.__new__(DocumentRetriever)
        retriever._vector_store = vsm.vector_store
        retriever._embeddings = vsm.embeddings
        retriever.index_dir = vsm.index_dir
        retriever.provider = "openai"

        scored_docs = retriever.retrieve_documents(state["query"], top_k=4)

        for doc in scored_docs:
            source_name = doc.get("source", "knowledge_base")
            if source_name not in sources:
                sources.append(source_name)

        context = "\n\n---\n\n".join(doc["content"] for doc in scored_docs)

    # Augmented generation
    prompt = (
        f"Answer the following question using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        f"If the context does not contain enough information, say so honestly "
        f"and provide the best answer you can."
    )
    response = llm.invoke(prompt)
    logger.info("Node [retriever_pipeline] generated answer from %d scored docs.", len(scored_docs))

    return {
        "documents": scored_docs,
        "response": response.content,
        "sources": sources or ["knowledge_base"],
    }



def web_search_pipeline(state: AdaptiveRAGState) -> dict:
    """
    Node 4b — Web Search Pipeline.

    Delegates to the dedicated ``websearch.web_search`` module for
    Tavily search and LLM-based summarization of results.
    """
    from websearch.web_search import search_web

    llm = load_model(state["model"])
    result = search_web(state["query"], llm, max_results=5)

    web_context = result.get("web_context", "")
    sources = result.get("sources", [])
    raw_results = result.get("raw_results", [])

    # Build document list from raw results for the frontend
    documents = [
        {
            "content": r.get("content", "")[:500],
            "source": r.get("url", "web_search"),
            "title": r.get("title", "Web Result"),
            "metadata": {"source": "web_search"},
        }
        for r in raw_results
    ]

    logger.info("Node [web_search_pipeline] — %d sources, summary=%d chars.", len(sources), len(web_context))

    return {
        "documents": documents,
        "response": web_context,
        "sources": sources or ["web_search"],
    }


def direct_llm_pipeline(state: AdaptiveRAGState) -> dict:
    """
    Node 4c — Direct LLM Pipeline.

    Delegates to the dedicated ``llm.direct_llm`` module which
    selects an intent-aware prompt template and the user's chosen model.
    """
    from llm.direct_llm import generate_direct_response

    # Extract intent from analysis if available (for prompt selection)
    intent = None
    analysis = state.get("analysis")
    if isinstance(analysis, dict):
        intent = analysis.get("intent")

    result = generate_direct_response(
        query=state["query"],
        model_name=state["model"],
        intent=intent,
    )

    logger.info("Node [direct_llm_pipeline] — model=%s, answer_len=%d.", result.get("model"), len(result.get("response", "")))

    return {
        "documents": [],
        "response": result["response"],
        "sources": result.get("sources", ["llm_knowledge"]),
    }


def generate_response_node(state: AdaptiveRAGState) -> dict:
    """
    Node 5 — Response Generator.

    Delegates to the dedicated ``response.response_generator`` module
    which refines the preliminary answer, injects context, and formats
    the output for the API layer.
    """
    from response.response_generator import generate_response

    llm = load_model(state["model"])

    result = generate_response(
        query=state.get("query", ""),
        preliminary_answer=state.get("response", ""),
        documents=state.get("documents", []),
        sources=state.get("sources", []),
        llm=llm,
        strategy=state.get("classification", "unknown"),
    )

    logger.info(
        "Node [generate_response] — strategy=%s, sources=%s, answer_len=%d",
        result.get("strategy"),
        result.get("sources"),
        len(result.get("answer", "")),
    )

    # Map the response generator output back to pipeline state keys
    return {
        "response": result["answer"],
        "sources": result["sources"],
    }


# ─── Graph Construction ──────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """
    Construct and return the Adaptive-RAG LangGraph StateGraph.

    The graph is NOT compiled here so the caller can inspect or
    modify it before compilation if needed.
    """
    graph = StateGraph(AdaptiveRAGState)

    # Add nodes
    graph.add_node("analyze_query_node", analyze_query_node)
    graph.add_node("classify_query_node", classify_query_node)
    graph.add_node("retriever_pipeline", retriever_pipeline)
    graph.add_node("web_search_pipeline", web_search_pipeline)
    graph.add_node("direct_llm_pipeline", direct_llm_pipeline)
    graph.add_node("generate_response_node", generate_response_node)

    # Linear edges
    graph.set_entry_point("analyze_query_node")
    graph.add_edge("analyze_query_node", "classify_query_node")

    # Conditional routing from classify → pipeline (uses dedicated router module)
    graph.add_conditional_edges(
        "classify_query_node",
        route_from_state,
        {
            "retriever_pipeline": "retriever_pipeline",
            "web_search_pipeline": "web_search_pipeline",
            "direct_llm_pipeline": "direct_llm_pipeline",
        },
    )

    # All pipelines converge → generate_response → END
    graph.add_edge("retriever_pipeline", "generate_response_node")
    graph.add_edge("web_search_pipeline", "generate_response_node")
    graph.add_edge("direct_llm_pipeline", "generate_response_node")
    graph.add_edge("generate_response_node", END)

    return graph


# ─── Public Entry Point ──────────────────────────────────────────────────────


def run_adaptive_rag(
    query: str,
    model: str = "gpt",
    vector_store_manager: VectorStoreManager | None = None,
) -> dict:
    """
    Execute the full Adaptive-RAG pipeline.

    This is the single function called by the service layer.  It
    builds the graph, injects the initial state, and returns the
    final state dictionary.

    Args:
        query:                The user's question.
        model:                LLM provider key ('gpt', 'gemini', 'claude').
        vector_store_manager: Initialised VectorStoreManager instance.

    Returns:
        Dictionary with keys: response, classification (strategy), sources, etc.
    """
    logger.info("Running Adaptive-RAG — query='%s', model='%s'", query[:80], model)

    graph = build_graph()
    app = graph.compile()

    initial_state: AdaptiveRAGState = {
        "query": query,
        "model": model,
        "analysis": {},
        "classification": "",
        "documents": [],
        "response": "",
        "sources": [],
        "vector_store_manager": vector_store_manager,
    }

    final_state = app.invoke(initial_state)

    logger.info(
        "Adaptive-RAG complete — strategy=%s, answer_len=%d",
        final_state.get("classification"),
        len(final_state.get("response", "")),
    )
    return final_state
