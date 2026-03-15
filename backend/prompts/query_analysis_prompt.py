"""
LLM prompt templates for Query Analysis and Query Classification.

Each template instructs the LLM to return a structured JSON response
that can be parsed by the helpers in utils.llm_helpers.
"""

from langchain_core.prompts import ChatPromptTemplate


# ─── Query Analysis Prompt ────────────────────────────────────────────────────

QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert query analysis engine for an Adaptive-RAG system.\n\n"
            "Your job is to analyze the user's question and determine:\n"
            "1. **intent** — the user's goal. Choose exactly one:\n"
            "   information_lookup | conversational | summarization | "
            "explanation | comparison | real_time_information\n"
            "2. **requires_retrieval** — true if the question refers to uploaded "
            "documents, specific files, or private knowledge that would be "
            "stored in a vector database.\n"
            "3. **requires_web_search** — true if the question asks about "
            "recent events, live data, current news, or anything that "
            "requires up-to-date internet information.\n"
            "4. **query_type** — a short label for the kind of question:\n"
            "   document_question | factual_question | current_events | "
            "opinion_request | general_chat\n\n"
            "Return ONLY a JSON object with these four keys. "
            "Do NOT include any explanation or commentary."
        ),
    ),
    ("human", "Analyze the following query:\n\n{query}"),
])


# ─── Query Classification Prompt ─────────────────────────────────────────────

QUERY_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a query classification engine for an Adaptive-RAG system.\n\n"
            "Given the user's query and previous analysis metadata, classify "
            "the query into exactly ONE of these categories:\n\n"
            "  retriever    — The user is asking about specific documents they "
            "uploaded or content stored in the knowledge base.\n"
            "  web_search   — The user needs real-time, current, or live "
            "information from the internet.\n"
            "  general_llm  — The user is asking a general knowledge question "
            "that can be answered directly by the LLM without any external "
            "retrieval.\n\n"
            "Return ONLY a JSON object with a single key 'classification' "
            "whose value is one of: retriever | web_search | general_llm.\n\n"
            "Do NOT include any explanation or commentary."
        ),
    ),
    (
        "human",
        (
            "Query: {query}\n\n"
            "Analysis metadata:\n{analysis_json}"
        ),
    ),
])
