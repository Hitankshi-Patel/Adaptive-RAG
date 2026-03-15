import streamlit as st
import requests
import json
import time

# --- CONSTANTS ---
API_BASE_URL = "http://localhost:8000"
QUERY_URL = f"{API_BASE_URL}/rag/query"
UPLOAD_URL = f"{API_BASE_URL}/rag/documents/upload"
HEALTH_URL = f"{API_BASE_URL}/health"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Adaptive-RAG AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Clean, modern typographic scale */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Query analysis card styling */
    .query-analysis-card {
        background-color: #f7f9fc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }
    .query-analysis-card:hover {
        transform: translateY(-2px);
    }
    
    /* Pipeline flow visualization */
    .pipeline-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        background: linear-gradient(145deg, #ffffff, #f0f4f8);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e1e4e8;
    }
    .pipeline-node {
        background: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        font-weight: 600;
        color: #1e293b;
        width: 80%;
        text-align: center;
        margin: 5px 0;
    }
    .pipeline-arrow {
        color: #64748b;
        font-size: 20px;
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "current_docs" not in st.session_state:
    st.session_state.current_docs = None

def clear_chat():
    st.session_state.messages = []
    st.session_state.current_analysis = None
    st.session_state.current_docs = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Adaptive-RAG")
    st.caption("v1.0.0 - AI Research Assistant")
    
    st.button("➕ New Chat", use_container_width=True, on_click=clear_chat)
    
    st.divider()
    
    st.markdown("### Example Questions")
    example_questions = [
        "What is Adaptive RAG?",
        "Explain query complexity classification",
        "How does iterative retrieval work?"
    ]
    
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.example_query = q

    st.divider()

    st.markdown("### Document Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        help="These documents will be indexed and used by the Retriever RAG strategy to answer your questions."
    )
    
    if uploaded_files:
        if st.button("📤 Upload & Index Documents", use_container_width=True):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        resp = requests.post(UPLOAD_URL, files=files, timeout=60)
                        resp.raise_for_status()
                        result = resp.json()
                        st.success(f"✅ **{result['filename']}** — {result['chunks_created']} chunks indexed.")
                    except Exception as e:
                        st.error(f"❌ Failed to upload {uploaded_file.name}: {e}")

    st.divider()

    st.markdown("### About Adaptive-RAG")
    st.info(
        "Adaptive-RAG is a Retrieval-Augmented Generation system that dynamically "
        "selects the best answering strategy depending on the complexity of the user's query.\n\n"
        "**Strategies:**\n"
        "- 🤖 Direct LLM (general knowledge)\n"
        "- 📚 Retriever RAG (document-based)\n"
        "- 🌐 Web Search (real-time info)"
    )
    
    with st.expander("⚙️ Settings"):
        st.text_input("Backend API URL", value=API_BASE_URL, key="api_url")
        st.toggle("Use Mock Backend", value=True, key="use_mock", help="Useful for UI testing without a real backend.")

# --- MAIN INTERFACE ---
st.title("Adaptive-RAG AI Assistant")
st.subheader("Query-Aware Retrieval-Augmented Generation System")

# Create a clean layout with columns for main chat vs system context/analysis
col_chat, col_context = st.columns([6, 4], gap="large")

with col_chat:
    st.markdown("### Conversation")
    
    # Model selector (right above the chat input)
    selected_model = st.selectbox(
        "🧠 Select LLM Model",
        options=["gpt", "gemini", "claude"],
        format_func=lambda x: {"gpt": "GPT (OpenAI)", "gemini": "Gemini (Google)", "claude": "Claude (Anthropic)"}[x],
        index=0,
        help="Choose which LLM provider to use for query processing.",
    )
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle input (either typed or from example button)
    user_query = st.chat_input("Ask a question about Adaptive-RAG or its strategies...")
    
    if "example_query" in st.session_state:
        user_query = st.session_state.example_query
        del st.session_state.example_query

    if user_query:
        # 1. Add and display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Add assistant response placeholder and loading indicators
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Step-based loading indicators
            with st.spinner("🔍 Analyzing query complexity..."):
                time.sleep(0.5)
            with st.spinner("🔀 Routing to optimal pipeline..."):
                time.sleep(0.3)
            with st.spinner("✨ Generating response..."):
                time.sleep(0.3)
            
            # 3. Call backend API or use mock
            try:
                if st.session_state.use_mock:
                    # Mock Response Generation
                    complexity = "Complex" if len(user_query) > 30 else "Simple"
                    strategy = "retriever" if complexity == "Complex" else "general_llm"
                    strategy_label = "Retriever RAG" if strategy == "retriever" else "Direct LLM"
                    docs = [
                        {
                            "title": "Adaptive RAG Research Paper",
                            "score": 0.95,
                            "content": "Adaptive RAG introduces a dynamically adjusting method to evaluate query complexity before deciding the retrieval path."
                        },
                        {
                            "title": "System Architecture Overview",
                            "score": 0.88,
                            "content": "The Strategy Router acts as a gateway, deciding between Direct LLM generation or initiating single to multi-hop retrieval workflows."
                        }
                    ] if complexity == "Complex" else [
                        {
                            "title": "Basic Definitions Reference",
                            "score": 0.99,
                            "content": "A simple query often does not require external retrieval and can be confidently answered by the LLM's parametric memory."
                        }
                    ]
                    
                    data = {
                        "answer": f"**Adaptive RAG** has analyzed your query: \n\n> *\"{user_query}\"*\n\nBased on our classification, we determined this is a **{complexity.lower()}** query requiring a **{strategy_label}** approach. Here is the generated response:\n\nAdaptive RAG optimizes response latency and accuracy by skipping unnecessary document retrieval for simple queries, while ensuring deep research is conducted for complex logical or factual questions.",
                        "strategy": strategy,
                        "sources": ["document_1.pdf", "document_2.pdf"] if strategy == "retriever" else ["llm_knowledge"],
                        "documents": docs
                    }
                else:
                    # Real Backend API Call
                    api_base = st.session_state.api_url
                    response = requests.post(
                        f"{api_base}/rag/query",
                        json={"query": user_query, "model": selected_model},
                        timeout=120,
                    )
                    response.raise_for_status()
                    data = response.json()
                    # Map strategy to display-friendly format
                    data["documents"] = data.get("documents", [])
                
                # Update Session State with Context
                answer = data.get("answer", "No answer found.")
                strategy_map = {
                    "retriever": "Retriever RAG",
                    "web_search": "Web Search",
                    "general_llm": "Direct LLM",
                }
                st.session_state.current_analysis = {
                    "complexity": strategy_map.get(data.get("strategy", ""), data.get("strategy", "Unknown")),
                    "strategy": strategy_map.get(data.get("strategy", ""), data.get("strategy", "Unknown")),
                }
                st.session_state.current_docs = data.get("documents", [])
                
                # Stream the response text
                def stream_text(text):
                    for word in text.split(" "):
                        yield word + " "
                        time.sleep(0.04)
                
                response_placeholder.write_stream(stream_text(answer))
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                # Fallback error handling
                error_msg = f"**Error calling backend API:**\n`{str(e)}`"
                response_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

with col_context:
    st.markdown("### System Dashboard")
    
    # ADAPTIVE PIPELINE FLOW
    st.markdown("#### Pipeline Visualization")
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-node">👤 User Query</div>
        <div class="pipeline-arrow">↓</div>
        <div class="pipeline-node">🔍 Query Analysis</div>
        <div class="pipeline-arrow">↓</div>
        <div class="pipeline-node">🏷️ Query Classification</div>
        <div class="pipeline-arrow">↓</div>
        <div class="pipeline-node">🔀 Strategy Router</div>
        <div class="pipeline-arrow">↓</div>
        <div class="pipeline-node" style="border: 1px dashed #94a3b8;">📚 Retriever · 🌐 Web Search · 🤖 Direct LLM</div>
        <div class="pipeline-arrow">↓</div>
        <div class="pipeline-node">✨ Response Generator</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # QUERY ANALYSIS PANEL
    st.markdown("#### Query Analysis")
    if st.session_state.current_analysis:
        st.markdown(f"""
        <div class="query-analysis-card">
            <h4 style="margin-top:0; color: #334155;">Latest Execution</h4>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #64748b; font-weight: 500;">Pipeline:</span>
                <span style="background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 4px; font-weight: 600;">{st.session_state.current_analysis['complexity']}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #64748b; font-weight: 500;">Strategy:</span>
                <span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 4px; font-weight: 600;">{st.session_state.current_analysis['strategy']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Submit a query to see analysis.")
        
    # RETRIEVED DOCUMENTS SECTION
    st.markdown("#### Retrieved Documents")
    if st.session_state.current_docs:
        for i, doc in enumerate(st.session_state.current_docs):
            title = doc.get("title", f"Document {i+1}")
            score = doc.get("score", "N/A")
            with st.expander(f"📄 {title}  (Score: {score})"):
                st.markdown(f"**Content Snippet:**")
                st.caption(doc.get("content", "No content snippet provided."))
    else:
        st.info("No documents retrieved for this query.")
