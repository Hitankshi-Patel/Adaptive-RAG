# Adaptive-RAG AI System

A full-stack **Adaptive Retrieval-Augmented Generation** system that dynamically selects the best answering strategy based on query complexity. Built with Streamlit, FastAPI, and LangGraph.

## Architecture

```
Streamlit UI  →  FastAPI Backend  →  LangGraph Orchestration
                                            ↓
                                     Query Analysis
                                            ↓
                                   Query Classification
                                            ↓
                                     Strategy Router
                                   ┌────────┼────────┐
                              Retriever   Web Search   Direct LLM
                                   └────────┼────────┘
                                            ↓
                                   Response Generator
```

## Project Structure

```text
Adaptive-RAG/
│
├── frontend/
│   ├── app.py                         # Streamlit application
│   └── requirements.txt
│
├── backend/
│   ├── main.py                        # FastAPI entry point
│   ├── requirements.txt
│   ├── api/
│   │   └── rag_routes.py              # REST endpoints
│   ├── schemas/
│   │   └── request_models.py          # Pydantic models
│   ├── analysis/
│   │   ├── query_analysis.py          # Query Analysis module
│   │   └── query_classification.py    # Query Classification engine
│   ├── prompts/
│   │   └── query_analysis_prompt.py   # LLM prompt templates
│   ├── pipelines/
│   │   ├── model_loader.py            # Multi-LLM loader (GPT/Gemini/Claude)
│   │   └── langgraph_pipeline.py      # 7-node LangGraph orchestration
│   ├── services/
│   │   ├── rag_service.py             # Query processing service
│   │   └── document_service.py        # Document upload service
│   ├── vectorstore/
│   │   └── vector_db.py               # FAISS vector store manager
│   └── utils/
│       ├── text_processing.py         # PDF/TXT extraction & chunking
│       └── llm_helpers.py             # Structured output parsing
│
└── README.md
```

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- API keys for one or more LLM providers

### 2. Environment Variables

Create a `.env` file inside the `backend/` directory:

```env
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
ANTHROPIC_API_KEY=your-anthropic-key
TAVILY_API_KEY=your-tavily-key
```

> You only need the key(s) for the model(s) you plan to use.

### 3. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 5. Run the Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 6. Run the Frontend

In a separate terminal:

```bash
cd frontend
streamlit run app.py
```

The UI will be available at `http://localhost:8501`.

### 7. Connect Frontend to Backend

In the Streamlit sidebar, open **⚙️ Settings** and turn **OFF** the "Use Mock Backend" toggle. Select your preferred LLM model from the dropdown in the chat area.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check |
| `/rag/query` | POST | Submit a query to the Adaptive-RAG pipeline |
| `/rag/documents/upload` | POST | Upload a PDF/TXT file for indexing |

### Example — Query

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Adaptive RAG?", "model": "gpt"}'
```

### Example — Document Upload

```bash
curl -X POST http://localhost:8000/rag/documents/upload \
  -F "file=@research_paper.pdf"
```

### Example — Health Check

```bash
curl http://localhost:8000/health
```

## Features

- **Multi-LLM Support** — GPT (OpenAI), Gemini (Google), Claude (Anthropic)
- **Adaptive Routing** — LangGraph dynamically routes queries through analysis → classification → optimal pipeline
- **Three Pipelines** — Document Retriever (FAISS), Web Search (Tavily), Direct LLM
- **Document Indexing** — Upload PDF/TXT files for vector-based retrieval
- **Modern UI** — Streamlit chat interface with streaming responses, pipeline visualization, and analysis dashboard
- **Mock Mode** — Test the UI without a running backend
