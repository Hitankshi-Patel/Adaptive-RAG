# Adaptive-RAG AI Assistant Frontend

A modern, professional frontend for the Adaptive-RAG system built with Streamlit. It demonstrates a Query-Aware Retrieval-Augmented Generation system.

## Project Structure

```text
Adaptive-RAG/
│
├── frontend/
│   ├── app.py              # Main Streamlit application
│   └── requirements.txt    # Python dependencies
├── backend/
│   └── mock_backend.py     # Example FastAPI backend for real API testing
└── README.md               # Instructions and documentation
```

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
Open your terminal and navigate to the frontend directory:
```bash
cd frontend
pip install -r requirements.txt
```

*(Optional) If you want to run the example API backend, install fastapi and uvicorn:*
```bash
pip install fastapi uvicorn
```

### 3. Run the Streamlit Application
Start the frontend by running (from the frontend directory):
```bash
streamlit run app.py
```
This will open the interface in your default web browser (usually http://localhost:8501). 

By default, the UI runs in a "Mock Mode" which simulates the delay and response of the API without needing the backend to actually be running.

### 4. Running the Example Backend API (Optional)
If you want to test the real `requests.post()` API integration:

1. Open a **second terminal**, navigate to the backend directory, and run:
   ```bash
   cd backend
   python mock_backend.py
   ```
2. In the Streamlit UI, open the **⚙️ Settings** expander in the bottom of the sidebar.
3. Turn **OFF** the "Use Mock Backend" toggle.
4. Now, any queries you ask will be routed through HTTP POST to `http://localhost:8000/ask`.

## Features Included

- **Modern UI Styling**: Incorporates CSS for a beautiful, robust data visualization styling (Rounded cards, soft shadows)
- **Chat Interface**: Retains state with `st.session_state` and uses native Streamlit chat UI components.
- **Loading Indicators**: Uses step-by-step artificial delays displaying "Analyzing query...", "Retrieving documents...", etc.
- **Analysis View Component**: Displays dynamic classification data seamlessly using colored highlight chips.
- **Pipeline Visualizer**: Provides an architectural understanding flow map of the data parsing algorithm directly on the dashboard.
- **Streaming Responses (Bonus)**: Uses `st.write_stream` generator to yield word tokens smoothly mimicking OpenAI's ChatGPT.
- **Expandable Document Viewers**: Uses `st.expander` to compress heavily retrieved text blocks for readability.
- **Mock/Real Interoperable Modes**: Allows graceful testing without setting up full infra.
