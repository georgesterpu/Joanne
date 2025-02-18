# Joanne: AI Assistant

#### Access here:
https://joanneai.streamlit.app/

----


This is a simple & minimalistic demo web application that leverages Retrieval-Augmented Generation (RAG) to analyse and compare insights from ConocoPhillips' 2023 and 2024 market research reports. This app allows users to interact with the reports using natural language queries and provides AI-powered insights backed by source data.

---


## Technology Stack

### Backend
- **Python**
- **FastAPI**
- **LangChain**
- **FAISS**
- **Hugging Face Embeddings**
- **Google Gemini LLM**

### Frontend
- **Streamlit**

### Deployment
- **Streamlit Cloud**: For hosting the frontend.
- **Render.com**: For hosting the backend API.

---

## Setup Instructions (locally)

### Prerequisites
- uv: https://docs.astral.sh/uv/getting-started/installation/
- A Google Gemini API key: https://aistudio.google.com/apikey

## Steps

#### Set up env
Rename .env.example to .env and update your GEMINI_API_KEY.

#### Start the backend
```bash
uv run uvicorn backend:app --reload
```
#### Start the front-end
```bash
uv run streamlit run frontend.py
```
#### Access the app
Typically at http://localhost:8501

---

## Future Work

- Online vector store management
- Streaming responses
- Chat archive

---
## Recommended Reading
"A Survey of Large Language Models"
https://arxiv.org/pdf/2303.18223
