# Multi-Agent Document Chat

This project implements a sophisticated, 3-Agent Retrieval-Augmented Generation (RAG) system for chatting with documents. Instead of relying on a single AI model to handle everything, the workload is divided into specific roles (Retriever, Analyst, and Answer Agents) to provide accurate, well-reasoned answers backed by exact sources.


## Project Structure

```text
Multi-Agent-Document-Chat/
├── app/
│   ├── agents/
│   │   ├── analyst/
│   │   │   ├── agent.py
│   │   │   └── tools/              # Calculator, data analysis, tables, comparisons
│   │   ├── answer/
│   │   │   ├── agent.py
│   │   │   └── tools/              # Citations and response formatting
│   │   ├── report/
│   │   │   ├── agent.py
│   │   │   └── tools/              # PDF document, layout, style, text, tables, cover
│   │   └── retriever/
│   │       ├── agent.py
│   │       └── tools/              # Search, query rewriting, filtering, reranking
│   ├── api/                        # Chat, voice, sources, and report endpoints
│   ├── core/                       # Settings, contracts, errors, and logging
│   ├── orchestrator/               # Graph workflow, nodes, and state
│   ├── pipeline/                   # Loading, parsing, cleaning, chunking, ingestion, indexing
│   ├── services/                   # OCR and multilingual speech transcription
│   └── main.py                     # FastAPI application and service initialization
├── data/                           # Runtime uploads, vector database, and model cache
├── tests/                          # Unit and API tests
├── .env.example                    # Example environment configuration
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── README.md
├── requirements.txt
└── streamlit_app.py                # Streamlit chat and document management UI
```

### Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (or via Poetry/Pipenv).
3. Set up your environment variables (API keys, DB connection strings) in a `.env` file.

### Running the App

#### Option 1: Docker Compose (All-in-One, Recommended)
Run both the Streamlit UI and the FastAPI backend in containerized environments with persistent Chroma vector storage and cached embedding models:
```bash
docker compose up --build
```
* **Streamlit UI**: Access at `http://localhost:8501`
* **FastAPI Backend & Docs**: Access at `http://localhost:8000/docs`

Voice questions can be recorded from the Streamlit chat in English or Arabic. Recognition defaults to automatic language detection; the language selector can also pin English or Arabic for better recognition. The API uses the multilingual `faster-whisper` `base` model on CPU with int8 compute, then sends the transcript through the same document chat workflow. The model is downloaded the first time a voice question is processed and cached in the container's Hugging Face cache volume. Audio is limited to 10 MB by default (`MAX_AUDIO_UPLOAD_MB`) and is not stored by the application.

The voice endpoint is `POST /chat/voice` and accepts a WAV `audio` file, optional `history` form data containing a JSON array of chat messages, and an optional `language` (`auto`, `en`, or `ar`; defaults to `auto`). Its response includes the recognized `transcript` and the normal chat answer and citation fields. Set `WHISPER_MODEL` to another multilingual faster-whisper model if you want to trade CPU use and model size for recognition quality; English-only models such as `base.en` do not support Arabic.

The document uploader also accepts PNG, JPG/JPEG, WEBP, and TIFF images. RapidOCR runs locally on CPU with Arabic and English recognition; its model files are downloaded on first OCR use and cached under `data/rapidocr_models` (mounted as `/app/data` in Docker). Extracted text is indexed through the normal document pipeline and citations refer to the image filename and page 1. Scanned PDF pages are not OCR processed.

To run only the Streamlit UI:
```bash
docker compose up --build streamlit
```

To run only the FastAPI backend:
```bash
docker compose up --build api
```

#### Option 2: Local Streamlit Interactive Web App
Launch the Streamlit user interface locally without Docker:
```bash
streamlit run streamlit_app.py
```

#### Option 3: Local FastAPI Backend Server
Run the REST API server locally:
```bash
uvicorn app.main:app --reload
```
API Documentation will be available at `http://localhost:8000/docs`.
