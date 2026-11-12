# Multi-Agent Document Chat

This project implements a sophisticated, 3-Agent Retrieval-Augmented Generation (RAG) system for chatting with documents. Instead of relying on a single AI model to handle everything, the workload is divided into specific roles (Retriever, Analyst, and Answer Agents) to provide accurate, well-reasoned answers backed by exact sources.


## Project Structure

```text
Multi-Agent-Document-Chat/
├── app/
│   ├── core/
│   │   ├── config.py            ← settings (max_rounds, top-k, models)
│   │   ├── contract.py          ← EvidenceChunk, EvidenceBundle, IngestResult
│   │   ├── errors.py            ← custom errors 
│   │   └── logging.py           ← shared logger, useful for tracing graph rounds
│   ├── pipeline/                ← Document Preparation Pipeline
│   │   ├── loader.py            ← Handles file uploads (PDFs, Word, text)
│   │   ├── parser.py            ← Extracts raw text from loaded files
│   │   ├── cleaner.py           ← Cleans up the text before chunking
│   │   ├── chunker.py           ← Breaks text into smaller, readable chunks
│   │   ├── embeddings.py        ← Converts chunks into searchable numerical data
│   │   └── vector_db.py         ← Stores embeddings for fast/precise searching
│   ├── agents/
│   │   ├── retriever/           ← Searches Vector DB without answering the question itself
│   │   │   ├── agent.py         ← Retriever agent execution logic
│   │   │   └── tools/
│   │   │       ├── query_rewriter.py    ← Transforms user question into a clearer query
│   │   │       ├── semantic_search.py   ← Retrieves chunks based on meaning using embeddings
│   │   │       ├── keyword_search.py    ← Retrieves documents via exact-term matching
│   │   │       ├── metadata_filter.py   ← Filters by document name, page, author, date
│   │   │       ├── reranker.py          ← Reorders initial results for relevance
│   │   │       └── context_selector.py  ← Selects precise evidence chunks to limit context size
│   │   ├── analyst/             ← Evaluates data, compares facts, performs heavy lifting
│   │   │   ├── agent.py         ← Analyst agent execution logic
│   │   │   └── tools/
│   │   │       ├── calculator.py          ← Performs accurate numerical calculations
│   │   │       ├── table_extractor.py     ← Extracts structured tables into rows/columns
│   │   │       ├── document_comparison.py ← Compares info/methodologies across multiple sources
│   │   │       └── data_analysis.py       ← Performs structured analysis on extracted data
│   │   └── answer/              ← Crafts the final response once evidence is sufficient
│   │       ├── agent.py         ← Answer agent execution logic
│   │       └── tools/
│   │           ├── citation_formatter.py  ← Attaches precise page and document citations
│   │           ├── source_formatter.py    ← Formats the list of evidence sources
│   │           └── response_formatter.py  ← Organizes insights into clear, readable text or tables
│   ├── orchestrator/            ← Coordinates the 3-Agent Workflow
│   │   ├── state.py             ← Tracks conversational state and evidence
│   │   ├── nodes.py             ← Graph nodes defining agent transitions
│   │   ├── evidence_assessment.py ← Evaluates if Analyst triggered the Feedback Loop for more evidence
│   │   └── orchestrator.py      ← Main workflow coordinator
│   ├── schemas/
│   │   └── analysis.py          ← AnalysisResult, FinalAnswer, Citation
│   └── main.py                  ← FastAPI entry point
├── tests/                       ← Unit and End-to-End tests
├── docker-compose.yml           ← Container orchestration (e.g., Vector DB, App)
└── .gitignore
```



### Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (or via Poetry/Pipenv).
3. Set up your environment variables (API keys, DB connection strings) in a `.env` file.

### Running the App
Run the main FastAPI server:
```bash
uvicorn app.main:app --reload
```
