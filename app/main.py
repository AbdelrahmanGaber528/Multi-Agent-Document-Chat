import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.errors import AppError, ConfigurationError
from app.core.logging import setup_logging
from app.api import chat, sources
from app.agents.retriever.agent import RetrieverAgent
from app.orchestrator.graph import build_graph
from app.pipeline.vector_db import VectorDatabase

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    logger.info("Starting %s (log_level=%s)", settings.app_name, settings.log_level)
    if not settings.llm_api_key:
        raise ConfigurationError("LLM_API_KEY is required to initialize the chat model.")

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
    vector_db = VectorDatabase(embedding_model=settings.embedding_model)
    retriever = RetrieverAgent(llm=llm, vector_db=vector_db)

    app.state.llm = llm
    app.state.vector_db = vector_db
    app.state.retriever = retriever
    app.state.workflow = build_graph(
        retrieve_evidence=lambda query, history: retriever.invoke(query),
        analyst_llm=llm,
        answer_llm=llm,
    )
    logger.info("LLM, vector database, retriever, and workflow initialized")
    try:
        yield
    finally:
        logger.info("Shutting down")

settings = get_settings()

app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.include_router(chat.router)
app.include_router(sources.router)


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
async def check_health():
    return {"status": "ok"}

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("%s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())
