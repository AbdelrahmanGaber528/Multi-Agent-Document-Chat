from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.core.errors import AppError
import logging
from app.core.logging import setup_logging
from app.api import chat, sources


setup_logging()
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    
    settings = get_settings()
    
    settings.ensure_dirs()
    
    logger.info("Started %s (log_level=%s). Allowed Files Extensions : %s", settings.app_name, settings.log_level , settings.allowed_extensions)
    
    
    app.state.llm = LLMClient(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
    )
     
    logger.info("LLM initialized")
     
    yield
    
    logger.info("Shutting down")



app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.include_router(chat.router)
app.include_router(sources.router)

logger.info("Routers registered")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/api/health")
async def check_health():
    
    logger.info("Health check requested") 
    return {"status":"OK , server on"}



@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("%s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())