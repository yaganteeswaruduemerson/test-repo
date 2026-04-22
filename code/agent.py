import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

import openai
from config import Config

# =========================
# CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are a planetary science information specialist. Your task is to deliver a comprehensive, scientifically accurate comparative analysis of Earth and Jupiter, focusing on their physical dimensions and spatial relationship to the Sun. For each planet, explicitly cite the equatorial diameter in both miles and kilometers, describe the scale difference (including how many Earths could fit inside Jupiter), and compare their average distances from the Sun. Ensure your response is clear, precise, and references only information from the provided knowledge base documents (Jupiter.pdf, Earth.pdf). If any requested measurement or comparison is not found in the retrieved content, inform the user and suggest consulting additional scientific resources. Format your output as a structured, professional summary suitable for educational or research purposes."
)
OUTPUT_FORMAT = (
    "- Structured summary with clear sections:\n"
    "  - Physical Dimensions (diameter in miles and kilometers)\n"
    "  - Scale Comparison (how many Earths fit inside Jupiter)\n"
    "  - Orbital Distances (average distance from Sun in miles and kilometers)\n"
    "  - Explanation of spatial relationship and orbital differences\n"
    "- Cite source document(s) for each fact\n"
    "- Use bullet points or paragraphs for clarity"
)
FALLBACK_RESPONSE = (
    "The requested measurements or comparisons are not available in the provided knowledge base documents. Please consult additional scientific resources for further information."
)
SELECTED_DOCUMENT_TITLES = ["Jupiter.pdf", "Earth.pdf"]
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# LOGGING CONFIG
# =========================

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# =========================
# INPUT/OUTPUT MODELS
# =========================

class AnalyzeResponse(BaseModel):
    success: bool = Field(..., description="Whether the analysis was successful")
    result: Optional[str] = Field(None, description="Structured comparative analysis summary")
    error: Optional[str] = Field(None, description="Error message if any")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input issues")

# =========================
# SANITIZER UTILITY
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# RETRIEVAL LAYER
# =========================

class AzureAISearchClient:
    """
    Handles low-level communication with Azure AI Search.
    """
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            endpoint = Config.AZURE_SEARCH_ENDPOINT
            api_key = Config.AZURE_SEARCH_API_KEY
            index_name = Config.AZURE_SEARCH_INDEX_NAME
            if not endpoint or not api_key or not index_name:
                raise ValueError("Azure AI Search credentials are not configured.")
            self._client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def search(self, query: str, filter: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        """
        Perform vector + keyword search with optional OData filter.
        """
        client = self._get_client()
        search_kwargs = {
            "search_text": query,
            "top": top_k,
            "select": ["chunk", "title"],
        }
        if filter:
            search_kwargs["filter"] = filter
        # The vector query will be injected by the ChunkRetriever
        return client.search(**search_kwargs)

class ChunkRetriever:
    """
    Orchestrates chunk retrieval using AzureAISearchClient.
    """
    def __init__(self, embedding_model: Optional[str] = None):
        self.search_client = AzureAISearchClient()
        self.embedding_model = embedding_model or Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, filter_titles: List[str], top_k: int) -> List[str]:
        """
        Retrieve relevant chunks from Azure AI Search using vector + keyword search and OData filter.
        """
        # Step 1: Embed the query using Azure OpenAI
        openai_client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        _t0 = _time.time()
        embedding_resp = await openai_client.embeddings.create(
            input=query,
            model=self.embedding_model,
        )
        try:
            trace_tool_call(
                tool_name="openai_client.embeddings.create",
                latency_ms=int((_time.time() - _t0) * 1000),
                output=str(embedding_resp)[:200] if embedding_resp else None,
                status="success",
            )
        except Exception:
            pass

        vector_query = VectorizedQuery(
            vector=embedding_resp.data[0].embedding,
            k_nearest_neighbors=top_k,
            fields="vector"
        )

        # Step 2: Build OData filter for selected document titles
        odata_filter = None
        if filter_titles:
            odata_parts = [f"title eq '{t}'" for t in filter_titles]
            odata_filter = " or ".join(odata_parts)

        # Step 3: Search
        client = self.search_client._get_client()
        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": ["chunk", "title"],
        }
        if odata_filter:
            search_kwargs["filter"] = odata_filter

        _t1 = _time.time()
        results = client.search(**search_kwargs)
        try:
            trace_tool_call(
                tool_name="search_client.search",
                latency_ms=int((_time.time() - _t1) * 1000),
                output=str(results)[:200] if results else None,
                status="success",
            )
        except Exception:
            pass

        context_chunks = [r["chunk"] for r in results if r.get("chunk")]
        return context_chunks

# =========================
# LLM SERVICE LAYER
# =========================

class LLMService:
    """
    Formats prompts, injects retrieved chunks as context, calls Azure OpenAI, returns structured response.
    """
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: List[str],
        parameters: Optional[dict] = None
    ) -> str:
        """
        Call LLM with system/user prompts and context, return structured summary.
        """
        client = self._get_client()
        # Compose the system message with output format appended
        system_message = f"{system_prompt}\n\nOutput Format: {OUTPUT_FORMAT}"
        # Compose the user message with context chunks
        context_str = "\n\n".join(context_chunks) if context_chunks else ""
        user_message = f"{user_prompt}\n\nContext:\n{context_str}" if context_str else user_prompt

        llm_kwargs = Config.get_llm_kwargs()
        if parameters:
            llm_kwargs.update(parameters)

        _t0 = _time.time()
        response = await client.chat.completions.create(
            model=Config.LLM_MODEL or "gpt-4.1",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            **llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.LLM_MODEL or "gpt-4.1",
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return content

# =========================
# ERROR HANDLER & LOGGER
# =========================

class ErrorHandler:
    """
    Handles error codes, fallback logic, logging, and user notifications.
    """
    def __init__(self, logger):
        self.logger = logger

    def handle_error(self, error_code: str, context: Optional[dict] = None) -> str:
        """
        Map error codes to user-facing messages or fallback responses.
        """
        if error_code == "DOC_NOT_FOUND":
            self.logger.log("error", "No relevant documents found", context or {})
            return FALLBACK_RESPONSE
        elif error_code == "MEASUREMENT_MISSING":
            self.logger.log("error", "Required measurements missing in retrieved content", context or {})
            return FALLBACK_RESPONSE
        else:
            self.logger.log("error", f"Unknown error code: {error_code}", context or {})
            return "An unexpected error occurred. Please try again later."

class Logger:
    """
    Audit logging of requests, responses, errors, and system events.
    """
    def __init__(self):
        self._logger = logging.getLogger("agent")

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def log(self, event_type: str, message: str, metadata: Optional[dict] = None) -> None:
        try:
            if event_type == "error":
                self._logger.error(f"{message} | {metadata}")
            elif event_type == "info":
                self._logger.info(f"{message} | {metadata}")
            else:
                self._logger.debug(f"{message} | {metadata}")
        except Exception:
            pass

# =========================
# AGENT ORCHESTRATOR
# =========================

class AgentOrchestrator:
    """
    Coordinates the flow: receives user input, applies document filters, invokes retrieval and LLM services, enforces business rules, handles errors.
    """
    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.logger = Logger()
        self.error_handler = ErrorHandler(self.logger)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def analyze(self) -> dict:
        """
        Main entry point for comparative analysis; orchestrates retrieval and LLM response generation.
        """
        async with trace_step(
            "retrieve_chunks",
            step_type="tool_call",
            decision_summary="Retrieve relevant planetary knowledge base chunks",
            output_fn=lambda r: f"chunks={len(r) if r else 0}",
        ) as step:
            try:
                context_chunks = await self.chunk_retriever.retrieve_chunks(
                    query=SYSTEM_PROMPT,
                    filter_titles=SELECTED_DOCUMENT_TITLES,
                    top_k=5
                )
                step.capture(context_chunks)
            except Exception as e:
                self.logger.log("error", "Chunk retrieval failed", {"error": str(e)})
                return {
                    "success": False,
                    "result": None,
                    "error": "Knowledge base retrieval failed.",
                    "tips": "Ensure the knowledge base is available and try again."
                }

        if not context_chunks or len(context_chunks) == 0:
            error_msg = self.error_handler.handle_error("DOC_NOT_FOUND", {})
            return {
                "success": False,
                "result": None,
                "error": error_msg,
                "tips": "Try updating the knowledge base or selecting different documents."
            }

        async with trace_step(
            "generate_llm_response",
            step_type="llm_call",
            decision_summary="Generate structured comparative analysis using LLM",
            output_fn=lambda r: f"response_length={len(r) if r else 0}",
        ) as step:
            try:
                raw_response = await self.llm_service.generate_response(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=SYSTEM_PROMPT,
                    context_chunks=context_chunks,
                    parameters={
                        "temperature": 0.7,
                        "max_tokens": 2000
                    }
                )
                step.capture(raw_response)
            except Exception as e:
                self.logger.log("error", "LLM generation failed", {"error": str(e)})
                error_msg = self.error_handler.handle_error("DOC_NOT_FOUND", {})
                return {
                    "success": False,
                    "result": None,
                    "error": error_msg,
                    "tips": "Try again later or check LLM service availability."
                }

        # Sanitize LLM output
        result = sanitize_llm_output(raw_response, content_type="text")
        if not result or FALLBACK_RESPONSE in result:
            error_msg = self.error_handler.handle_error("MEASUREMENT_MISSING", {})
            return {
                "success": False,
                "result": None,
                "error": error_msg,
                "tips": "Requested measurements may not be present in the selected documents."
            }

        return {
            "success": True,
            "result": result,
            "error": None,
            "tips": None
        }

# =========================
# MAIN AGENT CLASS
# =========================

class PlanetaryComparativeAnalysisAgent:
    """
    Main agent class. Orchestrates the planetary comparative analysis workflow.
    """
    def __init__(self):
        self.orchestrator = AgentOrchestrator()

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process(self) -> dict:
        """
        Entrypoint for the agent. Returns structured analysis or error.
        """
        return await self.orchestrator.analyze()

# =========================
# FASTAPI APP & ENDPOINTS
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Planetary Comparative Analysis Assistant",
    description="Compares Earth and Jupiter using authoritative knowledge base documents. Returns structured, cited scientific summaries.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

agent_instance = PlanetaryComparativeAnalysisAgent()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def analyze_endpoint():
    """
    Main endpoint for planetary comparative analysis.
    No user input required; uses SYSTEM_PROMPT and SELECTED_DOCUMENT_TITLES internally.
    """
    try:
        result = await agent_instance.process()
        return AnalyzeResponse(**result)
    except Exception as e:
        logger.error(f"Unhandled error in /analyze: {e}")
        return AnalyzeResponse(
            success=False,
            result=None,
            error="An unexpected error occurred. Please try again later.",
            tips="Check input and try again."
        )

@app.get("/status")
async def status_endpoint():
    """Returns agent status and configuration."""
    return {
        "agent_name": getattr(Config, "AGENT_NAME", "PlanetaryComparativeAnalysisAgent"),
        "project_name": getattr(Config, "PROJECT_NAME", "N/A"),
        "version": getattr(Config, "SERVICE_VERSION", "1.0.0"),
        "llm_model": getattr(Config, "LLM_MODEL", "gpt-4.1"),
        "rag_enabled": True,
        "selected_documents": SELECTED_DOCUMENT_TITLES,
    }

# =========================
# ERROR HANDLING FOR MALFORMED JSON
# =========================

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Malformed JSON in request: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "result": None,
            "error": "Malformed JSON or invalid request body.",
            "tips": "Check your JSON formatting (quotes, commas, brackets) and ensure all required fields are present."
        },
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Pydantic validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "result": None,
            "error": "Input validation failed.",
            "tips": "Check your input values and types."
        },
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "result": None,
            "error": "Internal server error.",
            "tips": "Try again later or contact support."
        },
    )

# =========================
# AGENT ENTRYPOINT
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())