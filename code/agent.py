try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Callable
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import requests
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_delay, wait_exponential, RetryError

# Observability wrappers are injected by the runtime (do not import or decorate @trace_agent manually)
# Use trace_step/trace_step_sync as instructed

# Load environment variables from .env if present
load_dotenv()

# Logging configuration
logger = logging.getLogger("CustomTranslatorFileAgent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
)
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# --- Configuration Management ---
class Config:
    """Centralized configuration management for Azure and LLM."""
    @staticmethod
    def get(key: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
        value = os.getenv(key)
        if value is None and required:
            raise ValueError(f"Missing required configuration: {key}")
        return value if value is not None else default

    @staticmethod
    def validate(required_keys=None):
        keys = required_keys or [
            "AZURE_BLOB_CONNECTION_STRING",
            "AZURE_BLOB_CONTAINER_NAME",
            "AZURE_TRANSLATOR_ENDPOINT",
            "AZURE_TRANSLATOR_KEY",
            "OPENAI_API_KEY"
        ]
        missing = [k for k in keys if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing required configuration keys: {missing}")

# --- Input Model and Validation ---
class TranslationRequest(BaseModel):
    filename: str = Field(..., max_length=256, description="Name of the file to translate.")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Filename must not be empty.")
        if len(v) > 256:
            raise ValueError("Filename too long (max 256 characters).")
        if any(c in v for c in ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']):
            raise ValueError("Filename contains invalid characters.")
        return v

# --- Logger Utility ---
class Logger:
    """Logs all API interactions, errors, and audit events."""
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def log_event(self, event_type: str, message: str, context: dict):
        try:
            logger.info(f"[{event_type}] {message} | Context: {context}")
        except Exception as e:
            # Logging must not interrupt main workflow
            pass

# --- Error Handler ---
class ErrorHandler:
    """Handles error mapping, formatting, escalation, and logging."""
    def __init__(self, logger: Logger):
        self.logger = logger

    def handle_error(self, error_code: str, context: dict) -> str:
        error_messages = {
            "BLOB_NOT_FOUND": "The requested file could not be found in the specified Azure Blob container. Please check the filename and try again.",
            "TRANSLATION_FAILED": "The translation service is unavailable or failed after multiple attempts. Please try again later or contact support.",
            "CONFIG_ERROR": "A configuration error occurred. Please contact support.",
            "VALIDATION_ERROR": "Input validation failed. Please check your request and try again.",
            "INTERNAL_ERROR": "An unexpected error occurred. Please try again later.",
        }
        msg = error_messages.get(error_code, "An unknown error occurred.")
        self.logger.log_event("error", f"{error_code}: {msg}", context)
        return msg

# --- Azure Blob Adapter ---
class AzureBlobAdapter:
    """Encapsulates Azure Blob Storage SDK interactions."""
    def __init__(self):
        self._connection_string = Config.get("AZURE_BLOB_CONNECTION_STRING", required=True)
        self._container_name = Config.get("AZURE_BLOB_CONTAINER_NAME", required=True)
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = BlobServiceClient.from_connection_string(self._connection_string)
        return self._client

    def file_exists(self, filename: str) -> bool:
        client = self._get_client()
        container_client = client.get_container_client(self._container_name)
        try:
            blob_client = container_client.get_blob_client(blob=filename)
            return blob_client.exists()
        except Exception as e:
            logger.warning(f"Error checking file existence: {e}")
            return False

    def create_sas_url(self, filename: str) -> str:
        client = self._get_client()
        try:
            sas_token = generate_blob_sas(
                account_name=client.account_name,
                container_name=self._container_name,
                blob_name=filename,
                account_key=client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=time.time() + 300  # 5 minutes
            )
            url = f"https://{client.account_name}.blob.core.windows.net/{self._container_name}/{filename}?{sas_token}"
            return url
        except Exception as e:
            logger.warning(f"Error generating SAS URL: {e}")
            raise

# --- File Validation Service ---
class FileValidationService:
    """Validates file existence in Azure Blob Storage."""
    def __init__(self, blob_adapter: AzureBlobAdapter):
        self.blob_adapter = blob_adapter

    def validate_file_exists(self, filename: str) -> bool:
        exists = self.blob_adapter.file_exists(filename)
        if not exists:
            logger.warning(f"File '{filename}' not found in blob storage.")
        return exists

# --- SAS URL Service ---
class SASUrlService:
    """Generates secure SAS URLs for files in Azure Blob Storage."""
    def __init__(self, blob_adapter: AzureBlobAdapter):
        self.blob_adapter = blob_adapter
        self._cache = {}  # Simple in-memory cache for 5 minutes

    def generate_sas_url(self, filename: str) -> str:
        now = time.time()
        # Cache SAS URLs for 5 minutes
        if filename in self._cache:
            cached_url, expiry = self._cache[filename]
            if expiry > now:
                return cached_url
        url = self.blob_adapter.create_sas_url(filename)
        self._cache[filename] = (url, now + 300)
        return url

# --- Azure Translator Adapter ---
class AzureTranslatorAdapter:
    """Encapsulates Azure Translator API calls."""
    def __init__(self):
        self._endpoint = Config.get("AZURE_TRANSLATOR_ENDPOINT", required=True)
        self._key = Config.get("AZURE_TRANSLATOR_KEY", required=True)

    async def submit_translation_request(self, sas_url: str) -> dict:
        # Example: POST to Azure Translator Document API
        url = f"{self._endpoint}/translator/text/batch/v1.0/batches"
        headers = {
            "Ocp-Apim-Subscription-Key": self._key,
            "Content-Type": "application/json"
        }
        body = {
            "inputs": [
                {
                    "storageSource": "AzureBlob",
                    "source": {
                        "sourceUrl": sas_url,
                        "language": "auto"
                    },
                    "targets": [
                        {
                            "targetUrl": sas_url,  # For demo, use same SAS URL; in real use, provide a writable SAS URL
                            "language": "en"
                        }
                    ]
                }
            ]
        }
        try:
            loop = asyncio.get_event_loop()
            # Use requests in thread executor for async
            def _post():
                _obs_t0 = _time.time()
                resp = requests.post(url, headers=headers, json=body, timeout=30)
                try:
                    trace_tool_call(
                        tool_name='requests.post',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                return resp
            resp = await loop.run_in_executor(None, _post)
            return {
                "status_code": resp.status_code,
                "json": resp.json() if resp.content else {},
                "text": resp.text
            }
        except Exception as e:
            logger.warning(f"Error calling Azure Translator: {e}")
            return {
                "status_code": 500,
                "json": {},
                "text": str(e)
            }

# --- Retry Handler ---
class RetryHandler:
    """Implements retry logic with exponential backoff for translation requests."""
    async def retry(self, func: Callable, max_duration: int, *args, **kwargs) -> dict:
        start_time = time.time()
        last_exc = None
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=2, min=2, max=10),
            stop=stop_after_delay(max_duration),
            reraise=True
        ):
            with attempt:
                result = await func(*args, **kwargs)
                if result.get("status_code") == 200:
                    return result
                last_exc = Exception(f"Non-200 status: {result.get('status_code')}")
                raise last_exc
        # If we get here, all retries failed
        raise last_exc if last_exc else Exception("Translation failed after retries.")

# --- Translation Service ---
class TranslationService:
    """Handles translation requests to Azure Translator, manages retry logic, and processes responses."""
    def __init__(self, translator_adapter: AzureTranslatorAdapter, retry_handler: RetryHandler):
        self.translator_adapter = translator_adapter
        self.retry_handler = retry_handler

    async def translate_file(self, sas_url: str) -> dict:
        async with trace_step(
            "translation_request", step_type="tool_call",
            decision_summary="Submit SAS URL to Azure Translator with retry logic",
            output_fn=lambda r: f"status_code={r.get('status_code','?')}"
        ) as step:
            try:
                result = await self.retry_handler.retry(
                    self.translator_adapter.submit_translation_request,
                    max_duration=100,
                    sas_url=sas_url
                )
                step.capture(result)
                return result
            except Exception as e:
                step.capture({"status_code": 500, "error": str(e)})
                return {"status_code": 500, "error": str(e)}

# --- Status Reporter ---
class StatusReporter:
    """Formats and delivers status updates and final responses to users."""
    def __init__(self, llm_client_factory: Callable[[], Any], model: str, system_prompt: str, temperature: float, max_tokens: int):
        self.llm_client_factory = llm_client_factory
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def report_status(self, status: str, details: dict) -> str:
        # Compose a professional summary using LLM
        user_message = f"Status: {status}\nDetails: {details}"
        async with trace_step(
            "generate_status_report", step_type="llm_call",
            decision_summary="Call LLM to summarize translation process for user",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            try:
                client = self.llm_client_factory()
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content
                step.capture(content)
                try:
                    trace_model_call(
                        provider="openai",
                        model_name=self.model,
                        prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                        completion_tokens=getattr(response.usage, "completion_tokens", 0),
                        latency_ms=0,
                        response_summary=content[:200] if content else ""
                    )
                except Exception:
                    pass
                return content
            except Exception as e:
                step.capture(f"LLM error: {e}")
                logger.warning(f"LLM status reporting failed: {e}")
                return None

# --- Agent Controller ---
class AgentController:
    """Entry point for user requests; manages input validation, workflow orchestration, and response formatting."""
    def __init__(
        self,
        file_validation_service: FileValidationService,
        sas_url_service: SASUrlService,
        translation_service: TranslationService,
        status_reporter: StatusReporter,
        error_handler: ErrorHandler
    ):
        self.file_validation_service = file_validation_service
        self.sas_url_service = sas_url_service
        self.translation_service = translation_service
        self.status_reporter = status_reporter
        self.error_handler = error_handler

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_translation_request(self, filename: str) -> dict:
        async with trace_step(
            "process_translation_request", step_type="process",
            decision_summary="Main workflow: validate file, generate SAS URL, submit translation, report status",
            output_fn=lambda r: f"success={r.get('success', '?')}"
        ) as step:
            try:
                # Step 1: Validate file existence
                with trace_step_sync(
                    "validate_file_exists", step_type="tool_call",
                    decision_summary="Check if file exists in Azure Blob Storage",
                    output_fn=lambda r: f"exists={r}"
                ) as substep:
                    exists = self.file_validation_service.validate_file_exists(filename)
                    substep.capture(exists)
                if not exists:
                    msg = self.error_handler.handle_error("BLOB_NOT_FOUND", {"filename": filename})
                    summary = await self.status_reporter.report_status("File Not Found", {"filename": filename})
                    step.capture({"success": False, "error": msg, "summary": summary})
                    return {
                        "success": False,
                        "error": msg,
                        "summary": summary or msg
                    }

                # Step 2: Generate SAS URL
                with trace_step_sync(
                    "generate_sas_url", step_type="tool_call",
                    decision_summary="Generate SAS URL for file",
                    output_fn=lambda r: f"url={r[:30]}..." if r else "url=None"
                ) as substep:
                    try:
                        sas_url = self.sas_url_service.generate_sas_url(filename)
                        substep.capture(sas_url)
                    except Exception as e:
                        msg = self.error_handler.handle_error("BLOB_NOT_FOUND", {"filename": filename, "error": str(e)})
                        summary = await self.status_reporter.report_status("SAS URL Generation Failed", {"filename": filename})
                        step.capture({"success": False, "error": msg, "summary": summary})
                        return {
                            "success": False,
                            "error": msg,
                            "summary": summary or msg
                        }

                # Step 3: Submit translation request (with retry)
                translation_result = await self.translation_service.translate_file(sas_url)
                status_code = translation_result.get("status_code")
                if status_code != 200:
                    msg = self.error_handler.handle_error("TRANSLATION_FAILED", {"filename": filename, "status_code": status_code})
                    summary = await self.status_reporter.report_status("Translation Failed", {"filename": filename, "details": translation_result})
                    step.capture({"success": False, "error": msg, "summary": summary})
                    return {
                        "success": False,
                        "error": msg,
                        "summary": summary or msg
                    }

                # Step 4: Success
                summary = await self.status_reporter.report_status("Translation Completed", {
                    "filename": filename,
                    "translation_response": translation_result.get("json", {})
                })
                step.capture({"success": True, "summary": summary, "result": translation_result.get("json", {})})
                return {
                    "success": True,
                    "summary": summary,
                    "result": translation_result.get("json", {})
                }
            except Exception as e:
                msg = self.error_handler.handle_error("INTERNAL_ERROR", {"filename": filename, "error": str(e)})
                summary = await self.status_reporter.report_status("Internal Error", {"filename": filename, "error": str(e)})
                step.capture({"success": False, "error": msg, "summary": summary})
                return {
                    "success": False,
                    "error": msg,
                    "summary": summary or msg
                }

# --- Main Agent Class ---
class CustomTranslatorFileAgent:
    """Main agent class composing all supporting classes."""
    def __init__(self):
        # Compose adapters and services
        self.logger = Logger()
        self.error_handler = ErrorHandler(self.logger)
        self.blob_adapter = AzureBlobAdapter()
        self.file_validation_service = FileValidationService(self.blob_adapter)
        self.sas_url_service = SASUrlService(self.blob_adapter)
        self.translator_adapter = AzureTranslatorAdapter()
        self.retry_handler = RetryHandler()
        self.translation_service = TranslationService(self.translator_adapter, self.retry_handler)
        self.status_reporter = StatusReporter(
            llm_client_factory=get_llm_client,
            model="gpt-4.1",
            system_prompt=(
                "You are a professional agent responsible for translating files stored in Azure Blob Storage. "
                "When provided with a filename, perform the following steps: 1. Validate that the file exists in the specified Azure Blob container. "
                "2. Generate a secure SAS URL for the file. 3. Submit the SAS URL to the Azure Translator service for translation. "
                "4. If the translation service does not return a status 200, retry the request for up to 100 seconds. "
                "5. Provide clear, concise, and professional updates on the process. 6. If the file is not found or translation fails after retries, "
                "return an informative error message. Always ensure sensitive information is not exposed in responses."
            ),
            temperature=0.7,
            max_tokens=2000
        )
        self.controller = AgentController(
            self.file_validation_service,
            self.sas_url_service,
            self.translation_service,
            self.status_reporter,
            self.error_handler
        )

    async def handle_request(self, filename: str) -> dict:
        return await self.controller.process_translation_request(filename)

# --- LLM Integration (OpenAI) ---
@with_content_safety(config=GUARDRAILS_CONFIG)
def get_llm_client():
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")
    return openai.AsyncOpenAI(api_key=api_key)

# --- FastAPI App ---
app = FastAPI(
    title="Custom Translator File Agent",
    description="Professional agent for translating files in Azure Blob Storage.",
    version="1.0.0"
)

# CORS (allow all origins for demo; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = CustomTranslatorFileAgent()

# --- Exception Handlers ---
@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Input validation failed.",
            "details": exc.errors(),
            "tips": "Ensure your JSON is well-formed and the filename is valid."
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "tips": "Check your request and try again."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.warning(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error.",
            "tips": "Please try again later or contact support."
        }
    )

# --- API Endpoint ---
@app.post("/translate", response_model=Dict[str, Any])
@with_content_safety(config=GUARDRAILS_CONFIG)
async def translate_file(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Malformed JSON in request body.",
                "tips": "Check for missing quotes, commas, or brackets in your JSON."
            }
        )
    try:
        req = TranslationRequest(**data)
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Input validation failed.",
                "details": ve.errors(),
                "tips": "Ensure your JSON is well-formed and the filename is valid."
            }
        )
    # Input size check
    if len(req.filename) > 256:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "success": False,
                "error": "Filename too long.",
                "tips": "Filename must be 256 characters or less."
            }
        )
    # Process translation request
    result = await agent.handle_request(req.filename)
    # Fallback response if needed
    if not result.get("success"):
        fallback = (
            "The requested file could not be found or the translation service is unavailable. "
            "Please verify the filename and try again, or contact support for assistance."
        )
        result.setdefault("summary", fallback)
    return JSONResponse(
        status_code=status.HTTP_200_OK if result.get("success") else status.HTTP_400_BAD_REQUEST,
        content=result
    )

# --- Main Entrypoint ---


async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        # Validate config only when running as main
        try:
        Config.validate()
        except Exception as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
        exit(1)
        logger.info("Starting Custom Translator File Agent API on http://0.0.0.0:8000")
        uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())