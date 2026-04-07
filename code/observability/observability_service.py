"""
Observability Service — in-memory trace accumulation and async DB persistence.

This module provides two key abstractions:

:class:`TraceContext`
    An in-memory accumulator for a single agent execution.  In the OTel path
    :class:`~observability.instrumentation.DatabaseSpanExporter` constructs
    and populates this automatically from the closed ``agent/`` span; no
    manual interaction is required in normal operation.
    :meth:`~TraceContext.to_trace_dict` serialises everything to a flat dict
    ready for the ``qo_observability_trace`` Azure SQL row.

:class:`ObservabilityService`
    Thin async persistence layer.  :meth:`~ObservabilityService.persist_trace`
    calls :meth:`~TraceContext.to_trace_dict`, maps the result to the
    :class:`~database.models.ObservabilityTrace` ORM model, and commits.
    Retries up to 3 times on ``SQLAlchemyError`` via ``tenacity``.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from observability.database.engine import ObsAsyncSessionType as AsyncSessionType
from observability.database.models import ObservabilityTrace, ObservabilityExecutionStatus

logger = logging.getLogger(__name__)


class TraceContext:
    """
    In-memory accumulator for a single agent execution trace.

    Instances are created at the start of an agent run (either by the
    ``@trace_agent`` decorator via :class:`~observability.instrumentation.DatabaseSpanExporter`,
    or directly in tests/backfill scripts) and progressively enriched as the
    execution proceeds.  ``to_trace_dict()`` produces the flat dict that maps
    to a single ``qo_observability_trace`` row.

    Thread-safety
        :class:`TraceContext` is not thread-safe.  Each agent execution should
        own its own instance.

    Attributes:
        agent_execution_id: Auto-generated UUID4 — primary key of the trace row.
        session_id:          Pipeline-level grouping ID shared by all agents in
                             one unit of work (e.g. one email batch).  Set via
                             :func:`~observability.observability_wrapper.set_trace_context_ids`.
        agent_name:          Name registered on the ``@trace_agent`` decorator.
        agent_version:       Semantic version of the agent (e.g. ``"2.0"``).
        environment:         Deployment tier — defaults to ``"production"``.
        steps:               Ordered list of step dicts built by :meth:`start_step` /
                             :meth:`end_step` or by the registry pop in
                             :class:`~observability.instrumentation.DatabaseSpanExporter`.
        model_calls:         List of model-call dicts appended by :meth:`add_model_call`.
        tool_calls:          List of tool-call dicts appended by :meth:`add_tool_call`.
    """
    
    def __init__(
        self,
        agent_name: str,
        session_id: Optional[UUID] = None,
        agent_version: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        """Initialise a new :class:`TraceContext` for one agent execution.

        ``session_id`` is **not** auto-generated - it must be supplied
        explicitly via
        :func:`~observability.observability_wrapper.set_trace_context_ids`
        at the pipeline entry point so that all agents within the same unit
        of work share the same grouping key.

        Args:
            agent_name:          Human-readable agent name (e.g. ``"EmailClassifierAgent"``).
            session_id:          Pipeline-level session UUID (optional).
            agent_version:       Semantic version string (optional).
            environment:         Deployment tier; defaults to ``"production"``.
        """
        self.agent_execution_id = uuid4()
        # session_id is intentionally NOT auto-generated.
        # It must be provided via set_trace_context_ids() so all agents
        # in one pipeline run share the same grouping key.
        self.session_id = session_id
        
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.environment = environment or "production"
        
        self.started_at = datetime.now(timezone.utc)
        self.ended_at: Optional[datetime] = None
        self.queue_time_ms: Optional[int] = None
        
        self.status = ObservabilityExecutionStatus.SUCCESS
        self.error_class: Optional[str] = None
        self.error_message: Optional[str] = None
        self.stack_trace: Optional[str] = None
        
        # Token tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
        # Cost tracking
        self.cost_amount: Optional[float] = None
        self.cost_currency: str = "USD"
        self.price_version: Optional[str] = None
        
        # Hierarchical data
        self.steps: List[Dict[str, Any]] = []
        self.model_calls: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        
        # Domain-specific context fields
        self.user_query: Optional[str] = None
        self.agent_response: Optional[str] = None
        self.is_evaluated: bool = False
        
        # Current step tracking
        self._current_step_index: int = -1
        
    def start_step(
        self,
        name: str,
        decision_summary: Optional[str] = None
    ) -> int:
        """Append a new step to the internal step list and return its index.

        Sets ``_current_step_index`` so that :meth:`add_model_call` and
        :meth:`add_tool_call` can stamp the correct ``step_index`` without
        requiring explicit ``step_index=`` arguments at every call site.

        Args:
            name:             Logical step label (e.g. ``"parse_input"``, ``"classify"``).
            decision_summary: Optional human-readable note about the step’s purpose.

        Returns:
            0-based index of the newly registered step.
        """
        self._current_step_index = len(self.steps)
        step = {
            'index': self._current_step_index,
            'name': name,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'ended_at': None,
            'status': 'running',
            'latency_ms': None,
            'retries': 0,
            'decision_summary': decision_summary,
        }
        self.steps.append(step)
        return self._current_step_index
    
    def end_step(
        self,
        step_index: Optional[int] = None,
        status: str = 'success',
        latency_ms: Optional[int] = None,
    ):
        """Finalise a step with its outcome timing.

        ``ended_at`` is set to the current UTC time.  If ``latency_ms`` is not
        provided it is derived from the difference between ``started_at`` and
        the current ``ended_at``.

        Args:
            step_index: Index of the step to finalise; defaults to the most
                        recently started step (``_current_step_index``).
            status:     Outcome string — typically ``"success"`` or ``"error"``.
            latency_ms: Explicit latency override; computed automatically when omitted.
        """
        idx = step_index if step_index is not None else self._current_step_index
        if 0 <= idx < len(self.steps):
            step = self.steps[idx]
            step['ended_at'] = datetime.now(timezone.utc).isoformat()
            step['status'] = status
            
            if latency_ms is not None:
                step['latency_ms'] = latency_ms
            elif step['started_at']:
                start = datetime.fromisoformat(step['started_at'])
                end = datetime.fromisoformat(step['ended_at'])
                step['latency_ms'] = int((end - start).total_seconds() * 1000)
    
    def add_model_call(
        self,
        provider: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        model_version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        cache_status: Optional[str] = None,
        status: str = 'success',
        error_class: Optional[str] = None,
        error_message: Optional[str] = None,
        token_usage_available: bool = True,
        token_usage_estimated: bool = False,
        model_call_type: str = 'chat',
        step_index: Optional[int] = None,
        started_at: Optional[str] = None,
        response_summary: Optional[str] = None,
        cost_usd: Optional[float] = None,
    ):
        """Append one LLM call record to ``model_calls``.

        Also accumulates ``prompt_tokens`` and ``completion_tokens`` onto the
        trace-level counters used to populate the ``tokens`` column, and sets
        ``model_name`` on the trace if not already set.

        Args:
            provider:              LLM provider identifier (e.g. ``"azure"``, ``"openai"``).
            model_name:            Model deployment name (e.g. ``"gpt-4.1"``).
            prompt_tokens:         Number of input/prompt tokens consumed.
            completion_tokens:     Number of output/completion tokens generated.
            latency_ms:            End-to-end call latency in milliseconds.
            model_version:         Optional model version string.
            parameters:            Optional dict of call parameters (temperature, etc.).
            cache_status:          ``"hit"`` / ``"miss"`` / ``None``.
            status:                ``"success"`` or ``"error"``.
            error_class:           Exception class name on failure.
            error_message:         Redacted error message on failure.
            token_usage_available: ``True`` when token counts are real measured values.
            token_usage_estimated: ``True`` when token counts are approximations.
            model_call_type:       Call semantic — ``"chat"`` (default) or ``"embedding"``.
            step_index:            Index of the containing step; defaults to current step.
            started_at:            ISO-8601 timestamp string for call start; derived from
                                   latency when omitted.
            response_summary:      Full LLM response text or structured output string.
            cost_usd:              Pre-computed cost in USD from LiteLLM response_cost; when
                                   present, ``_compute_cost_from_model_calls`` sums these
                                   directly instead of deriving cost from token rates.
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        
        _step_idx = step_index if step_index is not None else self._current_step_index
        if started_at:
            _model_started_at = datetime.fromisoformat(started_at)
            _model_ended_at = _model_started_at + timedelta(milliseconds=max(0, latency_ms or 0))
        else:
            _model_ended_at = datetime.now(timezone.utc)
            _model_started_at = _model_ended_at - timedelta(milliseconds=max(0, latency_ms or 0))
        model_call = {
            'model_call_id': str(uuid4()),
            'model_call_type': model_call_type,
            'step_index': _step_idx,
            'provider': provider,
            'model_name': model_name,
            'model_version': model_version,
            'parameters_summary': parameters or {},
            'cache_status': cache_status,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'started_at': _model_started_at.isoformat(),
            'ended_at': _model_ended_at.isoformat(),
            'status': status,
            'latency_ms': latency_ms,
            'error_class': error_class,
            'error_message': error_message,
            'token_usage_available': token_usage_available,
            'token_usage_estimated': token_usage_estimated,
            'response_summary': response_summary,
            'cost_usd': float(cost_usd) if cost_usd is not None else None,
        }
        self.model_calls.append(model_call)
    
    def add_tool_call(
        self,
        tool_name: str,
        latency_ms: int,
        tool_version: Optional[str] = None,
        args_summary: Optional[Dict[str, Any]] = None,
        output_summary: Optional[str] = None,
        status: str = 'success',
        error_class: Optional[str] = None,
        error_message: Optional[str] = None,
        step_index: Optional[int] = None,
        ended_at: Optional[str] = None,
    ):
        """Append one tool invocation record to ``tool_calls``.

        ``started_at`` is back-calculated from ``ended_at`` minus ``latency_ms``
        so that span-derived call entries have accurate timestamps.

        Args:
            tool_name:    Identifier of the tool (e.g. ``"eml_msg_parser"``).
            latency_ms:   End-to-end tool latency in milliseconds.
            tool_version: Optional version string for the tool.
            args_summary: Dict of arguments passed to the tool.
            output_summary: Tool output string (may include structured data).
            status:       ``"success"`` or ``"error"``.
            error_class:  Exception class name on failure.
            error_message: Error message on failure.
            step_index:   Index of the containing step; defaults to current step.
            ended_at:     ISO-8601 timestamp string for call end; uses now() when omitted.
        """
        _step_idx = step_index if step_index is not None else self._current_step_index
        if ended_at:
            _tool_ended_at = datetime.fromisoformat(ended_at)
        else:
            _tool_ended_at = datetime.now(timezone.utc)
        _tool_started_at = _tool_ended_at - timedelta(milliseconds=max(0, latency_ms or 0))
        tool_call = {
            'step_index': _step_idx,
            'tool_name': tool_name,
            'tool_version': tool_version,
            'args_summary': args_summary or {},
            'output_summary': output_summary,
            'started_at': _tool_started_at.isoformat(),
            'ended_at': _tool_ended_at.isoformat(),
            'status': status,
            'latency_ms': latency_ms,
            'error_class': error_class,
            'error_message': error_message,
        }
        self.tool_calls.append(tool_call)
    
    def set_cost(
        self,
        amount: float,
        currency: str = "USD",
        price_version: Optional[str] = None,
    ):
        """Record the monetary cost of this execution.

        Args:
            amount:        Cost amount in the specified currency.
            currency:      ISO 4217 currency code (default ``"USD"``).
            price_version: Pricing table version or date (e.g. ``"2024-03"``).
        """
        self.cost_amount = amount
        self.cost_currency = currency
        self.price_version = price_version
    
    def set_user_query(self, query: str):
        """Set the user query that triggered this execution."""
        self.user_query = query

    def set_agent_response(self, response: str):
        """Set the agent's final response."""
        self.agent_response = response
    
    def mark_evaluated(self):
        """Mark this trace as evaluated."""
        self.is_evaluated = True
    
    def finalize(self):
        """Finalize trace context at end of execution."""
        self.ended_at = datetime.now(timezone.utc)

        # Close any step left open by legacy direct-path callers.
        idx = self._current_step_index
        if 0 <= idx < len(self.steps):
            step = self.steps[idx]
            if step.get('ended_at') is None:
                now = datetime.now(timezone.utc)
                step['ended_at'] = now.isoformat()
                step['status'] = 'completed'
                if step.get('started_at'):
                    start = datetime.fromisoformat(step['started_at'])
                    step['latency_ms'] = int((now - start).total_seconds() * 1000)

        # Auto-compute cost from token counts if not already set explicitly
        if self.cost_amount is None and self.model_calls:
            self.cost_amount = self._compute_cost_from_model_calls()

    # Token pricing table (USD per 1 000 tokens, keyed by model name substring)
    # Rates reflect gpt-4.1 family pricing; updated via the config token_cost fields.
    _TOKEN_COST_TABLE = [
        # (model substring, input_per_1k, output_per_1k)
        ("gpt-5.2-pro", 0.021, 0.168),
        ("gpt-5.2", 0.00175, 0.014),
        ("gpt-5.1", 0.00125, 0.010),
        ("gpt-5-pro", 0.015, 0.120),
        ("gpt-5-mini", 0.00025, 0.002),
        ("gpt-5-nano", 0.00005, 0.0004),
        ("gpt-5", 0.00125, 0.010),
        ("gpt-4.1-mini", 0.0004, 0.0016),
        ("gpt-4.1-nano", 0.0001, 0.0004),
        ("gpt-4.1", 0.002, 0.008),
        ("gpt-4o-2024-05-13", 0.005, 0.015),
        ("gpt-4o-mini", 0.00015, 0.0006),
        ("gpt-4o", 0.0025, 0.010),
        ("gpt-4-turbo-2024-04-09", 0.010, 0.030),
        ("gpt-4-0125-preview", 0.010, 0.030),
        ("gpt-4-1106-preview", 0.010, 0.030),
        ("gpt-4-1106-vision-preview", 0.010, 0.030),
        ("gpt-4-0613", 0.030, 0.060),
        ("gpt-4-0314", 0.030, 0.060),
        ("gpt-4-32k", 0.060, 0.120),
        ("gpt-4-turbo", 0.010, 0.030),
        ("gpt-4", 0.010, 0.030),
        ("gpt-3.5-turbo-16k-0613", 0.003, 0.004),
        ("gpt-3.5-turbo-0613", 0.0015, 0.002),
        ("gpt-3.5-turbo-1106", 0.001, 0.002),
        ("gpt-3.5-turbo-0125", 0.0005, 0.0015),
        ("gpt-3.5-turbo-instruct", 0.0015, 0.002),
        ("gpt-3.5-0301", 0.0015, 0.002),
        ("gpt-3.5-turbo", 0.0005, 0.0015),
        ("gpt-3.5", 0.0005, 0.0015),
        ("o4-mini", 0.0011, 0.0044),
        ("o3-mini", 0.0011, 0.0044),
        ("o3-pro", 0.020, 0.080),
        ("o3", 0.002, 0.008),
        ("o1-mini", 0.0011, 0.0044),
        ("o1-pro", 0.150, 0.600),
        ("o1", 0.015, 0.060),
        ("davinci-002", 0.002, 0.002),
        ("babbage-002", 0.0004, 0.0004),
    ]

    def _compute_cost_from_model_calls(self) -> Optional[float]:
        """Derive total USD cost from accumulated model_calls.

        Prefers ``cost_usd`` values retrieved directly from LiteLLM
        ``_hidden_params["response_cost"]``.  Falls back to token-rate
        computation for any call where ``cost_usd`` is absent.
        """
        total_cost = 0.0
        found_any = False
        for call in self.model_calls:
            # Prefer the pre-computed cost supplied by LiteLLM
            cost_usd = call.get('cost_usd')
            if cost_usd is not None:
                total_cost += float(cost_usd)
                found_any = True
                continue
            # Fallback: derive from token counts × per-model rates
            prompt_tokens     = int(call.get("prompt_tokens", 0) or 0)
            completion_tokens = int(call.get("completion_tokens", 0) or 0)
            if prompt_tokens == 0 and completion_tokens == 0:
                continue
            model = (call.get("model_name") or "").lower()
            input_rate, output_rate = self._get_model_rates(model)
            total_cost += (prompt_tokens / 1000.0) * input_rate
            total_cost += (completion_tokens / 1000.0) * output_rate
            found_any = True
        return round(total_cost, 6) if found_any else None

    @staticmethod
    def _get_model_rates(model: str):
        """Return (input_$/1k, output_$/1k) for a model name string."""
        try:
            from observability.config import settings
            for llm_cfg in (settings.LLM_MODELS or []):
                name = (llm_cfg.get("model_name") or "").lower()
                if name and name in model or model in name:
                    in_cost  = llm_cfg.get("input_token_cost")
                    out_cost = llm_cfg.get("output_token_cost")
                    if in_cost is not None and out_cost is not None:
                        return float(in_cost) / 1000.0, float(out_cost) / 1000.0
        except Exception:
            pass
        # Fallback: static table match
        for substr, in_rate, out_rate in TraceContext._TOKEN_COST_TABLE:
            if substr in model:
                return in_rate, out_rate
        # Unknown model: use gpt-4.1 rates as conservative default
        return 0.003, 0.012
    
    def to_trace_dict(self) -> Dict[str, Any]:
        """Convert context to trace dictionary for persistence."""
        def _truncate_text(value: Optional[str], max_len: int) -> str:
            if not value:
                return ""
            text = str(value)
            return text if len(text) <= max_len else text[:max_len]

        def _json_len(value: Any) -> int:
            try:
                return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")))
            except Exception:
                return len(str(value))

        def _fit_json(value: Any, max_len: int = 8000) -> Any:
            """Best-effort JSON compaction to cap payload size before DB write."""
            if value is None:
                return None
            if _json_len(value) <= max_len:
                return value

            # Compact model_calls aggressively but keep token/error semantics.
            if isinstance(value, list) and value and isinstance(value[0], dict) and "model_name" in value[0]:
                compact_calls: List[Dict[str, Any]] = []
                for call in value:
                    compact = {
                        "step_index": call.get("step_index"),
                        "provider": call.get("provider"),
                        "model_name": call.get("model_name"),
                        "status": call.get("status"),
                        "prompt_tokens": int(call.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(call.get("completion_tokens", 0) or 0),
                        "total_tokens": int(call.get("total_tokens", 0) or 0),
                        "token_usage_available": bool(call.get("token_usage_available", False)),
                        "token_usage_estimated": bool(call.get("token_usage_estimated", False)),
                        "error_class": call.get("error_class"),
                        "error_message": _truncate_text(call.get("error_message"), 120) or None,
                    }
                    trial = compact_calls + [compact]
                    if _json_len(trial) > max_len:
                        break
                    compact_calls = trial
                return compact_calls if compact_calls else []

            if isinstance(value, list):
                compact_items = []
                for item in value:
                    compact = item
                    if isinstance(item, dict):
                        compact = {k: _truncate_text(v, 80) if isinstance(v, str) else v for k, v in item.items()}
                    trial = compact_items + [compact]
                    if _json_len(trial) > max_len:
                        break
                    compact_items = trial
                return compact_items

            if isinstance(value, dict):
                compact_dict = {}
                for k, v in value.items():
                    compact_v = _truncate_text(v, 120) if isinstance(v, str) else v
                    trial = dict(compact_dict)
                    trial[k] = compact_v
                    if _json_len(trial) > max_len:
                        break
                    compact_dict = trial
                return compact_dict

            return _truncate_text(str(value), max_len)

        total_latency_ms = None
        if self.ended_at:
            total_latency_ms = int((self.ended_at - self.started_at).total_seconds() * 1000)

        # Preserve null when queue time was not measured — null is semantically correct
        # (honest "not instrumented") and distinct from a genuine 0 (measured zero wait).
        queue_time_ms = self.queue_time_ms  # may be None, int >= 0
        user_query = _truncate_text(self.user_query or f"{self.agent_name} execution", 450)
        agent_response = _truncate_text(self.agent_response or f"status={self.status.value}", 450)

        # Preserve nullable error fields for successful traces.
        error_class = self.error_class
        error_message = _truncate_text(self.error_message, 450) if self.error_message else None
        stack_trace = _truncate_text(self.stack_trace, 450) if self.stack_trace else None
        
        # Only persist aggregate token counts when usage was actually reported.
        # Primary path: sum across model_calls where token_usage_available=True.
        # Fallback path: use self.prompt_tokens / self.completion_tokens which may
        #   have been set from span attributes by _span_to_trace_context even when
        #   model_calls is unavailable (e.g. edge cases before Fix 1 landed).
        token_measurements = [
            c for c in self.model_calls
            if c.get('token_usage_available') is True
        ]
        tokens_payload = None
        if token_measurements:
            prompt_total = sum(int(c.get('prompt_tokens', 0) or 0) for c in token_measurements)
            completion_total = sum(int(c.get('completion_tokens', 0) or 0) for c in token_measurements)
            tokens_payload = {
                'input_prompt': prompt_total,
                'output_completion': completion_total,
                'total': prompt_total + completion_total,
            }
        elif (self.prompt_tokens or 0) > 0 or (self.completion_tokens or 0) > 0:
            # Fallback: span-attribute totals (set by _span_to_trace_context from
            # agent-level OTel attributes when no structured model_calls exist).
            prompt_total = int(self.prompt_tokens or 0)
            completion_total = int(self.completion_tokens or 0)
            tokens_payload = {
                'input_prompt': prompt_total,
                'output_completion': completion_total,
                'total': prompt_total + completion_total,
            }

        return {
            'agent_execution_id': self.agent_execution_id,
            'session_id': self.session_id,
            'agent_name': self.agent_name,
            'agent_version': self.agent_version,
            'environment': self.environment,
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'total_latency_ms': total_latency_ms,
            'queue_time_ms': queue_time_ms,
            'status': self.status,
            'error_class': error_class,
            'error_message': error_message,
            'error_stack_summary': stack_trace,
            'tokens': _fit_json(tokens_payload),
            'cost': _fit_json({
                'amount': self.cost_amount,
                'currency': self.cost_currency,
                'price_version': self.price_version,
            } if self.cost_amount is not None else None),
            'steps': _fit_json(self.steps, max_len=8000),
            'model_calls': _fit_json(self.model_calls, max_len=8000),
            'tool_calls': _fit_json(self.tool_calls, max_len=8000),
            'user_query': user_query,
            'agent_response': agent_response,
            'is_evaluated': self.is_evaluated,
        }


class ObservabilityService:
    """
    Async persistence layer for :class:`TraceContext` objects.

    Wraps the SQLAlchemy session in a ``tenacity`` retry loop (up to 3
    attempts, exponential back-off) to handle transient ``SQLAlchemyError``
    failures without surfacing them to agent business logic.

    A singleton is lazily created by :func:`get_observability_service` and
    reused for the lifetime of the process.
    """
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(SQLAlchemyError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def persist_trace(
        self,
        trace_context: TraceContext,
        session: AsyncSessionType,
    ) -> bool:
        """
        Finalise and persist a :class:`TraceContext` as one ``qo_observability_trace`` row.

        Calls :meth:`~TraceContext.finalize` (sets ``ended_at``, detects primary
        model) then :meth:`~TraceContext.to_trace_dict` and maps the result to
        the :class:`~database.models.ObservabilityTrace` ORM model.
        Commits the session on success; rolls back on any error.

        Retried up to 3 times on :class:`sqlalchemy.exc.SQLAlchemyError` via
        the ``tenacity`` decorator before re-raising.

        Args:
            trace_context: Populated trace context to persist.
            session:       Active async database session.

        Returns:
            ``True`` on successful commit; ``False`` when a non-SQLAlchemy
            error is caught (error is logged and a degraded event is emitted).
        """
        try:
            # Finalize context
            trace_context.finalize()
            
            # Convert to dict
            trace_dict = trace_context.to_trace_dict()
            
            # Create ORM model
            trace = ObservabilityTrace(
                agent_execution_id=trace_dict['agent_execution_id'],
                session_id=trace_dict['session_id'],
                agent_name=trace_dict['agent_name'],
                agent_version=trace_dict.get('agent_version'),
                environment=trace_dict.get('environment'),
                started_at=trace_dict['started_at'],
                ended_at=trace_dict.get('ended_at'),
                total_latency_ms=trace_dict.get('total_latency_ms'),
                queue_time_ms=trace_dict.get('queue_time_ms'),
                status=trace_dict['status'],
                error_class=trace_dict.get('error_class'),
                error_message=trace_dict.get('error_message'),
                error_stack_summary=trace_dict.get('error_stack_summary'),
                tokens=trace_dict.get('tokens'),
                cost=trace_dict.get('cost'),
                steps=trace_dict.get('steps'),
                model_calls=trace_dict.get('model_calls'),
                tool_calls=trace_dict.get('tool_calls'),
                user_query=trace_dict.get('user_query'),
                agent_response=trace_dict.get('agent_response'),
                is_evaluated=trace_dict.get('is_evaluated', False),
            )
            
            session.add(trace)
            await session.commit()
            
            logger.info(
                f"Observability trace persisted: agent_execution_id={trace_context.agent_execution_id}, "
                f"agent={trace_context.agent_name}, status={trace_context.status.value}"
            )

            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error persisting trace: {e}", exc_info=True)
            await session.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error persisting trace: {e}", exc_info=True)
            await session.rollback()
            # Emit degraded event
            self._emit_degraded_event(trace_context, e)
            return False

    def _emit_degraded_event(self, trace_context: TraceContext, error: Exception):
        """
        Log a ``OBSERVABILITY_DEGRADED`` warning when trace persistence fails.

        Does not raise or propagate — observability failures must never block
        agent business logic.  The log entry includes ``agent_execution_id``,
        ``agent_name``, and the error class/message for post-hoc diagnosis.
        """
        try:
            logger.warning(
                f"OBSERVABILITY_DEGRADED: Failed to persist trace for agent_execution_id={trace_context.agent_execution_id}, "
                f"agent={trace_context.agent_name}, error={type(error).__name__}: {error}"
            )
            # Could also write to a separate degraded events table or file
        except Exception as e:
            logger.error(f"Failed to emit degraded event: {e}")


# Singleton service instance
_observability_service: Optional[ObservabilityService] = None


def get_observability_service() -> ObservabilityService:
    """
    Return (or lazily create) the process-level :class:`ObservabilityService` singleton.

    Returns:
        The singleton :class:`ObservabilityService` instance.
    """
    global _observability_service
    if _observability_service is None:
        _observability_service = ObservabilityService()
    return _observability_service
