"""
Observability Wrapper — decorator-based telemetry instrumentation for agents.

This module is the primary public interface for telemetry.  It exposes three
patterns for wrapping execution:

* ``@trace_agent`` (decorator / async context manager)
    Wraps an entire agent run.  Creates the root ``agent/<name>`` OTel span and
    feeds ``session_id`` / ``correlation_id`` from :data:`_active_session_id` /
    :data:`_active_correlation_id` into span attributes so the
    :class:`~observability.instrumentation.DatabaseSpanExporter` can stamp the
    DB row correctly when the span closes.

* ``@trace_step`` / ``@trace_step_sync`` (decorator / sync context manager)
    Wraps individual logical steps inside an agent.  Pushes a step-index frame
    on :data:`_step_stack` so that subsequent :func:`trace_model_call` and
    :func:`trace_tool_call` calls are stamped with the correct ``step_index``.

* :func:`trace_model_call`, :func:`trace_tool_call`
    Fire-and-forget recording functions called inside a step to log one LLM
    call or tool invocation.  Data is written into two thread-safe registry
    dicts keyed by ``trace_id``:

    * :data:`_token_registry`  — model calls
    * :data:`_tool_registry`   — tool calls

    When the ``agent/`` span closes the
    :class:`~observability.instrumentation.DatabaseSpanExporter` pops both
    registries and merges the data into the
    :class:`~observability.observability_service.TraceContext` that is then
    persisted to the ``qo_observability_trace`` table.

Context propagation
    :func:`set_trace_context_ids` stores ``session_id`` / ``correlation_id``
    in :class:`contextvars.ContextVar` slots.  These propagate automatically
    to child asyncio tasks and are picked up by ``@trace_agent`` at span start.
    Call :func:`clear_trace_context_ids` after the pipeline root completes.
"""

import asyncio
import contextvars
import functools
import inspect
import traceback
import time
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar
from uuid import UUID, uuid4
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Context-variable propagation for session_id / correlation_id.
#
# Call set_trace_context_ids() once at the pipeline entry point (e.g. before
# processing an email).  Both async and sync @trace_agent wrappers read these
# vars and stamp them as span attributes so the DatabaseSpanExporter can pick
# them up and write them to the DB.  ContextVar semantics ensure each asyncio
# task / thread sees its own values without interference.
# ---------------------------------------------------------------------------
_active_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'active_session_id', default=None
)
_active_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'active_correlation_id', default=None
)


def set_trace_context_ids(
    session_id: Optional[UUID] = None,
    correlation_id: Optional[UUID] = None,
) -> None:
    """Set shared session_id and correlation_id for the current async/thread context.

    Call this once per logical unit of work (e.g. per email) before invoking any
    agents.  Both IDs will be stamped onto every @trace_agent span that runs
    within this context, so all related traces share the same grouping keys.
    """
    _active_session_id.set(str(session_id) if session_id is not None else None)
    _active_correlation_id.set(str(correlation_id) if correlation_id is not None else None)


def clear_trace_context_ids() -> None:
    """Clear any active context IDs (call when the unit of work is finished)."""
    _active_session_id.set(None)
    _active_correlation_id.set(None)

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from observability.database.engine import ObsAsyncSessionType as AsyncSessionType
from observability.instrumentation import get_tracer


# ---------------------------------------------------------------------------
# Step registry — thread-safe accumulator keyed by (trace_id, span_id).
# trace_step_sync / trace_step write here; _span_to_trace_context reads here.
# This avoids relying on OTel BoundedAttributes round-trips that can be
# disrupted by SimpleSpanProcessor's context-suppress token swap.
# ---------------------------------------------------------------------------
_step_registry: Dict[tuple, List[Dict[str, Any]]] = {}
_step_registry_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Token registry — thread-safe accumulator keyed by trace_id (int).
#
# WHY this exists:
#   trace_model_call() is typically called from inside a trace_step_sync block,
#   so trace.get_current_span() returns the STEP span (e.g. step/classify),
#   NOT the parent agent span.  Writing token attributes directly to the
#   current span therefore loses them — the exporter filters out non-agent/
#   spans and never reads those attributes.
#
#   Keying by trace_id (a 128-bit int shared across every span in a trace)
#   allows accumulation regardless of call depth.  _span_to_trace_context
#   pops this registry when it processes the closing agent/ span.
# ---------------------------------------------------------------------------
_token_registry: Dict[int, List[Dict[str, Any]]] = {}  # trace_id → [model_call_dicts]
_token_registry_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Tool registry — thread-safe accumulator keyed by trace_id (int).
#
# Mirrors _token_registry for tool calls.  trace_tool_call() is invoked from
# inside step/ spans (or deeper), so its current span is never the agent/
# span.  Without this registry the DatabaseSpanExporter would silently discard
# all tool-call data (it only processes agent/ spans).  Popped by
# _span_to_trace_context via pop_tools_for_trace() when the agent/ span ends.
# ---------------------------------------------------------------------------
_tool_registry: Dict[int, List[Dict[str, Any]]] = {}  # trace_id → [tool_call_dicts]
_tool_registry_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Step-index stack — thread-local stack tracking the current step index.
#
# trace_step_sync / trace_step push the step_index on entry and pop on exit.
# trace_tool_call / trace_model_call read the top of the stack to stamp
# `step_index` onto their registry entries, so the exporter can automatically
# derive each step's output_summary from its last tool / model call without
# any explicit capture() call in business code.
# ---------------------------------------------------------------------------
_step_index_local = threading.local()


def _push_current_step(step_index: int) -> None:
    """Push *step_index* onto the thread-local step stack."""
    if not hasattr(_step_index_local, 'stack'):
        _step_index_local.stack = []
    _step_index_local.stack.append(step_index)


def _pop_current_step() -> None:
    """Pop the innermost step index from the thread-local stack."""
    stack = getattr(_step_index_local, 'stack', [])
    if stack:
        stack.pop()


def _get_current_step_index() -> int:
    """Return the active step index, or -1 when outside any step context."""
    stack = getattr(_step_index_local, 'stack', [])
    return stack[-1] if stack else -1


def _registry_key(span) -> Optional[tuple]:
    """Return (trace_id, span_id) key for a span, or None if unavailable."""
    try:
        ctx = span.get_span_context()
        if ctx and ctx.trace_id and ctx.span_id:
            return (ctx.trace_id, ctx.span_id)
    except Exception:
        pass
    return None


class _StepHandle:
    """Yielded by :func:`trace_step_sync` / :func:`trace_step`.

    The preferred (wrapper) pattern keeps extraction logic at the step
    *definition* site, not mixed into business code:

    .. code-block:: python

        with trace_step_sync(
            "classify",
            step_type="llm_call",
            output_fn=lambda r: f"classification={r.get('classification')}",
        ) as step:
            result = run_classification(data)
            step.capture(result)   # wrapper applies output_fn — no formatting here

    For a one-off extractor override pass ``fn=`` to :meth:`capture`::

        step.capture(result, fn=lambda r: r.get("label"))
    """
    __slots__ = ('_parent_span', '_step_index', '_output_fn', 'output_summary')

    def __init__(
        self,
        parent_span,
        step_index: int,
        output_fn: Optional[Callable[[Any], str]] = None,
    ) -> None:
        self._parent_span = parent_span
        self._step_index = step_index
        self._output_fn = output_fn
        self.output_summary: Optional[str] = None

    def capture(
        self,
        value: Any,
        fn: Optional[Callable[[Any], str]] = None,
    ) -> None:
        """Apply the extractor to *value* and store the result as ``output_summary``.

        Extractor priority:

        1. ``fn`` passed here — one-off override.
        2. ``output_fn`` declared on the context manager — the wrapper default.
        3. Bare ``str()`` fallback.

        Args:
            value:   Raw result object produced by this step.
            fn:      Optional one-off extractor ``(Any) -> str``.
        """
        extractor = fn or self._output_fn or str
        try:
            self.output_summary = extractor(value)
        except Exception:
            self.output_summary = str(value)


def _registry_claim_step(
    parent_span,
    step_name: str,
    step_type: str,
    decision_summary: Optional[str],
    start_dt: datetime,
) -> int:
    """Register a new step for *parent_span* and return its 0-based index."""
    key = _registry_key(parent_span)
    step_index = 0
    if key is not None:
        with _step_registry_lock:
            steps = _step_registry.setdefault(key, [])
            step_index = len(steps)
            steps.append({
                'index': step_index,
                'name': step_name,
                'step_type': step_type or 'unknown',
                'started_at': start_dt.isoformat(),
                'ended_at': None,
                'status': 'running',
                'latency_ms': None,
                'retries': 0,
                'decision_summary': decision_summary,
                'steps_status': 'measured',
            })
    return step_index


def _registry_finish_step(
    parent_span,
    step_index: int,
    status: str,
    latency_ms: int,
    error_type: Optional[str] = None,
    output_summary: Optional[str] = None,
) -> None:
    """Update the step entry in the registry with final status/timing."""
    key = _registry_key(parent_span)
    if key is None:
        return
    with _step_registry_lock:
        steps = _step_registry.get(key)
        if steps and step_index < len(steps):
            steps[step_index].update({
                'ended_at': datetime.now(timezone.utc).isoformat(),
                'status': status,
                'latency_ms': latency_ms,
            })
            if error_type:
                steps[step_index]['error_type'] = error_type
            if output_summary is not None:
                steps[step_index]['output_summary'] = output_summary


def pop_steps_for_span(trace_id: int, span_id: int) -> Optional[List[Dict[str, Any]]]:
    """
    Pop and return the measured step list for the given (trace_id, span_id).
    Called by DatabaseSpanExporter._span_to_trace_context when the agent span ends.
    Returns None when no measured steps exist (fall back to Phase B / derived).
    """
    key = (trace_id, span_id)
    with _step_registry_lock:
        return _step_registry.pop(key, None)


def pop_tokens_for_trace(trace_id: int) -> Optional[List[Dict[str, Any]]]:
    """
    Pop and return the accumulated model-call list for the given trace_id.
    Called by DatabaseSpanExporter._span_to_trace_context when the agent/ span ends.
    Returns None when no model calls were recorded for this trace.
    """
    with _token_registry_lock:
        return _token_registry.pop(trace_id, None)


def pop_tools_for_trace(trace_id: int) -> Optional[List[Dict[str, Any]]]:
    """
    Pop and return the accumulated tool-call list for the given trace_id.
    Called by DatabaseSpanExporter._span_to_trace_context when the agent/ span ends.
    Returns None when no tool calls were recorded for this trace.
    """
    with _tool_registry_lock:
        return _tool_registry.pop(trace_id, None)


# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


def trace_agent(
    agent_name: Optional[str] = None,
    agent_version: Optional[str] = None,
    environment: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to automatically trace agent execution using OpenTelemetry.
    
    Usage:
        @trace_agent(agent_name="EmailClassifierAgent", agent_version="1.0")
        async def classify_email(email_data: dict, session: AsyncSessionType):
            # Agent logic here
            return result
    
    Args:
        agent_name: Name of the agent (auto-detected from function name if not provided)
        agent_version: Version of the agent
        environment: Execution environment (dev/staging/production)
        tags: Additional tags for filtering/grouping
    
    Returns:
        Decorated function with automatic tracing
    """
    def decorator(func: F) -> F:
        # Infer agent name from function name if not provided
        inferred_agent_name = agent_name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get tracer
                tracer = get_tracer()
                
                if tracer is None:
                    # Tracing not available, run without tracing
                    return await func(*args, **kwargs)
                
                # Stamp enqueue time BEFORE opening the OTel span so that
                # span-setup overhead is captured as queue_time_ms (Option A).
                _enqueue_ns = time.perf_counter_ns()

                # Start OpenTelemetry span
                with tracer.start_as_current_span(f"agent/{inferred_agent_name}") as span:
                    _execution_start_ns = time.perf_counter_ns()
                    _queue_time_ms = max(0, int((_execution_start_ns - _enqueue_ns) / 1_000_000))
                    span.set_attribute("queue_time_ms", _queue_time_ms)
                    start_time = time.time()
                    
                    # Set span attributes
                    span.set_attribute("agent_name", inferred_agent_name)
                    if agent_version:
                        span.set_attribute("agent_version", agent_version)
                    if environment:
                        span.set_attribute("environment", environment)
                    else:
                        span.set_attribute("environment", _get_environment())
                    if tags:
                        for key, value in tags.items():
                            span.set_attribute(f"tag.{key}", str(value))

                    # Propagate shared pipeline IDs if set in the current context
                    _sid = _active_session_id.get()
                    if _sid:
                        span.set_attribute("session_id", _sid)
                    _cid = _active_correlation_id.get()
                    if _cid:
                        span.set_attribute("correlation_id", _cid)

                    # Extract user query before execution
                    user_query = _extract_user_query(args, kwargs, func)
                    if user_query:
                        span.set_attribute("user_query", user_query)
                    
                    result = None
                    try:
                        # Strip kwargs the wrapped function does not accept
                        # (e.g. `session` injected by FastAPI Depends).
                        _sig = inspect.signature(func)
                        _accepts_var_kw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in _sig.parameters.values()
                        )
                        _call_kwargs = kwargs if _accepts_var_kw else {
                            k: v for k, v in kwargs.items() if k in _sig.parameters
                        }
                        # Execute agent
                        result = await func(*args, **_call_kwargs)
                        
                        # Extract agent response after execution
                        agent_response = _extract_agent_response(result)
                        if agent_response:
                            span.set_attribute("agent_response", agent_response)
                        
                        # Set success status
                        span.set_status(Status(StatusCode.OK))
                        
                        # Calculate duration
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration_ms", int(duration_ms))
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("error_type", type(e).__name__)
                        span.set_attribute("error_message", str(e))
                        span.set_attribute("stack_trace", traceback.format_exc())
                        
                        # Calculate duration
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration_ms", int(duration_ms))
                        
                        raise
            
            return async_wrapper  # type: ignore
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get tracer
                tracer = get_tracer()

                if tracer is None:
                    # Tracing not available, run without tracing
                    return func(*args, **kwargs)

                # Stamp enqueue time BEFORE opening the OTel span so that
                # span-setup overhead is captured as queue_time_ms (Option A).
                _enqueue_ns = time.perf_counter_ns()

                # Start OpenTelemetry span
                with tracer.start_as_current_span(f"agent/{inferred_agent_name}") as span:
                    _execution_start_ns = time.perf_counter_ns()
                    _queue_time_ms = max(0, int((_execution_start_ns - _enqueue_ns) / 1_000_000))
                    span.set_attribute("queue_time_ms", _queue_time_ms)
                    start_time = time.time()
                    
                    # Set span attributes
                    span.set_attribute("agent_name", inferred_agent_name)
                    if agent_version:
                        span.set_attribute("agent_version", agent_version)
                    if environment:
                        span.set_attribute("environment", environment)
                    else:
                        span.set_attribute("environment", _get_environment())
                    if tags:
                        for key, value in tags.items():
                            span.set_attribute(f"tag.{key}", str(value))

                    # Propagate shared pipeline IDs if set in the current context
                    _sid = _active_session_id.get()
                    if _sid:
                        span.set_attribute("session_id", _sid)
                    _cid = _active_correlation_id.get()
                    if _cid:
                        span.set_attribute("correlation_id", _cid)

                    # Extract user query before execution
                    user_query = _extract_user_query(args, kwargs, func)
                    if user_query:
                        span.set_attribute("user_query", user_query)
                    
                    try:
                        # Strip kwargs the wrapped function does not accept
                        # (e.g. `session` injected by FastAPI Depends).
                        _sig = inspect.signature(func)
                        _accepts_var_kw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in _sig.parameters.values()
                        )
                        _call_kwargs = kwargs if _accepts_var_kw else {
                            k: v for k, v in kwargs.items() if k in _sig.parameters
                        }
                        # Execute agent
                        result = func(*args, **_call_kwargs)
                        
                        # Extract agent response after execution
                        agent_response = _extract_agent_response(result)
                        if agent_response:
                            span.set_attribute("agent_response", agent_response)
                        
                        # Set success status
                        span.set_status(Status(StatusCode.OK))
                        
                        # Calculate duration
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration_ms", int(duration_ms))

                        return result
                    except Exception as e:
                        # Record error
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("error_type", type(e).__name__)
                        span.set_attribute("error_message", str(e))
                        span.set_attribute("stack_trace", traceback.format_exc())
                        
                        # Calculate duration
                        duration_ms = (time.time() - start_time) * 1000
                        span.set_attribute("duration_ms", int(duration_ms))

                        raise

            return sync_wrapper  # type: ignore
    
    return decorator


@asynccontextmanager
async def trace_step(
    step_name: str,
    decision_summary: Optional[str] = None,
    step_type: Optional[str] = None,
    output_fn: Optional[Callable[[Any], str]] = None,
):
    """
    Context manager for tracing individual steps within an agent using OpenTelemetry.

    Encodes step data as flat ``step.<i>.*`` attributes on the **parent** agent
    span so that ``_span_to_trace_context`` can reconstruct a properly ordered
    ``steps`` list at export time.  A child span is still emitted for
    distributed-tracing visibility, but the structured step payload lives on
    the parent – so it is never silently discarded by the exporter filter that
    only processes ``agent/`` spans.

    Declare *how* to format the step output via ``output_fn`` — an
    ``(Any) -> str`` callable applied to whatever value you pass to
    ``step.capture(result)`` inside the block.  This keeps extraction logic
    at the wrapper definition rather than embedded in business code.

    Usage::

        async with trace_step(
            "llm_classification",
            decision_summary="Classifying email type",
            output_fn=lambda r: f"classification={r.get('classification')}",
        ) as step:
            result = await llm.classify(email_text)
            step.capture(result)

    Args:
        step_name:      Name of the step (e.g. ``"parse_email"``, ``"llm_call"``).
        decision_summary: Optional human-readable summary of the step's purpose.
        step_type:      Semantic type (``"llm_call"``, ``"tool_call"``, ``"parse"``, …).
        output_fn:      ``(Any) -> str`` extractor applied by ``step.capture()``.
    """
    parent_span = trace.get_current_span()
    start_time = time.time()
    start_dt = datetime.now(timezone.utc)

    # Register step in the module-level registry.
    step_index = _registry_claim_step(
        parent_span, step_name, step_type or 'unknown', decision_summary, start_dt
    )
    handle = _StepHandle(parent_span, step_index, output_fn=output_fn)
    _push_current_step(step_index)

    tracer = get_tracer()
    if tracer is None:
        try:
            yield handle
            latency_ms = int((time.time() - start_time) * 1000)
            _registry_finish_step(parent_span, step_index, 'success', latency_ms,
                                  output_summary=handle.output_summary)
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            _registry_finish_step(parent_span, step_index, 'failure', latency_ms, type(e).__name__,
                                  output_summary=handle.output_summary)
            raise
        finally:
            _pop_current_step()
        return

    # Emit a child span for distributed-tracing visibility (Jaeger / OTLP)
    with tracer.start_as_current_span(f'step/{step_name}') as child_span:
        child_span.set_attribute('step_name', step_name)
        child_span.set_attribute('step_index', step_index)
        if step_type:
            child_span.set_attribute('step_type', step_type)
        if decision_summary:
            child_span.set_attribute('decision_summary', decision_summary)

        try:
            yield handle
            latency_ms = int((time.time() - start_time) * 1000)
            child_span.set_attribute('latency_ms', latency_ms)
            child_span.set_status(Status(StatusCode.OK))
            _registry_finish_step(parent_span, step_index, 'success', latency_ms,
                                  output_summary=handle.output_summary)
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            child_span.set_attribute('latency_ms', latency_ms)
            child_span.set_status(Status(StatusCode.ERROR, str(e)))
            child_span.set_attribute('error_type', type(e).__name__)
            child_span.set_attribute('error_message', str(e))
            _registry_finish_step(parent_span, step_index, 'failure', latency_ms, type(e).__name__,
                                  output_summary=handle.output_summary)
            raise
        finally:
            _pop_current_step()


from contextlib import contextmanager as _contextmanager


@_contextmanager
def trace_step_sync(
    step_name: str,
    decision_summary: Optional[str] = None,
    step_type: Optional[str] = None,
    output_fn: Optional[Callable[[Any], str]] = None,
):
    """
    Synchronous context manager for tracing individual steps within a sync agent.

    Stores step data in a module-level registry keyed by the parent agent span-id
    so that ``_span_to_trace_context`` can reliably reconstruct a ``steps`` list
    at export time, regardless of OTel context-manager internals.

    Declare *how* to format the step output via ``output_fn`` — an
    ``(Any) -> str`` callable applied to whatever value you pass to
    ``step.capture(result)`` inside the block.  This keeps extraction logic
    at the wrapper definition rather than embedded in business code::

        with trace_step_sync(
            "classify",
            step_type="llm_call",
            output_fn=lambda r: f"classification={r.get('classification')}",
        ) as step:
            result = run_classification(data)
            step.capture(result)   # output_fn applied here, no formatting in business code

    Args:
        step_name:        Logical name (e.g. ``"parse_input"``, ``"classify"``).
        decision_summary: Optional human-readable note about what this step does.
        step_type:        Semantic category: ``"parse"``, ``"llm_call"``, ``"process"``,
                          ``"format"``, ``"tool_call"``, ``"plan"``, ``"final"``.
        output_fn:        ``(Any) -> str`` extractor applied by ``step.capture()``.
    """
    parent_span = trace.get_current_span()
    start_time = time.time()
    start_dt = datetime.now(timezone.utc)

    # Register step in the module-level registry.
    step_index = _registry_claim_step(
        parent_span, step_name, step_type or 'unknown', decision_summary, start_dt
    )
    handle = _StepHandle(parent_span, step_index, output_fn=output_fn)
    _push_current_step(step_index)

    tracer = get_tracer()
    if tracer is None:
        try:
            yield handle
            latency_ms = int((time.time() - start_time) * 1000)
            _registry_finish_step(parent_span, step_index, 'success', latency_ms,
                                  output_summary=handle.output_summary)
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            _registry_finish_step(parent_span, step_index, 'failure', latency_ms, type(e).__name__,
                                  output_summary=handle.output_summary)
            raise
        finally:
            _pop_current_step()
        return

    # Emit a child span for distributed-tracing visibility.
    with tracer.start_as_current_span(f'step/{step_name}') as child_span:
        child_span.set_attribute('step_name', step_name)
        child_span.set_attribute('step_index', step_index)
        if step_type:
            child_span.set_attribute('step_type', step_type)
        if decision_summary:
            child_span.set_attribute('decision_summary', decision_summary)

        try:
            yield handle
            latency_ms = int((time.time() - start_time) * 1000)
            child_span.set_attribute('latency_ms', latency_ms)
            child_span.set_status(Status(StatusCode.OK))
            _registry_finish_step(parent_span, step_index, 'success', latency_ms,
                                  output_summary=handle.output_summary)
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            child_span.set_attribute('latency_ms', latency_ms)
            child_span.set_status(Status(StatusCode.ERROR, str(e)))
            child_span.set_attribute('error_type', type(e).__name__)
            child_span.set_attribute('error_message', str(e))
            _registry_finish_step(parent_span, step_index, 'failure', latency_ms, type(e).__name__,
                                  output_summary=handle.output_summary)
            raise
        finally:
            _pop_current_step()


def trace_model_call(
    provider: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    model_version: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    cache_status: Optional[str] = None,
    status: str = 'success',
    error: Optional[Exception] = None,
    token_usage_available: bool = True,
    token_usage_estimated: bool = False,
    started_at: Optional[datetime] = None,
    model_call_type: str = 'chat',
    response_summary: Optional[str] = None,
    cost_usd: Optional[float] = None,
):
    """
    Record a model call using OpenTelemetry span.
    
    Usage:
        response = await llm.generate(prompt)
        trace_model_call(
            provider="openai",
            model_name="gpt-4",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=response.latency_ms,
        )
    
    Args:
        provider:              LLM provider identifier (e.g. ``"azure"``, ``"openai"``).
        model_name:            Deployment or model name (e.g. ``"gpt-4.1"``).
        prompt_tokens:         Number of input/prompt tokens consumed.
        completion_tokens:     Number of output/completion tokens generated.
        latency_ms:            End-to-end call duration in milliseconds.
        model_version:         Optional model version string.
        parameters:            Optional dict of call parameters (temperature, max_tokens, etc.).
        cache_status:          ``"hit"`` / ``"miss"`` / ``None`` — prompt-cache result.
        status:                ``"success"`` or ``"error"``.
        error:                 Exception instance on failure; used to populate
                               ``error_class`` and ``error_message``.
        token_usage_available: ``True`` when token counts are real measured values.
        token_usage_estimated: ``True`` when token counts are approximations rather
                               than exact measurements from the provider.
        started_at:            :class:`~datetime.datetime` marking when the LLM call
                               began; used to compute the model call ``started_at`` /
                               ``ended_at`` timestamps stored in the DB.
        model_call_type:       Semantic call category — ``"chat"`` (default),
                               ``"completion"``, ``"embedding"``, etc.
        response_summary:      Full LLM response text or structured representation.
                               Stored without truncation in
                               ``model_calls[].response_summary`` and appended to
                               the ``llm_call`` event ``payload_summary``.
    """
    tracer = get_tracer()
    if tracer is None:
        return

    # -----------------------------------------------------------------------
    # Accumulate this model call in the trace-level token registry.
    #
    # We use trace_id (not span_id) as key because this function is called
    # from inside trace_step_sync blocks where the current span is a STEP
    # span, not the agent span.  All spans of the same logical trace share
    # the same trace_id, so the agent span can pop this registry when it
    # closes even though it was never the "current" span during the LLM call.
    # -----------------------------------------------------------------------
    current_span = trace.get_current_span()
    span_ctx = current_span.get_span_context() if current_span else None
    if span_ctx and span_ctx.trace_id and span_ctx.trace_id != 0:
        call_entry = {
            'step_index': _get_current_step_index(),
            'provider': str(provider),
            'model_name': str(model_name),
            'model_version': str(model_version) if model_version else None,
            'prompt_tokens': int(prompt_tokens or 0),
            'completion_tokens': int(completion_tokens or 0),
            'total_tokens': int(prompt_tokens or 0) + int(completion_tokens or 0),
            'latency_ms': int(latency_ms or 0),
            'status': str(status),
            'error_class': type(error).__name__ if error is not None else None,
            'error_message': str(error) if error is not None else None,
            'token_usage_available': bool(token_usage_available),
            'token_usage_estimated': bool(token_usage_estimated),
            'started_at': started_at.isoformat() if started_at is not None else None,
            'model_call_type': str(model_call_type),
            'response_summary': str(response_summary) if response_summary else None,
            'cost_usd': float(cost_usd) if cost_usd is not None else None,
        }
        with _token_registry_lock:
            _token_registry.setdefault(span_ctx.trace_id, []).append(call_entry)

    # Also propagate key attributes onto the current span for distributed-tracing
    # visibility (e.g. Jaeger).  Note: the current span may be a step span
    # rather than the agent span — the registry above is the authoritative path
    # for the database exporter.
    if current_span and current_span.is_recording():
        # Set model_name on the current span (if not already set)
        if not current_span.attributes or 'model_name' not in current_span.attributes:
            current_span.set_attribute("model_name", model_name)
        if not current_span.attributes or 'llm_provider' not in current_span.attributes:
            current_span.set_attribute("llm_provider", provider)
        current_span.set_attribute("prompt_tokens", prompt_tokens)
        current_span.set_attribute("completion_tokens", completion_tokens)
        current_span.set_attribute("total_tokens", prompt_tokens + completion_tokens)
        current_span.set_attribute("llm_status", status)
        current_span.set_attribute("parameter.token_usage_available", str(token_usage_available).lower())
        current_span.set_attribute("parameter.token_usage_estimated", str(token_usage_estimated).lower())
        if error is not None:
            current_span.set_attribute("llm_error_type", type(error).__name__)
            current_span.set_attribute("llm_error_message", str(error))

    # Create a span for the model call
    with tracer.start_as_current_span(f"llm/{provider}/{model_name}") as span:
        span.set_attribute("provider", provider)
        span.set_attribute("model_name", model_name)
        span.set_attribute("operation", f"{provider}_call")
        
        if model_version:
            span.set_attribute("model_version", model_version)
        
        # Set token usage
        span.set_attribute("prompt_tokens", prompt_tokens)
        span.set_attribute("completion_tokens", completion_tokens)
        span.set_attribute("total_tokens", prompt_tokens + completion_tokens)
        
        # Set latency
        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("duration_ms", latency_ms)
        
        # Set parameters
        if parameters:
            for key, value in parameters.items():
                if value is not None:
                    span.set_attribute(f"parameter.{key}", str(value))
        
        # Set cache status
        if cache_status:
            span.set_attribute("cache_status", cache_status)
        
        # Set status
        if status == 'success':
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR))
            if error:
                span.set_attribute("error_type", type(error).__name__)
                span.set_attribute("error_message", str(error))


def trace_tool_call(
    tool_name: str,
    latency_ms: int,
    tool_version: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    output: Optional[Any] = None,
    status: str = 'success',
    error: Optional[Exception] = None,
):
    """
    Record a tool call using OpenTelemetry span.
    
    Usage:
        result = await email_validator.validate(email_data)
        trace_tool_call(
            tool_name="email_validator",
            latency_ms=latency,
            args={"email": email_data},
            output=result,
        )
    
    Args:
        tool_name: Tool identifier
        latency_ms: Call latency in milliseconds
        tool_version: Tool version
        args: Tool arguments
        output: Tool output
        status: Call status (success/failure)
        error: Exception if failed
    """
    tracer = get_tracer()

    # -----------------------------------------------------------------------
    # Accumulate this tool call in the trace-level tool registry.
    #
    # trace_tool_call() is typically called from inside a trace_step_sync
    # block, so the current span is a STEP span, not the agent/ span.
    # The DatabaseSpanExporter only processes agent/ spans and would discard
    # a bare tool/{name} child span.  Keying by trace_id (shared across all
    # spans in the trace) lets _span_to_trace_context reliably pop this data
    # when the agent/ span closes — identical to how _token_registry works
    # for model calls.
    # -----------------------------------------------------------------------
    current_span = trace.get_current_span()
    span_ctx = current_span.get_span_context() if current_span else None
    if span_ctx and span_ctx.trace_id and span_ctx.trace_id != 0:
        import json as _json
        args_summary: Optional[str] = None
        if args:
            try:
                args_summary = _json.dumps(args, default=str)
            except Exception:
                args_summary = str(args)
        output_summary: Optional[str] = str(output) if output is not None else None
        call_entry = {
            'step_index': _get_current_step_index(),
            'tool_name': str(tool_name),
            'tool_version': str(tool_version) if tool_version else None,
            'args_summary': args_summary,
            'output_summary': output_summary,
            'ended_at': datetime.now(timezone.utc).isoformat(),
            'status': str(status),
            'latency_ms': int(latency_ms or 0),
            'error_class': type(error).__name__ if error is not None else None,
            'error_message': str(error) if error is not None else None,
        }
        with _tool_registry_lock:
            _tool_registry.setdefault(span_ctx.trace_id, []).append(call_entry)

    if tracer is None:
        return

    with tracer.start_as_current_span(f"tool/{tool_name}") as span:
        span.set_attribute("tool_name", tool_name)
        
        if tool_version:
            span.set_attribute("tool_version", tool_version)
        
        span.set_attribute("latency_ms", latency_ms)
        
        # Set args (summarized)
        if args:
            import json
            try:
                args_summary_str = json.dumps(args, default=str)
                span.set_attribute("args_summary", args_summary_str)
            except Exception:
                pass
        
        # Set output (summarized)
        if output is not None:
            output_summary_str = str(output)
            span.set_attribute("output_summary", output_summary_str)
        
        # Set status
        if status == 'success':
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR))
            if error:
                span.set_attribute("error_type", type(error).__name__)
                span.set_attribute("error_message", str(error))


# Helper functions

def _extract_user_query(args: tuple, kwargs: dict, func: Callable) -> Optional[str]:
    """
    Extract the primary input/query from a traced function's arguments.

    Generic — not tied to any specific agent domain.  The heuristics are:
      1. Look for a kwarg (or positional arg) whose name appears in
         ``_QUERY_PARAM_NAMES``.  Extend that set when new agents are added.
      2. File-path strings (param name ends with ``_file`` / ``_path``, or the
         value contains an OS path separator) are returned as their basename
         only — long temp-directory paths confuse downstream consumers such as
         Foundry's JSONL eval data source.
      3. Dict values are summarised by scanning a priority list of common
         content keys; the full JSON is used as a fallback.
      4. If no named parameter matches, the first positional non-session arg
         is used as a last resort.
    """
    import os as _os

    # Names of parameters that carry the primary agent input.
    # Add domain-specific names here as new agents are onboarded.
    _QUERY_PARAM_NAMES = {
        'query', 'user_query', 'input', 'prompt', 'message',
        'text', 'content',
        # File-based input (generic)
        'file', 'file_path', 'filepath', 'filename', 'msg_file',
        # Structured input objects (generic)
        'data', 'payload', 'request', 'request_data',
        # Domain-specific — extend as needed
        'email_data', 'email_json',
    }

    # Param-name suffixes that indicate the value is a filesystem path.
    _FILE_SUFFIXES = ('_file', '_path', '_filepath', '_filename')

    def _is_file_param(name: str) -> bool:
        return name.endswith(_FILE_SUFFIXES)

    def _is_file_value(value: str) -> bool:
        """True when the string looks like an absolute or temp file path."""
        return _os.sep in value or (len(value) > 60 and '/' in value)

    def _extract_value(param_name: str, value: Any) -> Optional[str]:
        if isinstance(value, str):
            if _is_file_param(param_name) or _is_file_value(value):
                return _os.path.basename(value) or value
            return value
        if isinstance(value, dict):
            for key in ['query', 'content', 'message', 'text', 'body', 'subject', 'title']:
                if key in value and isinstance(value[key], str):
                    return f"{key.title()}: {value[key]}"
            try:
                import json as _json
                return _json.dumps(value, default=str)
            except Exception:
                return str(value)
        return str(value)

    # --- 1. Check kwargs ---
    for param_name in _QUERY_PARAM_NAMES:
        if param_name in kwargs:
            result = _extract_value(param_name, kwargs[param_name])
            if result is not None:
                return result

    # --- 2. Check positional args via function signature ---
    try:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        for idx, arg in enumerate(args):
            if idx < len(param_names):
                param_name = param_names[idx]
                if param_name in _QUERY_PARAM_NAMES:
                    result = _extract_value(param_name, arg)
                    if result is not None:
                        return result
    except Exception:
        pass

    # --- 3. Final fallback: first non-session positional argument ---
    for arg in args:
        if arg is None:
            continue
        if 'session' in type(arg).__name__.lower():
            continue
        if isinstance(arg, str):
            return arg
        if isinstance(arg, dict):
            try:
                import json as _json
                return _json.dumps(arg, default=str)
            except Exception:
                return str(arg)
        return str(arg)

    return None


def _extract_agent_response(result: Any) -> Optional[str]:
    """
    Extract agent response from return value.
    """
    if result is None:
        return None
    
    # If string, check if it's JSON first
    if isinstance(result, str):
        # Try to parse as JSON and extract key fields
        try:
            import json
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                # Extract key fields from classifier or other JSON responses
                summary_parts = []
                for key in ['classification', 'success', 'result', 'status', 'output']:
                    if key in parsed:
                        summary_parts.append(f"{key}: {parsed[key]}")
                if summary_parts:
                    summary = "; ".join(summary_parts)
                    # If the summary is too sparse for LLM evaluators, fall
                    # back to the full JSON so Foundry has enough context.
                    if len(summary) < 100:
                        try:
                            return json.dumps(parsed, default=str)
                        except Exception:
                            pass
                    return summary
        except:
            pass
        # If not JSON or parsing failed, return as-is
        return result
    
    # If dict, try to extract key fields
    if isinstance(result, dict):
        # Try to create a summary of the response
        summary_parts = []
        
        # Common response field names to summarize
        for key in ['classification', 'success', 'result', 'status', 'output', 'answer', 'response', 'data', 'error']:
            if key in result:
                summary_parts.append(f"{key}: {result[key]}")
        
        # If we found summary parts, return them
        if summary_parts:
            summary = "; ".join(summary_parts)
            # Add count of other fields
            other_fields = [k for k in result.keys() if k not in ['classification', 'success', 'result', 'status', 'output', 'answer', 'response']]
            if other_fields:
                summary += f" (+ {len(other_fields)} other fields)"
            return summary
        
        # Otherwise return JSON representation
        import json
        try:
            return json.dumps(result, default=str)
        except:
            return str(result)
    
    # For other types, return string representation
    return str(result)


def _get_environment() -> str:
    """Get current environment from config or default."""
    try:
        from observability.config import settings
        # Try to infer from config
        if hasattr(settings, 'ENVIRONMENT'):
            return settings.ENVIRONMENT
        # Otherwise infer from other settings
        if hasattr(settings, 'USE_KEY_VAULT') and settings.USE_KEY_VAULT:
            return "production"
        return "development"
    except Exception:
        return "unknown"



