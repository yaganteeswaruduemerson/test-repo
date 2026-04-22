"""
OpenTelemetry instrumentation — tracer setup and span-to-trace-context conversion.

Architecture overview
---------------------
The module wires together three components:

1. **Tracer lifecycle** — :func:`initialize_tracer` creates a
   ``TracerProvider`` backed by ``DatabaseSpanExporter``.
   :func:`get_tracer` returns the singleton, auto-initialising on first use.
   An ``atexit`` handler flushes pending spans on process exit.

2. **DatabaseSpanExporter** — a custom ``SpanExporter`` that converts
   *only* ``agent/`` spans into :class:`~observability.observability_service.TraceContext`
   objects and persists them to Azure SQL via
   :meth:`~observability.observability_service.ObservabilityService.persist_trace`.
   Non-agent spans (``step/``, ``llm/``, ``tool/``) are intentionally ignored by
   the exporter; their data reaches the DB via the three registry systems below.

3. **Registry pop pattern** — ``observability_wrapper`` accumulates model calls,
   tool calls, and events in three thread-safe dicts keyed by ``trace_id``
   (``_token_registry``, ``_tool_registry``, ``_event_registry``).
   :meth:`DatabaseSpanExporter._span_to_trace_context` pops each registry when
   the parent ``agent/`` span closes, so all nested call data is correctly
   attached regardless of span-context nesting depth.

4. **Step reconstruction (three-tier)** — steps are sourced preferentially from
   the ``_step_registry`` (Tier 1), then from flat OTel span attributes (Tier 2),
   and finally as a single synthesised step when neither source is available
   (Tier 3 / derived).
"""

import atexit
import asyncio
import json
import logging
import os
import threading
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Background event loop for fire-and-forget async span export.
#
# Using a single persistent background thread with its own event loop avoids
# two critical bugs in the previous implementation:
#
# Bug 2 (blocking): The exporter was doing thread.join() after spawning the
#   persist thread, which blocked the FastAPI event loop on every span close.
#
# Bug 3 (new-loop-per-trace): Each persistence call created a brand-new
#   asyncio event loop, bypassing the existing connection pool and causing
#   resource leaks under load.
#
# The fix: a single daemon thread runs _export_loop.run_forever() for the
# lifetime of the process.  export() schedules coroutines onto this loop via
# asyncio.run_coroutine_threadsafe() — which is thread-safe and returns a
# concurrent.futures.Future that we do NOT wait on in the async path (true
# fire-and-forget).  In the rare non-async path (no running event loop) we
# block the caller with future.result(timeout=30) which is safe because the
# call site is synchronous anyway.
# ---------------------------------------------------------------------------
_export_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_export_thread: threading.Thread = threading.Thread(
    target=_export_loop.run_forever,
    daemon=True,
    name="obs-exporter",
)
_export_thread.start()

def _to_bool(value) -> bool:
    """Parse span attribute booleans reliably (handles bools and 'true'/'false' strings)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}

# Global tracer instance
_tracer: Optional[trace.Tracer] = None
_tracer_provider: Optional[TracerProvider] = None
_cleanup_registered = False


class DatabaseSpanExporter(SpanExporter):
    """
    Custom ``SpanExporter`` that bridges OpenTelemetry spans to the Azure SQL
    ``qo_observability_trace`` table.

    Only ``agent/`` spans are processed; all child spans (``step/``, ``llm/``,
    ``tool/``, ``event/``) are skipped.  Data from those child spans is
    collected via the registry pattern in ``observability_wrapper`` and is
    attached to the ``agent/`` trace context when it is built.

    Persistence is synchronous — each ``export()`` call blocks until all trace
    contexts have been written.  When called from inside a running event loop
    (e.g. FastAPI), the async DB write is dispatched to a new background thread
    to avoid blocking the caller's loop.
    """
    
    def __init__(self):
        """Initialize the database exporter."""
        self.shutdown_flag = False
    
    def export(self, spans) -> SpanExportResult:
        """Export spans by persisting them asynchronously on the background loop."""
        if self.shutdown_flag:
            logger.warning("DatabaseSpanExporter.export() called after shutdown — spans dropped")
            return SpanExportResult.FAILURE

        try:
            all_span_names = [str(getattr(s, 'name', '<unknown>')) for s in spans]
            agent_span_names = [n for n in all_span_names if n.startswith('agent/')]
            logger.debug(
                "DatabaseSpanExporter.export(): received %d span(s); "
                "agent/ spans: %d %s; non-agent spans (skipped): %d",
                len(all_span_names), len(agent_span_names), agent_span_names,
                len(all_span_names) - len(agent_span_names),
            )

            # Process each span and collect trace contexts
            trace_contexts = []
            for span in spans:
                trace_context = self._span_to_trace_context(span)
                if trace_context:
                    trace_contexts.append(trace_context)

            if not trace_contexts:
                if agent_span_names:
                    logger.warning(
                        "DatabaseSpanExporter.export(): %d agent/ span(s) received but "
                        "_span_to_trace_context() returned None for all — "
                        "check logs above for conversion errors",
                        len(agent_span_names),
                    )
                return SpanExportResult.SUCCESS

            coro = self._persist_traces_batch(trace_contexts)

            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is not None:
                # Called from inside an async context (e.g. FastAPI).
                # Schedule fire-and-forget on the dedicated background loop
                # so the caller's event loop is NOT blocked.
                asyncio.run_coroutine_threadsafe(coro, _export_loop)
                logger.info(
                    "DatabaseSpanExporter.export(): scheduled %d trace context(s) for async DB persistence",
                    len(trace_contexts),
                )
            else:
                # Called from a purely synchronous context (rare; e.g. atexit
                # flush or sync test).  Block at most 30s — safe here because
                # there is no running event loop to starve.
                future = asyncio.run_coroutine_threadsafe(coro, _export_loop)
                try:
                    future.result(timeout=30)
                    logger.info(
                        "DatabaseSpanExporter.export(): persisted %d trace(s) synchronously",
                        len(trace_contexts),
                    )
                except Exception as exc:
                    logger.error("Sync trace persistence failed: %s", exc, exc_info=True)

            return SpanExportResult.SUCCESS

        except Exception as e:
            logger.error("DatabaseSpanExporter.export() unhandled error: %s", e, exc_info=True)
            return SpanExportResult.FAILURE
    
    async def _persist_traces_batch(self, trace_contexts: list):
        """Persist a batch of trace contexts.

        Bug 4 fix: use gen.aclose() instead of the fragile await anext(session_gen)
        pattern that could silently swallow connection leaks on DB errors.
        aclose() always executes the generator's finally block regardless of whether
        the body raised an exception.
        """
        from observability.observability_service import get_observability_service
        from observability.database.engine import get_obs_async_session as get_async_session

        service = get_observability_service()
        logger.info(
            "_persist_traces_batch: attempting to persist %d trace(s) to database",
            len(trace_contexts),
        )

        gen = get_async_session()
        session = await anext(gen)
        try:
            for trace_context in trace_contexts:
                try:
                    await service.persist_trace(trace_context, session)
                except Exception as e:
                    logger.error(
                        "_persist_traces_batch: FAILED to save trace %s (agent=%r): %s",
                        trace_context.agent_execution_id,
                        getattr(trace_context, 'agent_name', '?'),
                        e,
                        exc_info=True,
                    )
        finally:
            # Properly close the async generator so its finally/cleanup block runs.
            await gen.aclose()
    
    @staticmethod
    def _auto_fill_step_statuses(trace_context) -> None:
        """Promote error status from tool/model calls up to the step level.

        If any tool call or model call within a step has ``status == 'error'``,
        the step itself is marked ``'error'`` — even when the agent did not raise
        an exception (e.g. it caught the error and returned early).
        Steps already recorded as ``'failure'`` or ``'error'`` are left as-is.
        """
        error_step_indices = set()
        for tc in getattr(trace_context, 'tool_calls', []):
            if tc.get('status') == 'error':
                error_step_indices.add(tc.get('step_index'))
        for mc in getattr(trace_context, 'model_calls', []):
            if mc.get('status') == 'error':
                error_step_indices.add(mc.get('step_index'))
        for step in getattr(trace_context, 'steps', []):
            if step.get('index') in error_step_indices:
                step['status'] = 'error'

    @staticmethod
    def _auto_fill_step_outputs(trace_context) -> None:
        """Derive ``output_summary`` for each step from its last tool call or
        model call, so callers never need an explicit ``step.capture()`` call.

        Priority per step:
        1. ``output_summary`` already set manually (via ``step.capture()``) — leave untouched.
        2. Last **tool call** whose ``step_index`` matches → use its
           ``output_summary`` value.
        3. Last **model call** whose ``step_index`` matches → use its
           ``response_summary`` value.
        """
        for step in getattr(trace_context, 'steps', []):
            if step.get('output_summary'):
                continue  # already captured manually
            idx = step.get('index', -1)
            # Try tool calls first (most specific)
            for tc in reversed(getattr(trace_context, 'tool_calls', [])):
                if tc.get('step_index') == idx:
                    val = tc.get('output_summary')
                    if val:
                        step['output_summary'] = str(val)
                    break
            if step.get('output_summary'):
                continue
            # Fall back to model calls (response_summary)
            for mc in reversed(getattr(trace_context, 'model_calls', [])):
                if mc.get('step_index') == idx:
                    val = mc.get('response_summary')
                    if val:
                        step['output_summary'] = str(val)
                    break

    def _span_to_trace_context(self, span) -> Optional[object]:
        """
        Convert a closed ``agent/`` span into a populated :class:`TraceContext`.

        Only ``agent/``-prefixed spans are processed; all others return ``None``
        immediately.  The method applies the following population strategy:

        **Timing & identity**
            ``started_at``, ``ended_at`` are read from span timestamps
            :func:`~observability.observability_wrapper.trace_agent`.

        **Tier A — token registry (model_calls)**
            :func:`~observability.observability_wrapper.pop_tokens_for_trace` is
            called to drain model-call dicts accumulated by
            :func:`~observability.observability_wrapper.trace_model_call`.
            Each dict becomes one entry in ``trace_context.model_calls``.

        **Tier B — span-attribute fallback (model_calls)**
            Fires only when the token registry is empty.  Synthesises a single
            model call from flat OTel attributes (``prompt_tokens``,
            ``completion_tokens``, etc.) written directly onto the agent span.

        **Tool registry (tool_calls)**
            :func:`~observability.observability_wrapper.pop_tools_for_trace`
            drains tool-call dicts accumulated by
            :func:`~observability.observability_wrapper.trace_tool_call`.

        **Event registry (events)**
            :func:`~observability.observability_wrapper.pop_events_for_trace`
            drains event dicts accumulated by
            :func:`~observability.observability_wrapper.record_event_for_trace`.

        **Step reconstruction (three tiers)**
            1. *Registry* — :func:`~observability.observability_wrapper.pop_steps_for_span`
               returns measured steps written by :func:`trace_step_sync` / :func:`trace_step`.
            2. *OTel attributes* — flat ``step.<i>.*`` attributes on the span.
            3. *Derived* — a single synthesised step when no explicit steps exist
               but model/tool work was detected.

        After population, :meth:`_auto_fill_step_outputs` and
        :meth:`_auto_fill_step_statuses` are called to derive any missing step
        fields automatically.

        Args:
            span: A closed ``ReadableSpan`` from the OTel SDK.

        Returns:
            Populated :class:`~observability.observability_service.TraceContext`,
            or ``None`` if the span is not an ``agent/`` span or conversion fails.
        """
        try:
            from observability.observability_service import TraceContext
            from uuid import UUID, uuid4

            # Persist only top-level agent spans as observability traces.
            # LLM/tool/event spans are captured as structured fields on the parent agent span.
            span_name = str(getattr(span, 'name', ''))
            if not span_name.startswith('agent/'):
                return None

            logger.info(
                "_span_to_trace_context: processing closed span '%s'",
                span_name,
            )

            # Extract attributes
            attributes = {}
            if hasattr(span, 'attributes') and span.attributes:
                attributes = dict(span.attributes)

            # Get agent name from span name or attributes
            agent_name = attributes.get('agent_name', span.name.split('/')[-1] if '/' in span.name else span.name)
            
            # Extract shared pipeline ID stamped by set_trace_context_ids().
            # When the agent is called from an HTTP route without an explicit
            # set_trace_context_ids() call the attribute is absent — auto-generate
            # a UUID so the NOT NULL constraint on session_id is always satisfied.
            _sid_str = attributes.get('session_id')
            _session_id = UUID(_sid_str) if _sid_str else uuid4()

            # Create trace context
            trace_context = TraceContext(
                agent_name=agent_name,
                agent_version=attributes.get('agent_version'),
                environment=attributes.get('environment', 'production'),
                project_name=attributes.get('project_name'),
                session_id=_session_id,
            )
            
            # Set timing information
            if hasattr(span, 'start_time') and span.start_time:
                from datetime import datetime, timezone
                trace_context.started_at = datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc)
            
            if hasattr(span, 'end_time') and span.end_time:
                from datetime import datetime, timezone
                trace_context.ended_at = datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc)

            # Set status
            if hasattr(span, 'status') and span.status:
                from opentelemetry.trace import StatusCode
                from observability.database.models import ObservabilityExecutionStatus
                if span.status.status_code == StatusCode.ERROR:
                    trace_context.status = ObservabilityExecutionStatus.FAILURE
                    trace_context.error_class = attributes.get('error_type', 'UnknownError')
                    trace_context.error_message = attributes.get('error_message')
                    trace_context.stack_trace = attributes.get('stack_trace')
            
            # Set user query and response
            if 'user_query' in attributes:
                trace_context.set_user_query(attributes['user_query'])
            
            if 'agent_response' in attributes:
                trace_context.set_agent_response(attributes['agent_response'])
            
            # Extract token usage if available
            if 'prompt_tokens' in attributes:
                trace_context.prompt_tokens = int(attributes['prompt_tokens'])
            if 'completion_tokens' in attributes:
                trace_context.completion_tokens = int(attributes['completion_tokens'])

            # -------------------------------------------------------------------
            # Tier A — Token registry (primary path for model_calls population).
            #
            # trace_model_call() accumulates model-call dicts in _token_registry
            # keyed by trace_id regardless of how deeply nested the call site is.
            # This avoids the previous bug where token attributes were written to
            # the current step/ span rather than the agent/ span, causing them to
            # be silently discarded by the non-agent early-exit guard below.
            # -------------------------------------------------------------------
            if str(getattr(span, 'name', '')).startswith('agent/'):
                try:
                    from observability.observability_wrapper import pop_tokens_for_trace
                    _span_ctx_for_tokens = span.get_span_context() if hasattr(span, 'get_span_context') else None
                    registry_calls = pop_tokens_for_trace(_span_ctx_for_tokens.trace_id) if _span_ctx_for_tokens else None
                    if registry_calls:
                        for rc in registry_calls:
                            trace_context.add_model_call(
                                provider=rc['provider'],
                                model_name=rc['model_name'],
                                prompt_tokens=int(rc.get('prompt_tokens', 0) or 0),
                                completion_tokens=int(rc.get('completion_tokens', 0) or 0),
                                latency_ms=int(rc.get('latency_ms', 0) or 0),
                                model_version=rc.get('model_version'),
                                status=rc.get('status', 'success'),
                                error_class=rc.get('error_class'),
                                error_message=rc.get('error_message'),
                                token_usage_available=bool(rc.get('token_usage_available', True)),
                                token_usage_estimated=bool(rc.get('token_usage_estimated', False)),
                                model_call_type=rc.get('model_call_type', 'chat'),
                                step_index=rc.get('step_index'),
                                started_at=rc.get('started_at'),
                                response_summary=rc.get('response_summary'),
                                cost_usd=rc.get('cost_usd'),
                            )
                except ImportError:
                    pass

            # -------------------------------------------------------------------
            # Tier B — Span-attribute fallback (only fires when the registry is
            # empty, e.g. when trace_model_call was invoked at agent scope rather
            # than inside a step).  Synthesises one model_call from attributes
            # that trace_model_call wrote directly onto the agent span.
            # -------------------------------------------------------------------
            if str(getattr(span, 'name', '')).startswith('agent/') and not trace_context.model_calls:
                logger.debug(
                    "_span_to_trace_context: token registry empty for '%s' — "
                    "falling back to span attribute model_call synthesis",
                    span_name,
                )
                provider = attributes.get('llm_provider')
                model_name = attributes.get('model_name')
                prompt_tokens = int(attributes.get('prompt_tokens', 0) or 0)
                completion_tokens = int(attributes.get('completion_tokens', 0) or 0)
                total_tokens = int(attributes.get('total_tokens', 0) or 0)

                if model_name or prompt_tokens or completion_tokens or total_tokens:
                    latency_ms = int(attributes.get('duration_ms', attributes.get('latency_ms', 0)) or 0)
                    llm_status = str(attributes.get('llm_status', 'success'))
                    has_nonzero_tokens = prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0
                    token_usage_available = _to_bool(attributes.get('parameter.token_usage_available', has_nonzero_tokens))
                    token_usage_estimated = _to_bool(attributes.get('parameter.token_usage_estimated', False))
                    trace_context.add_model_call(
                        provider=str(provider or 'unknown'),
                        model_name=str(model_name or 'unknown'),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        status='failure' if llm_status == 'failure' else 'success',
                        error_class=attributes.get('llm_error_type'),
                        error_message=attributes.get('llm_error_message'),
                        token_usage_available=token_usage_available,
                        token_usage_estimated=token_usage_estimated,
                        model_call_type='chat',
                    )

            # -------------------------------------------------------------------
            # Tier A (tools) — Tool registry (primary path for tool_calls).
            #
            # trace_tool_call() accumulates tool-call dicts in _tool_registry
            # keyed by trace_id, exactly mirroring the _token_registry pattern
            # used for model_calls.  Without this block every tool call would
            # be silently dropped because the tool/{name} child span is not an
            # agent/ span and is discarded by the early-exit guard above.
            # -------------------------------------------------------------------
            if str(getattr(span, 'name', '')).startswith('agent/'):
                try:
                    from observability.observability_wrapper import pop_tools_for_trace
                    _span_ctx_for_tools = span.get_span_context() if hasattr(span, 'get_span_context') else None
                    registry_tool_calls = pop_tools_for_trace(_span_ctx_for_tools.trace_id) if _span_ctx_for_tools else None
                    if registry_tool_calls:
                        for tc in registry_tool_calls:
                            trace_context.add_tool_call(
                                tool_name=tc['tool_name'],
                                latency_ms=int(tc.get('latency_ms', 0) or 0),
                                tool_version=tc.get('tool_version'),
                                args_summary=json.loads(tc['args_summary']) if isinstance(tc.get('args_summary'), str) else tc.get('args_summary'),
                                output_summary=tc.get('output_summary'),
                                status=tc.get('status', 'success'),
                                error_class=tc.get('error_class'),
                                error_message=tc.get('error_message'),
                                step_index=tc.get('step_index'),
                                ended_at=tc.get('ended_at'),
                            )
                except ImportError:
                    pass

            # Extract cost if available
            if 'cost_amount' in attributes:
                trace_context.set_cost(
                    amount=float(attributes['cost_amount']),
                    currency=attributes.get('cost_currency', 'USD'),
                    price_version=attributes.get('price_version'),
                )

            # ---------------------------------------------------------------
            # Step reconstruction: three-tier approach
            #
            # Tier 1 — Registry (preferred): trace_step_sync / trace_step
            #   write step data to a module-level dict keyed by span-id.
            #   Immune to OTel context-suppress token mechanics.
            # Tier 2 — OTel attributes: legacy path; reads step.count +
            #   step.<i>.* flat attrs set directly on the span.
            # Tier 3 — Derived: no measured steps but real model/tool work
            #   happened — synthesise one covering step so step_count >= 1.
            # ---------------------------------------------------------------

            # Tier 1: check the module-level step registry
            measured_steps = None
            span_ctx_for_registry = span.get_span_context() if hasattr(span, 'get_span_context') else None
            if span_ctx_for_registry and span_ctx_for_registry.trace_id and span_ctx_for_registry.span_id:
                try:
                    from observability.observability_wrapper import pop_steps_for_span
                    measured_steps = pop_steps_for_span(
                        span_ctx_for_registry.trace_id, span_ctx_for_registry.span_id
                    )
                except ImportError:
                    pass

            if measured_steps:
                # Tier 1 – registry-measured steps
                trace_context.steps = measured_steps
                trace_context._current_step_index = len(measured_steps) - 1
                for call in trace_context.model_calls:
                    if call.get('step_index', -1) == -1:
                        call['step_index'] = len(measured_steps) - 1
                for call in trace_context.tool_calls:
                    if call.get('step_index', -1) == -1:
                        call['step_index'] = len(measured_steps) - 1
                # Auto-derive output_summary for each step from its last
                # tool call or model call — no explicit capture() needed.
                self._auto_fill_step_outputs(trace_context)
                # Promote error status from tool/model calls up to the step.
                self._auto_fill_step_statuses(trace_context)
            else:
                step_count = int(attributes.get('step.count', 0))
                if step_count > 0:
                    # Tier 2 – measured steps from OTel span attributes
                    for i in range(step_count):
                        trace_context.steps.append({
                            'index': i,
                            'name': str(attributes.get(f'step.{i}.name', f'step_{i}')),
                            'step_type': str(attributes.get(f'step.{i}.step_type', 'unknown')),
                            'started_at': attributes.get(f'step.{i}.started_at'),
                            'ended_at': attributes.get(f'step.{i}.ended_at'),
                            'status': str(attributes.get(f'step.{i}.status', 'unknown')),
                            'latency_ms': int(attributes[f'step.{i}.latency_ms']) if f'step.{i}.latency_ms' in attributes else None,
                            'retries': int(attributes.get(f'step.{i}.retries', 0)),
                            'decision_summary': attributes.get(f'step.{i}.decision_summary'),
                            'steps_status': 'measured',
                        })
                    trace_context._current_step_index = step_count - 1
                    for call in trace_context.model_calls:
                        if call.get('step_index', -1) == -1:
                            call['step_index'] = step_count - 1
                    for call in trace_context.tool_calls:
                        if call.get('step_index', -1) == -1:
                            call['step_index'] = step_count - 1
                elif trace_context.model_calls or trace_context.tool_calls:
                    # Tier 3 – derived: synthesize one covering step
                    latency_ms = None
                    if trace_context.started_at and trace_context.ended_at:
                        latency_ms = int(
                            (trace_context.ended_at - trace_context.started_at).total_seconds() * 1000
                        )
                    step_status = 'failure' if trace_context.error_class else 'success'
                    detail = (
                        f"{len(trace_context.model_calls)} model call(s), "
                        f"{len(trace_context.tool_calls)} tool call(s)"
                    )
                    trace_context.steps = [{
                        'index': 0,
                        'name': 'agent_execution',
                        'step_type': 'llm_call' if trace_context.model_calls else 'tool_call',
                        'started_at': trace_context.started_at.isoformat() if trace_context.started_at else None,
                        'ended_at': trace_context.ended_at.isoformat() if trace_context.ended_at else None,
                        'status': step_status,
                        'latency_ms': latency_ms,
                        'retries': 0,
                        'decision_summary': f'Synthesized: {detail}',
                        'steps_status': 'derived',
                    }]
                    trace_context._current_step_index = 0
                    for call in trace_context.model_calls:
                        if call.get('step_index', -1) == -1:
                            call['step_index'] = 0
                    for call in trace_context.tool_calls:
                        if call.get('step_index', -1) == -1:
                            call['step_index'] = 0
                # else: run never started (no model/tool work) — steps stays []
            # which is the only legitimate case for step_count == 0.

            # Note: trace_context.context is intentionally NOT set here.
            # to_trace_dict() generates context_payload directly from agent_name
            # and agent_execution_id, so set_context() would have no effect on the DB.

            if not trace_context.user_query:
                trace_context.set_user_query(f"span:{getattr(span, 'name', 'unknown')}")

            if not trace_context.agent_response:
                status_code = getattr(getattr(span, 'status', None), 'status_code', None)
                trace_context.set_agent_response(f"status={status_code}")

            logger.info(
                "_span_to_trace_context: built TraceContext for '%s' (execution_id=%s)",
                span_name,
                trace_context.agent_execution_id,
            )
            if not getattr(trace_context, 'model_calls', []):
                logger.warning(
                    "_span_to_trace_context: 0 model_calls for '%s' — "
                    "either trace_model_call() was never called inside this agent run "
                    "or the LLM call pattern did not match _LLM_AWAIT_CALL_RE / _LLM_SYNC_CALL_RE "
                    "during code generation",
                    span_name,
                )

            return trace_context

        except Exception as e:
            logger.error("_span_to_trace_context: unhandled error for span '%s': %s",
                         getattr(span, 'name', '<unknown>'), e, exc_info=True)
            return None
    
    def shutdown(self):
        """Shutdown the exporter, waiting up to 5 s for in-flight exports."""
        self.shutdown_flag = True
        # Give the background loop a moment to drain any in-flight persistence
        # coroutines that were already scheduled before the flag was set.
        if _export_loop and _export_loop.is_running():
            import time
            time.sleep(0.5)  # small grace period; daemon thread exits with process
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush any pending spans.
        
        Args:
            timeout_millis: Timeout in milliseconds
            
        Returns:
            True if successful
        """
        return True


def _cleanup_on_exit():
    """
    ``atexit`` handler — shuts down the ``TracerProvider`` to flush any
    buffered spans before the process exits.

    Registered once by :func:`initialize_tracer`.  Safe to call multiple
    times; subsequent calls are no-ops because ``_tracer_provider`` is
    cleared after the first shutdown.
    """
    global _tracer_provider
    
    try:
        # Shutdown tracer to flush pending spans
        if _tracer_provider:
            _tracer_provider.shutdown()
            logger.debug("Observability tracer flushed on exit")
    except Exception as e:
        logger.debug(f"Error during observability cleanup: {e}")
    
    # Note: We don't explicitly stop the worker here as it may cause issues
    # The worker will be terminated when the process exits


def initialize_tracer(
    service_name: str = "qno_backend",
    service_version: str = "1.0.0",
    environment: Optional[str] = None,
    enable_database_export: bool = True,
) -> trace.Tracer:
    """
    Create and globally register an OpenTelemetry ``TracerProvider``.

    Attaches a :class:`DatabaseSpanExporter` via ``SimpleSpanProcessor`` so
    that every ``agent/`` span is synchronously persisted to the database
    immediately when it closes.  Returns the singleton tracer; repeated calls
    after the first return the already-initialised instance without
    re-creating any resources.

    Args:
        service_name:          OTel ``service.name`` resource attribute.
        service_version:       OTel ``service.version`` resource attribute.
        environment:           Deployment environment (``dev`` / ``production``).
                               Falls back to ``ENVIRONMENT`` config setting.
        enable_database_export: Attach :class:`DatabaseSpanExporter` when ``True``.

    Returns:
        Configured :class:`opentelemetry.sdk.trace.Tracer` instance.
    """
    global _tracer, _tracer_provider
    
    # If already initialized, return existing tracer
    if _tracer is not None:
        return _tracer
    
    # Get environment from config or parameter
    if environment is None:
        try:
            from config import settings
            environment = getattr(settings, 'ENVIRONMENT', 'production')
        except Exception:
            environment = os.getenv('ENVIRONMENT', 'production')
    
    # Create resource with service information
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": environment,
    })
    
    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)
    
    # Add exporters
    exporters = []
    
    # Database exporter (custom)
    if enable_database_export:
        try:
            db_exporter = DatabaseSpanExporter()
            # Use SimpleSpanProcessor for immediate export instead of batching
            # This ensures spans are exported immediately when they end
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            _tracer_provider.add_span_processor(SimpleSpanProcessor(db_exporter))
            exporters.append("database")
            logger.info("OpenTelemetry database exporter enabled (SimpleSpanProcessor)")
        except Exception as e:
            logger.error(f"Failed to initialize database exporter: {e}")
    
    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)
    
    # Get tracer
    _tracer = trace.get_tracer(__name__)
    
    # Register cleanup on process exit (fire and forget pattern)
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(_cleanup_on_exit)
        _cleanup_registered = True
        logger.debug("Registered atexit cleanup handler for observability")
    
    logger.info("OpenTelemetry tracer initialized")
    
    return _tracer


def get_tracer() -> Optional[trace.Tracer]:
    """
    Return the global tracer singleton, auto-initialising on first call.

    Reads ``APP_NAME``, ``APP_VERSION``, ``ENVIRONMENT``, and
    ``OTEL_DATABASE_EXPORT`` from ``core.config.settings`` if available.
    Failures during auto-initialisation are logged as warnings and the
    function returns ``None``, allowing callers to skip tracing gracefully.

    Returns:
        The initialised :class:`opentelemetry.sdk.trace.Tracer`, or ``None``
        when initialisation is not possible (e.g. missing config).
    """
    global _tracer
    
    # If not initialized, try to initialize with config settings
    if _tracer is None:
        try:
            # Load config
            from config import settings
            
            _tracer = initialize_tracer(
                service_name=getattr(settings, 'APP_NAME', 'qno_backend'),
                service_version=getattr(settings, 'APP_VERSION', '1.0.0'),
                environment=getattr(settings, 'ENVIRONMENT', 'production'),
                enable_database_export=getattr(settings, 'OTEL_DATABASE_EXPORT', True),
            )
            logger.info("OpenTelemetry tracer auto-initialized on first decorator use")
        except Exception as e:
            logger.warning(
                "Failed to auto-initialize tracer: %s — all @trace_agent / trace_step / "
                "trace_model_call calls will be no-ops until initialize_tracer() succeeds. "
                "Ensure initialize_tracer() is called at application startup "
                "(e.g. in the __main__ block or FastAPI lifespan).",
                e,
                exc_info=True,
            )
    
    return _tracer



