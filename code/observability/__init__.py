"""
Observability package for the NGCC Quote-to-Order backend.

Provides end-to-end execution tracing via OpenTelemetry with a custom database
exporter.  All trace data is written to the ``qo_observability_trace`` Azure SQL
table and is structured into four columns — ``steps``, ``model_calls``,
``tool_calls``, and ``events`` — each storing a JSON array.

Public API (import from this package or the individual modules directly):

- :func:`~observability.observability_wrapper.trace_agent`      — agent-level decorator
- :func:`~observability.observability_wrapper.trace_step_sync`  — sync step context manager
- :func:`~observability.observability_wrapper.trace_step`       — async step context manager
- :func:`~observability.observability_wrapper.trace_model_call` — record one LLM call
- :func:`~observability.observability_wrapper.trace_tool_call`  — record one tool call
- :func:`~observability.observability_wrapper.set_trace_context_ids` — stamp session/correlation IDs
- :class:`~observability.observability_service.TraceContext`    — in-memory trace accumulator
- :class:`~observability.observability_service.ObservabilityService` — async DB persistence
- :class:`~observability.instrumentation.DatabaseSpanExporter`  — OTel → DB bridge
- :class:`~observability.evaluation_background_service.EvaluationBackgroundService`
                                                               — Foundry eval pipeline
- :func:`~observability.evaluation_background_service.start_evaluation_worker`
                                                               — start DB-polling asyncio worker at app startup
- :func:`~observability.evaluation_background_service.stop_evaluation_worker`
                                                               — graceful shutdown at app shutdown
"""
