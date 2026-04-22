"""
Service layer for ObservabilityTrace CRUD operations.

All methods are async and accept an injected AsyncSessionType (uses
AsyncSessionWrapper wrapping a sync session in a thread pool for Azure SQL).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, func, desc, asc

from observability.database.engine import ObsAsyncSessionType as AsyncSessionType
from observability.database.models import ObservabilityTrace, ObservabilityExecutionStatus

logger = logging.getLogger(__name__)


@dataclass
class TraceFilters:
    """Optional filter bag for list_and_count – keeps the method under 13 params."""
    agent_name: Optional[str] = None
    agent_names: Optional[List[str]] = field(default=None)  # multi-agent filter (OR semantics)
    status: Optional[str] = None
    environment: Optional[str] = None
    session_id: Optional[UUID] = None
    is_evaluated: Optional[bool] = None
    started_at_from: Optional[datetime] = None
    started_at_to: Optional[datetime] = None
    project_name: Optional[str] = None


class ObservabilityTraceService:
    """
    Async data-access service for the ``qo_observability_trace`` table.

    All methods are static and accept an injected
    :class:`~database.engine.AsyncSessionType`; there is no shared mutable
    state, so no instantiation is required (call methods as
    ``ObservabilityTraceService.get_by_id(...)``).

    Public interface:

    * :meth:`get_by_id` — fetch one trace row by primary key.
    * :meth:`list_and_count` — paginated, filtered, sorted query returning
      ``(items, total_count)`` for API list endpoints.

    Filtering is expressed through the :class:`TraceFilters` parameter bag;
    unset fields (``None``) are simply ignored so callers only specify the
    dimensions they care about.
    """

    # ------------------------------------------------------------------
    # Read one
    # ------------------------------------------------------------------

    @staticmethod
    async def get_by_id(
        agent_execution_id: UUID,
        session: AsyncSessionType,
    ) -> Optional[ObservabilityTrace]:
        """Return a single trace or None."""
        stmt = select(ObservabilityTrace).where(
            ObservabilityTrace.agent_execution_id == agent_execution_id
        )
        result = await session.execute(stmt)
        return result.scalars().first()

    # ------------------------------------------------------------------
    # List / search
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where_clauses(f: TraceFilters) -> list:
        """
        Translate a :class:`TraceFilters` instance into SQLAlchemy WHERE clauses.

        Only non-``None`` filter fields produce clauses; ``None`` fields are
        left out so callers can safely leave unused filters at their default
        ``None`` values.

        Args:
            f: Populated :class:`TraceFilters` dataclass.

        Returns:
            List of SQLAlchemy binary expressions suitable for
            ``stmt.where(*clauses)``; may be empty when no filters are active.
        """
        clauses = []
        if f.agent_name is not None:
            clauses.append(ObservabilityTrace.agent_name == f.agent_name)
        # agent_names (multi-select) uses IN — takes precedence over single agent_name when both are set
        if f.agent_names is not None and len(f.agent_names) > 0:
            clauses.append(ObservabilityTrace.agent_name.in_(f.agent_names))
        if f.status is not None:
            clauses.append(
                ObservabilityTrace.status == ObservabilityExecutionStatus(f.status)
            )
        if f.environment is not None:
            clauses.append(ObservabilityTrace.environment == f.environment)
        if f.session_id is not None:
            clauses.append(ObservabilityTrace.session_id == f.session_id)
        if f.is_evaluated is not None:
            clauses.append(ObservabilityTrace.is_evaluated == f.is_evaluated)
        if f.started_at_from is not None:
            clauses.append(ObservabilityTrace.started_at >= f.started_at_from)
        if f.started_at_to is not None:
            clauses.append(ObservabilityTrace.started_at <= f.started_at_to)
        if f.project_name is not None:
            clauses.append(ObservabilityTrace.project_name == f.project_name)
        return clauses

    @staticmethod
    async def list_and_count(
        session: AsyncSessionType,
        *,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "started_at",
        sort_order: str = "desc",
        filters: Optional[TraceFilters] = None,
    ) -> Tuple[List[ObservabilityTrace], int]:
        """
        Return a paginated ``(items, total_count)`` tuple matching the given filters.

        Executes two SQL queries: a ``COUNT(*)`` for the total and a
        ``SELECT … LIMIT/OFFSET`` for the current page.  Sort direction is
        clamped to the ``allowed_sort_columns`` allow-list to prevent column
        injection.

        Args:
            session:    Active async database session.
            page:       1-based page number (default 1).
            page_size:  Rows per page (default 20).
            sort_by:    Column name to sort by; must be one of ``started_at``,
                        ``created_at``, ``updated_at``, ``agent_name``,
                        ``status``, ``total_latency_ms`` (defaults to
                        ``started_at`` for unknown values).
            sort_order: ``"desc"`` (default) or ``"asc"``.
            filters:    Optional :class:`TraceFilters` instance; ``None`` means
                        no filtering — all rows are returned.

        Returns:
            A 2-tuple ``(items, total_count)`` where *items* is the
            :class:`list` of :class:`~database.models.ObservabilityTrace` ORM
            objects for the requested page and *total_count* is the full
            matching row count (before pagination).
        """
        f = filters or TraceFilters()

        allowed_sort_columns: Dict[str, Any] = {
            "started_at": ObservabilityTrace.started_at,
            "updated_at": ObservabilityTrace.updated_at,
            "agent_name": ObservabilityTrace.agent_name,
            "status": ObservabilityTrace.status,
            "total_latency_ms": ObservabilityTrace.total_latency_ms,
        }
        sort_col = allowed_sort_columns.get(sort_by, ObservabilityTrace.started_at)
        where_clauses = ObservabilityTraceService._build_where_clauses(f)

        # Count query
        count_stmt = select(func.count()).select_from(ObservabilityTrace)
        if where_clauses:
            count_stmt = count_stmt.where(*where_clauses)
        count_result = await session.execute(count_stmt)
        total = count_result.scalars().first() or 0

        # Data query
        order_fn = desc if sort_order.lower() == "desc" else asc
        offset = (page - 1) * page_size
        data_stmt = (
            select(ObservabilityTrace)
            .order_by(order_fn(sort_col))
            .offset(offset)
            .limit(page_size)
        )
        if where_clauses:
            data_stmt = data_stmt.where(*where_clauses)
        data_result = await session.execute(data_stmt)
        items = data_result.scalars().all()

        return list(items), int(total)
