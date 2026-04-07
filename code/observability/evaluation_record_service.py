"""
Service layer for EvaluationRecord CRUD operations.

All methods are async and accept an injected AsyncSessionType (works with
both the real AsyncSession for PostgreSQL/SQLite and AsyncSessionWrapper for
Azure SQL).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, func, desc, asc

from observability.database.engine import ObsAsyncSessionType as AsyncSessionType
from observability.database.models import EvaluationRecord

logger = logging.getLogger(__name__)


@dataclass
class EvalFilters:
    """Optional filter bag for list_and_count – keeps the method under 13 params."""
    agent_execution_id: Optional[UUID] = None
    evaluated_at_from: Optional[datetime] = None
    evaluated_at_to: Optional[datetime] = None


class EvaluationRecordService:
    """
    Async data-access service for the ``qo_evaluation_record`` table.

    All methods are static and accept an injected
    :class:`~database.engine.AsyncSessionType`; no instantiation is required.

    Public interface:

    * :meth:`get_by_id` — fetch one evaluation record by its primary key
      (``evaluation_id`` UUID).
    * :meth:`get_by_execution_id` — look up the evaluation record associated
      with a particular agent execution (``agent_execution_id`` UUID).
    * :meth:`list_and_count` — paginated, filtered, sorted query returning
      ``(items, total_count)`` for API list endpoints.

    Filtering is expressed through the :class:`EvalFilters` parameter bag;
    unset fields (``None``) are ignored so callers only specify the dimensions
    they care about.
    """

    # ------------------------------------------------------------------
    # Read one
    # ------------------------------------------------------------------

    @staticmethod
    async def get_by_id(
        evaluation_id: UUID,
        session: AsyncSessionType,
    ) -> Optional[EvaluationRecord]:
        """Return a single evaluation record or None."""
        stmt = select(EvaluationRecord).where(
            EvaluationRecord.evaluation_id == evaluation_id
        )
        result = await session.execute(stmt)
        return result.scalars().first()

    @staticmethod
    async def get_by_execution_id(
        agent_execution_id: UUID,
        session: AsyncSessionType,
    ) -> Optional[EvaluationRecord]:
        """Return the evaluation record for a given agent_execution_id or None."""
        stmt = select(EvaluationRecord).where(
            EvaluationRecord.agent_execution_id == agent_execution_id
        )
        result = await session.execute(stmt)
        return result.scalars().first()

    # ------------------------------------------------------------------
    # List / search
    # ------------------------------------------------------------------

    @staticmethod
    async def list_and_count(
        session: AsyncSessionType,
        *,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "evaluated_at",
        sort_order: str = "desc",
        filters: Optional[EvalFilters] = None,
    ) -> Tuple[List[EvaluationRecord], int]:
        """
        Return a paginated ``(items, total_count)`` tuple matching the given filters.

        Executes two SQL queries: a ``COUNT(*)`` for the total and a
        ``SELECT … LIMIT/OFFSET`` for the current page.  Sort column is
        validated against an allow-list to prevent column injection.

        Args:
            session:    Active async database session.
            page:       1-based page number (default 1).
            page_size:  Rows per page (default 20).
            sort_by:    Column to sort by; must be one of
                        ``evaluated_at`` (defaults to ``evaluated_at``
                        for unknown values).
            sort_order: ``"desc"`` (default) or ``"asc"``.
            filters:    Optional :class:`EvalFilters`; ``None`` returns all rows.

        Returns:
            A 2-tuple ``(items, total_count)`` where *items* is the
            :class:`list` of :class:`~database.models.EvaluationRecord` ORM
            objects for the requested page and *total_count* is the full
            matching row count before pagination.
        """
        f = filters or EvalFilters()

        allowed_sort_columns: Dict[str, Any] = {
            "evaluated_at": EvaluationRecord.evaluated_at,
        }
        sort_col = allowed_sort_columns.get(
            sort_by, EvaluationRecord.evaluated_at
        )

        where_clauses = []
        if f.agent_execution_id is not None:
            where_clauses.append(EvaluationRecord.agent_execution_id == f.agent_execution_id)
        if f.evaluated_at_from is not None:
            where_clauses.append(
                EvaluationRecord.evaluated_at >= f.evaluated_at_from
            )
        if f.evaluated_at_to is not None:
            where_clauses.append(
                EvaluationRecord.evaluated_at <= f.evaluated_at_to
            )

        # Count query
        count_stmt = select(func.count()).select_from(EvaluationRecord)
        if where_clauses:
            count_stmt = count_stmt.where(*where_clauses)
        count_result = await session.execute(count_stmt)
        total = count_result.scalars().first() or 0

        # Data query
        order_fn = desc if sort_order.lower() == "desc" else asc
        offset = (page - 1) * page_size
        data_stmt = (
            select(EvaluationRecord)
            .order_by(order_fn(sort_col))
            .offset(offset)
            .limit(page_size)
        )
        if where_clauses:
            data_stmt = data_stmt.where(*where_clauses)
        data_result = await session.execute(data_stmt)
        items = data_result.scalars().all()

        return list(items), int(total)
