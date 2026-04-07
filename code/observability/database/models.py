"""
SQLAlchemy ORM models for the observability database.

Contains all tables related to agent execution traces and evaluation records.
These models use ObsBase (observability/database/base.py) so they are registered
in a completely separate SQLAlchemy metadata from the main application database.
"""

from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from uuid import uuid4

from observability.database.base import ObsBase

# ---------------------------------------------------------------------------
# Database-agnostic type helpers
# ---------------------------------------------------------------------------

def _get_db_type() -> str:
    try:
        from observability.config import settings
        return settings.OBS_DATABASE_TYPE.lower()
    except Exception:
        return "sqlite"


_db_type = _get_db_type()

if _db_type == "postgresql":
    from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSONB
    _UUIDType = PostgresUUID(as_uuid=True)
    _JSONBType = JSONB
elif _db_type in ("azure_sql", "mssql"):
    from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
    _UUIDType = UNIQUEIDENTIFIER
    _JSONBType = JSON
else:
    _UUIDType = String(36)
    _JSONBType = JSON


def get_uuid_type():
    return _UUIDType


def get_jsonb_type():
    return _JSONBType


TABLE_PREFIX = ""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ObservabilityExecutionStatus(Enum):
    """Execution status for observability traces."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ObservabilityTrace(ObsBase):
    """
    Observability Traces — authoritative ingestion store for agent execution telemetry.
    One row per agent execution run.
    """
    __tablename__ = f"{TABLE_PREFIX}observability_trace"

    # ── Identity ───────────────────────────────────────────────────────────
    agent_execution_id = Column(get_uuid_type(), primary_key=True, default=uuid4)
    session_id         = Column(get_uuid_type(), nullable=False, index=True)

    # ── Agent metadata ─────────────────────────────────────────────────────
    agent_name    = Column(String(255), nullable=False, index=True)
    agent_version = Column(String(100), nullable=True)
    environment   = Column(String(100), nullable=True, index=True)

    # ── Timing ────────────────────────────────────────────────────────────
    started_at       = Column(DateTime(timezone=True), nullable=False, index=True)
    ended_at         = Column(DateTime(timezone=True), nullable=True)
    total_latency_ms = Column(BigInteger, nullable=True)
    queue_time_ms    = Column(BigInteger, nullable=True)

    # ── Status & error ────────────────────────────────────────────────────
    status              = Column(SQLEnum(ObservabilityExecutionStatus), nullable=False, index=True)
    error_class         = Column(String(255), nullable=True, index=True)
    error_message       = Column(Text, nullable=True)
    error_stack_summary = Column(Text, nullable=True)

    # ── Metrics ───────────────────────────────────────────────────────────
    tokens = Column(get_jsonb_type(), nullable=True)
    cost   = Column(get_jsonb_type(), nullable=True)

    # ── Execution data ────────────────────────────────────────────────────
    steps       = Column(get_jsonb_type(), nullable=True)
    model_calls = Column(get_jsonb_type(), nullable=True)
    tool_calls  = Column(get_jsonb_type(), nullable=True)

    # ── Domain context ────────────────────────────────────────────────────
    user_query     = Column(Text, nullable=True)
    agent_response = Column(Text, nullable=True)
    is_evaluated   = Column(Boolean, nullable=False, default=False)

    # ── Audit ─────────────────────────────────────────────────────────────
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    evaluation_records = relationship("EvaluationRecord", back_populates="trace", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_obs_trace_session_started", "session_id", "started_at"),
        Index("idx_obs_trace_agent_status",    "agent_name", "status"),
        Index("idx_obs_trace_env_started",     "environment", "started_at"),
        Index("idx_obs_trace_status_started",  "status", "started_at"),
    )


class EvaluationRecord(ObsBase):
    """
    Evaluation Records — one row per evaluated agent execution trace.
    """
    __tablename__ = f"{TABLE_PREFIX}evaluation_record"

    # ── Identity ───────────────────────────────────────────────────────────
    evaluation_id      = Column(get_uuid_type(), primary_key=True, default=uuid4)
    agent_execution_id = Column(
        get_uuid_type(),
        ForeignKey(f"{TABLE_PREFIX}observability_trace.agent_execution_id"),
        nullable=False, index=True, unique=True,
    )

    # ── Evaluation timing ─────────────────────────────────────────────────
    evaluated_at = Column(DateTime(timezone=True), nullable=False)


    # ── Scores (F2 & F5) ──────────────────────────────────────────────────
    scores       = Column(get_jsonb_type(), nullable=True)
    level_scores = Column(get_jsonb_type(), nullable=True)

    # ── Goal & workflow (F7) ──────────────────────────────────────────────
    goal_summary               = Column(get_jsonb_type(), nullable=True)
    workflow_deviation_summary = Column(get_jsonb_type(), nullable=True)
    failure_points             = Column(get_jsonb_type(), nullable=True)

    # ── Remediation ───────────────────────────────────────────────────────────
    remediation_hints = Column(get_jsonb_type(), nullable=True)

    # ── Evaluator ─────────────────────────────────────────────────────────
    evaluator_metadata = Column(get_jsonb_type(), nullable=True)

    # ── Context ───────────────────────────────────────────────────────────
    persona = Column(String(255), nullable=True)

    # Relationships
    trace = relationship("ObservabilityTrace", back_populates="evaluation_records")

    __table_args__ = (
        Index("idx_eval_agent_execution",  "agent_execution_id"),
        Index("idx_eval_evaluated",        "evaluated_at"),
    )
