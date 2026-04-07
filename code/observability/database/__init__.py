"""
Observability database package.

Contains all database-related code for the observability subsystem:
  - base.py   — ObsBase declarative base (separate SQLAlchemy metadata)
  - models.py — ObservabilityTrace, EvaluationRecord ORM models
  - engine.py — Engine factory, session management, async wrappers
"""

from observability.database.base import ObsBase, get_obs_table_schema
from observability.database.engine import (
    create_obs_database_engine,
    get_obs_session_factory,
    get_obs_session,
    get_obs_async_session,
    obs_health_check,
    close_obs_engine,
    ObsAsyncSessionWrapper,
    ObsAsyncSessionType,
)
from observability.database.models import (
    ObservabilityTrace,
    ObservabilityExecutionStatus,
    EvaluationRecord,
)

__all__ = [
    "ObsBase",
    "get_obs_table_schema",
    "create_obs_database_engine",
    "get_obs_session_factory",
    "get_obs_session",
    "get_obs_async_session",
    "obs_health_check",
    "close_obs_engine",
    "ObsAsyncSessionWrapper",
    "ObsAsyncSessionType",
    "ObservabilityTrace",
    "ObservabilityExecutionStatus",
    "EvaluationRecord",
]
