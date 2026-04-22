"""
SQLAlchemy engine factory and session management for the observability database.

This module provides a completely separate database connection pool for
AgentOps / observability data, independent of the main application database.

Supports Azure SQL Server only. The async session interface wraps a sync
session running in a dedicated thread pool.

Configuration is driven by OBS_* settings in core/config.py:
  OBS_DATABASE_TYPE (must be "azure_sql")
  OBS_AZURE_SQL_SERVER, OBS_AZURE_SQL_DATABASE, …

Public API:
  - get_obs_async_session()  — async generator (FastAPI Depends-compatible)
  - obs_health_check()       — async ping
  - close_obs_engine()       — dispose engine on shutdown
  - ObsAsyncSessionType      — Union type alias
"""

import asyncio
import concurrent.futures
import os
from typing import Optional, AsyncGenerator, Union, Any
from urllib.parse import quote_plus

import pyodbc
from sqlalchemy import create_engine, Engine, text, event
from sqlalchemy.engine import Result
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession  # For type alias only
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
    retry_if_exception,
)

from config import settings
from logging import getLogger as get_logger

logger = get_logger("observability.database.engine")

# ---------------------------------------------------------------------------
# Transient Azure SQL error codes (same set as main engine)
# ---------------------------------------------------------------------------
_AZURE_SQL_TRANSIENT_ERRORS = frozenset({
    40613, 40197, 40501, 49918, 49919, 49920, 4060, 10929, 10928, 10060,
    18456, 233, -1,
})


def _is_transient_azure_sql_error(exc: BaseException) -> bool:
    import pyodbc as _pyodbc
    if isinstance(exc, _pyodbc.Error):
        msg = str(exc)
        for code in _AZURE_SQL_TRANSIENT_ERRORS:
            if str(code) in msg:
                return True
        sqlstate = exc.args[0] if exc.args else None
        if sqlstate in ("08S01", "08001", "HYT00", "HY000"):
            return True
    from sqlalchemy.exc import OperationalError, DBAPIError
    if isinstance(exc, (OperationalError, DBAPIError)):
        orig = getattr(exc, "orig", None)
        if orig is not None:
            return _is_transient_azure_sql_error(orig)
        msg = str(exc)
        for code in _AZURE_SQL_TRANSIENT_ERRORS:
            if str(code) in msg:
                return True
    return False


# ---------------------------------------------------------------------------
# Dedicated thread pool for observability DB async wrapper
# ---------------------------------------------------------------------------
_obs_db_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=20, thread_name_prefix="obs-db-pool"
)

# ---------------------------------------------------------------------------
# Pool parameters
# ---------------------------------------------------------------------------
_POOL_SIZE     = 10
_MAX_OVERFLOW  = 10
_POOL_RECYCLE  = 1800
_QUERY_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Sync engine and session
# ---------------------------------------------------------------------------
_obs_engine: Optional[Engine] = None
_obs_SessionLocal: Optional[sessionmaker] = None


# ---------------------------------------------------------------------------
# ODBC / URL helpers
# ---------------------------------------------------------------------------

def _detect_odbc_driver() -> str:
    drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
    if not drivers:
        available = ", ".join(pyodbc.drivers()) if pyodbc.drivers() else "none"
        raise RuntimeError(
            f"No SQL Server ODBC driver found. Available: {available}"
        )
    preferred = [d for d in drivers if "17" in d or "18" in d]
    return preferred[0] if preferred else drivers[-1]


def _escape_odbc_value(value: str) -> str:
    return "{" + (value or "").replace("}", "}}") + "}"


def _normalize_sql_server(server: str, port: str) -> str:
    server_value = (server or "").strip()
    if server_value.lower().startswith("tcp:"):
        server_value = server_value[4:]
    if "," in server_value:
        return server_value
    return f"{server_value},{port}"


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def get_obs_database_url() -> str:
    """Build synchronous SQLAlchemy URL for the observability database (Azure SQL only)."""
    db_type = settings.OBS_DATABASE_TYPE.lower()
    
    if db_type != "azure_sql":
        raise ValueError(
            f"Unsupported OBS_DATABASE_TYPE: {db_type}. Only 'azure_sql' is supported."
        )
    
    try:
        driver = _detect_odbc_driver()
        username = (settings.OBS_AZURE_SQL_USERNAME or "").strip()
        password = settings.OBS_AZURE_SQL_PASSWORD or ""
        server   = (settings.OBS_AZURE_SQL_SERVER   or "").strip()
        database = (settings.OBS_AZURE_SQL_DATABASE or "").strip()
        port     = str(settings.OBS_AZURE_SQL_PORT  or "1433").strip()

        if not all((username, password, server, database)):
            raise RuntimeError(
                "Missing OBS Azure SQL config. Required: OBS_AZURE_SQL_USERNAME, "
                "OBS_AZURE_SQL_PASSWORD, OBS_AZURE_SQL_SERVER, OBS_AZURE_SQL_DATABASE."
            )

        server_host = _normalize_sql_server(server, port)
        trust_cert  = getattr(settings, "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE", "no")
        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER=tcp:{server_host};"
            f"DATABASE={_escape_odbc_value(database)};"
            f"UID={_escape_odbc_value(username)};"
            f"PWD={_escape_odbc_value(password)};"
            "Encrypt=yes;"
            f"TrustServerCertificate={trust_cert};"
            f"Connection Timeout={_QUERY_TIMEOUT};"
        )
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}"
    except Exception as e:
        logger.error(f"Failed to build observability Azure SQL URL: {e}")
        raise


# ---------------------------------------------------------------------------
# Engine creation
# ---------------------------------------------------------------------------

def create_obs_database_engine() -> Engine:
    """Create (or return cached) sync SQLAlchemy engine for observability DB."""
    global _obs_engine

    if _obs_engine is None:
        database_url = get_obs_database_url()

        pool_kwargs: dict = {
            "pool_size": _POOL_SIZE,
            "max_overflow": _MAX_OVERFLOW,
            "pool_pre_ping": True,
            "pool_recycle": _POOL_RECYCLE,
            "connect_args": {
                "timeout": _QUERY_TIMEOUT,
                "autocommit": False,
                "attrs_before": {
                    pyodbc.SQL_ATTR_LOGIN_TIMEOUT: _QUERY_TIMEOUT,
                },
            },
            "fast_executemany": True,
        }

        try:
            _obs_engine = create_engine(database_url, **pool_kwargs)

            @event.listens_for(_obs_engine, "before_cursor_execute", retval=True)
            def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                return statement, parameters

            with _obs_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Observability DB connection test successful")

        except Exception as e:
            logger.error(f"Failed to create observability DB engine (server={settings.OBS_AZURE_SQL_SERVER}): {e}")
            raise

    return _obs_engine


def get_obs_session_factory() -> sessionmaker:
    """Get or create sync session factory for observability DB."""
    global _obs_SessionLocal
    if _obs_SessionLocal is None:
        engine = create_obs_database_engine()
        _obs_SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
        )
    return _obs_SessionLocal


def get_obs_session() -> Session:
    """Get a new sync session for observability DB."""
    return get_obs_session_factory()()


# ---------------------------------------------------------------------------
# AsyncSessionWrapper
# ---------------------------------------------------------------------------

class _AsyncResultWrapper:
    def __init__(self, rows: list, rowcount: int = 0):
        self._rows = rows
        self.rowcount = rowcount

    def scalars(self) -> "_AsyncScalarsWrapper":
        return _AsyncScalarsWrapper(self._rows)

    def unique(self) -> "_AsyncResultWrapper":
        seen: set = set()
        unique_rows = []
        for row in self._rows:
            rid = id(row)
            if rid not in seen:
                seen.add(rid)
                unique_rows.append(row)
        return _AsyncResultWrapper(unique_rows, self.rowcount)

    def first(self) -> Any:
        return self._rows[0] if self._rows else None

    def all(self) -> list:
        return self._rows

    def scalar_one_or_none(self) -> Any:
        if not self._rows:
            return None
        if len(self._rows) == 1:
            return self._rows[0]
        raise ValueError("Multiple rows found when one or none was required")


class _AsyncScalarsWrapper:
    def __init__(self, rows: list):
        self._rows = rows

    def first(self) -> Any:
        return self._rows[0] if self._rows else None

    def all(self) -> list:
        return self._rows


class ObsAsyncSessionWrapper:
    """
    Async facade over a sync Session for the observability database.
    Mirrors AsyncSessionWrapper in database/engine.py.
    """

    def __init__(self, sync_session: Session):
        self._session = sync_session

    async def _run_in_db_pool(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_obs_db_executor, fn, *args)

    async def execute(self, statement: Any, **kwargs: Any) -> _AsyncResultWrapper:
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=0.5, max=10),
            retry=retry_if_exception(_is_transient_azure_sql_error),
            reraise=True,
        )
        def _run() -> tuple:
            try:
                result: Result = self._session.execute(statement, **kwargs)
                rowcount = getattr(result, "rowcount", 0)
                try:
                    rows = list(result.scalars().all())
                except Exception:
                    rows = []
                return rows, rowcount
            except Exception:
                try:
                    self._session.rollback()
                except Exception:
                    pass
                raise

        rows, rowcount = await self._run_in_db_pool(_run)
        return _AsyncResultWrapper(rows, rowcount)

    async def commit(self) -> None:
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=0.5, max=10),
            retry=retry_if_exception(_is_transient_azure_sql_error),
            reraise=True,
        )
        def _commit():
            self._session.commit()

        await self._run_in_db_pool(_commit)

    async def rollback(self) -> None:
        await self._run_in_db_pool(self._session.rollback)

    async def refresh(self, instance: Any, attribute_names: Optional[list] = None) -> None:
        if attribute_names is not None:
            await self._run_in_db_pool(self._session.refresh, instance, attribute_names)
        else:
            await self._run_in_db_pool(self._session.refresh, instance)

    def add(self, instance: Any) -> None:
        self._session.add(instance)

    def add_all(self, instances: list) -> None:
        self._session.add_all(instances)

    async def flush(self) -> None:
        await self._run_in_db_pool(self._session.flush)

    async def delete(self, instance: Any) -> None:
        await self._run_in_db_pool(self._session.delete, instance)

    async def close(self) -> None:
        await self._run_in_db_pool(self._session.close)


# ---------------------------------------------------------------------------
# Session type alias
# ---------------------------------------------------------------------------
ObsAsyncSessionType = Union[AsyncSession, ObsAsyncSessionWrapper]


# ---------------------------------------------------------------------------
# Session creation with retry
# ---------------------------------------------------------------------------

def _is_obs_session_retriable_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return (
        "session factory not initialized" in s
        or "connection pool" in s
        or "too many connections" in s
        or "all pooled connections were in use" in s
        or "login timeout" in s
        or "timeout expired" in s
    )


@retry(
    stop=stop_after_delay(900),
    wait=wait_fixed(30),
    retry=retry_if_exception(_is_obs_session_retriable_error),
    reraise=True,
)
def _create_obs_sync_session_with_retry() -> Session:
    return get_obs_session_factory()()


async def get_obs_async_session() -> AsyncGenerator[ObsAsyncSessionType, None]:
    """
    Async generator yielding a session for the observability database.

    Yields ObsAsyncSessionWrapper wrapping a sync session running in thread pool.

    Usage (FastAPI Depends):
        async def endpoint(session = Depends(get_obs_async_session)): ...
    """
    sync_session = _create_obs_sync_session_with_retry()
    wrapper = ObsAsyncSessionWrapper(sync_session)
    try:
        yield wrapper
    except Exception:
        await wrapper.rollback()
        raise
    finally:
        await wrapper.rollback()
        await wrapper.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def obs_health_check() -> bool:
    """Ping the observability database."""
    try:
        engine = create_obs_database_engine()

        def _ping() -> bool:
            with engine.connect() as conn:
                return conn.execute(text("SELECT 1")).scalar() == 1

        return await asyncio.to_thread(_ping)
    except Exception as e:
        logger.error(f"Observability DB health check failed: {e}")
        return False


async def close_obs_engine() -> None:
    """Dispose observability engine and shut down its thread pool (call on app shutdown)."""
    global _obs_engine, _obs_SessionLocal
    if _obs_engine is not None:
        _obs_engine.dispose()
        _obs_engine = None
        _obs_SessionLocal = None
        logger.info("Observability DB engine closed")
    _obs_db_executor.shutdown(wait=False)
    logger.info("Observability DB thread pool shut down")
