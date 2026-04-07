"""
SQLAlchemy declarative base for the observability database.

All observability ORM models (ObservabilityTrace, EvaluationRecord, …)
inherit from this base so they live in a **separate** SQLAlchemy metadata
registry — completely isolated from the main application database.
"""

from sqlalchemy.orm import DeclarativeBase
from observability.config import settings


def get_obs_table_schema() -> str | None:
    """
    Get the schema name for observability tables based on database type.

    Returns:
        Schema name string for Azure SQL, None for other database types.
    """
    if settings.OBS_DATABASE_TYPE.lower() == "azure_sql":
        return settings.OBS_AZURE_SQL_SCHEMA or "dbo"
    return None


class ObsBase(DeclarativeBase):
    """Base class for all observability ORM models."""

    def __init_subclass__(cls, **kwargs):
        """Automatically inject schema into __table_args__ for Azure SQL models."""
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "__tablename__"):
            schema = get_obs_table_schema()
            if schema:
                existing_args = getattr(cls, "__table_args__", None)

                if existing_args is None:
                    cls.__table_args__ = {"schema": schema}
                elif isinstance(existing_args, tuple):
                    if len(existing_args) > 0 and isinstance(existing_args[-1], dict):
                        args_dict = existing_args[-1].copy()
                        if "schema" not in args_dict:
                            args_dict["schema"] = schema
                        cls.__table_args__ = existing_args[:-1] + (args_dict,)
                    else:
                        cls.__table_args__ = existing_args + ({"schema": schema},)
                elif isinstance(existing_args, dict):
                    if "schema" not in existing_args:
                        existing_args = existing_args.copy()
                        existing_args["schema"] = schema
                        cls.__table_args__ = existing_args
