"""Standalone config for the observability package (generated agent bundle).

Reads observability/evaluation configuration from environment variables so the
module works without the main FastAPI application's core.config dependency.
"""
import os


class _ObsSettings:  # noqa: D101
    # Observability database
    OBS_DATABASE_TYPE: str = os.getenv("OBS_DATABASE_TYPE", "sqlite")
    OBS_AZURE_SQL_SERVER: str = os.getenv("OBS_AZURE_SQL_SERVER", "")
    OBS_AZURE_SQL_DATABASE: str = os.getenv("OBS_AZURE_SQL_DATABASE", "")
    OBS_AZURE_SQL_SCHEMA: str = os.getenv("OBS_AZURE_SQL_SCHEMA", "dbo")
    OBS_AZURE_SQL_USERNAME: str = os.getenv("OBS_AZURE_SQL_USERNAME", "")
    OBS_AZURE_SQL_PASSWORD: str = os.getenv("OBS_AZURE_SQL_PASSWORD", "")
    OBS_AZURE_SQL_DRIVER: str = os.getenv(
        "OBS_AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server"
    )
    OBS_SQLITE_PATH: str = os.getenv("OBS_SQLITE_PATH", "observability.db")

    # Azure AI Foundry evaluation
    AZURE_AI_FOUNDRY_ENDPOINT: str = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT", "")
    EVAL_MODEL_DEPLOYMENT_NAME: str = os.getenv("EVAL_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    EVAL_POLL_INTERVAL_SECONDS: int = int(os.getenv("EVAL_POLL_INTERVAL_SECONDS", "60"))
    EVAL_BATCH_SIZE: int = int(os.getenv("EVAL_BATCH_SIZE", "10"))

    # OpenTelemetry / service identity
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "agent")
    SERVICE_VERSION: str = os.getenv("SERVICE_VERSION", "1.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")

    # Azure auth (used by engine.py for managed-identity / connection-string)
    AZURE_CLIENT_ID: str = os.getenv("AZURE_CLIENT_ID", "")
    AZURE_TENANT_ID: str = os.getenv("AZURE_TENANT_ID", "")
    AZURE_CLIENT_SECRET: str = os.getenv("AZURE_CLIENT_SECRET", "")


settings = _ObsSettings()
