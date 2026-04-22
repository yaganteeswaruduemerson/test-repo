

FROM python:3.11-slim

# Install system dependencies for pip packages (incl. pyodbc, FastAPI, SQLAlchemy, cryptography, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
        unixodbc-dev \
        curl \
        libssl-dev \
        libffi-dev \
        libsasl2-dev \
        libpq-dev \
        python3-dev \
        gnupg2 \
        apt-transport-https \
        ca-certificates \
        wget \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY ./code/ /app/

# Symlink for validation config path compatibility
RUN mkdir -p /app/code && ln -s /app/validation_config.json /app/code/validation_config.json

# Add non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup --no-create-home appuser
USER appuser

ENV PYTHONUNBUFFERED=1  PYTHONDONTWRITEBYTECODE=1  AGENT_NAME=""

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Detect FastAPI app variable 'app' in agent.py, so use uvicorn
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]

