# Planetary Comparative Analysis Assistant

A professional agent that compares Earth and Jupiter using authoritative knowledge base documents. It retrieves relevant scientific facts from Azure AI Search and generates a structured, cited summary using Azure OpenAI, with observability and content safety guardrails.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy `.env.example` to `.env` and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

**Direct execution:**
```
python code/agent.py
```

**As a FastAPI server:**
```
uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME`
- `AGENT_ID`
- `PROJECT_NAME`
- `PROJECT_ID`
- `USE_KEY_VAULT`
- `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE`

**General**
- `ENVIRONMENT`

**Azure Key Vault**
- `KEY_VAULT_URI`
- `AZURE_USE_DEFAULT_CREDENTIAL`

**Azure Authentication**
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

**LLM Configuration**
- `MODEL_PROVIDER`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`

**API Keys / Secrets**
- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `AZURE_CONTENT_SAFETY_KEY`

**Service Endpoints**
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_CONTENT_SAFETY_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`

**Observability DB**
- `OBS_DATABASE_TYPE`
- `OBS_AZURE_SQL_SERVER`
- `OBS_AZURE_SQL_DATABASE`
- `OBS_AZURE_SQL_PORT`
- `OBS_AZURE_SQL_USERNAME`
- `OBS_AZURE_SQL_PASSWORD`
- `OBS_AZURE_SQL_SCHEMA`

**Agent-Specific**
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`
- `SERVICE_NAME`
- `SERVICE_VERSION`
- `VALIDATION_CONFIG_PATH`
- `LLM_MODELS`
- `VERSION`

**Content Safety (Optional)**
- `CONTENT_SAFETY_ENABLED`
- `CONTENT_SAFETY_SEVERITY_THRESHOLD`

See `.env.example` for details and required/optional status.

---

## API Endpoints

### **GET** `/health`
- **Description:** Health check endpoint.
- **Response:**
  ```
  {
    "status": "ok"
  }
  ```

---

### **POST** `/analyze`
- **Description:** Runs a planetary comparative analysis of Earth and Jupiter using the internal knowledge base. No request body required.
- **Request body:** _None_
- **Response:**
  ```
  {
    "success": true|false,
    "result": "string (structured summary, present if success)",
    "error": "string (present if failed, else null)",
    "tips": "string (optional, present if error)"
  }
  ```

---

### **GET** `/status`
- **Description:** Returns agent status and configuration.
- **Response:**
  ```
  {
    "agent_name": "string",
    "project_name": "string",
    "version": "string",
    "llm_model": "string",
    "rag_enabled": true,
    "selected_documents": ["Jupiter.pdf", "Earth.pdf"]
  }
  ```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t planetary-comparative-analysis-assistant -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name planetary-comparative-analysis-assistant planetary-comparative-analysis-assistant
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs planetary-comparative-analysis-assistant
```

### 7. Stop the container:
```
docker stop planetary-comparative-analysis-assistant
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Planetary Comparative Analysis Assistant** — Scientific, cited planetary comparisons at your fingertips.