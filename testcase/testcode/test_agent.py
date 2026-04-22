# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")

# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")


import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from agent import (
    app,
    AnalyzeResponse,
    ChunkRetriever,
    LLMService,
    ErrorHandler,
    Logger,
    sanitize_llm_output,
    FALLBACK_RESPONSE,
    SELECTED_DOCUMENT_TITLES,
    SYSTEM_PROMPT,
    OUTPUT_FORMAT
)

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

@pytest.mark.functional
def test_analyze_endpoint_returns_structured_summary_on_success(client):
    """
    Functional: /analyze returns structured summary when all dependencies succeed.
    """
    # Patch retrieval and LLM to simulate success
    chunks = [
        "Physical Dimensions: Jupiter's diameter is 86,881 miles. Earth is 7,917 miles.",
        "Scale Comparison: Over 1,300 Earths fit inside Jupiter.",
        "Orbital Distances: Jupiter is 483 million miles from the Sun. Earth is 93 million miles."
    ]
    llm_response = (
        "Physical Dimensions:\n- Jupiter: 86,881 miles (139,820 km)\n- Earth: 7,917 miles (12,742 km)\n\n"
        "Scale Comparison:\n- Over 1,300 Earths fit inside Jupiter.\n\n"
        "Orbital Distances:\n- Jupiter: 483 million miles\n- Earth: 93 million miles"
    )
    with patch("agent.ChunkRetriever.retrieve_chunks", new=AsyncMock(return_value=chunks)), \
         patch("agent.LLMService.generate_response", new=AsyncMock(return_value=llm_response)):
        response = client.post("/analyze")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"] is not None
        assert "Physical Dimensions" in data["result"]
        assert "Scale Comparison" in data["result"]
        assert "Orbital Distances" in data["result"]
        assert data["error"] is None
        assert data["tips"] is None

@pytest.mark.functional
def test_analyze_endpoint_returns_fallback_when_measurements_missing(client):
    """
    Functional: /analyze returns fallback error when measurements are missing.
    """
    chunks = ["Some unrelated text without measurements."]
    fallback = FALLBACK_RESPONSE
    with patch("agent.ChunkRetriever.retrieve_chunks", new=AsyncMock(return_value=chunks)), \
         patch("agent.LLMService.generate_response", new=AsyncMock(return_value=fallback)):
        response = client.post("/analyze")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["result"] is None
        assert "requested measurements or comparisons are not available" in data["error"].lower()
        assert "requested measurements may not be present" in (data["tips"] or "").lower()

@pytest.mark.unit
@pytest.mark.asyncio
async def test_chunkretriever_retrieve_chunks_returns_correct_chunks_with_odata_filter():
    """
    Unit: ChunkRetriever.retrieve_chunks applies correct OData filter and returns correct chunks.
    """
    # Prepare mocks
    mock_embedding = MagicMock()
    mock_embedding.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_openai_client = MagicMock()
    mock_openai_client.embeddings.create = AsyncMock(return_value=mock_embedding)

    # Patch openai.AsyncAzureOpenAI and AzureAISearchClient
    with patch("agent.openai.AsyncAzureOpenAI", return_value=mock_openai_client), \
         patch("agent.AzureAISearchClient") as mock_search_client_cls:
        mock_search_client = MagicMock()
        # Simulate Azure Search returning chunks from selected docs only
        mock_results = [
            {"chunk": "Chunk from Jupiter", "title": "Jupiter.pdf"},
            {"chunk": "Chunk from Earth", "title": "Earth.pdf"},
        ]
        mock_search_client._get_client.return_value = MagicMock()
        mock_search_client._get_client.return_value.search.return_value = mock_results
        mock_search_client_cls.return_value = mock_search_client

        retriever = ChunkRetriever()
        result = await retriever.retrieve_chunks(
            query="planetary comparison",
            filter_titles=["Jupiter.pdf", "Earth.pdf"],
            top_k=5
        )
        assert isinstance(result, list)
        assert all(chunk in ["Chunk from Jupiter", "Chunk from Earth"] for chunk in result)
        # Check OData filter construction
        odata_filter = " or ".join([f"title eq '{t}'" for t in ["Jupiter.pdf", "Earth.pdf"]])
        # The filter is passed to search; check call args
        search_call = mock_search_client._get_client.return_value.search
        called_kwargs = search_call.call_args.kwargs
        assert called_kwargs["filter"] == odata_filter

@pytest.mark.unit
@pytest.mark.asyncio
async def test_llmservice_generate_response_returns_llm_output_with_correct_prompt_formatting():
    """
    Unit: LLMService.generate_response sends correct prompts and returns LLM output.
    """
    # Patch openai.AsyncAzureOpenAI
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Physical Dimensions:\nScale Comparison:\nOrbital Distances:"))]
    mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    with patch("agent.openai.AsyncAzureOpenAI", return_value=mock_client):
        llm = LLMService()
        context_chunks = ["Earth: 7,917 miles", "Jupiter: 86,881 miles"]
        params = {"temperature": 0.7, "max_tokens": 2000}
        result = await llm.generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=SYSTEM_PROMPT,
            context_chunks=context_chunks,
            parameters=params
        )
        assert result is not None
        # System prompt includes OUTPUT_FORMAT
        system_message = f"{SYSTEM_PROMPT}\n\nOutput Format: {OUTPUT_FORMAT}"
        user_message = f"{SYSTEM_PROMPT}\n\nContext:\n" + "\n\n".join(context_chunks)
        # The actual call to openai should have these messages
        call_args = mock_client.chat.completions.create.call_args.kwargs
        assert any(system_message in m["content"] for m in call_args["messages"])
        assert any("Context:" in m["content"] for m in call_args["messages"])
        # Output contains expected sections
        assert "Physical Dimensions" in result
        assert "Scale Comparison" in result
        assert "Orbital Distances" in result

@pytest.mark.unit
def test_errorhandler_handle_error_returns_correct_fallback_messages():
    """
    Unit: ErrorHandler.handle_error maps error codes to correct fallback/generic messages.
    """
    logger = Logger()
    handler = ErrorHandler(logger)
    # DOC_NOT_FOUND
    msg1 = handler.handle_error("DOC_NOT_FOUND")
    assert msg1 == FALLBACK_RESPONSE
    # MEASUREMENT_MISSING
    msg2 = handler.handle_error("MEASUREMENT_MISSING")
    assert msg2 == FALLBACK_RESPONSE
    # Unknown code
    msg3 = handler.handle_error("UNKNOWN_CODE")
    assert "unexpected error" in msg3.lower()

@pytest.mark.functional
def test_status_endpoint_returns_agent_configuration(client):
    """
    Functional: /status endpoint returns correct agent metadata and selected documents.
    """
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "agent_name" in data
    assert data["agent_name"] == getattr(__import__("agent").Config, "AGENT_NAME", "Planetary Comparative Analysis Assistant")
    assert "project_name" in data
    assert "version" in data
    assert "llm_model" in data
    assert data["rag_enabled"] is True
    assert "selected_documents" in data
    assert "Jupiter.pdf" in data["selected_documents"]
    assert "Earth.pdf" in data["selected_documents"]

@pytest.mark.unit
def test_sanitizer_utility_strips_markdown_fences_and_wrappers():
    """
    Unit: sanitize_llm_output removes markdown fences, wrappers, and sign-offs.
    """
    raw = (
        "```markdown\n"
        "Here is the answer:\n"
        "Physical Dimensions: ...\n"
        "Scale Comparison: ...\n"
        "Orbital Distances: ...\n"
        "Let me know if you need more help!\n"
        "```"
    )
    cleaned = sanitize_llm_output(raw, content_type="text")
    assert "```" not in cleaned
    assert not cleaned.startswith("Here is the answer")
    assert not cleaned.endswith("Let me know if you need more help!")
    # Blank lines collapsed
    assert "\n\n\n" not in cleaned

@pytest.mark.functional
def test_api_returns_422_on_malformed_json(client):
    """
    Functional: API returns 422 and helpful tips on malformed JSON.
    """
    # /analyze expects no body, but simulate malformed JSON
    response = client.post("/analyze", data="{bad json")
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert "Malformed JSON" in data["error"]
    assert "Check your JSON formatting" in data["tips"]

@pytest.mark.unit
def test_logger_log_handles_all_event_types_and_does_not_interrupt_flow():
    """
    Unit: Logger.log handles info, error, debug and does not raise even if logging fails.
    """
    logger = Logger()
    # Patch the underlying logger to raise exception
    logger._logger = MagicMock()
    logger._logger.error.side_effect = Exception("fail")
    logger._logger.info.side_effect = Exception("fail")
    logger._logger.debug.side_effect = Exception("fail")
    # Should not raise
    logger.log("error", "msg", {"foo": "bar"})
    logger.log("info", "msg", {"foo": "bar"})
    logger.log("debug", "msg", {"foo": "bar"})