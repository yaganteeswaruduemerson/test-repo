import pytest
import asyncio
from unittest.mock import Mock, patch
from agent import CustomTranslatorFileAgentAgent

class TestCustomTranslatorFileAgentAgent:
    
    @pytest.fixture
    def agent(self):
        return CustomTranslatorFileAgentAgent()
    
    @pytest.mark.asyncio
    async def test_process_message(self, agent):
        # Test basic message processing
        response = await agent.process_message("Hello!")
        assert response is not None
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        # Test error handling
        with patch.object(agent, 'llm_client') as mock_llm:
            mock_llm.generate.side_effect = Exception("API Error")
            response = await agent.process_message("Test")
            assert "error" in response.lower() or "sorry" in response.lower()
    
    def test_configuration_loading(self, agent):
        # Test configuration loading
        assert agent.config is not None
        assert hasattr(agent.config, 'llm_provider')