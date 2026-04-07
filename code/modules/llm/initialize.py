"""Initialize and register all LLM provider types."""

import logging
from .registry import LLMProviderRegistry
from .openai import OpenAIProvider
from .github import GitHubProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider
from .azure import AzureProvider

logger = logging.getLogger(__name__)


def initialize_llm_providers():
    """Register all available LLM provider types.
    
    Returns:
        Number of registered providers
    """
    # Clear existing providers to allow re-initialization
    LLMProviderRegistry._providers.clear()
    
    # Register all providers
    providers = [
        OpenAIProvider,
        GitHubProvider,
        AnthropicProvider,
        GoogleProvider,
        AzureProvider,
    ]
    
    registered_count = 0
    for provider_class in providers:
        try:
            LLMProviderRegistry.register(provider_class)
            registered_count += 1
            logger.debug(f"Registered LLM provider: {provider_class.provider_name}")
        except Exception as e:
            logger.error(f"Failed to register {provider_class.__name__}: {str(e)}")
    
    logger.info(f"Successfully registered {registered_count} LLM provider types")
    return registered_count


# Auto-initialize when module is imported
_providers_registered = initialize_llm_providers()

