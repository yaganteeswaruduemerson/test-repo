"""LLM Manager module for managing multiple LLM providers."""

# Import initialization to register all providers
from .initialize import initialize_llm_providers

# Export main classes
from .manager import LLMManager
from .base.base_llm_provider import BaseLLMProvider, ToolExecutionError, LLMResponse
from .registry import LLMProviderRegistry, LLMProviderFactory

# Export provider classes
from .openai import OpenAIProvider
from .github import GitHubProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider
from .azure import AzureProvider

__all__ = [
    "LLMManager",
    "BaseLLMProvider",
    "ToolExecutionError",
    "LLMResponse",
    "LLMProviderRegistry",
    "LLMProviderFactory",
    "OpenAIProvider",
    "GitHubProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "AzureProvider",
    "initialize_llm_providers",
]

