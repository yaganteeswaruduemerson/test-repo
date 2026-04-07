"""LLM Manager for creating and using LLM providers."""

import logging
from typing import Dict, Any, Callable, Optional, List, Union
from .registry import LLMProviderFactory, LLMProviderRegistry
from .base.base_llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMManager:
    """Manager for LLM providers that delegates to registered providers."""
    
    def __init__(self, provider_name: str, api_key: str, models_config: Optional[List[Dict[str, Any]]] = None):
        """Initialize LLM manager with a specific provider.
        
        Args:
            provider_name: Name of the LLM provider (e.g., 'openai', 'github', 'anthropic')
            api_key: API key for the provider
            models_config: Optional models configuration. If not provided, will be loaded from settings.
            
        Raises:
            ValueError: If provider is not found or API key is invalid
        """
        self.provider_name = provider_name.lower()
        self.provider: Optional[BaseLLMProvider] = LLMProviderFactory.create_provider(
            provider_name=self.provider_name,
            api_key=api_key,
            models_config=models_config
        )
        
        if self.provider is None:
            available = list(LLMProviderRegistry.list_providers().keys())
            raise ValueError(
                f"LLM provider '{provider_name}' not found. "
                f"Available providers: {', '.join(available)}"
            )
    
    def invoke(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: Optional[float] = 0.1,
        model: Optional[str] = None,
        tools: bool = False,
        funcs: Optional[List[Callable]] = None,
        tool_choice: Union[str, Dict[str, Any]] = 'auto',
        parse: bool = False,
        parser: Optional[dict] = None,
        standard_output: bool = True,
        max_tool_calls: Optional[int] = 3,
        image_path: Optional[str] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        Invoke the LLM model with support for iterative tool calling and vision.
        
        This method delegates to the underlying provider's invoke method.
        
        Args:
            user_prompt: The user's prompt
            system_prompt: The system prompt/instructions
            temperature: Sampling temperature (0.0 to 1.0)
            model: The model to use (provider-specific)
            tools: Whether to enable tools/function calling
            funcs: List of callable functions available to the model
            tool_choice: Tool choice strategy ('auto', 'required', 'none')
            parse: Whether to parse the response with a custom parser
            parser: Parser configuration dictionary
            standard_output: Whether to return standardized output format
            max_tool_calls: Maximum number of tool call iterations (default: 3)
            image_path: Optional path to local image file for vision capabilities
            
        Returns:
            Either raw response object or standardized dictionary with content, tool_calls, and token_usage
        """
        return self.provider.invoke(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            model=model,
            tools=tools,
            funcs=funcs,
            tool_choice=tool_choice,
            parse=parse,
            parser=parser,
            standard_output=standard_output,
            max_tool_calls=max_tool_calls,
            image_path=image_path
        )
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get properties for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary or None if model not found
        """
        return self.provider.get_model_info(model_name)
    
    def list_models(self) -> List[str]:
        """
        List all available model names for this provider.
        
        Returns:
            List of model names
        """
        return self.provider.list_models()
    
    def get_model_property(self, model_name: str, property_name: str) -> Optional[Any]:
        """
        Get specific property value for a model.
        
        Args:
            model_name: Name of the model
            property_name: Name of the property to retrieve
            
        Returns:
            Property value or None if model or property not found
        """
        return self.provider.get_model_property(model_name, property_name)
    
    def has_model(self, model_name: str) -> bool:
        """
        Check if model exists in configuration.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model exists, False otherwise
        """
        return self.provider.has_model(model_name)
    
    def get_default_model(self) -> Optional[str]:
        """
        Get default model name for this provider.
        
        Returns:
            Default model name or None if no models configured
        """
        return self.provider.get_default_model()

