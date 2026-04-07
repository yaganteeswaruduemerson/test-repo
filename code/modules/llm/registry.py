"""LLM provider registry and factory."""

import logging
from typing import Dict, Type, Optional, List, Any
from .base.base_llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMProviderRegistry:
    """Registry for all available LLM providers."""
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    
    @classmethod
    def register(cls, provider_class: Type[BaseLLMProvider], provider_name: str = None):
        """Register a provider class.
        
        Args:
            provider_class: The provider class to register
            provider_name: Optional name override (defaults to class provider_name attribute)
        """
        name = provider_name or provider_class.provider_name
        cls._providers[name.lower()] = provider_class
        logger.debug(f"Registered LLM provider: {name}")
    
    @classmethod
    def get_provider_class(cls, provider_name: str) -> Optional[Type[BaseLLMProvider]]:
        """Get provider class by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider class or None if not found
        """
        return cls._providers.get(provider_name.lower())
    
    @classmethod
    def list_providers(cls) -> Dict[str, Dict[str, Any]]:
        """List all registered provider types.
        
        Returns:
            Dictionary mapping provider names to their metadata
        """
        result = {}
        for name, provider_class in cls._providers.items():
            result[name] = {
                "provider_name": provider_class.provider_name,
            }
        return result
    
    @classmethod
    def get_provider_info(cls, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider type.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider information dictionary or None if not found
        """
        provider_class = cls.get_provider_class(provider_name)
        if not provider_class:
            return None
        
        return {
            "provider_name": provider_class.provider_name,
        }
    
    @classmethod
    def load_from_config(cls, providers_config: List[Dict[str, Any]]) -> int:
        """Load and register providers from configuration.
        
        Args:
            providers_config: List of provider configurations, each with:
                - module_path: Full module path (e.g., "modules.llm_manager.openai")
                - class_name: Class name (e.g., "OpenAIProvider")
                - enabled: Boolean to enable/disable the provider (optional, defaults to True)
                - provider_name: Optional name override
                
        Returns:
            Number of successfully registered providers
        """
        registered_count = 0
        
        for provider_config in providers_config:
            # Skip if disabled
            if not provider_config.get("enabled", True):
                logger.debug(f"Skipping disabled provider: {provider_config.get('class_name')}")
                continue
            
            module_path = provider_config.get("module_path")
            class_name = provider_config.get("class_name")
            provider_name = provider_config.get("provider_name")
            
            if not module_path or not class_name:
                logger.warning(f"Invalid provider config: missing module_path or class_name")
                continue
            
            try:
                # Import the module
                module = __import__(module_path, fromlist=[class_name])
                provider_class = getattr(module, class_name)
                
                # Verify it's a BaseLLMProvider subclass
                if not issubclass(provider_class, BaseLLMProvider):
                    logger.warning(f"{class_name} is not a subclass of BaseLLMProvider")
                    continue
                
                # Register the provider
                cls.register(provider_class, provider_name)
                registered_count += 1
                logger.debug(f"Registered provider: {provider_class.provider_name}")
                
            except ImportError as e:
                logger.warning(f"Failed to import {module_path}.{class_name}: {str(e)}. Provider will not be available.")
            except AttributeError as e:
                logger.warning(f"Class {class_name} not found in {module_path}: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to register {module_path}.{class_name}: {str(e)}")
        
        logger.info(f"Loaded {registered_count} providers from configuration")
        return registered_count


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    @staticmethod
    def _load_models_config(provider_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load models configuration from settings for a provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            List of model configuration dictionaries or None if not found
        """
        try:
            from config import settings
            
            if not hasattr(settings, 'LLM_PROVIDERS') or not settings.LLM_PROVIDERS:
                return None
            
            # Search for matching provider
            for provider_config in settings.LLM_PROVIDERS:
                if provider_config.get('provider_name', '').lower() == provider_name.lower():
                    return provider_config.get('models', [])
            
            return None
        except Exception as e:
            logger.warning(f"Failed to load models config for {provider_name}: {str(e)}")
            return None
    
    @staticmethod
    def create_provider(
        provider_name: str,
        api_key: str,
        models_config: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[BaseLLMProvider]:
        """Create a provider instance.
        
        Args:
            provider_name: Name of the provider
            api_key: API key for the provider
            models_config: Optional models configuration. If not provided, will be loaded from settings.
            
        Returns:
            Provider instance or None if provider not found
        """
        provider_class = LLMProviderRegistry.get_provider_class(provider_name)
        if not provider_class:
            return None
        
        # Load models config from settings if not provided
        if models_config is None:
            models_config = LLMProviderFactory._load_models_config(provider_name)
        
        return provider_class(api_key=api_key, models_config=models_config)

