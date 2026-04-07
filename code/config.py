
# config.py

import os
from dotenv import load_dotenv
from typing import Optional, Dict

# Load environment variables from .env if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Centralized configuration management for the Custom Translator File Agent.
    Handles environment variable loading, API key management, LLM config, domain settings,
    validation, error handling, and default values/fallbacks.
    """

    # Required configuration keys for domain logic
    REQUIRED_KEYS = [
        "AZURE_BLOB_CONNECTION_STRING",
        "AZURE_BLOB_CONTAINER_NAME",
        "AZURE_TRANSLATOR_ENDPOINT",
        "AZURE_TRANSLATOR_KEY",
        "OPENAI_API_KEY"
    ]

    # LLM configuration defaults
    LLM_CONFIG_DEFAULTS = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are a professional agent responsible for translating files stored in Azure Blob Storage. "
            "When provided with a filename, perform the following steps: 1. Validate that the file exists in the specified Azure Blob container. "
            "2. Generate a secure SAS URL for the file. 3. Submit the SAS URL to the Azure Translator service for translation. "
            "4. If the translation service does not return a status 200, retry the request for up to 100 seconds. "
            "5. Provide clear, concise, and professional updates on the process. 6. If the file is not found or translation fails after retries, "
            "return an informative error message. Always ensure sensitive information is not exposed in responses."
        ),
        "user_prompt_template": (
            "Please provide the filename of the document you wish to translate. "
            "The agent will validate the file, generate a secure access link, and process the translation. "
            "You will receive updates on the status and results."
        ),
        "few_shot_examples": [
            "File 'report2023.docx' found. SAS URL generated. Translation in progress... Translation completed successfully.",
            "Error: The file 'missingfile.pdf' was not found in the specified blob container. Please check the filename and try again."
        ]
    }

    # Domain-specific settings
    DOMAIN = "general"
    AGENT_NAME = "Custom Translator File Agent"
    PERSONALITY = "professional"

    @classmethod
    def get(cls, key: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
        """Get a configuration value from environment variables."""
        value = os.getenv(key)
        if value is None:
            if required:
                raise ConfigError(f"Missing required configuration: {key}")
            return default
        return value

    @classmethod
    def validate(cls):
        """Validate that all required configuration keys are present."""
        missing = [k for k in cls.REQUIRED_KEYS if not os.getenv(k)]
        if missing:
            raise ConfigError(f"Missing required configuration keys: {missing}")

    @classmethod
    def get_llm_config(cls) -> Dict:
        """Return LLM configuration, using environment overrides if present."""
        config = cls.LLM_CONFIG_DEFAULTS.copy()
        # Allow environment variable overrides for model, temperature, max_tokens
        config["provider"] = os.getenv("LLM_PROVIDER", config["provider"])
        config["model"] = os.getenv("LLM_MODEL", config["model"])
        config["temperature"] = float(os.getenv("LLM_TEMPERATURE", config["temperature"]))
        config["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", config["max_tokens"]))
        config["system_prompt"] = os.getenv("LLM_SYSTEM_PROMPT", config["system_prompt"])
        return config

    @classmethod
    def get_domain_settings(cls) -> Dict:
        """Return domain-specific settings."""
        return {
            "domain": cls.DOMAIN,
            "agent_name": cls.AGENT_NAME,
            "personality": cls.PERSONALITY
        }

    @classmethod
    def get_azure_blob_settings(cls) -> Dict:
        """Return Azure Blob Storage settings."""
        return {
            "connection_string": cls.get("AZURE_BLOB_CONNECTION_STRING"),
            "container_name": cls.get("AZURE_BLOB_CONTAINER_NAME")
        }

    @classmethod
    def get_azure_translator_settings(cls) -> Dict:
        """Return Azure Translator API settings."""
        return {
            "endpoint": cls.get("AZURE_TRANSLATOR_ENDPOINT"),
            "key": cls.get("AZURE_TRANSLATOR_KEY")
        }

    @classmethod
    def get_openai_api_key(cls) -> str:
        """Return OpenAI API key, raise error if missing."""
        key = cls.get("OPENAI_API_KEY")
        if not key:
            raise ConfigError("OPENAI_API_KEY is required for LLM operations.")
        return key

    @classmethod
    def get_all_settings(cls) -> Dict:
        """Return all relevant settings for debugging or diagnostics."""
        try:
            cls.validate()
        except Exception as e:
            raise ConfigError(f"Configuration validation failed: {e}")
        return {
            "llm_config": cls.get_llm_config(),
            "domain_settings": cls.get_domain_settings(),
            "azure_blob": cls.get_azure_blob_settings(),
            "azure_translator": cls.get_azure_translator_settings(),
            "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY"))
        }

# Error handling for missing API keys or configuration
try:
    Config.validate()
except ConfigError as ce:
    # Comment out the next line if you want to suppress error on import
    # print(f"Configuration error: {ce}")
    raise

# Default/fallback values are handled in get() and get_llm_config()
