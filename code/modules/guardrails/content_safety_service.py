"""Runtime Azure Content Safety service used inside generated artifacts."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CONTENT_SAFETY_TEXT_LIMIT = 10000


class ContentSafetyService:
    """Lightweight Content Safety wrapper for generated code runtime."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self.enabled = self._to_bool(
            self._config.get("content_safety_enabled", os.getenv("CONTENT_SAFETY_ENABLED", "false")),
            False,
        )
        self.endpoint = self._config.get(
            "content_safety_endpoint",
            os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT", ""),
        )
        self.key = self._config.get(
            "content_safety_key",
            os.getenv("AZURE_CONTENT_SAFETY_KEY", ""),
        )
        self.severity_threshold = self._to_int(
            self._config.get("content_safety_severity_threshold", os.getenv("CONTENT_SAFETY_SEVERITY_THRESHOLD", 2)),
            2,
        )

        self._client = None
        if self.enabled:
            try:
                from azure.ai.contentsafety import ContentSafetyClient
                from azure.core.credentials import AzureKeyCredential

                self._client = ContentSafetyClient(
                    self.endpoint,
                    AzureKeyCredential(self.key),
                )
            except Exception as error:
                logger.warning("Content Safety disabled at runtime (client init failed): %s", error)
                self.enabled = False

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def _to_int(value: Any, default: int = 2) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def analyze_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze text and return category severities including Jailbreak when available."""
        if not self.enabled or self._client is None:
            return None

        if not text or not text.strip():
            return {
                "categories": {},
                "max_severity": 0,
                "safe": True,
                "truncated": False,
            }

        truncated = False
        if len(text) > CONTENT_SAFETY_TEXT_LIMIT:
            text = text[:CONTENT_SAFETY_TEXT_LIMIT]
            truncated = True

        try:
            from azure.ai.contentsafety.models import AnalyzeTextOptions

            response = self._client.analyze_text(AnalyzeTextOptions(text=text))
            categories: Dict[str, int] = {}
            max_severity = 0
            for category in response.categories_analysis:
                category_name = str(category.category)
                severity = int(category.severity)
                categories[category_name] = severity
                max_severity = max(max_severity, severity)

            return {
                "categories": categories,
                "max_severity": max_severity,
                "safe": max_severity < self.severity_threshold,
                "truncated": truncated,
            }
        except Exception as error:
            logger.warning("Content Safety analysis failed at runtime: %s", error)
            return None

    def is_safe(self, text: str) -> Tuple[bool, Optional[str]]:
        """Return (is_safe, reason) based on configured severity threshold."""
        if not self.enabled:
            return True, None

        analysis = self.analyze_text(text)
        if analysis is None:
            return True, None

        if analysis.get("safe", True):
            return True, None

        categories = analysis.get("categories", {})
        blocked = [
            f"{cat}({sev})"
            for cat, sev in categories.items()
            if sev >= self.severity_threshold
        ]
        return False, (
            "Content violates runtime safety policy. "
            f"Flagged categories: {', '.join(blocked)}. "
            f"Threshold: {self.severity_threshold}."
        )


_content_safety_service: Optional[ContentSafetyService] = None


def get_content_safety_service(config: Optional[Dict[str, Any]] = None) -> ContentSafetyService:
    """Get or create singleton runtime content safety service."""
    global _content_safety_service
    if config is not None:
        return ContentSafetyService(config=config)
    if _content_safety_service is None:
        _content_safety_service = ContentSafetyService()
    return _content_safety_service
