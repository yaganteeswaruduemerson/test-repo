"""Runtime guardrails service used by generated artifacts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .content_safety_service import get_content_safety_service

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_safe: bool
    violations: List[str]
    details: Dict[str, Any]


class PIIDetector:
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "phone": r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    @classmethod
    def detect(cls, text: str) -> Dict[str, List[str]]:
        detected: Dict[str, List[str]] = {}
        for pii_type, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if matches and isinstance(matches[0], tuple):
                    matches = ["-".join(m) for m in matches]
                detected[pii_type] = matches
        return detected


class CredentialScanner:
    PATTERNS = {
        "api_key": r"(?i)(api[_-]?key\s*[=:]\s*[\"']?)([A-Za-z0-9_\-]{20,})",
        "secret_key": r"(?i)(secret[_-]?key\s*[=:]\s*[\"']?)([A-Za-z0-9_\-]{20,})",
        "password": r"(?i)(password\s*[=:]\s*[\"']?)([^\"'\s]{8,})",
        "connection_string": r"(?i)(connection[_-]?string\s*[=:]\s*[\"']?)([^\"']{30,})",
        "private_key": r"-----BEGIN (?:RSA|OPENSSH|DSA|EC|PGP) PRIVATE KEY-----",
    }

    @classmethod
    def scan(cls, code: str) -> Dict[str, List[str]]:
        detected: Dict[str, List[str]] = {}
        for cred_type, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, code, re.MULTILINE)
            if matches:
                if isinstance(matches[0], tuple):
                    detected[cred_type] = [f"{m[0]}***REDACTED***" for m in matches]
                else:
                    detected[cred_type] = ["***REDACTED***" for _ in matches]
        return detected


class ToxicCodeDetector:
    DANGEROUS_PATTERNS = {
        "file_deletion": r"\b(?:os\.remove|os\.unlink|shutil\.rmtree|Path\.unlink)\s*\(",
        "system_commands": r"\b(?:os\.system|subprocess\.(?:call|run|Popen))\s*\(",
        "code_execution": r"\b(?:eval|exec|compile|__import__)\s*\(",
        "infinite_loop": r"\bwhile\s+True\s*:(?!\s*#)",
    }

    @classmethod
    def detect(cls, code: str) -> Dict[str, List[str]]:
        detected: Dict[str, List[str]] = {}
        for pattern_type, pattern in cls.DANGEROUS_PATTERNS.items():
            matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE)
            if matches:
                detected[pattern_type] = matches if isinstance(matches, list) else [matches]
        return detected


class GuardrailsService:
    """Runtime guardrails service controlled by GUARDRAILS_CONFIG."""

    DEFAULTS: Dict[str, Any] = {
        "runtime_enabled": True,
        "check_pii_input": True,
        "check_toxicity": True,
        "check_jailbreak": True,
        "check_output": True,
        "check_credentials_output": True,
        "check_toxic_code_output": True,
        "sanitize_pii": False,
        "content_safety_enabled": False,
        "content_safety_severity_threshold": 2,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULTS, **(config or {})}
        self.pii_detector = PIIDetector()
        self.credential_scanner = CredentialScanner()
        self.toxic_code_detector = ToxicCodeDetector()
        self.content_safety = get_content_safety_service(config=self.config)

    def validate_input(self, text: str) -> ValidationResult:
        violations: List[str] = []
        details: Dict[str, Any] = {}

        if not self.config.get("runtime_enabled", True):
            return ValidationResult(True, violations, details)

        if self.config.get("check_pii_input", True):
            pii_detected = self.pii_detector.detect(text)
            if pii_detected:
                violations.append("PII_DETECTED")
                details["pii"] = pii_detected

        if (self.config.get("check_toxicity", True) or self.config.get("check_jailbreak", True)) and self.content_safety.enabled:
            analysis = self.content_safety.analyze_text(text)
            if analysis:
                categories = analysis.get("categories", {})
                threshold = self.content_safety.severity_threshold

                if self.config.get("check_jailbreak", True):
                    jailbreak_level = categories.get("Jailbreak", 0)
                    if jailbreak_level >= threshold:
                        violations.append("JAILBREAK_DETECTED")
                        details["jailbreak"] = jailbreak_level

                if self.config.get("check_toxicity", True):
                    toxic_categories = {
                        cat: sev
                        for cat, sev in categories.items()
                        if cat != "Jailbreak" and sev >= threshold
                    }
                    if toxic_categories:
                        violations.append("TOXIC_CONTENT")
                        details["toxicity"] = toxic_categories

        return ValidationResult(len(violations) == 0, violations, details)

    def validate_output_text(self, text: str) -> ValidationResult:
        violations: List[str] = []
        details: Dict[str, Any] = {}

        if not self.config.get("runtime_enabled", True):
            return ValidationResult(True, violations, details)

        if self.config.get("check_output", True) and self.content_safety.enabled:
            analysis = self.content_safety.analyze_text(text)
            if analysis:
                categories = analysis.get("categories", {})
                threshold = self.content_safety.severity_threshold
                toxic_categories = {
                    cat: sev
                    for cat, sev in categories.items()
                    if sev >= threshold
                }
                if toxic_categories:
                    violations.append("UNSAFE_OUTPUT_TEXT")
                    details["toxicity"] = toxic_categories

        return ValidationResult(len(violations) == 0, violations, details)

    def validate_output_code(self, code: str) -> ValidationResult:
        violations: List[str] = []
        details: Dict[str, Any] = {}

        if not self.config.get("runtime_enabled", True):
            return ValidationResult(True, violations, details)

        if self.config.get("check_credentials_output", True):
            credentials_found = self.credential_scanner.scan(code)
            if credentials_found:
                violations.append("HARDCODED_CREDENTIALS")
                details["credentials"] = credentials_found

        if self.config.get("check_toxic_code_output", True):
            toxic_patterns = self.toxic_code_detector.detect(code)
            if toxic_patterns:
                violations.append("DANGEROUS_CODE")
                details["toxic_patterns"] = toxic_patterns

        return ValidationResult(len(violations) == 0, violations, details)

    def sanitize_text(self, text: str) -> str:
        sanitized = text
        sanitized = re.sub(self.pii_detector.PATTERNS["email"], "[EMAIL_REDACTED]", sanitized)
        sanitized = re.sub(self.pii_detector.PATTERNS["ssn"], "[SSN_REDACTED]", sanitized)
        sanitized = re.sub(self.pii_detector.PATTERNS["phone"], "[PHONE_REDACTED]", sanitized)
        sanitized = re.sub(self.pii_detector.PATTERNS["credit_card"], "[CARD_REDACTED]", sanitized)
        return sanitized

    def sanitize_code(self, code: str) -> str:
        return self.sanitize_text(code)


_guardrails_service: Optional[GuardrailsService] = None


def get_guardrails_service(config: Optional[Dict[str, Any]] = None) -> GuardrailsService:
    """Get singleton guardrails service or config-scoped instance."""
    global _guardrails_service
    if config is not None:
        return GuardrailsService(config=config)
    if _guardrails_service is None:
        _guardrails_service = GuardrailsService()
    return _guardrails_service
