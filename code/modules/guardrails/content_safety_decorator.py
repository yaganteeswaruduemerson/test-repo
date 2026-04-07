"""Runtime decorator that applies guardrails checks inside generated artifacts."""

from __future__ import annotations

import functools
import inspect
import copy
import json
from typing import Any, Callable, Dict, Optional

from .guardrails_service import get_guardrails_service


PROMPT_KWARG_KEYS = (
    "user_prompt",
    "prompt",
    "message",
    "input",
    "body",
    "payload",
    "email_json",
    "data",
    "query",
    "question",
    "user_input",
    "input_text",
    "text",
    "request",
    "instruction",
    "content",
)

MAX_GUARDRAILS_TEXT_CHARS = 20000


def _to_bool(value: Any, default: bool = False) -> bool:
    """Coerce bool-like values, keeping a fallback default."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _clip_text(value: str) -> str:
    """Limit very large payloads passed into validators to avoid excessive scan cost."""
    if len(value) <= MAX_GUARDRAILS_TEXT_CHARS:
        return value
    return value[:MAX_GUARDRAILS_TEXT_CHARS]


def _serialize_candidate(value: Any) -> Optional[str]:
    """Convert common payload shapes to text for runtime guardrails checks."""
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"

    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, default=str)
        except Exception:
            return str(value)

    if hasattr(value, "__fspath__"):
        return str(value)

    return None


def _extract_input_text(inner_func: Callable, args: tuple, kwargs: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Extract a representative input text across string and structured payload-style arguments."""
    keys_from_config = config.get("guardrails_input_keys", PROMPT_KWARG_KEYS)
    if isinstance(keys_from_config, str):
        input_keys = (keys_from_config,)
    elif isinstance(keys_from_config, (list, tuple, set)):
        input_keys = tuple(str(item) for item in keys_from_config)
    else:
        input_keys = PROMPT_KWARG_KEYS

    scan_all_inputs = _to_bool(config.get("scan_all_inputs", True), True)

    chunks = []
    seen = set()

    def _add(label: str, value: Any) -> None:
        serialized = _serialize_candidate(value)
        if not serialized:
            return
        text = f"{label}: {serialized}" if label else serialized
        if text in seen:
            return
        seen.add(text)
        chunks.append(text)

    bound_arguments: Dict[str, Any] = {}
    try:
        bound = inspect.signature(inner_func).bind_partial(*args, **kwargs)
        bound_arguments = dict(bound.arguments)
    except Exception:
        bound_arguments = {}

    excluded_keys = {"self", "cls", "guardrails_config", "GUARDRAILS_CONFIG"}

    if bound_arguments:
        for key in input_keys:
            if key in bound_arguments:
                _add(key, bound_arguments[key])

        if scan_all_inputs:
            for key, value in bound_arguments.items():
                if key in excluded_keys or key in input_keys:
                    continue
                _add(key, value)
    else:
        for key in input_keys:
            if key in kwargs:
                _add(key, kwargs[key])

        if scan_all_inputs:
            for key, value in kwargs.items():
                if key in excluded_keys or key in input_keys:
                    continue
                _add(key, value)

            # Fallback for positional payloads when signature binding is not available.
            positional_start = 1 if args and not isinstance(args[0], str) else 0
            for index in range(positional_start, len(args)):
                _add(f"arg{index}", args[index])

    if not chunks:
        return ""

    return _clip_text("\n".join(chunks))


def _extract_prompt(args: tuple, kwargs: Dict[str, Any]) -> str:
    """Extract likely user prompt argument from function call."""
    for key in PROMPT_KWARG_KEYS:
        if key in kwargs and isinstance(kwargs[key], str):
            return kwargs[key]

    # Prefer positional args after self, then fallback to first positional for plain functions.
    for index in range(1, len(args)):
        if isinstance(args[index], str):
            return args[index]

    if args and isinstance(args[0], str):
        return args[0]

    return ""


def _resolve_guardrail_config(
    inner_func: Callable,
    args: tuple,
    kwargs: Dict[str, Any],
    decorator_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve runtime config from decorator arg, kwargs, instance, or module globals."""
    if isinstance(decorator_config, dict):
        return decorator_config

    for key in ("guardrails_config", "GUARDRAILS_CONFIG"):
        candidate = kwargs.get(key)
        if isinstance(candidate, dict):
            return candidate

    if args:
        bound_instance = args[0]
        for attr in ("guardrails_config", "GUARDRAILS_CONFIG"):
            candidate = getattr(bound_instance, attr, None)
            if isinstance(candidate, dict):
                return candidate

    module_config = inner_func.__globals__.get("GUARDRAILS_CONFIG")
    if isinstance(module_config, dict):
        return module_config

    return None


def _replace_prompt(args: tuple, kwargs: Dict[str, Any], prompt: str) -> tuple[tuple, Dict[str, Any]]:
    """Replace extracted prompt in kwargs or positional args."""
    updated_kwargs = dict(kwargs)
    updated_args = args

    for key in PROMPT_KWARG_KEYS:
        if key in updated_kwargs and isinstance(updated_kwargs[key], str):
            updated_kwargs[key] = prompt
            return updated_args, updated_kwargs

    mutable_args = list(args)
    for index in range(1, len(mutable_args)):
        if isinstance(mutable_args[index], str):
            mutable_args[index] = prompt
            return tuple(mutable_args), updated_kwargs

    if mutable_args and isinstance(mutable_args[0], str):
        mutable_args[0] = prompt
        updated_args = tuple(mutable_args)

    return updated_args, updated_kwargs


def _extract_response_text(response: Any) -> Optional[str]:
    """Extract representative text from common LLM response payload shapes."""
    if response is None:
        return None

    if isinstance(response, str):
        return _clip_text(response)

    if isinstance(response, dict):
        texts = []
        for key in ("content", "text", "message", "response", "output", "result"):
            if key not in response:
                continue
            value = response[key]
            extracted = _extract_response_text(value)
            if extracted:
                texts.append(extracted)

        if texts:
            return _clip_text(" ".join(texts))

        try:
            return _clip_text(json.dumps(response, default=str))
        except Exception:
            return _clip_text(str(response))

    if isinstance(response, list):
        texts = []
        for item in response:
            extracted = _extract_response_text(item)
            if extracted:
                texts.append(extracted)
        if texts:
            return _clip_text(" ".join(texts))

        try:
            return _clip_text(json.dumps(response, default=str))
        except Exception:
            return _clip_text(str(response))

    return None


def _sanitize_response(response: Any, sanitizer: Callable[[str], str]) -> Any:
    """Sanitize text fields in common response payloads while preserving structure."""
    if isinstance(response, str):
        return sanitizer(response)

    if isinstance(response, dict):
        sanitized = copy.deepcopy(response)
        for key in ("content", "text", "message", "response", "output", "result"):
            if key in sanitized:
                sanitized[key] = _sanitize_response(sanitized[key], sanitizer)
        return sanitized

    if isinstance(response, list):
        return [_sanitize_response(item, sanitizer) for item in response]

    return response


def with_content_safety(func: Optional[Callable] = None, *, config: Optional[Dict[str, Any]] = None):
    """Decorator for runtime prompt/output safety checks."""

    def _decorate(inner_func: Callable):
        if inspect.iscoroutinefunction(inner_func):
            @functools.wraps(inner_func)
            async def _async_wrapper(*args, **kwargs):
                guardrail_config = _resolve_guardrail_config(inner_func, args, kwargs, config)
                if not guardrail_config:
                    return await inner_func(*args, **kwargs)

                guardrails_service = get_guardrails_service(config=guardrail_config)

                input_text = _extract_input_text(inner_func, args, kwargs, guardrail_config)
                if input_text:
                    input_validation = guardrails_service.validate_input(input_text)
                    if not input_validation.is_safe:
                        raise ValueError(
                            f"Input blocked by runtime guardrails: {input_validation.violations}"
                        )

                prompt = _extract_prompt(args, kwargs)
                if prompt and _to_bool(guardrails_service.config.get("sanitize_pii", False), False):
                    sanitized_prompt = guardrails_service.sanitize_text(prompt)
                    args, kwargs = _replace_prompt(args, kwargs, sanitized_prompt)

                result = await inner_func(*args, **kwargs)

                response_text = _extract_response_text(result)
                if response_text:
                    output_validation = guardrails_service.validate_output_text(response_text)
                    if not output_validation.is_safe:
                        raise ValueError(
                            f"Output blocked by runtime guardrails: {output_validation.violations}"
                        )

                    # Optional, opt-in code/payload check to avoid over-blocking non-code agents.
                    if _to_bool(guardrail_config.get("check_output_code", False), False):
                        output_payload_validation = guardrails_service.validate_output_code(response_text)
                        if not output_payload_validation.is_safe:
                            raise ValueError(
                                "Output payload blocked by runtime guardrails: "
                                f"{output_payload_validation.violations}"
                            )

                    if _to_bool(guardrails_service.config.get("sanitize_pii", False), False):
                        return _sanitize_response(result, guardrails_service.sanitize_text)
                return result

            return _async_wrapper

        @functools.wraps(inner_func)
        def _sync_wrapper(*args, **kwargs):
            guardrail_config = _resolve_guardrail_config(inner_func, args, kwargs, config)
            if not guardrail_config:
                return inner_func(*args, **kwargs)

            guardrails_service = get_guardrails_service(config=guardrail_config)

            input_text = _extract_input_text(inner_func, args, kwargs, guardrail_config)
            if input_text:
                input_validation = guardrails_service.validate_input(input_text)
                if not input_validation.is_safe:
                    raise ValueError(
                        f"Input blocked by runtime guardrails: {input_validation.violations}"
                    )

            prompt = _extract_prompt(args, kwargs)
            if prompt and _to_bool(guardrails_service.config.get("sanitize_pii", False), False):
                sanitized_prompt = guardrails_service.sanitize_text(prompt)
                args, kwargs = _replace_prompt(args, kwargs, sanitized_prompt)

            result = inner_func(*args, **kwargs)

            response_text = _extract_response_text(result)
            if response_text:
                output_validation = guardrails_service.validate_output_text(response_text)
                if not output_validation.is_safe:
                    raise ValueError(
                        f"Output blocked by runtime guardrails: {output_validation.violations}"
                    )

                # Optional, opt-in code/payload check to avoid over-blocking non-code agents.
                if _to_bool(guardrail_config.get("check_output_code", False), False):
                    output_payload_validation = guardrails_service.validate_output_code(response_text)
                    if not output_payload_validation.is_safe:
                        raise ValueError(
                            "Output payload blocked by runtime guardrails: "
                            f"{output_payload_validation.violations}"
                        )

                if _to_bool(guardrails_service.config.get("sanitize_pii", False), False):
                    return _sanitize_response(result, guardrails_service.sanitize_text)
            return result

        return _sync_wrapper

    if func is not None:
        return _decorate(func)
    return _decorate
