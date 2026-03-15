"""
LLM helper utilities for structured output parsing.

Provides safe JSON extraction from LLM responses and reusable
chain construction helpers for the analysis / classification modules.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def parse_llm_json_response(response_text: str, defaults: dict | None = None) -> dict:
    """
    Safely extract a JSON object from an LLM response string.

    LLMs sometimes wrap JSON in markdown code fences or add commentary.
    This function strips that away and attempts a clean parse, falling
    back to caller-supplied defaults on failure.

    Args:
        response_text: Raw text output from the LLM.
        defaults:      Fallback dict returned when parsing fails.

    Returns:
        Parsed dictionary.
    """
    if defaults is None:
        defaults = {}

    # Strip markdown code fences if present
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        logger.warning("LLM returned non-dict JSON; using defaults.")
        return defaults
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON: %s", cleaned[:200])
        return defaults
