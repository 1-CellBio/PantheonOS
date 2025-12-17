"""
JSON parsing utilities for ACE learning module.

Simple JSON extraction from LLM text responses.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar, Type

from pydantic import BaseModel

from pantheon.utils.log import logger


T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str) -> dict | None:
    """
    Extract JSON object from text response.
    
    Tries direct parse first, then strips common prefixes/suffixes.
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Fallback: extract from ```json ... ``` code block (common LLM behavior)
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    return None


def parse_to_model(text: str, model_class: Type[T], default_factory: callable) -> T:
    """Parse text to a Pydantic model, return default on failure."""
    try:
        data = extract_json_from_text(text)
        if data is None:
            logger.warning(f"Failed to extract JSON for {model_class.__name__}")
            return default_factory()
        return model_class.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to parse {model_class.__name__}: {e}")
        return default_factory()
