"""Truncation utilities for tool output management."""

from typing import Any


def truncate_string(content: str, max_length: int) -> str:
    """Truncate string preserving head and tail with info.
    
    Args:
        content: String to truncate
        max_length: Maximum allowed length
        
    Returns:
        Truncated string with head...truncated...tail format
    """
    if len(content) <= max_length:
        return content
    
    truncated_chars = len(content) - max_length
    suffix = f"\n[Truncated: {truncated_chars:,} chars removed, total {len(content):,} chars]"
    
    # Calculate available space for content
    available = max_length - len(suffix) - 20  # 20 for "...truncated..."
    half = available // 2
    
    head = content[:half]
    tail = content[-half:]
    
    return f"{head}\n\n...truncated...\n\n{tail}{suffix}"


def truncate_dict_values(data: dict, max_total: int) -> dict:
    """Truncate large string values in dict.
    
    Distributes max_total across dict keys, truncating each value
    that exceeds its share.
    
    Args:
        data: Dictionary to process
        max_total: Maximum total characters across all values
        
    Returns:
        Dictionary with truncated values
    """
    if not data:
        return data
    
    max_per_value = max_total // max(len(data), 1)
    result = {}
    
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_per_value:
            half = max_per_value // 2 - 20
            if half > 0:
                result[key] = f"{value[:half]}...truncated...{value[-half:]}"
            else:
                result[key] = value[:max_per_value]
        elif isinstance(value, dict):
            result[key] = truncate_dict_values(value, max_per_value)
        elif isinstance(value, list):
            result[key] = [
                truncate_dict_values(v, max_per_value // max(len(value), 1)) 
                if isinstance(v, dict) else v 
                for v in value
            ]
        else:
            result[key] = value
    
    return result


def smart_truncate_result(result: Any, max_length: int, filter_base64_fn=None) -> str:
    """Smart truncation with optional base64 filtering.
    
    Args:
        result: Tool result to truncate (usually dict)
        max_length: Maximum content length
        filter_base64_fn: Optional function to filter base64 from dicts
        
    Returns:
        Truncated string representation
    """
    if isinstance(result, dict):
        # Filter base64 if function provided
        if filter_base64_fn:
            result = filter_base64_fn(result)
        # Truncate dict values
        result = truncate_dict_values(result, max_length)
    
    # Convert to string
    content = repr(result)
    
    # Apply string-level truncation if still too long
    if len(content) > max_length:
        content = truncate_string(content, max_length)
    
    return content
