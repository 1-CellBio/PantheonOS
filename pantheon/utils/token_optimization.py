from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pantheon.utils.log import logger

PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"
PREVIEW_SIZE_BYTES = 2000
BYTES_PER_TOKEN = 4
MAX_TOOL_RESULTS_PER_MESSAGE_CHARS = 200_000
TIME_BASED_MC_GAP_THRESHOLD_MINUTES = 60
TIME_BASED_MC_KEEP_RECENT = 5
STATE_KEY = "token_optimization"
COMPACTABLE_TOOL_SUFFIXES = {
    "read_file",
    "view_file",
    "write_file",
    "update_file",
    "apply_patch",
    "glob",
    "grep",
    "grep_search",
    "find_by_name",
    "shell",
    "bash",
    "web_fetch",
    "web_search",
    "web_crawl",
}


@dataclass
class ContentReplacementState:
    seen_ids: set[str]
    replacements: dict[str, str]


@dataclass
class ToolMessageCandidate:
    tool_use_id: str
    content: str
    size: int


@dataclass(frozen=True)
class TimeBasedMicrocompactConfig:
    enabled: bool
    gap_threshold_minutes: int
    keep_recent: int


@dataclass(frozen=True)
class CacheSafeRuntimeParams:
    model: str
    model_params_raw: dict[str, Any]
    model_params_normalized: Any
    response_format_raw: Any | None
    response_format_normalized: Any | None


def create_content_replacement_state() -> ContentReplacementState:
    return ContentReplacementState(seen_ids=set(), replacements={})


def get_time_based_microcompact_config() -> TimeBasedMicrocompactConfig:
    return TimeBasedMicrocompactConfig(
        enabled=True,
        gap_threshold_minutes=TIME_BASED_MC_GAP_THRESHOLD_MINUTES,
        keep_recent=TIME_BASED_MC_KEEP_RECENT,
    )


def normalize_cache_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(key): normalize_cache_safe_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [normalize_cache_safe_value(item) for item in value]
    if isinstance(value, set):
        return sorted(normalize_cache_safe_value(item) for item in value)
    if hasattr(value, "model_json_schema"):
        try:
            return normalize_cache_safe_value(value.model_json_schema())
        except TypeError:
            pass
    if hasattr(value, "__qualname__") and hasattr(value, "__module__"):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def build_cache_safe_runtime_params(
    model: str,
    model_params: dict[str, Any] | None,
    response_format: Any | None,
) -> CacheSafeRuntimeParams:
    raw_model_params = dict(model_params or {})
    return CacheSafeRuntimeParams(
        model=model,
        model_params_raw=raw_model_params,
        model_params_normalized=normalize_cache_safe_value(raw_model_params),
        response_format_raw=response_format,
        response_format_normalized=normalize_cache_safe_value(response_format),
    )


def _normalize_state_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return data


def load_content_replacement_state(memory: Any | None) -> ContentReplacementState:
    if memory is None:
        return create_content_replacement_state()
    payload = _normalize_state_payload(memory.extra_data.get(STATE_KEY))
    seen_ids = {
        str(item)
        for item in payload.get("seen_ids", [])
        if isinstance(item, str) and item
    }
    replacements = {
        str(tool_use_id): str(replacement)
        for tool_use_id, replacement in payload.get("replacements", {}).items()
        if isinstance(tool_use_id, str) and tool_use_id
    }
    return ContentReplacementState(seen_ids=seen_ids, replacements=replacements)


def save_content_replacement_state(
    memory: Any | None,
    state: ContentReplacementState,
) -> None:
    if memory is None:
        return
    payload = {
        "seen_ids": sorted(state.seen_ids),
        "replacements": dict(sorted(state.replacements.items())),
    }
    if memory.extra_data.get(STATE_KEY) == payload:
        return
    memory.extra_data[STATE_KEY] = payload
    memory.mark_dirty()


def _format_file_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def generate_preview(content: str, max_bytes: int) -> tuple[str, bool]:
    if len(content) <= max_bytes:
        return content, False
    truncated = content[:max_bytes]
    last_newline = truncated.rfind("\n")
    cut_point = last_newline if last_newline > max_bytes * 0.5 else max_bytes
    return content[:cut_point], True


def _get_tool_results_dir(memory: Any | None, base_dir: Path | None) -> Path:
    if base_dir is not None:
        root = Path(base_dir)
    else:
        from pantheon.settings import get_settings

        root = get_settings().tmp_dir / "tool-results"
    memory_id = getattr(memory, "id", None) or "default"
    return root / str(memory_id)


def _get_tool_result_path(
    tool_use_id: str,
    memory: Any | None,
    base_dir: Path | None,
) -> Path:
    return _get_tool_results_dir(memory, base_dir) / f"{tool_use_id}.txt"


def persist_tool_result(
    content: str,
    tool_use_id: str,
    memory: Any | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    directory = _get_tool_results_dir(memory, base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = _get_tool_result_path(tool_use_id, memory, base_dir)
    if not filepath.exists():
        filepath.write_text(content, encoding="utf-8")
    preview, has_more = generate_preview(content, PREVIEW_SIZE_BYTES)
    return {
        "filepath": str(filepath),
        "original_size": len(content),
        "preview": preview,
        "has_more": has_more,
    }


def build_large_tool_result_message(result: dict[str, Any]) -> str:
    message = f"{PERSISTED_OUTPUT_TAG}\n"
    message += (
        f"Output too large ({_format_file_size(result['original_size'])}). "
        f"Full output saved to: {result['filepath']}\n\n"
    )
    message += f"Preview (first {_format_file_size(PREVIEW_SIZE_BYTES)}):\n"
    message += result["preview"]
    message += "\n...\n" if result["has_more"] else "\n"
    message += PERSISTED_OUTPUT_CLOSING_TAG
    return message


def _is_already_externalized(content: str) -> bool:
    return (
        content.startswith(PERSISTED_OUTPUT_TAG)
        or content == TIME_BASED_MC_CLEARED_MESSAGE
        or "Full content saved to:" in content
    )


def build_tool_name_map(messages: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            function = tool_call.get("function") or {}
            tool_name = function.get("name")
            if tool_call_id and tool_name:
                result[str(tool_call_id)] = str(tool_name)
    return result


def get_tool_name_for_message(message: dict, tool_name_map: dict[str, str]) -> str | None:
    tool_use_id = message.get("tool_call_id")
    if isinstance(tool_use_id, str) and tool_use_id:
        mapped = tool_name_map.get(tool_use_id)
        if mapped:
            return mapped

    tool_name = message.get("tool_name")
    if isinstance(tool_name, str) and tool_name:
        return tool_name
    return None


def normalize_tool_name(tool_name: str | None) -> str:
    if not tool_name:
        return ""
    if "__" in tool_name:
        return tool_name.rsplit("__", 1)[-1]
    return tool_name


def is_compactable_tool_name(tool_name: str | None) -> bool:
    return normalize_tool_name(tool_name) in COMPACTABLE_TOOL_SUFFIXES


def collect_candidates_from_message(message: dict) -> list[ToolMessageCandidate]:
    if message.get("role") != "tool":
        return []
    tool_use_id = message.get("tool_call_id")
    content = message.get("content")
    if not isinstance(tool_use_id, str) or not tool_use_id:
        return []
    if not isinstance(content, str) or not content:
        return []
    if _is_already_externalized(content):
        return []
    return [
        ToolMessageCandidate(
            tool_use_id=tool_use_id,
            content=content,
            size=len(content),
        )
    ]


def collect_candidates_by_message(messages: list[dict]) -> list[list[ToolMessageCandidate]]:
    groups: list[list[ToolMessageCandidate]] = []
    current: list[ToolMessageCandidate] = []
    seen_assistant_ids: set[str] = set()

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
        current = []

    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool":
            current.extend(collect_candidates_from_message(message))
            continue
        if role == "assistant":
            assistant_id = str(message.get("id") or f"assistant-{index}")
            if assistant_id not in seen_assistant_ids:
                flush()
                seen_assistant_ids.add(assistant_id)
            continue
        flush()
    flush()
    return groups


def partition_by_prior_decision(
    candidates: list[ToolMessageCandidate],
    state: ContentReplacementState,
) -> tuple[list[tuple[ToolMessageCandidate, str]], list[ToolMessageCandidate], list[ToolMessageCandidate]]:
    must_reapply: list[tuple[ToolMessageCandidate, str]] = []
    frozen: list[ToolMessageCandidate] = []
    fresh: list[ToolMessageCandidate] = []
    for candidate in candidates:
        replacement = state.replacements.get(candidate.tool_use_id)
        if replacement is not None:
            must_reapply.append((candidate, replacement))
        elif candidate.tool_use_id in state.seen_ids:
            frozen.append(candidate)
        else:
            fresh.append(candidate)
    return must_reapply, frozen, fresh


def select_fresh_to_replace(
    fresh: list[ToolMessageCandidate],
    frozen_size: int,
    limit: int,
) -> list[ToolMessageCandidate]:
    sorted_candidates = sorted(fresh, key=lambda item: item.size, reverse=True)
    selected: list[ToolMessageCandidate] = []
    remaining = frozen_size + sum(item.size for item in fresh)
    for candidate in sorted_candidates:
        if remaining <= limit:
            break
        selected.append(candidate)
        remaining -= candidate.size
    return selected


def replace_tool_message_contents(
    messages: list[dict],
    replacement_map: dict[str, str],
) -> list[dict]:
    result: list[dict] = []
    for message in messages:
        if message.get("role") != "tool":
            result.append(message)
            continue
        tool_use_id = message.get("tool_call_id")
        replacement = replacement_map.get(str(tool_use_id))
        if replacement is None:
            result.append(message)
            continue
        new_message = dict(message)
        new_message["content"] = replacement
        result.append(new_message)
    return result


def apply_tool_result_budget(
    messages: list[dict],
    memory: Any | None = None,
    base_dir: Path | None = None,
    per_message_limit: int = MAX_TOOL_RESULTS_PER_MESSAGE_CHARS,
    skip_tool_names: set[str] | None = None,
) -> list[dict]:
    state = load_content_replacement_state(memory)
    candidates_by_message = collect_candidates_by_message(messages)
    skip_tool_names = skip_tool_names or set()
    name_by_tool_use_id = (
        build_tool_name_map(messages) if skip_tool_names else {}
    )
    replacement_map: dict[str, str] = {}
    changed = False

    for candidates in candidates_by_message:
        must_reapply, frozen, fresh = partition_by_prior_decision(candidates, state)

        for candidate, replacement in must_reapply:
            replacement_map[candidate.tool_use_id] = replacement

        if not fresh:
            for candidate in candidates:
                state.seen_ids.add(candidate.tool_use_id)
            continue

        skipped = [
            candidate
            for candidate in fresh
            if name_by_tool_use_id.get(candidate.tool_use_id) in skip_tool_names
        ]
        for candidate in skipped:
            state.seen_ids.add(candidate.tool_use_id)

        eligible = [candidate for candidate in fresh if candidate not in skipped]
        frozen_size = sum(candidate.size for candidate in frozen)
        fresh_size = sum(candidate.size for candidate in eligible)
        selected = (
            select_fresh_to_replace(eligible, frozen_size, per_message_limit)
            if frozen_size + fresh_size > per_message_limit
            else []
        )

        selected_ids = {candidate.tool_use_id for candidate in selected}
        for candidate in candidates:
            if candidate.tool_use_id not in selected_ids:
                state.seen_ids.add(candidate.tool_use_id)

        for candidate in selected:
            persisted = persist_tool_result(
                candidate.content,
                candidate.tool_use_id,
                memory=memory,
                base_dir=base_dir,
            )
            replacement = build_large_tool_result_message(persisted)
            state.seen_ids.add(candidate.tool_use_id)
            state.replacements[candidate.tool_use_id] = replacement
            replacement_map[candidate.tool_use_id] = replacement
            changed = True

    save_content_replacement_state(memory, state)
    if not replacement_map:
        return messages
    return replace_tool_message_contents(messages, replacement_map)


def _parse_timestamp(timestamp: Any) -> float | None:
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    if isinstance(timestamp, str):
        try:
            from datetime import datetime

            return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _collect_compactable_tool_message_ids(messages: list[dict]) -> list[str]:
    tool_name_map = build_tool_name_map(messages)
    ids: list[str] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_use_id = message.get("tool_call_id")
        content = message.get("content")
        if not isinstance(tool_use_id, str) or not tool_use_id:
            continue
        if not isinstance(content, str) or not content:
            continue
        if _is_already_externalized(content):
            continue
        tool_name = get_tool_name_for_message(message, tool_name_map)
        if not is_compactable_tool_name(tool_name):
            continue
        ids.append(tool_use_id)
    return ids


def evaluate_time_based_trigger(
    messages: list[dict],
    *,
    is_main_thread: bool,
    config: TimeBasedMicrocompactConfig | None = None,
) -> tuple[float, TimeBasedMicrocompactConfig] | None:
    resolved_config = config or get_time_based_microcompact_config()
    if not resolved_config.enabled or not is_main_thread:
        return None

    last_assistant_timestamp: float | None = None
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        parsed = _parse_timestamp(message.get("timestamp"))
        if parsed is not None:
            last_assistant_timestamp = parsed
            break

    if last_assistant_timestamp is None:
        return None

    import time

    gap_minutes = (time.time() - last_assistant_timestamp) / 60.0
    if gap_minutes < resolved_config.gap_threshold_minutes:
        return None
    return gap_minutes, resolved_config


def microcompact_messages(
    messages: list[dict],
    *,
    is_main_thread: bool = True,
    config: TimeBasedMicrocompactConfig | None = None,
) -> list[dict]:
    trigger = evaluate_time_based_trigger(
        messages,
        is_main_thread=is_main_thread,
        config=config,
    )
    if trigger is None:
        return messages
    gap_minutes, resolved_config = trigger

    compactable_ids = _collect_compactable_tool_message_ids(messages)
    if not compactable_ids:
        return messages

    keep_count = max(1, resolved_config.keep_recent)
    keep_set = set(compactable_ids[-keep_count:])
    clear_set = {tool_use_id for tool_use_id in compactable_ids if tool_use_id not in keep_set}
    if not clear_set:
        return messages

    changed = False
    tokens_saved = 0
    result: list[dict] = []
    for message in messages:
        if message.get("role") != "tool":
            result.append(message)
            continue
        tool_use_id = message.get("tool_call_id")
        if tool_use_id not in clear_set:
            result.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str) or _is_already_externalized(content):
            result.append(message)
            continue
        tokens_saved += max(1, len(content) // BYTES_PER_TOKEN)
        new_message = dict(message)
        new_message["content"] = TIME_BASED_MC_CLEARED_MESSAGE
        result.append(new_message)
        changed = True

    if changed:
        logger.info(
            "[token optimization] time-based microcompact cleared {} tool messages after {:.1f} minute gap (~{} tokens saved, kept last {})",
            len(clear_set),
            gap_minutes,
            tokens_saved,
            len(keep_set),
        )
    return result if changed else messages


def apply_token_optimizations(
    messages: list[dict],
    memory: Any | None = None,
    base_dir: Path | None = None,
    *,
    is_main_thread: bool = True,
) -> list[dict]:
    optimized = apply_tool_result_budget(
        messages,
        memory=memory,
        base_dir=base_dir,
    )
    optimized = microcompact_messages(
        optimized,
        is_main_thread=is_main_thread,
    )
    return optimized


def project_memory_messages_for_llm(messages: list[dict]) -> list[dict]:
    """Project stored history into the LLM-facing view."""
    from copy import deepcopy

    filtered = [message for message in messages if message.get("role") != "system"]

    last_compression_idx = -1
    for index, message in enumerate(filtered):
        if message.get("role") == "compression":
            last_compression_idx = index

    if last_compression_idx >= 0:
        filtered = filtered[last_compression_idx:]

    result: list[dict] = []
    for message in filtered:
        msg = deepcopy(message)
        if msg.get("role") == "compression":
            msg["role"] = "user"
            if not isinstance(msg.get("content"), str):
                msg["content"] = str(msg.get("content", ""))
        result.append(msg)
    return result


def build_llm_view(
    messages: list[dict],
    memory: Any | None = None,
    base_dir: Path | None = None,
    *,
    is_main_thread: bool = True,
) -> list[dict]:
    """Build the projected prompt view from raw history."""
    if not messages:
        return []

    system_message = next(
        (message for message in messages if message.get("role") == "system"),
        None,
    )
    non_system_messages = [
        message for message in messages if message.get("role") != "system"
    ]
    projected = project_memory_messages_for_llm(non_system_messages)
    optimized = apply_token_optimizations(
        projected,
        memory=memory,
        base_dir=base_dir,
        is_main_thread=is_main_thread,
    )
    if system_message is not None:
        return [system_message, *optimized]
    return optimized


def stabilize_tool_definitions(tools: list[dict]) -> list[dict]:
    """Return deterministic tool definitions for cache-stable prompts."""

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            normalized = {key: normalize(value[key]) for key in sorted(value)}
            required = normalized.get("required")
            if isinstance(required, list) and all(
                isinstance(item, str) for item in required
            ):
                normalized["required"] = sorted(required)
            return normalized
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    stabilized = [normalize(tool) for tool in tools]
    stabilized.sort(
        key=lambda tool: (
            str(tool.get("function", {}).get("name", "")),
            json.dumps(tool, ensure_ascii=False, sort_keys=True),
        )
    )
    return stabilized


def extract_persisted_file_paths(messages: list[dict]) -> list[str]:
    pattern = re.compile(r"Full output saved to:\s*(.+)")
    paths: list[str] = []
    seen: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for match in pattern.findall(content):
            path = match.strip()
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def build_recent_context_block(
    messages: list[dict],
    max_messages: int = 6,
    max_chars_per_message: int = 1200,
) -> str:
    relevant = [
        message
        for message in messages
        if message.get("role") in {"user", "assistant", "tool"}
    ]
    if not relevant:
        return ""

    def trim(text: str) -> str:
        if len(text) <= max_chars_per_message:
            return text
        return text[:max_chars_per_message] + "\n...[truncated recent context]..."

    blocks: list[str] = []
    for message in relevant[-max_messages:]:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = str(message.get("role", "unknown")).upper()
        blocks.append(f"[{role}]\n{trim(content.strip())}")
    return "\n\n".join(blocks)


def build_delegation_context_message(
    history: list[dict],
    instruction: str,
    summary_text: str | None = None,
) -> str:
    projected = build_llm_view(history, is_main_thread=False)
    parts: list[str] = []
    if summary_text:
        parts.append(f"Context Summary:\n{summary_text}")

    recent_context = build_recent_context_block(projected)
    if recent_context:
        parts.append(f"Recent Context:\n{recent_context}")

    file_paths = extract_persisted_file_paths(projected)
    if file_paths:
        parts.append(
            "Referenced Files:\n" + "\n".join(f"- {path}" for path in file_paths[:10])
        )

    parts.append(f"Task: {instruction}")
    return "\n\n".join(parts)


def estimate_total_tokens_from_chars(messages: list[dict]) -> int:
    total_chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
    return int(total_chars / BYTES_PER_TOKEN)
