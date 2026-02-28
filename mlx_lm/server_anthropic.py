# Anthropic Messages API (/v1/messages) support for mlx-lm server.
#
# The endpoint adapts Anthropic request/response wire format to the shared
# generation pipeline used by the OpenAI-compatible handlers.

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .server import CompletionRequest
from .server_common import (
    load_json_body,
    make_progress_callback,
    run_generation_loop,
    write_json_response,
)

ANTHROPIC_VERSION = "2023-06-01"
LOGGER = logging.getLogger(__name__)


# Request conversion (Anthropic -> OpenAI internal)


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_text(content: Any) -> str:
    if isinstance(content, str) or content is None:
        return _string(content)
    if isinstance(content, dict):
        return _string(content.get("text"))
    if not isinstance(content, list):
        return _string(content)

    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" or "text" in block:
            parts.append(_string(block.get("text")))
    return "".join(parts)


def _append_text_message(out: List[Dict[str, Any]], role: str, text: str) -> None:
    if not text:
        return
    if out and out[-1].get("role") == role and "tool_calls" not in out[-1]:
        out[-1]["content"] += text
        return
    out.append({"role": role, "content": text})


def _tool_use_to_openai_call(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = block.get("name")
    if not isinstance(name, str) or not name:
        LOGGER.warning("Ignoring assistant tool_use block with missing/invalid name.")
        return None

    tool_input = block.get("input", {})
    if not isinstance(tool_input, dict):
        LOGGER.warning(
            "Coercing assistant tool_use input to empty object because input is not an object."
        )
        tool_input = {}

    tool_use_id = block.get("id")
    if not isinstance(tool_use_id, str) or not tool_use_id:
        tool_use_id = f"toolu_{uuid.uuid4().hex}"

    return {
        "id": tool_use_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(tool_input, ensure_ascii=False),
        },
    }


def _tool_result_to_openai_message(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tool_use_id = block.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id:
        LOGGER.warning(
            "Ignoring user tool_result block with missing/invalid tool_use_id."
        )
        return None

    out: Dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_use_id,
        "content": _extract_text(block.get("content")),
    }
    if isinstance(block.get("is_error"), bool):
        out["is_error"] = block["is_error"]
    return out


def convert_anthropic_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    out: List[Dict[str, Any]] = []

    system_text = _extract_text(body.get("system"))
    if system_text:
        out.append({"role": "system", "content": system_text})

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object")

        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(
                "Unsupported role. Anthropic messages support only `user` and `assistant`."
            )

        content = message.get("content")
        if isinstance(content, str) or content is None:
            _append_text_message(out, role, _string(content))
            continue

        if not isinstance(content, list):
            _append_text_message(out, role, _string(content))
            continue

        if role == "assistant":
            assistant_text = ""
            assistant_tool_calls: List[Dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_call = _tool_use_to_openai_call(block)
                    if tool_call is not None:
                        assistant_tool_calls.append(tool_call)
                    continue

                if block_type == "text" or "text" in block:
                    assistant_text += _string(block.get("text"))

            if assistant_tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                        "tool_calls": assistant_tool_calls,
                    }
                )
            else:
                _append_text_message(out, "assistant", assistant_text)
            continue

        user_text = ""
        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "tool_result":
                _append_text_message(out, "user", user_text)
                user_text = ""
                tool_message = _tool_result_to_openai_message(block)
                if tool_message is not None:
                    out.append(tool_message)
                continue

            if block_type == "text" or "text" in block:
                user_text += _string(block.get("text"))

        _append_text_message(out, "user", user_text)

    return out


def convert_anthropic_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")

    out: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            LOGGER.warning(
                "Ignoring invalid tool definition because entry is not an object."
            )
            continue

        name = tool.get("name")
        if not isinstance(name, str) or not name:
            LOGGER.warning(
                "Ignoring invalid tool definition because name is missing/invalid."
            )
            continue

        input_schema = tool.get("input_schema")
        if not isinstance(input_schema, dict):
            LOGGER.warning(
                "Tool `%s` has invalid input_schema; using empty object schema.", name
            )
            input_schema = {"type": "object", "properties": {}}

        function: Dict[str, Any] = {"name": name, "parameters": input_schema}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]

        out.append({"type": "function", "function": function})

    return out


# Tool parsing


def _parse_generated_tool_uses(
    raw_text: str,
    ctx: Any,
    tools: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    parser = getattr(ctx, "tool_parser", None)
    if parser is None:
        return []

    try:
        parsed = parser(raw_text, tools)
    except Exception as e:
        LOGGER.warning("Tool parser failed; dropping parsed tool uses: %s", e)
        return []

    parsed_calls = parsed if isinstance(parsed, list) else [parsed]

    out: List[Dict[str, Any]] = []
    for call in parsed_calls:
        if not isinstance(call, dict):
            LOGGER.warning("Ignoring parsed tool call because it is not an object.")
            continue

        name = call.get("name")
        if not isinstance(name, str) or not name:
            LOGGER.warning(
                "Ignoring parsed tool call because name is missing/invalid."
            )
            continue

        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                LOGGER.warning(
                    "Ignoring parsed tool call `%s` because arguments JSON is invalid.",
                    name,
                )
                continue
        if not isinstance(arguments, dict):
            LOGGER.warning(
                "Ignoring parsed tool call `%s` because arguments is not an object.",
                name,
            )
            continue

        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"toolu_{uuid.uuid4().hex}"

        out.append(
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": arguments,
            }
        )
    return out


# Stop reason mapping


def anthropic_stop_reason(
    finish_reason: Optional[str],
    stop_sequence: Optional[str],
) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "stop":
        return "stop_sequence" if stop_sequence is not None else "end_turn"
    return "end_turn"


# Response builders


def build_response(
    request_id: str,
    model: str,
    finish_reason: Optional[str],
    prompt_token_count: int,
    completion_token_count: int,
    stop_sequence: Optional[str],
    content_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": anthropic_stop_reason(finish_reason, stop_sequence),
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": prompt_token_count,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": completion_token_count,
        },
    }


def error_payload(message: str, error_type: str) -> Dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }


def _effective_finish_reason(
    finish_reason: Optional[str],
    has_tool_use: bool,
) -> Optional[str]:
    if finish_reason == "tool_calls" and not has_tool_use:
        return "stop"
    return finish_reason


def _write_invalid_request(handler: Any, message: str) -> None:
    write_json_response(
        handler,
        status_code=400,
        payload=error_payload(message, "invalid_request_error"),
    )


def _write_server_error(handler: Any) -> None:
    write_json_response(
        handler,
        status_code=500,
        payload=error_payload("Internal server error", "api_error"),
    )


def _start_generation(
    handler: Any,
    request: CompletionRequest,
    args: Any,
    progress_callback: Optional[Any] = None,
) -> Optional[Tuple[Any, Any]]:
    try:
        return handler.response_generator.generate(
            request,
            args,
            progress_callback=progress_callback,
        )
    except Exception as e:
        # Existing OpenAI-compatible handlers also use HTTP 404 here.
        write_json_response(
            handler,
            status_code=404,
            payload=error_payload(str(e), "api_error"),
        )
        return None


def _build_request(body: Dict[str, Any]) -> CompletionRequest:
    return CompletionRequest(
        "chat",
        "",
        convert_anthropic_messages(body),
        convert_anthropic_tools(body.get("tools")),
        None,
    )


def _load_request_body(handler: Any) -> Optional[Dict[str, Any]]:
    success, decoded = load_json_body(handler)
    if not success:
        _write_invalid_request(handler, f"Invalid JSON in request body: {decoded}")
        return None
    if not isinstance(decoded, dict):
        _write_invalid_request(handler, "Request body must be a JSON object")
        return None
    handler.body = decoded
    return decoded


def _handle_non_stream_completion(handler: Any, request: CompletionRequest, args: Any) -> None:
    generated = _start_generation(handler, request, args)
    if generated is None:
        return
    ctx, response = generated

    content_blocks: List[Dict[str, Any]] = []
    has_tool_use = False

    def append_text(text_segment: str) -> None:
        if not text_segment:
            return
        if content_blocks and content_blocks[-1].get("type") == "text":
            content_blocks[-1]["text"] += text_segment
        else:
            content_blocks.append({"type": "text", "text": text_segment})

    def append_tool_uses(raw_tool_text: str) -> None:
        nonlocal has_tool_use
        for tool_use in _parse_generated_tool_uses(raw_tool_text, ctx, request.tools):
            has_tool_use = True
            content_blocks.append(tool_use)

    try:
        result = run_generation_loop(
            ctx,
            response,
            args.stop_words,
            on_text_segment=append_text,
            on_tool_call_done=append_tool_uses,
        )
    except Exception:
        logging.exception("Unexpected error in Anthropic completion")
        _write_server_error(handler)
        return

    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    out = build_response(
        request_id=handler.request_id,
        model=handler.requested_model,
        finish_reason=_effective_finish_reason(result.finish_reason, has_tool_use),
        prompt_token_count=len(ctx.prompt),
        completion_token_count=len(result.tokens),
        stop_sequence=result.stop_sequence,
        content_blocks=content_blocks,
    )

    write_json_response(
        handler,
        status_code=200,
        payload=out,
        headers={
            "anthropic-version": ANTHROPIC_VERSION,
            "request-id": handler.request_id,
        },
        flush=True,
    )


def _handle_stream_completion(handler: Any, request: CompletionRequest, args: Any) -> None:
    headers_sent = False

    def emit(event: str, data: Dict[str, Any], *, safe: bool = False) -> bool:
        try:
            handler.wfile.write(f"event: {event}\n".encode())
            handler.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            handler.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            if safe:
                return False
            raise

    def emit_ping(*_args: Any) -> None:
        if headers_sent:
            emit("ping", {"type": "ping"}, safe=True)

    progress_callback = make_progress_callback(True, lambda _p, _t: emit_ping())

    generated = _start_generation(
        handler,
        request,
        args,
        progress_callback=progress_callback,
    )
    if generated is None:
        return
    ctx, response = generated

    handler.send_response(200)
    handler.send_header("Content-type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("anthropic-version", ANTHROPIC_VERSION)
    handler.send_header("request-id", handler.request_id)
    handler._set_cors_headers()
    handler.end_headers()
    headers_sent = True

    emit(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": handler.request_id,
                "type": "message",
                "role": "assistant",
                "model": handler.requested_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": len(ctx.prompt),
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        },
        safe=True,
    )

    block_index = 0
    text_block_open = False
    has_tool_use = False

    def close_text() -> None:
        nonlocal block_index, text_block_open
        if not text_block_open:
            return
        emit(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
            safe=True,
        )
        text_block_open = False
        block_index += 1

    def emit_text(text_segment: str) -> None:
        nonlocal text_block_open
        if not text_segment:
            return
        if not text_block_open:
            emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                },
                safe=True,
            )
            text_block_open = True
        emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": text_segment},
            },
            safe=True,
        )

    def emit_tool_uses(raw_tool_text: str) -> None:
        nonlocal block_index, has_tool_use
        close_text()
        for tool_use in _parse_generated_tool_uses(raw_tool_text, ctx, request.tools):
            has_tool_use = True
            emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_use["id"],
                        "name": tool_use["name"],
                        "input": {},
                    },
                },
                safe=True,
            )
            emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(
                            tool_use["input"], ensure_ascii=False
                        ),
                    },
                },
                safe=True,
            )
            emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_index},
                safe=True,
            )
            block_index += 1

    try:
        result = run_generation_loop(
            ctx,
            response,
            args.stop_words,
            on_text_segment=emit_text,
            on_tool_call_start=close_text,
            on_tool_call_done=emit_tool_uses,
            on_hidden_progress=lambda: emit_ping(),
        )
    except Exception:
        logging.exception("Unexpected error in Anthropic stream")
        emit(
            "error",
            error_payload("Internal server error", "api_error"),
            safe=True,
        )
        return

    close_text()

    finish_reason = _effective_finish_reason(result.finish_reason, has_tool_use)
    emit(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": anthropic_stop_reason(
                    finish_reason,
                    result.stop_sequence,
                ),
                "stop_sequence": result.stop_sequence,
            },
            "usage": {"output_tokens": len(result.tokens)},
        },
        safe=True,
    )
    emit("message_stop", {"type": "message_stop"}, safe=True)


def handle_post(handler: Any) -> None:
    """Handle POST /v1/messages using APIHandler primitives."""
    body = _load_request_body(handler)
    if body is None:
        return

    # Anthropic uses stop_sequences instead of stop.
    args = handler._parse_and_build_args(
        body.get("stop_sequences"),
        body.get("max_tokens"),
    )
    handler.request_id = f"msg_{uuid.uuid4().hex}"

    try:
        request = _build_request(body)
    except ValueError as e:
        _write_invalid_request(handler, str(e))
        return

    if handler.stream:
        _handle_stream_completion(handler, request, args)
        return

    _handle_non_stream_completion(handler, request, args)
