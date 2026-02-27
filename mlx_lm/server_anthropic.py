# Anthropic Messages API (/v1/messages) support for mlx-lm server.
#
# Translates Anthropic request/response shapes into the OpenAI-compatible
# internal representation used by the core server, then formats generation
# output back into the Anthropic wire format.

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .server_common import (
    load_json_body,
    make_progress_callback,
    run_generation_loop,
    write_json_response,
)
from .server import CompletionRequest


# Request conversion (Anthropic -> OpenAI internal)


def _expect_object(value: Any, error_message: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(error_message)
    return value


def _content_to_text(content, *, field_name: str) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        text_fragments = []
        for fragment in content:
            fragment = _expect_object(
                fragment, f"Expected {field_name} content blocks to be objects."
            )
            if fragment.get("type") != "text":
                raise ValueError(
                    f"Only 'text' content type is supported in {field_name}."
                )
            text = fragment.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"text block is missing a valid `text` field in {field_name}."
                )
            text_fragments.append(text)
        return "".join(text_fragments)
    raise ValueError(
        f"Expected {field_name} to be a string or list of text blocks, got {type(content)}."
    )


def _text_from_block(block: Dict[str, Any], *, error_message: str) -> str:
    text = block.get("text")
    if not isinstance(text, str):
        raise ValueError(error_message)
    return text


def _append_merged_text_message(
    out: List[Dict[str, Any]], role: str, content: str
) -> None:
    if not content:
        return
    if out and out[-1].get("role") == role and "tool_calls" not in out[-1]:
        out[-1]["content"] += content
    else:
        out.append({"role": role, "content": content})


def _anthropic_tool_use_to_openai_tool_call(block: Dict[str, Any]) -> Dict[str, Any]:
    name = block.get("name")
    if not isinstance(name, str):
        raise ValueError("tool_use block is missing a valid `name`.")
    tool_input = block.get("input", {})
    if not isinstance(tool_input, dict):
        raise ValueError("tool_use block `input` must be an object.")
    tool_use_id = block.get("id")
    if tool_use_id is not None and not isinstance(tool_use_id, str):
        raise ValueError("tool_use block `id` must be a string.")
    return {
        "id": tool_use_id or f"toolu_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(tool_input, ensure_ascii=False),
        },
    }


def _anthropic_tool_result_to_openai_tool_message(
    block: Dict[str, Any],
) -> Dict[str, Any]:
    tool_use_id = block.get("tool_use_id")
    if not isinstance(tool_use_id, str):
        raise ValueError("tool_result block is missing a valid `tool_use_id`.")
    is_error = block.get("is_error")
    if is_error is not None and not isinstance(is_error, bool):
        raise ValueError("tool_result block `is_error` must be a boolean.")

    out = {
        "role": "tool",
        "tool_call_id": tool_use_id,
        "content": _content_to_text(
            block.get("content"), field_name="tool_result.content"
        ),
    }
    if isinstance(is_error, bool):
        out["is_error"] = is_error
    return out


def _content_blocks(content: Any) -> List[Dict[str, Any]]:
    if not isinstance(content, list):
        raise ValueError("messages[].content must be a string or list")
    return [_expect_object(block, "messages[].content[] must be an object") for block in content]


def _collect_assistant_content(
    blocks: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    assistant_text_fragments: List[str] = []
    assistant_tool_calls: List[Dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            assistant_text_fragments.append(
                _text_from_block(
                    block,
                    error_message="text block is missing a valid `text` field.",
                )
            )
            continue
        if block_type == "tool_use":
            assistant_tool_calls.append(_anthropic_tool_use_to_openai_tool_call(block))
            continue
        raise ValueError(
            f"Unsupported content block type `{block_type}` for assistant message."
        )
    return "".join(assistant_text_fragments), assistant_tool_calls


def _append_user_content_blocks(
    out: List[Dict[str, Any]],
    blocks: List[Dict[str, Any]],
) -> None:
    current_user_text = ""
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            current_user_text += _text_from_block(
                block,
                error_message="text block is missing a valid `text` field.",
            )
            continue
        if block_type == "tool_result":
            _append_merged_text_message(out, "user", current_user_text)
            current_user_text = ""
            out.append(_anthropic_tool_result_to_openai_tool_message(block))
            continue
        raise ValueError(
            f"Unsupported content block type `{block_type}` for user message."
        )
    _append_merged_text_message(out, "user", current_user_text)


def convert_anthropic_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    system = body.get("system")
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    out: List[Dict[str, Any]] = []
    if system is not None:
        out.append(
            {
                "role": "system",
                "content": _content_to_text(system, field_name="system"),
            }
        )

    for message in messages:
        message = _expect_object(message, "Each message must be an object")
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"Unsupported role `{role}`. Anthropic messages support only `user` and `assistant`."
            )

        content = message.get("content")
        if isinstance(content, str) or content is None:
            _append_merged_text_message(out, role, content or "")
            continue
        blocks = _content_blocks(content)
        if role == "assistant":
            assistant_text, assistant_tool_calls = _collect_assistant_content(blocks)
            if assistant_tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                        "tool_calls": assistant_tool_calls,
                    }
                )
            else:
                _append_merged_text_message(out, "assistant", assistant_text)
            continue

        _append_user_content_blocks(out, blocks)
    return out


def convert_anthropic_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")

    out: List[Dict[str, Any]] = []
    for tool in tools:
        tool = _expect_object(tool, "Each tool must be an object")
        tool_type = tool.get("type")
        if tool_type not in (None, "custom"):
            raise ValueError(
                f"Unsupported tool type `{tool_type}`. Only client tools are supported."
            )
        name = tool.get("name")
        if not isinstance(name, str):
            raise ValueError("Tool `name` must be a string.")
        input_schema = tool.get("input_schema", {"type": "object", "properties": {}})
        if not isinstance(input_schema, dict):
            raise ValueError("Tool `input_schema` must be an object.")

        function: Dict[str, Any] = {"name": name, "parameters": input_schema}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        out.append({"type": "function", "function": function})

    return out


def _run_generation_with_error_mapping(
    ctx: Any,
    response: Any,
    stop_words: Any,
    *,
    on_value_error: Any,
    on_unexpected_error: Any,
    log_message: str,
    **loop_kwargs: Any,
) -> Optional[Any]:
    try:
        return run_generation_loop(
            ctx,
            response,
            stop_words,
            **loop_kwargs,
        )
    except ValueError as e:
        on_value_error(str(e))
        return None
    except Exception:
        logging.exception(log_message)
        on_unexpected_error()
        return None


def _start_generation(
    handler: Any,
    request: CompletionRequest,
    args: Any,
    progress_callback: Optional[Any] = None,
) -> Optional[Any]:
    try:
        return handler.response_generator.generate(
            request,
            args,
            progress_callback=progress_callback,
        )
    except Exception as e:
        # NOTE: Existing OpenAI-compatible endpoints also use HTTP 404 here.
        # Keeping this status avoids cross-endpoint behavior drift.
        write_json_response(
            handler,
            status_code=404,
            payload=error_payload(str(e), "api_error"),
        )
        return None


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


class _SSEWriter:
    def __init__(self, handler: Any):
        self.handler = handler
        self.headers_sent = False

    def send_headers(self) -> None:
        self.handler.send_response(200)
        self.handler.send_header("Content-type", "text/event-stream")
        self.handler.send_header("Cache-Control", "no-cache")
        self.handler.send_header("anthropic-version", "2023-06-01")
        self.handler.send_header("request-id", self.handler.request_id)
        self.handler._set_cors_headers()
        self.handler.end_headers()
        self.headers_sent = True

    def emit(self, event: str, data: Dict[str, Any]) -> None:
        self.handler.wfile.write(f"event: {event}\n".encode())
        self.handler.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.handler.wfile.flush()

    def emit_safely(self, event: str, data: Dict[str, Any]) -> None:
        try:
            self.emit(event, data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def ping(self) -> None:
        if not self.headers_sent:
            return
        self.emit_safely("ping", {"type": "ping"})


class _StreamBlockEmitter:
    def __init__(self, sse: _SSEWriter):
        self.sse = sse
        self.block_index = 0
        self.text_block_open = False

    def emit_text(self, text_segment: str) -> None:
        if not text_segment:
            return
        if not self.text_block_open:
            self.sse.emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            self.text_block_open = True
        self.sse.emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": {"type": "text_delta", "text": text_segment},
            },
        )

    def close_text(self) -> None:
        if not self.text_block_open:
            return
        self.sse.emit(
            "content_block_stop",
            {"type": "content_block_stop", "index": self.block_index},
        )
        self.text_block_open = False
        self.block_index += 1

    def emit_tool_use(self, tool_use: Dict[str, Any]) -> None:
        partial_json = json.dumps(tool_use["input"], ensure_ascii=False)
        self.sse.emit(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_use["id"],
                    "name": tool_use["name"],
                    "input": {},
                },
            },
        )
        self.sse.emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json,
                },
            },
        )
        self.sse.emit(
            "content_block_stop",
            {"type": "content_block_stop", "index": self.block_index},
        )
        self.block_index += 1


class _ToolUseTracker:
    def __init__(self, tools: Optional[List[Any]]):
        self.tools = tools
        self.has_tool_use = False

    def emit(self, raw_tool_text: str, on_emit: Any, ctx: Any) -> None:
        for tool_use in _parse_generated_tool_uses(raw_tool_text, ctx, self.tools):
            self.has_tool_use = True
            on_emit(tool_use)

    def effective_finish_reason(self, result: Any) -> Optional[str]:
        if result.finish_reason == "tool_calls" and not self.has_tool_use:
            return "stop"
        return result.finish_reason


def _build_request_from_body(body: Dict[str, Any]) -> CompletionRequest:
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


def _handle_non_stream_completion(
    handler: Any,
    request: CompletionRequest,
    args: Any,
    emit_parsed_tool_uses: Any,
    effective_finish_reason: Any,
) -> None:
    generated = _start_generation(handler, request, args)
    if generated is None:
        return
    ctx, response = generated

    content_blocks: List[Dict[str, Any]] = []

    def append_text(text_segment: str) -> None:
        if not text_segment:
            return
        if content_blocks and content_blocks[-1].get("type") == "text":
            content_blocks[-1]["text"] += text_segment
        else:
            content_blocks.append({"type": "text", "text": text_segment})

    result = _run_generation_with_error_mapping(
        ctx,
        response,
        args.stop_words,
        on_text_segment=append_text,
        on_tool_call_done=lambda raw: emit_parsed_tool_uses(
            raw, lambda tu: content_blocks.append(tu), ctx
        ),
        on_value_error=lambda message: _write_invalid_request(handler, message),
        on_unexpected_error=lambda: _write_server_error(handler),
        log_message="Unexpected error in Anthropic completion",
    )
    if result is None:
        return

    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    out = build_response(
        request_id=handler.request_id,
        model=handler.requested_model,
        finish_reason=effective_finish_reason(result),
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
            "anthropic-version": "2023-06-01",
            "request-id": handler.request_id,
        },
        flush=True,
    )


def _handle_stream_completion(
    handler: Any,
    request: CompletionRequest,
    args: Any,
    emit_parsed_tool_uses: Any,
    effective_finish_reason: Any,
) -> None:
    sse = _SSEWriter(handler)

    progress_callback = make_progress_callback(
        True, lambda _processed_tokens, _total_tokens: sse.ping()
    )

    generated = _start_generation(
        handler,
        request,
        args,
        progress_callback=progress_callback,
    )
    if generated is None:
        return
    ctx, response = generated

    sse.send_headers()
    sse.emit(
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
    )

    block_emitter = _StreamBlockEmitter(sse)
    result = _run_generation_with_error_mapping(
        ctx,
        response,
        args.stop_words,
        on_text_segment=block_emitter.emit_text,
        on_tool_call_done=lambda raw: emit_parsed_tool_uses(
            raw, block_emitter.emit_tool_use, ctx
        ),
        on_hidden_progress=sse.ping,
        on_tool_call_start=block_emitter.close_text,
        on_value_error=lambda message: sse.emit_safely(
            "error", error_payload(message, "invalid_request_error")
        ),
        on_unexpected_error=lambda: sse.emit_safely(
            "error", error_payload("Internal server error", "api_error")
        ),
        log_message="Unexpected error in Anthropic stream",
    )
    if result is None:
        return

    block_emitter.close_text()

    sse.emit(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": anthropic_stop_reason(
                    effective_finish_reason(result), result.stop_sequence
                ),
                "stop_sequence": result.stop_sequence,
            },
            "usage": {"output_tokens": len(result.tokens)},
        },
    )
    sse.emit("message_stop", {"type": "message_stop"})


def handle_post(handler: Any) -> None:
    """Handle POST /v1/messages using APIHandler primitives."""
    body = _load_request_body(handler)
    if body is None:
        return

    # Anthropic uses stop_sequences instead of stop
    args = handler._parse_and_build_args(
        body.get("stop_sequences"),
        body.get("max_tokens"),
    )

    handler.request_id = f"msg_{uuid.uuid4().hex}"

    try:
        request = _build_request_from_body(body)
    except ValueError as e:
        _write_invalid_request(handler, str(e))
        return

    tool_use_tracker = _ToolUseTracker(request.tools)

    if not handler.stream:
        _handle_non_stream_completion(
            handler,
            request,
            args,
            tool_use_tracker.emit,
            tool_use_tracker.effective_finish_reason,
        )
        return

    _handle_stream_completion(
        handler,
        request,
        args,
        tool_use_tracker.emit,
        tool_use_tracker.effective_finish_reason,
    )


# Tool parsing


def _parse_generated_tool_uses(
    raw_text: str, ctx: Any, tools: Optional[List[Any]]
) -> List[Dict[str, Any]]:
    """Parse raw tool-call text into Anthropic tool_use content blocks."""
    if ctx.tool_parser is None:
        raise ValueError("Model does not support tool calling.")

    parsed = ctx.tool_parser(raw_text, tools)
    parsed_calls = parsed if isinstance(parsed, list) else [parsed]

    out = []
    for call in parsed_calls:
        if not isinstance(call, dict):
            raise ValueError("Parsed tool call must be an object.")
        name = call.get("name")
        if not isinstance(name, str):
            raise ValueError("Parsed tool call is missing `name`.")

        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Parsed tool call `arguments` must be an object.")

        call_id = call.get("id")
        if call_id is not None and not isinstance(call_id, str):
            raise ValueError("Parsed tool call `id` must be a string.")
        out.append(
            {
                "type": "tool_use",
                "id": call_id or f"toolu_{uuid.uuid4().hex}",
                "name": name,
                "input": arguments,
            }
        )
    return out


# Stop reason mapping


def anthropic_stop_reason(
    finish_reason: Optional[str], stop_sequence: Optional[str]
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
) -> dict:
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
