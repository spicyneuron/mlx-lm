# Anthropic Messages API (/v1/messages) support for mlx-lm server.
#
# Translates Anthropic request/response shapes into the OpenAI-compatible
# internal representation used by the core server, then formats generation
# output back into the Anthropic wire format.

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .server_common import (
    load_json_body,
    make_progress_callback,
    write_json_response,
)
from .server import CompletionRequest, sequence_overlap, stopping_criteria


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
        if not isinstance(content, list):
            raise ValueError("messages[].content must be a string or list")

        if role == "assistant":
            assistant_text = ""
            assistant_tool_calls = []
            for block in content:
                block = _expect_object(block, "messages[].content[] must be an object")
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if not isinstance(text, str):
                        raise ValueError("text block is missing a valid `text` field.")
                    assistant_text += text
                elif block_type == "tool_use":
                    assistant_tool_calls.append(
                        _anthropic_tool_use_to_openai_tool_call(block)
                    )
                else:
                    raise ValueError(
                        f"Unsupported content block type `{block_type}` for assistant message."
                    )

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

        # role == "user"
        current_user_text = ""
        for block in content:
            block = _expect_object(block, "messages[].content[] must be an object")
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError("text block is missing a valid `text` field.")
                current_user_text += text
            elif block_type == "tool_result":
                _append_merged_text_message(out, "user", current_user_text)
                current_user_text = ""
                out.append(_anthropic_tool_result_to_openai_tool_message(block))
            else:
                raise ValueError(
                    f"Unsupported content block type `{block_type}` for user message."
                )
        _append_merged_text_message(out, "user", current_user_text)
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


def handle_post(handler: Any) -> None:
    """Handle POST /v1/messages using APIHandler primitives."""
    success, decoded = load_json_body(handler)
    if not success:
        write_json_response(
            handler,
            status_code=400,
            payload=error_payload(
                f"Invalid JSON in request body: {decoded}",
                "invalid_request_error",
            ),
        )
        return
    handler.body = decoded

    if not isinstance(handler.body, dict):
        write_json_response(
            handler,
            status_code=400,
            payload=error_payload(
                "Request body must be a JSON object",
                "invalid_request_error",
            ),
        )
        return

    # Anthropic uses stop_sequences instead of stop
    args = handler._parse_and_build_args(
        handler.body.get("stop_sequences"),
        handler.body.get("max_tokens"),
    )

    handler.request_id = f"msg_{uuid.uuid4().hex}"

    try:
        request = CompletionRequest(
            "chat",
            "",
            convert_anthropic_messages(handler.body),
            convert_anthropic_tools(handler.body.get("tools")),
            None,
        )
    except ValueError as e:
        write_json_response(
            handler,
            status_code=400,
            payload=error_payload(str(e), "invalid_request_error"),
        )
        return

    def write_sse_event(event: str, data: Dict[str, Any]) -> None:
        handler.wfile.write(f"event: {event}\n".encode())
        handler.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        handler.wfile.flush()

    def write_sse_ping() -> None:
        try:
            write_sse_event("ping", {"type": "ping"})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    progress_callback = (
        make_progress_callback(
            handler.stream, lambda _processed_tokens, _total_tokens: write_sse_ping()
        )
        if handler.stream
        else None
    )

    try:
        ctx, response = handler.response_generator.generate(
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
        return

    if not handler.stream:
        content_blocks: List[Dict[str, Any]] = []

        def append_text(text_segment: str) -> None:
            if not text_segment:
                return
            if content_blocks and content_blocks[-1].get("type") == "text":
                content_blocks[-1]["text"] += text_segment
            else:
                content_blocks.append({"type": "text", "text": text_segment})

        try:
            result = run_generation_loop(
                request=request,
                ctx=ctx,
                response=response,
                stop_words=args.stop_words,
                on_text_segment=append_text,
                on_tool_use=lambda tu: content_blocks.append(tu),
            )
        except ValueError as e:
            write_json_response(
                handler,
                status_code=400,
                payload=error_payload(str(e), "invalid_request_error"),
            )
            return
        except Exception:
            logging.exception("Unexpected error in Anthropic completion")
            write_json_response(
                handler,
                status_code=500,
                payload=error_payload("Internal server error", "api_error"),
            )
            return

        if not content_blocks:
            content_blocks = [{"type": "text", "text": ""}]

        out = build_response(
            request_id=handler.request_id,
            model=handler.requested_model,
            finish_reason=result.finish_reason,
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
        return

    handler.send_response(200)
    handler.send_header("Content-type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("anthropic-version", "2023-06-01")
    handler.send_header("request-id", handler.request_id)
    handler._set_cors_headers()
    handler.end_headers()

    write_sse_event(
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

    block_index = 0
    text_block_open = False

    def emit_text_segment(text_segment: str) -> None:
        nonlocal text_block_open, block_index
        if not text_segment:
            return
        if not text_block_open:
            write_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            text_block_open = True
        write_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": text_segment},
            },
        )

    def close_text_block() -> None:
        nonlocal text_block_open, block_index
        if text_block_open:
            write_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_index},
            )
            text_block_open = False
            block_index += 1

    def emit_tool_use_block(tool_use: Dict[str, Any]) -> None:
        nonlocal block_index
        partial_json = json.dumps(tool_use["input"], ensure_ascii=False)
        write_sse_event(
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
        )
        write_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json,
                },
            },
        )
        write_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )
        block_index += 1

    try:
        result = run_generation_loop(
            request=request,
            ctx=ctx,
            response=response,
            stop_words=args.stop_words,
            on_text_segment=emit_text_segment,
            on_tool_use=emit_tool_use_block,
            on_hidden_progress=write_sse_ping,
            on_tool_call_start=close_text_block,
        )
    except ValueError as e:
        try:
            write_sse_event(
                "error",
                error_payload(str(e), "invalid_request_error"),
            )
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        return
    except Exception:
        logging.exception("Unexpected error in Anthropic stream")
        try:
            write_sse_event(
                "error", error_payload("Internal server error", "api_error")
            )
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        return

    close_text_block()

    write_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": anthropic_stop_reason(
                    result.finish_reason, result.stop_sequence
                ),
                "stop_sequence": result.stop_sequence,
            },
            "usage": {"output_tokens": len(result.tokens)},
        },
    )
    write_sse_event("message_stop", {"type": "message_stop"})


# Generation loop and response formatting


@dataclass
class GenerationResult:
    tokens: List[int]
    finish_reason: str
    stop_sequence: Optional[str]
    has_tool_use: bool


def trim_visible_stop_text(
    text: str, stop_sequence: Optional[str], trim_text_length: int
) -> str:
    if trim_text_length <= 0 or not stop_sequence:
        return text
    if not text.endswith(stop_sequence):
        return text
    return text[: max(0, len(text) - trim_text_length)]


def _parse_generated_tool_uses(
    tool_calls: List[str], ctx: Any, tools: Optional[List[Any]]
) -> List[Dict[str, Any]]:
    if not tool_calls:
        return []
    if ctx.tool_parser is None:
        raise ValueError("Model does not support tool calling.")

    parsed_calls = []
    for tool_text in tool_calls:
        parsed = ctx.tool_parser(tool_text, tools)
        if isinstance(parsed, list):
            parsed_calls.extend(parsed)
        else:
            parsed_calls.append(parsed)

    out = []
    for parsed in parsed_calls:
        if not isinstance(parsed, dict):
            raise ValueError("Parsed tool call must be an object.")
        name = parsed.get("name")
        if not isinstance(name, str):
            raise ValueError("Parsed tool call is missing `name`.")

        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Parsed tool call `arguments` must be an object.")

        call_id = parsed.get("id")
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


def run_generation_loop(
    request: CompletionRequest,
    ctx: Any,
    response: Any,
    stop_words: List[str],
    on_text_segment: Callable[[str], None],
    on_tool_use: Callable[[Dict[str, Any]], None],
    on_hidden_progress: Callable[[], None] = lambda: None,
    on_tool_call_start: Callable[[], None] = lambda: None,
) -> GenerationResult:
    """Token-by-token generation loop that emits Anthropic content blocks.

    Parallels handle_completion's token loop in server.py. They share most
    logic (stop checking, tool parsing, reasoning state) but diverge on
    output formatting.
    TODO: fold both loops behind shared callbacks to avoid drift.
    """
    tokens: List[int] = []
    segment = ""
    finish_reason = "length"
    stop_sequence = None
    tool_text = ""
    has_tool_use = False
    in_tool_call = False
    in_reasoning = False
    if ctx.has_thinking:
        for i in range(len(ctx.prompt) - 1, -1, -1):
            if ctx.prompt[i] == ctx.think_end_id:
                break
            elif ctx.prompt[i] == ctx.think_start_id:
                in_reasoning = True
                break

    def flush_segment():
        nonlocal segment
        if not segment:
            return
        on_text_segment(segment)
        segment = ""

    def emit_tool_uses(raw_tool_text: str):
        nonlocal has_tool_use
        parsed_tool_uses = _parse_generated_tool_uses(
            [raw_tool_text], ctx, request.tools
        )
        for tool_use in parsed_tool_uses:
            has_tool_use = True
            on_tool_use(tool_use)

    for gen in response:
        tokens.append(gen.token)
        visible = ""

        if in_reasoning:
            if gen.text == ctx.think_end:
                in_reasoning = False
        elif ctx.has_tool_calling and gen.text == ctx.tool_call_start:
            flush_segment()
            on_tool_call_start()
            in_tool_call = True
        elif in_tool_call:
            if gen.text == ctx.tool_call_end:
                emit_tool_uses(tool_text)
                tool_text = ""
                in_tool_call = False
            else:
                tool_text += gen.text
        else:
            visible = gen.text

        segment += visible

        if gen.finish_reason is not None:
            finish_reason = gen.finish_reason

        stop_condition = stopping_criteria(
            tokens,
            ctx.eos_token_ids,
            ctx.stop_token_sequences,
            stop_words,
        )
        if stop_condition.stop_met:
            finish_reason = "stop"
            if stop_condition.trim_length > 0:
                stop_sequence = stop_condition.stop_word
                tokens = tokens[: len(tokens) - stop_condition.trim_length]
            segment = trim_visible_stop_text(
                segment, stop_sequence, stop_condition.trim_text_length
            )
            ctx.stop()
            flush_segment()
            break

        if any(
            sequence_overlap(tokens, sequence) for sequence in ctx.stop_token_sequences
        ):
            if not visible and not segment:
                on_hidden_progress()
            continue

        if segment:
            flush_segment()
        elif not visible:
            on_hidden_progress()

    if in_tool_call and tool_text:
        emit_tool_uses(tool_text)

    flush_segment()

    if has_tool_use:
        finish_reason = "tool_calls"

    return GenerationResult(
        tokens=tokens,
        finish_reason=finish_reason,
        stop_sequence=stop_sequence,
        has_tool_use=has_tool_use,
    )


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
