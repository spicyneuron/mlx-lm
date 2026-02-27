# Copyright © 2026 Apple Inc.

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple


# Stopping criteria


class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int
    trim_text_length: int
    stop_word: Optional[str] = None


def stopping_criteria(
    tokens: List[int],
    eos_token_ids: set,
    stop_id_sequences: List[List[int]],
    stop_words: List[str],
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined
    conditions.

    Args:
        tokens (List[int]): The current sequence of generated tokens.
        eos_token_ids (set): The token IDs that represents the
          end-of-sequence. If the last token in ``tokens`` is in the set,
          the generation should stop.
        stop_id_sequences (List[List[[int]]): A list of integer lists, each
          representing a sequence of token IDs. If the end of the `tokens`
          list matches any of these sequences, the generation should stop.
        stop_words (List[str]): The stop words that correspond to the
            ``stop_id_sequences``.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has
          been met (`stop_met`) and how many tokens should be trimmed from the
          end if it has (`trim_length`) as well as the text that should be
          trimmed.
    """
    if tokens and tokens[-1] in eos_token_ids:
        return StopCondition(stop_met=True, trim_length=0, trim_text_length=0)

    for stop_ids, stop_word in zip(stop_id_sequences, stop_words):
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StopCondition(
                    stop_met=True,
                    trim_length=len(stop_ids),
                    trim_text_length=len(stop_word),
                    stop_word=stop_word,
                )

    return StopCondition(stop_met=False, trim_length=0, trim_text_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2

    Args:
        s1 (Sequence): The first sequence
        s2 (Sequence): The second sequence

    Returns:
        bool: If the two sequences have overlap
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


# Generation loop


@dataclass
class GenerationResult:
    tokens: List[int]
    finish_reason: str
    stop_sequence: Optional[str]
    made_tool_call: bool


def trim_visible_stop_text(
    text: str, stop_sequence: Optional[str], trim_text_length: int
) -> str:
    if trim_text_length <= 0 or not stop_sequence:
        return text
    if not text.endswith(stop_sequence):
        return text
    return text[: max(0, len(text) - trim_text_length)]


_noop = lambda *_args, **_kwargs: None


def run_generation_loop(
    ctx: Any,
    response: Any,
    stop_words: List[str],
    on_text_segment: Callable[[str], None],
    on_reasoning_segment: Callable[[str], None] = _noop,
    on_tool_call_start: Callable[[], None] = _noop,
    on_tool_call_done: Callable[[str], None] = _noop,
    on_hidden_progress: Callable[[], None] = _noop,
    on_token: Callable[[Any], None] = _noop,
) -> GenerationResult:
    """Shared token-by-token generation loop.

    Drives the core state machine (reasoning, tool calls, stop conditions)
    and delegates all formatting to caller-supplied callbacks.

    Callbacks:
        on_text_segment(text)    -- visible text ready to emit/accumulate
        on_reasoning_segment(text) -- reasoning text (hidden for Anthropic)
        on_tool_call_start()     -- tool-call delimiter opened
        on_tool_call_done(raw)   -- raw tool-call body text, caller parses
        on_hidden_progress()     -- nothing visible, keepalive opportunity
        on_token(gen)            -- every token (for logprobs collection)
    """
    tokens: List[int] = []
    segment = ""
    finish_reason = "length"
    stop_sequence = None
    tool_text = ""
    made_tool_call = False
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

    for gen in response:
        tokens.append(gen.token)
        on_token(gen)
        visible = ""

        if in_reasoning:
            if gen.text == ctx.think_end:
                in_reasoning = False
            else:
                on_reasoning_segment(gen.text)
        elif ctx.has_tool_calling and gen.text == ctx.tool_call_start:
            flush_segment()
            on_tool_call_start()
            made_tool_call = True
            in_tool_call = True
        elif in_tool_call:
            if gen.text == ctx.tool_call_end:
                on_tool_call_done(tool_text)
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
            sequence_overlap(tokens, seq) for seq in ctx.stop_token_sequences
        ):
            if not visible and not segment:
                on_hidden_progress()
            continue

        if segment:
            flush_segment()
        elif not visible:
            on_hidden_progress()

    if in_tool_call and tool_text:
        on_tool_call_done(tool_text)

    flush_segment()

    if made_tool_call and finish_reason == "stop":
        finish_reason = "tool_calls"

    return GenerationResult(
        tokens=tokens,
        finish_reason=finish_reason,
        stop_sequence=stop_sequence,
        made_tool_call=made_tool_call,
    )


# HTTP helpers


def load_json_body(handler: Any) -> Tuple[bool, Any]:
    content_length = int(handler.headers["Content-Length"])
    raw_body = handler.rfile.read(content_length)
    try:
        return True, json.loads(raw_body.decode())
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError: {e} - Raw body: {raw_body.decode()}")
        return False, e


def write_json_response(
    handler: Any,
    status_code: int,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    flush: bool = False,
) -> None:
    response_json = json.dumps(payload).encode()
    handler.send_response(status_code)
    handler.send_header("Content-type", "application/json")
    if headers:
        for name, value in headers.items():
            handler.send_header(name, value)
    handler.send_header("Content-Length", str(len(response_json)))
    handler._set_cors_headers()
    handler.end_headers()
    handler.wfile.write(response_json)
    if flush:
        handler.wfile.flush()


def make_progress_callback(
    stream: bool, on_stream_progress: Callable[[int, int], None]
) -> Callable[[int, int], None]:
    def callback(processed_tokens: int, total_tokens: int) -> None:
        logging.info(
            f"Prompt processing progress: {processed_tokens}/{total_tokens}"
        )
        if stream:
            on_stream_progress(processed_tokens, total_tokens)

    return callback
