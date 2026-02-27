# Copyright © 2026 Apple Inc.

import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple


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
