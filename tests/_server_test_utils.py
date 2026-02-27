# Copyright © 2026 Apple Inc.

import json
import threading
from http.server import HTTPServer
from types import SimpleNamespace
from typing import Any, Tuple, cast

from mlx_lm.server import APIHandler, LRUPromptCache, ResponseGenerator
from mlx_lm.utils import load


def _load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any]:
    loaded = load(model_path)
    return loaded[0], loaded[1]


class DummyModelProvider:
    def __init__(self, with_draft: bool = False):
        hf_model_path = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = _load_model_and_tokenizer(hf_model_path)
        self.model_key = (hf_model_path, None)
        self.is_batchable = True

        self.draft_model = None
        self.draft_model_key = None
        self.cli_args: Any = SimpleNamespace(
            adapter_path=None,
            chat_template=None,
            use_default_chat_template=False,
            trust_remote_code=False,
            draft_model=None,
            num_draft_tokens=3,
            temp=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            max_tokens=512,
            chat_template_args={},
            model=None,
            decode_concurrency=32,
            prompt_concurrency=8,
            prompt_cache_size=10,
            prompt_cache_bytes=1 << 63,
            prompt_cache_total_bytes=None,
        )

        if with_draft:
            self.draft_model, _ = _load_model_and_tokenizer(hf_model_path)
            self.draft_model_key = hf_model_path
            self.cli_args.draft_model = hf_model_path

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer


class ServerAPITestBase:
    model_provider_kwargs = {}

    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            cast(Any, DummyModelProvider(**cls.model_provider_kwargs)),
            LRUPromptCache(),
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()


def collect_sse_events(response):
    events = []
    current_event = None
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith(":"):
            continue
        if line.startswith("event: "):
            current_event = line[len("event: "):]
            continue
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                continue
            events.append((current_event, json.loads(payload)))
    return events


def event_payloads(events, event_name):
    return [payload for event, payload in events if event == event_name]


def visible_event_names(events):
    return [event for event, _ in events if event != "ping"]


def text_deltas(events):
    return [
        payload["delta"]["text"]
        for event, payload in events
        if event == "content_block_delta"
        and payload.get("delta", {}).get("type") == "text_delta"
    ]


def tool_use_start_payloads(events):
    return [
        payload
        for payload in event_payloads(events, "content_block_start")
        if payload.get("content_block", {}).get("type") == "tool_use"
    ]


def tool_use_delta_payloads(events):
    return [
        payload
        for payload in event_payloads(events, "content_block_delta")
        if payload.get("delta", {}).get("type") == "input_json_delta"
    ]


def message_delta(events):
    return [payload for event, payload in events if event == "message_delta"][-1]


def text_from_content_blocks(content):
    return "".join(block["text"] for block in content if block.get("type") == "text")


def make_ctx(**overrides):
    defaults = dict(
        has_thinking=False,
        think_start_id=-1,
        think_end_id=-1,
        think_end="",
        has_tool_calling=False,
        tool_call_start="",
        tool_call_end="",
        eos_token_ids=set(),
        stop_token_sequences=[],
        prompt=[1, 2, 3],
        stop=lambda: None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_gen(text, token, finish_reason=None):
    return SimpleNamespace(
        text=text,
        token=token,
        logprob=0.0,
        finish_reason=finish_reason,
        top_tokens=(),
    )


def make_fake_generate(chunks, on_call=None, **ctx_overrides):
    def fake_generate(request, generation_args, progress_callback=None):
        if on_call:
            on_call(request, generation_args, progress_callback)
        ctx = make_ctx(**ctx_overrides)

        def iterator():
            for idx, text in enumerate(chunks, start=1):
                yield make_gen(
                    text,
                    idx,
                    finish_reason="stop" if idx == len(chunks) else None,
                )

        return ctx, iterator()

    return fake_generate


def make_fake_tool_generate(chunks, tool_parser):
    return make_fake_generate(
        chunks,
        has_tool_calling=True,
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        tool_parser=tool_parser,
    )


def collect_sse_payloads(response):
    return [payload for _, payload in collect_sse_events(response)]
