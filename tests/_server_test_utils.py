# Copyright © 2026 Apple Inc.

import http
import json
import threading

from mlx_lm.server import APIHandler, LRUPromptCache, ResponseGenerator
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False):
        hf_model_path = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(hf_model_path)
        self.model_key = (hf_model_path, None)
        self.is_batchable = True

        self.draft_model = None
        self.draft_model_key = None
        self.cli_args = type(
            "obj",
            (object,),
            {
                "adapter_path": None,
                "chat_template": None,
                "use_default_chat_template": False,
                "trust_remote_code": False,
                "draft_model": None,
                "num_draft_tokens": 3,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
                "model": None,
                "decode_concurrency": 32,
                "prompt_concurrency": 8,
                "prompt_cache_size": 10,
                "prompt_cache_bytes": 1 << 63,
                "prompt_cache_total_bytes": None,
            },
        )

        if with_draft:
            self.draft_model, _ = load(hf_model_path)
            self.draft_model_key = hf_model_path
            self.cli_args.draft_model = hf_model_path

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer


class ServerAPITestBase:
    model_provider_kwargs = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(**cls.model_provider_kwargs),
            LRUPromptCache(),
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
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
        super().tearDownClass()


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


def collect_sse_payloads(response):
    return [payload for _, payload in collect_sse_events(response)]
