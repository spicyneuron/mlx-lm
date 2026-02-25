# Copyright © 2024 Apple Inc.

import http
import io
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import requests

from mlx_lm.models.cache import KVCache
from mlx_lm.server import APIHandler, LRUPromptCache, ResponseGenerator
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.model_key = (HF_MODEL_PATH, None)
        self.is_batchable = True

        # Add draft model support
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
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH
            self.cli_args.draft_model = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer


class MockCache:
    def __init__(self, value):
        self.value = value

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
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

    @staticmethod
    def _collect_sse_events(response):
        events = []
        current_event = None
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith(":"):
                continue
            if line.startswith("event: "):
                current_event = line[len("event: ") :]
                continue
            if line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
                events.append((current_event, payload))
        return events

    @staticmethod
    def _make_tool_ctx(tool_parser):
        return SimpleNamespace(
            has_thinking=False,
            think_start_id=-1,
            think_end_id=-1,
            think_end="",
            has_tool_calling=True,
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=tool_parser,
            eos_token_ids=set(),
            stop_token_sequences=[],
            prompt=[1, 2, 3],
            stop=lambda: None,
        )

    def _make_fake_tool_generate(self, chunks, tool_parser):
        def fake_generate(request, generation_args, progress_callback=None):
            ctx = self._make_tool_ctx(tool_parser)

            def iterator():
                final_idx = len(chunks)
                for idx, text in enumerate(chunks, start=1):
                    yield SimpleNamespace(
                        text=text,
                        token=idx,
                        logprob=0.0,
                        finish_reason="stop" if idx == final_idx else None,
                        top_tokens=(),
                    )

            return ctx, iterator()

        return fake_generate

    @staticmethod
    def _anthropic_tool_request(stream=False):
        body = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Use a tool"}],
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        }
        if stream:
            body["stream"] = True
        return body

    def _assert_anthropic_error(
        self, body, error_type=None, message_substring=None
    ):
        self.assertEqual(body["type"], "error")
        self.assertIn("error", body)
        self.assertIsInstance(body["error"], dict)
        if error_type is not None:
            self.assertEqual(body["error"]["type"], error_type)
        if message_substring is not None:
            self.assertIn(message_substring, body["error"]["message"])

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "seed": 999,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = json.loads(response.text)

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        first_text = response_body["choices"][0]["text"]
        self.assertEqual(
            first_text,
            json.loads(requests.post(url, json=post_data).text)["choices"][0]["text"],
        )

    def test_handle_completions_requires_prompt(self):
        url = f"http://localhost:{self.port}/v1/completions"
        response = requests.post(url, json={"model": "default_model", "max_tokens": 10})
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.text)
        self.assertIn("error", body)
        self.assertIn("prompt", body["error"])

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_requires_messages(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        response = requests.post(url, json={"model": "chat_model", "max_tokens": 10})
        self.assertEqual(response.status_code, 400)
        response_body = json.loads(response.text)
        self.assertIn("error", response_body)
        self.assertIn("messages", response_body["error"])

    def test_handle_chat_completions_with_query_string(self):
        url = f"http://localhost:{self.port}/v1/chat/completions?api-version=2026-02-25"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_content_fragments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_null_tool_content(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_anthropic_messages(self):
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("anthropic-version"), "2023-06-01")

        response_body = json.loads(response.text)
        self.assertTrue(response_body["id"].startswith("msg_"))
        self.assertEqual(response_body["type"], "message")
        self.assertEqual(response_body["role"], "assistant")
        self.assertEqual(response_body["model"], "chat_model")
        self.assertIsInstance(response_body["content"], list)
        self.assertEqual(response_body["content"][0]["type"], "text")
        self.assertIn("text", response_body["content"][0])
        self.assertIn(
            response_body["stop_reason"], {"end_turn", "max_tokens", "stop_sequence"}
        )
        self.assertIn("stop_sequence", response_body)
        self.assertIn("usage", response_body)
        self.assertIn("input_tokens", response_body["usage"])
        self.assertIn("output_tokens", response_body["usage"])

    def test_handle_anthropic_messages_requires_messages(self):
        url = f"http://localhost:{self.port}/v1/messages"
        response = requests.post(url, json={"model": "chat_model", "max_tokens": 10})
        self.assertEqual(response.status_code, 400)
        response_body = json.loads(response.text)
        self._assert_anthropic_error(
            response_body,
            error_type="invalid_request_error",
            message_substring="messages",
        )

    def test_handle_anthropic_messages_requires_object_body(self):
        url = f"http://localhost:{self.port}/v1/messages"
        response = requests.post(url, json=["not", "an", "object"])
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.text)
        self._assert_anthropic_error(
            body,
            error_type="invalid_request_error",
            message_substring="JSON object",
        )

    def test_handle_anthropic_messages_with_query_string(self):
        url = (
            f"http://localhost:{self.port}/v1/messages"
            "?beta=true&anthropic-version=2023-06-01"
        )
        post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["type"], "message")
        self.assertEqual(response_body["role"], "assistant")

    def test_handle_anthropic_messages_rejects_text_block_missing_text_field(self):
        url = f"http://localhost:{self.port}/v1/messages"
        response = requests.post(
            url,
            json={
                "model": "chat_model",
                "max_tokens": 10,
                "system": [{"type": "text"}],
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        self.assertEqual(response.status_code, 400)
        response_body = json.loads(response.text)
        self._assert_anthropic_error(
            response_body,
            error_type="invalid_request_error",
            message_substring="valid `text`",
        )

    def test_handle_anthropic_messages_uses_stop_sequences_field(self):
        url = f"http://localhost:{self.port}/v1/messages"
        captured = {"stop_words": None}

        def fake_generate(request, generation_args, progress_callback=None):
            captured["stop_words"] = generation_args.stop_words
            ctx = SimpleNamespace(
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

            def iterator():
                yield SimpleNamespace(
                    text="Hello",
                    token=1,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stop": ["ignored-stop"],
                    "stop_sequences": ["take-this-stop"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["stop_words"], ["take-this-stop"])

    def test_handle_anthropic_messages_stop_sequences_stops_generation(self):
        url = f"http://localhost:{self.port}/v1/messages"

        def fake_generate(request, generation_args, progress_callback=None):
            ctx = SimpleNamespace(
                has_thinking=False,
                think_start_id=-1,
                think_end_id=-1,
                think_end="",
                has_tool_calling=False,
                tool_call_start="",
                tool_call_end="",
                eos_token_ids=set(),
                stop_token_sequences=[[2]],
                prompt=[1, 2, 3],
                stop=lambda: None,
            )

            def iterator():
                yield SimpleNamespace(
                    text="Hello ",
                    token=1,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="STOP",
                    token=2,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="ignored",
                    token=3,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stop_sequences": ["STOP"],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.text)
        text_blocks = [
            block["text"] for block in body["content"] if block.get("type") == "text"
        ]
        self.assertEqual(body["stop_reason"], "stop_sequence")
        self.assertEqual(body["stop_sequence"], "STOP")
        self.assertEqual("".join(text_blocks), "Hello ")

    def test_handle_anthropic_messages_defaults_model_and_max_tokens(self):
        url = f"http://localhost:{self.port}/v1/messages"
        captured = {"model": None, "max_tokens": None}

        def fake_generate(request, generation_args, progress_callback=None):
            captured["model"] = generation_args.model.model
            captured["max_tokens"] = generation_args.max_tokens
            ctx = SimpleNamespace(
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

            def iterator():
                yield SimpleNamespace(
                    text="Hello",
                    token=1,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.text)
        self.assertEqual(body["model"], "default_model")
        self.assertEqual(captured["model"], "default_model")
        self.assertEqual(
            captured["max_tokens"], self.response_generator.cli_args.max_tokens
        )

    def test_handle_anthropic_messages_non_stream_generate_error_uses_anthropic_schema(
        self,
    ):
        url = f"http://localhost:{self.port}/v1/messages"
        with patch.object(
            self.response_generator,
            "generate",
            side_effect=RuntimeError("generation failed"),
        ):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

        self.assertEqual(response.status_code, 404)
        body = json.loads(response.text)
        self._assert_anthropic_error(
            body,
            error_type="api_error",
            message_substring="generation failed",
        )

    def test_handle_anthropic_messages_stream_generate_error_uses_anthropic_schema(
        self,
    ):
        url = f"http://localhost:{self.port}/v1/messages"
        with patch.object(
            self.response_generator,
            "generate",
            side_effect=RuntimeError("generation failed"),
        ):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

        self.assertEqual(response.status_code, 404)
        body = json.loads(response.text)
        self._assert_anthropic_error(
            body,
            error_type="api_error",
            message_substring="generation failed",
        )

    def test_convert_anthropic_messages_with_tool_blocks(self):
        from mlx_lm.server import convert_anthropic_messages

        converted = convert_anthropic_messages(
            {
                "messages": [
                    {"role": "user", "content": "Check weather."},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Looking it up."},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "get_weather",
                                "input": {"location": "sf"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "72F",
                            }
                        ],
                    },
                ]
            }
        )

        self.assertEqual(converted[0], {"role": "user", "content": "Check weather."})
        self.assertEqual(converted[1]["role"], "assistant")
        self.assertEqual(converted[1]["content"], "Looking it up.")
        self.assertEqual(converted[1]["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(
            converted[1]["tool_calls"][0]["function"]["name"], "get_weather"
        )
        self.assertEqual(converted[2]["role"], "tool")
        self.assertEqual(converted[2]["tool_call_id"], "toolu_1")
        self.assertEqual(converted[2]["content"], "72F")

    def test_convert_anthropic_messages_with_system_string(self):
        from mlx_lm.server import convert_anthropic_messages

        converted = convert_anthropic_messages(
            {
                "system": "You are concise.",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        self.assertEqual(converted[0], {"role": "system", "content": "You are concise."})
        self.assertEqual(converted[1], {"role": "user", "content": "Hello"})

    def test_convert_anthropic_messages_merges_adjacent_user_messages(self):
        from mlx_lm.server import convert_anthropic_messages

        converted = convert_anthropic_messages(
            {
                "messages": [
                    {"role": "user", "content": "Hello "},
                    {"role": "user", "content": "there"},
                    {"role": "assistant", "content": "General Kenobi"},
                ]
            }
        )

        self.assertEqual(
            converted,
            [
                {"role": "user", "content": "Hello there"},
                {"role": "assistant", "content": "General Kenobi"},
            ],
        )

    def test_convert_anthropic_messages_interleaved_tool_result_and_text(self):
        from mlx_lm.server import convert_anthropic_messages

        converted = convert_anthropic_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Before "},
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "Result1",
                            },
                            {"type": "text", "text": "After1 "},
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_2",
                                "content": [{"type": "text", "text": "Result2"}],
                            },
                            {"type": "text", "text": "After2"},
                        ],
                    }
                ]
            }
        )

        self.assertEqual(
            converted,
            [
                {"role": "user", "content": "Before "},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "Result1"},
                {"role": "user", "content": "After1 "},
                {"role": "tool", "tool_call_id": "toolu_2", "content": "Result2"},
                {"role": "user", "content": "After2"},
            ],
        )

    def test_convert_anthropic_messages_multi_turn_text(self):
        from mlx_lm.server import convert_anthropic_messages

        converted = convert_anthropic_messages(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "How are you?"},
                ]
            }
        )

        self.assertEqual(
            converted,
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ],
        )

    def test_convert_anthropic_tools(self):
        from mlx_lm.server import convert_anthropic_tools

        converted = convert_anthropic_tools(
            [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ]
        )
        self.assertEqual(converted[0]["type"], "function")
        self.assertEqual(converted[0]["function"]["name"], "get_weather")
        self.assertIn("parameters", converted[0]["function"])

    def test_convert_anthropic_tools_none(self):
        from mlx_lm.server import convert_anthropic_tools

        self.assertIsNone(convert_anthropic_tools(None))

    def test_convert_anthropic_tools_empty_list(self):
        from mlx_lm.server import convert_anthropic_tools

        self.assertEqual(convert_anthropic_tools([]), [])

    def test_convert_anthropic_tools_rejects_unsupported_type(self):
        from mlx_lm.server import convert_anthropic_tools

        with self.assertRaisesRegex(ValueError, "Unsupported tool type"):
            convert_anthropic_tools(
                [{"type": "server", "name": "get_weather", "input_schema": {}}]
            )

    def test_handle_anthropic_messages_with_blocks_and_system(self):
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "system": [{"type": "text", "text": "You are a helpful assistant."}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Say hi."}],
                }
            ],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertEqual(response_body["type"], "message")
        self.assertEqual(response_body["role"], "assistant")
        self.assertEqual(response_body["content"][0]["type"], "text")

    def test_handle_anthropic_messages_with_tools_non_stream(self):
        url = f"http://localhost:{self.port}/v1/messages"

        def fake_generate(request, generation_args, progress_callback=None):
            ctx = SimpleNamespace(
                has_thinking=False,
                think_start_id=-1,
                think_end_id=-1,
                think_end="",
                has_tool_calling=True,
                tool_call_start="<tool_call>",
                tool_call_end="</tool_call>",
                tool_parser=lambda text, tools: json.loads(text),
                eos_token_ids=set(),
                stop_token_sequences=[],
                prompt=[1, 2, 3],
                stop=lambda: None,
            )

            def iterator():
                yield SimpleNamespace(
                    text="<tool_call>",
                    token=1,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text='{"name":"get_weather","arguments":{"location":"sf"}}',
                    token=2,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="</tool_call>",
                    token=3,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Use a tool"}],
                    "tools": [
                        {
                            "name": "get_weather",
                            "input_schema": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.text)
        self.assertEqual(body["stop_reason"], "tool_use")
        self.assertEqual(body["content"][0]["type"], "tool_use")
        self.assertEqual(body["content"][0]["name"], "get_weather")
        self.assertTrue(body["content"][0]["id"].startswith("toolu_"))
        self.assertEqual(body["content"][0]["input"], {"location": "sf"})

    def test_handle_anthropic_messages_streaming(self):
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        response = requests.post(url, json=post_data, stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("anthropic-version"), "2023-06-01")
        events = self._collect_sse_events(response)

        event_names = [event for event, _ in events]
        visible_event_names = [event for event in event_names if event != "ping"]
        self.assertEqual(visible_event_names[0], "message_start")
        self.assertEqual(visible_event_names[1], "content_block_start")
        self.assertIn("content_block_stop", visible_event_names)
        self.assertEqual(visible_event_names[-2], "message_delta")
        self.assertEqual(visible_event_names[-1], "message_stop")
        self.assertLess(
            visible_event_names.index("content_block_stop"),
            visible_event_names.index("message_delta"),
        )

        message_start = events[0][1]
        self.assertEqual(message_start["type"], "message_start")
        self.assertEqual(message_start["message"]["type"], "message")
        self.assertEqual(message_start["message"]["role"], "assistant")
        self.assertEqual(message_start["message"]["usage"]["output_tokens"], 0)

        message_delta = events[-2][1]
        self.assertEqual(message_delta["type"], "message_delta")
        self.assertIn(
            message_delta["delta"]["stop_reason"],
            {"end_turn", "max_tokens", "stop_sequence"},
        )
        self.assertIn("stop_sequence", message_delta["delta"])
        self.assertIn("output_tokens", message_delta["usage"])
        self.assertGreaterEqual(message_delta["usage"]["output_tokens"], 0)

    def test_handle_anthropic_messages_streaming_tool_use(self):
        url = f"http://localhost:{self.port}/v1/messages"

        def fake_generate(request, generation_args, progress_callback=None):
            ctx = SimpleNamespace(
                has_thinking=False,
                think_start_id=-1,
                think_end_id=-1,
                think_end="",
                has_tool_calling=True,
                tool_call_start="<tool_call>",
                tool_call_end="</tool_call>",
                tool_parser=lambda text, tools: json.loads(text),
                eos_token_ids=set(),
                stop_token_sequences=[],
                prompt=[1, 2, 3],
                stop=lambda: None,
            )

            def iterator():
                yield SimpleNamespace(
                    text="<tool_call>",
                    token=1,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text='{"name":"get_weather","arguments":{"location":"sf"}}',
                    token=2,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="</tool_call>",
                    token=3,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Use a tool"}],
                    "tools": [
                        {
                            "name": "get_weather",
                            "input_schema": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                            },
                        }
                    ],
                },
                stream=True,
            )
            self.assertEqual(response.status_code, 200)
            events = self._collect_sse_events(response)

        tool_starts = [
            payload
            for event, payload in events
            if event == "content_block_start"
            and payload["content_block"]["type"] == "tool_use"
        ]
        self.assertEqual(len(tool_starts), 1)
        self.assertEqual(tool_starts[0]["content_block"]["name"], "get_weather")
        self.assertTrue(tool_starts[0]["content_block"]["id"].startswith("toolu_"))
        self.assertEqual(tool_starts[0]["content_block"]["input"], {})
        tool_deltas = [
            payload
            for event, payload in events
            if event == "content_block_delta"
            and payload["delta"]["type"] == "input_json_delta"
        ]
        self.assertEqual(len(tool_deltas), 1)
        self.assertEqual(
            json.loads(tool_deltas[0]["delta"]["partial_json"]), {"location": "sf"}
        )
        message_delta = [payload for event, payload in events if event == "message_delta"][
            -1
        ]
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_handle_anthropic_messages_non_stream_interleaved_text_tool_order(self):
        url = f"http://localhost:{self.port}/v1/messages"
        fake_generate = self._make_fake_tool_generate(
            [
                "Before ",
                "<tool_call>",
                '{"name":"get_weather","arguments":{"location":"sf"}}',
                "</tool_call>",
                "After",
            ],
            lambda text, tools: json.loads(text),
        )

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(url, json=self._anthropic_tool_request())

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.text)
        self.assertEqual(
            [block["type"] for block in body["content"]], ["text", "tool_use", "text"]
        )
        self.assertEqual(body["content"][0]["text"], "Before ")
        self.assertEqual(body["content"][1]["name"], "get_weather")
        self.assertEqual(body["content"][2]["text"], "After")
        self.assertEqual(body["stop_reason"], "tool_use")

    def test_handle_anthropic_messages_streaming_interleaved_text_tool_order(self):
        url = f"http://localhost:{self.port}/v1/messages"
        fake_generate = self._make_fake_tool_generate(
            [
                "Before ",
                "<tool_call>",
                '{"name":"get_weather","arguments":{"location":"sf"}}',
                "</tool_call>",
                "After",
            ],
            lambda text, tools: json.loads(text),
        )

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json=self._anthropic_tool_request(stream=True),
                stream=True,
            )
            self.assertEqual(response.status_code, 200)
            events = self._collect_sse_events(response)

        block_start_types = [
            payload["content_block"]["type"]
            for event, payload in events
            if event == "content_block_start"
        ]
        self.assertEqual(block_start_types, ["text", "tool_use", "text"])
        message_delta = [payload for event, payload in events if event == "message_delta"][
            -1
        ]
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_handle_anthropic_messages_non_stream_tool_parse_error_returns_400(self):
        url = f"http://localhost:{self.port}/v1/messages"

        def bad_parser(text, tools):
            raise ValueError("bad tool call json")

        fake_generate = self._make_fake_tool_generate(
            ["<tool_call>", "not-json", "</tool_call>"],
            bad_parser,
        )

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(url, json=self._anthropic_tool_request())

        self.assertEqual(response.status_code, 400)
        body = json.loads(response.text)
        self._assert_anthropic_error(
            body,
            error_type="invalid_request_error",
            message_substring="bad tool call json",
        )

    def test_handle_anthropic_messages_streaming_tool_parse_error_emits_error_event(self):
        url = f"http://localhost:{self.port}/v1/messages"

        def bad_parser(text, tools):
            raise ValueError("bad tool call json")

        fake_generate = self._make_fake_tool_generate(
            ["<tool_call>", "not-json", "</tool_call>"],
            bad_parser,
        )

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json=self._anthropic_tool_request(stream=True),
                stream=True,
            )
            self.assertEqual(response.status_code, 200)
            events = self._collect_sse_events(response)

        self.assertTrue(any(event == "error" for event, _ in events))
        error_payload = [payload for event, payload in events if event == "error"][-1]
        self.assertEqual(error_payload["type"], "error")
        self.assertIn("bad tool call json", error_payload["error"]["message"])
        # Stream terminates on error without a trailing message_stop, matching
        # the real Anthropic API behavior.
        self.assertNotEqual(events[-1][0], "message_stop")

    def test_handle_anthropic_messages_no_tool_use_stop_reason_when_tool_list_empty(self):
        url = f"http://localhost:{self.port}/v1/messages"
        fake_generate = self._make_fake_tool_generate(
            ["<tool_call>", '{"name":"noop","arguments":{}}', "</tool_call>"],
            lambda text, tools: [],
        )

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(url, json=self._anthropic_tool_request())

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.text)
        self.assertNotEqual(body["stop_reason"], "tool_use")
        self.assertEqual(body["content"], [{"type": "text", "text": ""}])

    def test_handle_anthropic_messages_rejects_non_text_content(self):
        url = f"http://localhost:{self.port}/v1/messages"
        post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/a.png",
                            },
                        }
                    ],
                }
            ],
        }
        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 400)
        response_body = json.loads(response.text)
        self._assert_anthropic_error(response_body, error_type="invalid_request_error")

    def test_handle_anthropic_messages_non_stream_text_completion(self):
        url = f"http://localhost:{self.port}/v1/messages"
        response = requests.post(
            url,
            json={
                "model": "chat_model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.text)
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertIn("content", body)
        self.assertGreater(len(body["content"]), 0)
        self.assertEqual(body["content"][0]["type"], "text")
        self.assertIsInstance(body["content"][0]["text"], str)
        self.assertIn(body["stop_reason"], {"end_turn", "max_tokens", "stop_sequence"})
        self.assertIn("usage", body)
        self.assertIn("input_tokens", body["usage"])
        self.assertIn("output_tokens", body["usage"])

    def test_handle_anthropic_messages_malformed_request_returns_anthropic_error(self):
        url = f"http://localhost:{self.port}/v1/messages"
        # Missing required "messages" field triggers an error during request
        # construction; verify we get back an Anthropic-shaped error envelope.
        response = requests.post(
            url,
            json={
                "model": "chat_model",
                "max_tokens": 10,
            },
        )

        self.assertIn(response.status_code, {400, 404})
        body = json.loads(response.text)
        self._assert_anthropic_error(body)

    def test_handle_models(self):
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["object"], "list")
        self.assertIsInstance(response_body["data"], list)
        self.assertGreater(len(response_body["data"]), 0)
        model = response_body["data"][0]
        self.assertIn("id", model)
        self.assertEqual(model["object"], "model")
        self.assertIn("created", model)

    def test_sequence_overlap(self):
        from mlx_lm.server import sequence_overlap

        self.assertTrue(sequence_overlap([1], [1]))
        self.assertTrue(sequence_overlap([1, 2], [1, 2]))
        self.assertTrue(sequence_overlap([1, 3], [3, 4]))
        self.assertTrue(sequence_overlap([1, 2, 3], [2, 3]))

        self.assertFalse(sequence_overlap([1], [2]))
        self.assertFalse(sequence_overlap([1, 2], [3, 4]))
        self.assertFalse(sequence_overlap([1, 2, 3], [4, 1, 2, 3]))

    def test_trim_visible_stop_text(self):
        from mlx_lm.server import _trim_visible_stop_text

        self.assertEqual(_trim_visible_stop_text("hello STOP", "STOP", 4), "hello ")
        self.assertEqual(_trim_visible_stop_text("hello", "STOP", 4), "hello")
        self.assertEqual(_trim_visible_stop_text("hello", None, 4), "hello")

    def test_matched_stop_sequence(self):
        matched = APIHandler._matched_stop_sequence(
            tokens=[1, 2, 3],
            stop_id_sequences=[[9], [2, 3]],
            stop_words=["x", "STOP"],
        )
        self.assertEqual(matched, "STOP")

        unmatched = APIHandler._matched_stop_sequence(
            tokens=[1, 2, 3],
            stop_id_sequences=[[4, 5]],
            stop_words=["NOPE"],
        )
        self.assertIsNone(unmatched)

    def test_handle_anthropic_messages_streaming_uses_progress_callback(self):
        url = f"http://localhost:{self.port}/v1/messages"
        captured = {"called": False, "has_callback": False}

        def fake_generate(request, generation_args, progress_callback=None):
            captured["called"] = True
            captured["has_callback"] = progress_callback is not None
            ctx = SimpleNamespace(
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
            )

            def iterator():
                yield SimpleNamespace(
                    text="Hello",
                    token=1,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
                stream=True,
            )
            self.assertEqual(response.status_code, 200)
            for _ in response.iter_lines():
                pass

        self.assertTrue(captured["called"])
        self.assertTrue(captured["has_callback"])

    def test_handle_anthropic_messages_non_stream_uses_progress_callback(self):
        url = f"http://localhost:{self.port}/v1/messages"
        captured = {"called": False, "has_callback": False, "callback_name": None}

        def fake_generate(request, generation_args, progress_callback=None):
            captured["called"] = True
            captured["has_callback"] = progress_callback is not None
            captured["callback_name"] = getattr(progress_callback, "__name__", None)
            ctx = SimpleNamespace(
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

            def iterator():
                yield SimpleNamespace(
                    text="Hello",
                    token=1,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )
            self.assertEqual(response.status_code, 200)

        self.assertTrue(captured["called"])
        self.assertTrue(captured["has_callback"])
        self.assertEqual(captured["callback_name"], "_anthropic_keepalive_callback")

    def test_handle_anthropic_messages_non_stream_hides_thinking_text(self):
        url = f"http://localhost:{self.port}/v1/messages"

        def fake_generate(request, generation_args, progress_callback=None):
            ctx = SimpleNamespace(
                has_thinking=True,
                think_start_id=11,
                think_end_id=12,
                think_end="</think>",
                has_tool_calling=False,
                tool_call_start="",
                tool_call_end="",
                eos_token_ids=set(),
                stop_token_sequences=[],
                prompt=[11],
                stop=lambda: None,
            )

            def iterator():
                yield SimpleNamespace(
                    text="internal reasoning",
                    token=1,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="</think>",
                    token=2,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="Hello",
                    token=3,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        text_blocks = [
            block["text"]
            for block in response_body["content"]
            if block.get("type") == "text"
        ]
        self.assertEqual("".join(text_blocks), "Hello")
        self.assertNotIn("internal reasoning", "".join(text_blocks))

    def test_handle_anthropic_messages_streaming_hides_thinking_text(self):
        url = f"http://localhost:{self.port}/v1/messages"

        def fake_generate(request, generation_args, progress_callback=None):
            ctx = SimpleNamespace(
                has_thinking=True,
                think_start_id=11,
                think_end_id=12,
                think_end="</think>",
                has_tool_calling=False,
                tool_call_start="",
                tool_call_end="",
                eos_token_ids=set(),
                stop_token_sequences=[],
                prompt=[11],
                stop=lambda: None,
            )

            def iterator():
                yield SimpleNamespace(
                    text="internal reasoning",
                    token=1,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="</think>",
                    token=2,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="Hello",
                    token=3,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
                stream=True,
            )
            self.assertEqual(response.status_code, 200)
            events = self._collect_sse_events(response)

        text_deltas = [
            payload["delta"]["text"]
            for event, payload in events
            if event == "content_block_delta"
            and payload.get("delta", {}).get("type") == "text_delta"
        ]
        self.assertEqual("".join(text_deltas), "Hello")
        self.assertNotIn("internal reasoning", "".join(text_deltas))

    def test_handle_anthropic_messages_streaming_hidden_keepalive(self):
        url = f"http://localhost:{self.port}/v1/messages"

        def fake_generate(request, generation_args, progress_callback=None):
            ctx = SimpleNamespace(
                has_thinking=False,
                think_start_id=-1,
                think_end_id=-1,
                think_end="",
                has_tool_calling=True,
                tool_call_start="<tool_call>",
                tool_call_end="</tool_call>",
                tool_parser=lambda text, tools: {"name": "x", "arguments": {}},
                eos_token_ids=set(),
                stop_token_sequences=[],
                prompt=[1, 2, 3],
                stop=lambda: None,
            )

            def iterator():
                yield SimpleNamespace(
                    text="<tool_call>",
                    token=1,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text='{"name":"x"}',
                    token=2,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="</tool_call>",
                    token=3,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="Hello",
                    token=4,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
                stream=True,
            )
            self.assertEqual(response.status_code, 200)
            events = self._collect_sse_events(response)

        self.assertTrue(any(event == "ping" for event, _ in events))
        self.assertFalse(any(event == "error" for event, _ in events))
        text_deltas = [
            payload["delta"]["text"]
            for event, payload in events
            if event == "content_block_delta"
            and payload.get("delta", {}).get("type") == "text_delta"
        ]
        self.assertIn("Hello", "".join(text_deltas))

    def test_handle_chat_completions_streaming_hidden_keepalive(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        def fake_generate(request, generation_args, progress_callback=None):
            ctx = SimpleNamespace(
                has_tool_calling=True,
                tool_call_start="<tool_call>",
                tool_call_end="</tool_call>",
                tool_parser=lambda tool_text, tools: {"name": "x", "arguments": {}},
                has_thinking=False,
                think_start_id=-1,
                think_end="",
                think_end_id=-1,
                eos_token_ids=set(),
                stop_token_sequences=[],
                prompt=[1, 2, 3],
                stop=lambda: None,
            )

            def iterator():
                yield SimpleNamespace(
                    text="<tool_call>",
                    token=1,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text='{"name":"x"}',
                    token=2,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="</tool_call>",
                    token=3,
                    logprob=0.0,
                    finish_reason=None,
                    top_tokens=(),
                )
                yield SimpleNamespace(
                    text="Done",
                    token=4,
                    logprob=0.0,
                    finish_reason="stop",
                    top_tokens=(),
                )

            return ctx, iterator()

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = requests.post(
                url,
                json={
                    "model": "chat_model",
                    "max_tokens": 10,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
                stream=True,
            )
            self.assertEqual(response.status_code, 200)
            lines = [line.decode("utf-8") for line in response.iter_lines() if line]

        self.assertTrue(any(line.startswith(": keepalive hidden") for line in lines))


class TestServerWithDraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(with_draft=True), LRUPromptCache()
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

    def test_handle_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_handle_chat_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_streaming_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk_data = json.loads(data[6:])  # Skip the "data: " prefix
                    self.assertIn("choices", chunk_data)
                    self.assertEqual(len(chunk_data["choices"]), 1)
                    self.assertIn("delta", chunk_data["choices"][0])
                    chunk_count += 1

        # Make sure we got some streaming chunks
        self.assertGreater(chunk_count, 0)

    def test_prompt_cache_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        # First request to initialize cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about"},
            ],
        }

        first_response = requests.post(url, json=chat_post_data)
        self.assertEqual(first_response.status_code, 200)

        # Second request with same prefix should use cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about dragons."},
            ],
        }

        second_response = requests.post(url, json=chat_post_data)
        self.assertEqual(second_response.status_code, 200)

        # Both responses should have content
        first_response_body = json.loads(first_response.text)
        second_response_body = json.loads(second_response.text)

        self.assertIn("choices", first_response_body)
        self.assertIn("choices", second_response_body)
        self.assertIn("message", first_response_body["choices"][0])
        self.assertIn("message", second_response_body["choices"][0])
        self.assertIn("content", first_response_body["choices"][0]["message"])
        self.assertIn("content", second_response_body["choices"][0]["message"])

        # Ensure both generated content
        self.assertIsNotNone(first_response_body["choices"][0]["message"]["content"])
        self.assertIsNotNone(second_response_body["choices"][0]["message"]["content"])


class TestKeepalive(unittest.TestCase):
    def test_keepalive_callback(self):
        """Test keepalive callback sends SSE comments and handles errors"""
        from unittest.mock import Mock

        # Mock handler-like object
        mock_wfile = io.BytesIO()
        handler = Mock(spec=["stream", "wfile", "_write_sse_keepalive"])
        handler.wfile = mock_wfile
        handler._write_sse_keepalive = lambda message: APIHandler._write_sse_keepalive(
            handler, message
        )

        # Test streaming enabled
        handler.stream = True
        APIHandler._keepalive_callback(handler, 1024, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, ": keepalive 1024/4096\n\n")

        # Test streaming disabled
        handler.stream = False
        mock_wfile.seek(0)
        mock_wfile.truncate(0)
        APIHandler._keepalive_callback(handler, 2048, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, "")

        # Test error handling
        handler.stream = True
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError("Connection broken")

        # Should not raise exception
        try:
            APIHandler._keepalive_callback(handler, 3072, 4096)
        except Exception as e:
            self.fail(f"Callback should handle BrokenPipeError: {e}")


class TestLRUPromptCache(unittest.TestCase):
    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertTrue(c is None)
        self.assertEqual(t, tokens)

        c = [KVCache()]
        c[0].update_and_fetch(*get_kv(24))
        cache.insert_cache(model, t, c)

        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache._lru), 0)

        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)

        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)
        self.assertEqual(len(cache._lru), 1)

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [1, 2], [MockCache("test1")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])
        cache.insert_cache(model, [3, 4], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])

    def test_lru_bytes(self):
        cache = LRUPromptCache(max_size=100, max_bytes=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("aaa")])
        cache.insert_cache(model, [3, 4], [MockCache("bbb")])
        cache.insert_cache(model, [4, 5], [MockCache("ccc")])
        cache.insert_cache(model, [6, 7], [MockCache("ddd")])

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.nbytes, 9)

        cache.trim_to(n_bytes=7)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 6)

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, None)
        self.assertEqual(t, [3, 4])


if __name__ == "__main__":
    unittest.main()
