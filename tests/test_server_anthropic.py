# Copyright © 2024 Apple Inc.

import json
import unittest
from unittest.mock import patch

import requests

import mlx_lm.server_anthropic as server_anthropic
from mlx_lm.server_anthropic import convert_anthropic_messages, convert_anthropic_tools
from mlx_lm.server_common import stopping_criteria, trim_visible_stop_text
from tests._server_test_utils import (
    ServerAPITestBase,
    collect_sse_events,
    event_payloads,
    make_fake_generate,
    make_fake_tool_generate,
    message_delta,
    text_deltas,
    text_from_content_blocks,
    tool_use_delta_payloads,
    tool_use_start_payloads,
    visible_event_names,
)


class TestAnthropicServer(ServerAPITestBase, unittest.TestCase):

    # Test helpers

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

    @staticmethod
    def _basic_request(message="Hello!", stream=False):
        body = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": message}],
        }
        if stream:
            body["stream"] = True
        return body

    def _messages_url(self, query=""):
        url = f"http://localhost:{self.port}/v1/messages"
        if query:
            return f"{url}?{query}"
        return url

    def _post_messages(self, body, *, stream=False, query=""):
        return requests.post(self._messages_url(query), json=body, stream=stream)

    def _post_raw(self, data, headers):
        return requests.post(self._messages_url(), data=data, headers=headers)

    @staticmethod
    def _json_body(response):
        return json.loads(response.text)

    def _assert_anthropic_error(self, body, error_type=None, message_substring=None):
        self.assertEqual(body["type"], "error")
        self.assertIn("error", body)
        self.assertIsInstance(body["error"], dict)
        if error_type is not None:
            self.assertEqual(body["error"]["type"], error_type)
        if message_substring is not None:
            self.assertIn(message_substring, body["error"]["message"])

    def _assert_tool_use_starts(self, starts, expected_name, expected_count):
        self.assertEqual(len(starts), expected_count)
        for payload in starts:
            self.assertEqual(payload["content_block"]["name"], expected_name)
            self.assertTrue(payload["content_block"]["id"].startswith("toolu_"))
            self.assertEqual(payload["content_block"]["input"], {})

    def _assert_stream_tool_inputs(self, events, expected_inputs):
        starts = tool_use_start_payloads(events)
        self._assert_tool_use_starts(starts, expected_name="get_weather", expected_count=len(expected_inputs))
        deltas = tool_use_delta_payloads(events)
        self.assertEqual(len(deltas), len(expected_inputs))
        for delta, expected in zip(deltas, expected_inputs):
            self.assertEqual(json.loads(delta["delta"]["partial_json"]), expected)

    def _assert_non_stream_tool_inputs(self, body, expected_inputs):
        tool_blocks = [block for block in body["content"] if block["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), len(expected_inputs))
        for block, expected in zip(tool_blocks, expected_inputs):
            self.assertEqual(block["name"], "get_weather")
            self.assertTrue(block["id"].startswith("toolu_"))
            self.assertEqual(block["input"], expected)

    # Endpoint tests

    def test_handle_anthropic_messages(self):
        response = self._post_messages(self._basic_request())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("anthropic-version"), "2023-06-01")

        body = self._json_body(response)
        self.assertTrue(body["id"].startswith("msg_"))
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertEqual(body["model"], "chat_model")
        self.assertIsInstance(body["content"], list)
        self.assertEqual(body["content"][0]["type"], "text")
        self.assertIn("text", body["content"][0])
        self.assertIn(body["stop_reason"], {"end_turn", "max_tokens", "stop_sequence"})
        self.assertIn("stop_sequence", body)
        self.assertIn("usage", body)
        self.assertIn("input_tokens", body["usage"])
        self.assertIn("output_tokens", body["usage"])

    def test_handle_anthropic_messages_requires_messages(self):
        response = self._post_messages({"model": "chat_model", "max_tokens": 10})
        self.assertEqual(response.status_code, 400)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="invalid_request_error",
            message_substring="messages",
        )

    def test_handle_anthropic_messages_requires_object_body(self):
        response = self._post_messages(["not", "an", "object"])
        self.assertEqual(response.status_code, 400)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="invalid_request_error",
            message_substring="JSON object",
        )

    def test_handle_anthropic_messages_with_query_string(self):
        response = self._post_messages(
            self._basic_request(),
            query="beta=true&anthropic-version=2023-06-01",
        )
        self.assertEqual(response.status_code, 200)
        body = self._json_body(response)
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")

    def test_handle_anthropic_messages_rejects_text_block_missing_text_field(self):
        response = self._post_messages(
            {
                "model": "chat_model",
                "max_tokens": 10,
                "system": [{"type": "text"}],
                "messages": [{"role": "user", "content": "Hello!"}],
            }
        )
        self.assertEqual(response.status_code, 400)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="invalid_request_error",
            message_substring="valid `text`",
        )

    def test_handle_anthropic_messages_uses_stop_sequences_field(self):
        captured = {"stop_words": None}

        def on_call(req, args, cb):
            captured["stop_words"] = args.stop_words

        fake = make_fake_generate(["Hello"], on_call=on_call)

        with patch.object(self.response_generator, "generate", side_effect=fake):
            response = self._post_messages(
                {
                    "model": "chat_model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stop": ["ignored-stop"],
                    "stop_sequences": ["take-this-stop"],
                }
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["stop_words"], ["take-this-stop"])

    def test_handle_anthropic_messages_stop_sequences_stops_generation(self):
        fake = make_fake_generate(
            ["Hello ", "STOP", "ignored"],
            stop_token_sequences=[[2]],
        )

        with patch.object(self.response_generator, "generate", side_effect=fake):
            response = self._post_messages(
                {
                    "model": "chat_model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stop_sequences": ["STOP"],
                }
            )

        self.assertEqual(response.status_code, 200)
        body = self._json_body(response)
        self.assertEqual(body["stop_reason"], "stop_sequence")
        self.assertEqual(body["stop_sequence"], "STOP")
        self.assertEqual(text_from_content_blocks(body["content"]), "Hello ")

    def test_handle_anthropic_messages_defaults_model_and_max_tokens(self):
        captured = {"model": None, "max_tokens": None}

        def on_call(req, args, cb):
            captured["model"] = args.model.model
            captured["max_tokens"] = args.max_tokens

        fake = make_fake_generate(["Hello"], on_call=on_call)

        with patch.object(self.response_generator, "generate", side_effect=fake):
            response = self._post_messages(
                {
                    "messages": [{"role": "user", "content": "Hello!"}],
                }
            )

        self.assertEqual(response.status_code, 200)
        body = self._json_body(response)
        self.assertEqual(body["model"], "default_model")
        self.assertEqual(captured["model"], "default_model")
        self.assertEqual(captured["max_tokens"], self.response_generator.cli_args.max_tokens)

    def test_handle_anthropic_messages_generate_error_uses_anthropic_schema(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                request_body = self._basic_request(stream=stream)
                with patch.object(
                    self.response_generator,
                    "generate",
                    side_effect=RuntimeError("generation failed"),
                ):
                    response = self._post_messages(request_body)

                self.assertEqual(response.status_code, 404)
                self._assert_anthropic_error(
                    self._json_body(response),
                    error_type="api_error",
                    message_substring="generation failed",
                )

    def test_handle_anthropic_messages_unexpected_loop_error_is_server_error(self):
        fake = make_fake_generate(["Hello"])

        with patch.object(self.response_generator, "generate", side_effect=fake):
            with patch.object(
                server_anthropic,
                "run_generation_loop",
                side_effect=RuntimeError("unexpected"),
            ):
                response = self._post_messages(self._basic_request())

        self.assertEqual(response.status_code, 500)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="api_error",
            message_substring="Internal server error",
        )

    def test_handle_anthropic_messages_with_blocks_and_system(self):
        response = self._post_messages(
            {
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
        )
        self.assertEqual(response.status_code, 200)
        body = self._json_body(response)
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertEqual(body["content"][0]["type"], "text")

    def test_handle_anthropic_messages_tool_use(self):
        fake = make_fake_tool_generate(
            [
                "<tool_call>",
                '{"name":"get_weather","arguments":{"location":"sf"}}',
                "</tool_call>",
            ],
            lambda text, tools: json.loads(text),
        )

        for stream in (False, True):
            with self.subTest(stream=stream):
                with patch.object(self.response_generator, "generate", side_effect=fake):
                    response = self._post_messages(
                        self._anthropic_tool_request(stream=stream),
                        stream=stream,
                    )
                self.assertEqual(response.status_code, 200)

                if stream:
                    events = collect_sse_events(response)
                    self._assert_stream_tool_inputs(events, [{"location": "sf"}])
                    self.assertEqual(message_delta(events)["delta"]["stop_reason"], "tool_use")
                    continue

                body = self._json_body(response)
                self.assertEqual(body["stop_reason"], "tool_use")
                self._assert_non_stream_tool_inputs(body, [{"location": "sf"}])

    def test_handle_anthropic_messages_streaming(self):
        response = self._post_messages(self._basic_request(stream=True), stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("anthropic-version"), "2023-06-01")

        events = collect_sse_events(response)
        event_names = visible_event_names(events)
        self.assertEqual(event_names[0], "message_start")
        self.assertEqual(event_names[1], "content_block_start")
        self.assertIn("content_block_stop", event_names)
        self.assertEqual(event_names[-2], "message_delta")
        self.assertEqual(event_names[-1], "message_stop")
        self.assertLess(
            event_names.index("content_block_stop"),
            event_names.index("message_delta"),
        )

        message_start = events[0][1]
        self.assertEqual(message_start["type"], "message_start")
        self.assertEqual(message_start["message"]["type"], "message")
        self.assertEqual(message_start["message"]["role"], "assistant")
        self.assertEqual(message_start["message"]["usage"]["output_tokens"], 0)

        delta = message_delta(events)
        self.assertEqual(delta["type"], "message_delta")
        self.assertIn(
            delta["delta"]["stop_reason"],
            {"end_turn", "max_tokens", "stop_sequence"},
        )
        self.assertIn("stop_sequence", delta["delta"])
        self.assertIn("output_tokens", delta["usage"])
        self.assertGreaterEqual(delta["usage"]["output_tokens"], 0)

    def test_handle_anthropic_messages_interleaved_text_tool_order(self):
        fake_generate = make_fake_tool_generate(
            [
                "Before ",
                "<tool_call>",
                '{"name":"get_weather","arguments":{"location":"sf"}}',
                "</tool_call>",
                "After",
            ],
            lambda text, tools: json.loads(text),
        )

        for stream in (False, True):
            with self.subTest(stream=stream):
                with patch.object(
                    self.response_generator, "generate", side_effect=fake_generate
                ):
                    response = self._post_messages(
                        self._anthropic_tool_request(stream=stream),
                        stream=stream,
                    )
                    self.assertEqual(response.status_code, 200)

                if stream:
                    events = collect_sse_events(response)
                    block_types = [
                        payload["content_block"]["type"]
                        for event, payload in events
                        if event == "content_block_start"
                    ]
                    self.assertEqual(block_types, ["text", "tool_use", "text"])
                    self.assertEqual(
                        message_delta(events)["delta"]["stop_reason"],
                        "tool_use",
                    )
                else:
                    body = self._json_body(response)
                    self.assertEqual(
                        [block["type"] for block in body["content"]],
                        ["text", "tool_use", "text"],
                    )
                    self.assertEqual(body["content"][0]["text"], "Before ")
                    self.assertEqual(body["content"][1]["name"], "get_weather")
                    self.assertEqual(body["content"][2]["text"], "After")
                    self.assertEqual(body["stop_reason"], "tool_use")

    def test_handle_anthropic_messages_tool_parse_error(self):
        def bad_parser(text, tools):
            raise ValueError("bad tool call json")

        fake_generate = make_fake_tool_generate(
            ["<tool_call>", "not-json", "</tool_call>"],
            bad_parser,
        )

        for stream in (False, True):
            with self.subTest(stream=stream):
                with patch.object(self.response_generator, "generate", side_effect=fake_generate):
                    response = self._post_messages(
                        self._anthropic_tool_request(stream=stream),
                        stream=stream,
                    )

                if not stream:
                    self.assertEqual(response.status_code, 400)
                    self._assert_anthropic_error(
                        self._json_body(response),
                        error_type="invalid_request_error",
                        message_substring="bad tool call json",
                    )
                    continue

                self.assertEqual(response.status_code, 200)
                events = collect_sse_events(response)
                self.assertTrue(any(event == "error" for event, _ in events))
                payload = event_payloads(events, "error")[-1]
                self.assertEqual(payload["type"], "error")
                self.assertIn("bad tool call json", payload["error"]["message"])
                self.assertNotEqual(events[-1][0], "message_stop")

    def test_handle_anthropic_messages_no_tool_use_stop_reason_when_tool_list_empty(self):
        fake_generate = make_fake_tool_generate(
            ["<tool_call>", '{"name":"noop","arguments":{}}', "</tool_call>"],
            lambda text, tools: [],
        )

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = self._post_messages(self._anthropic_tool_request())

        self.assertEqual(response.status_code, 200)
        body = self._json_body(response)
        self.assertNotEqual(body["stop_reason"], "tool_use")
        self.assertEqual(body["content"], [{"type": "text", "text": ""}])

    def test_handle_anthropic_messages_rejects_non_text_content(self):
        response = self._post_messages(
            {
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
        )
        self.assertEqual(response.status_code, 400)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="invalid_request_error",
        )

    def test_handle_anthropic_messages_unexpected_request_factory_error_returns_500(self):
        with patch.object(
            server_anthropic,
            "handle_post",
            side_effect=RuntimeError("unexpected"),
        ):
            response = self._post_messages(self._basic_request())

        self.assertEqual(response.status_code, 500)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="api_error",
            message_substring="Internal server error",
        )

    def test_handle_anthropic_messages_malformed_json_returns_anthropic_error(self):
        response = self._post_raw(
            data=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="invalid_request_error",
            message_substring="Invalid JSON",
        )

    def test_handle_anthropic_messages_multiple_tool_calls(self):
        fake_generate = make_fake_tool_generate(
            [
                "<tool_call>",
                '{"name":"get_weather","arguments":{"location":"sf"}}',
                "</tool_call>",
                "<tool_call>",
                '{"name":"get_weather","arguments":{"location":"nyc"}}',
                "</tool_call>",
            ],
            lambda text, tools: json.loads(text),
        )

        for stream in (False, True):
            with self.subTest(stream=stream):
                with patch.object(
                    self.response_generator, "generate", side_effect=fake_generate
                ):
                    response = self._post_messages(
                        self._anthropic_tool_request(stream=stream),
                        stream=stream,
                    )
                    self.assertEqual(response.status_code, 200)

                if stream:
                    events = collect_sse_events(response)
                    self._assert_stream_tool_inputs(
                        events,
                        [{"location": "sf"}, {"location": "nyc"}],
                    )
                    starts = tool_use_start_payloads(events)
                    self.assertNotEqual(
                        starts[0]["content_block"]["id"],
                        starts[1]["content_block"]["id"],
                    )
                    self.assertEqual(
                        message_delta(events)["delta"]["stop_reason"],
                        "tool_use",
                    )
                else:
                    body = self._json_body(response)
                    self._assert_non_stream_tool_inputs(
                        body,
                        [{"location": "sf"}, {"location": "nyc"}],
                    )
                    tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
                    self.assertNotEqual(tool_blocks[0]["id"], tool_blocks[1]["id"])
                    self.assertEqual(body["stop_reason"], "tool_use")

    def test_handle_anthropic_messages_progress_callback_by_stream_mode(self):
        for stream, expected_callback in ((False, False), (True, True)):
            with self.subTest(stream=stream):
                captured = {"called": False, "has_callback": False}

                def on_call(req, args, cb):
                    captured["called"] = True
                    captured["has_callback"] = cb is not None

                fake = make_fake_generate(["Hello"], on_call=on_call)
                request_body = self._basic_request(stream=stream)

                with patch.object(self.response_generator, "generate", side_effect=fake):
                    response = self._post_messages(request_body, stream=stream)
                    self.assertEqual(response.status_code, 200)
                    if stream:
                        for _ in response.iter_lines():
                            pass

                self.assertTrue(captured["called"])
                self.assertEqual(captured["has_callback"], expected_callback)

    def test_handle_anthropic_messages_hides_thinking_text(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                fake = make_fake_generate(
                    ["internal reasoning", "</think>", "Hello"],
                    has_thinking=True,
                    think_start_id=11,
                    think_end_id=12,
                    think_end="</think>",
                    prompt=[11],
                )
                request_body = self._basic_request(stream=stream)

                with patch.object(self.response_generator, "generate", side_effect=fake):
                    response = self._post_messages(request_body, stream=stream)
                    self.assertEqual(response.status_code, 200)
                    if stream:
                        visible_text = "".join(text_deltas(collect_sse_events(response)))
                    else:
                        visible_text = text_from_content_blocks(
                            self._json_body(response)["content"]
                        )

                self.assertEqual(visible_text, "Hello")
                self.assertNotIn("internal reasoning", visible_text)

    def test_handle_anthropic_messages_streaming_hidden_keepalive(self):
        fake = make_fake_generate(
            ["<tool_call>", '{"name":"x"}', "</tool_call>", "Hello"],
            has_tool_calling=True,
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=lambda text, tools: {"name": "x", "arguments": {}},
        )

        with patch.object(self.response_generator, "generate", side_effect=fake):
            response = self._post_messages(self._basic_request(stream=True), stream=True)
            self.assertEqual(response.status_code, 200)
            events = collect_sse_events(response)

        self.assertTrue(any(event == "ping" for event, _ in events))
        self.assertFalse(any(event == "error" for event, _ in events))
        self.assertIn("Hello", "".join(text_deltas(events)))


class TestAnthropicConversion(unittest.TestCase):
    def test_convert_anthropic_messages_with_tool_blocks(self):
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
        self.assertEqual(converted[1]["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(converted[2]["role"], "tool")
        self.assertEqual(converted[2]["tool_call_id"], "toolu_1")
        self.assertEqual(converted[2]["content"], "72F")

    def test_convert_anthropic_messages_tool_use_only_assistant(self):
        converted = convert_anthropic_messages(
            {
                "messages": [
                    {"role": "user", "content": "Check weather."},
                    {
                        "role": "assistant",
                        "content": [
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

        self.assertEqual(converted[1]["role"], "assistant")
        self.assertEqual(converted[1]["content"], "")
        self.assertEqual(len(converted[1]["tool_calls"]), 1)
        self.assertEqual(converted[1]["tool_calls"][0]["id"], "toolu_1")

    def test_convert_anthropic_messages_tool_result_is_error(self):
        converted = convert_anthropic_messages(
            {
                "messages": [
                    {"role": "user", "content": "Check weather."},
                    {
                        "role": "assistant",
                        "content": [
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
                                "content": "Connection timed out",
                                "is_error": True,
                            }
                        ],
                    },
                ]
            }
        )

        self.assertEqual(converted[2]["role"], "tool")
        self.assertEqual(converted[2]["tool_call_id"], "toolu_1")
        self.assertEqual(converted[2]["content"], "Connection timed out")
        self.assertTrue(converted[2]["is_error"])

    def test_convert_anthropic_messages_with_system_string(self):
        converted = convert_anthropic_messages(
            {
                "system": "You are concise.",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        self.assertEqual(converted[0], {"role": "system", "content": "You are concise."})
        self.assertEqual(converted[1], {"role": "user", "content": "Hello"})

    def test_convert_anthropic_messages_merges_adjacent_user_messages(self):
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
        self.assertIsNone(convert_anthropic_tools(None))

    def test_convert_anthropic_tools_empty_list(self):
        self.assertEqual(convert_anthropic_tools([]), [])

    def test_convert_anthropic_tools_rejects_unsupported_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tool type"):
            convert_anthropic_tools(
                [{"type": "server", "name": "get_weather", "input_schema": {}}]
            )

    def test_convert_anthropic_tools_accepts_custom_type(self):
        converted = convert_anthropic_tools(
            [
                {
                    "type": "custom",
                    "name": "get_weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ]
        )
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["type"], "function")
        self.assertEqual(converted[0]["function"]["name"], "get_weather")


class TestServerCommonUtilities(unittest.TestCase):
    def test_trim_visible_stop_text(self):
        self.assertEqual(trim_visible_stop_text("hello STOP", "STOP", 4), "hello ")
        self.assertEqual(trim_visible_stop_text("hello", "STOP", 4), "hello")
        self.assertEqual(trim_visible_stop_text("hello", None, 4), "hello")

    def test_stopping_criteria_returns_matched_stop_word(self):
        matched = stopping_criteria(
            tokens=[1, 2, 3],
            eos_token_ids=set(),
            stop_id_sequences=[[9], [2, 3]],
            stop_words=["x", "STOP"],
        )
        self.assertTrue(matched.stop_met)
        self.assertEqual(matched.stop_word, "STOP")

        unmatched = stopping_criteria(
            tokens=[1, 2, 3],
            eos_token_ids=set(),
            stop_id_sequences=[[4, 5]],
            stop_words=["NOPE"],
        )
        self.assertFalse(unmatched.stop_met)
        self.assertIsNone(unmatched.stop_word)


if __name__ == "__main__":
    unittest.main()
