# Copyright © 2024 Apple Inc.

import json
import unittest
from unittest.mock import patch

import requests

from mlx_lm.server_anthropic import convert_anthropic_messages, convert_anthropic_tools
from tests._server_test_utils import (
    ServerAPITestBase,
    collect_sse_events,
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

    @staticmethod
    def _tool_request(stream=False):
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

    def _messages_url(self, query=""):
        url = f"http://localhost:{self.port}/v1/messages"
        return f"{url}?{query}" if query else url

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
        if error_type is not None:
            self.assertEqual(body["error"]["type"], error_type)
        if message_substring is not None:
            self.assertIn(message_substring, body["error"]["message"])

    def _assert_stream_tool_inputs(self, events, expected_inputs):
        starts = tool_use_start_payloads(events)
        self.assertEqual(len(starts), len(expected_inputs))
        for payload in starts:
            self.assertEqual(payload["content_block"]["name"], "get_weather")
            self.assertTrue(payload["content_block"]["id"].startswith("toolu_"))
            self.assertEqual(payload["content_block"]["input"], {})

        deltas = tool_use_delta_payloads(events)
        self.assertEqual(len(deltas), len(expected_inputs))
        for delta, expected in zip(deltas, expected_inputs):
            self.assertEqual(json.loads(delta["delta"]["partial_json"]), expected)

    def _assert_non_stream_tool_inputs(self, body, expected_inputs):
        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), len(expected_inputs))
        for block, expected in zip(tool_blocks, expected_inputs):
            self.assertEqual(block["name"], "get_weather")
            self.assertTrue(block["id"].startswith("toolu_"))
            self.assertEqual(block["input"], expected)

    def test_handle_anthropic_messages_non_stream(self):
        response = self._post_messages(self._basic_request())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("anthropic-version"), "2023-06-01")

        body = self._json_body(response)
        self.assertTrue(body["id"].startswith("msg_"))
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertEqual(body["model"], "chat_model")
        self.assertIsInstance(body["content"], list)
        self.assertIn(body["stop_reason"], {"end_turn", "max_tokens", "stop_sequence"})
        self.assertIn("usage", body)
        self.assertIn("input_tokens", body["usage"])
        self.assertIn("output_tokens", body["usage"])

    def test_handle_anthropic_messages_stream(self):
        response = self._post_messages(self._basic_request(stream=True), stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("anthropic-version"), "2023-06-01")

        events = collect_sse_events(response)
        event_names = visible_event_names(events)
        self.assertEqual(event_names[0], "message_start")
        self.assertIn("content_block_start", event_names)
        self.assertIn("content_block_stop", event_names)
        self.assertEqual(event_names[-2], "message_delta")
        self.assertEqual(event_names[-1], "message_stop")

        delta = message_delta(events)
        self.assertIn(delta["delta"]["stop_reason"], {"end_turn", "max_tokens", "stop_sequence"})
        self.assertIn("output_tokens", delta["usage"])

    def test_handle_anthropic_messages_requires_messages_list(self):
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

    def test_handle_anthropic_messages_rejects_non_object_message(self):
        response = self._post_messages(
            {
                "model": "chat_model",
                "max_tokens": 10,
                "messages": ["not-an-object"],
            }
        )
        self.assertEqual(response.status_code, 400)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="invalid_request_error",
            message_substring="message",
        )

    def test_handle_anthropic_messages_rejects_unsupported_role(self):
        response = self._post_messages(
            {
                "model": "chat_model",
                "max_tokens": 10,
                "messages": [{"role": "system", "content": "x"}],
            }
        )
        self.assertEqual(response.status_code, 400)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="invalid_request_error",
            message_substring="Unsupported role",
        )

    def test_handle_anthropic_messages_rejects_non_list_tools(self):
        response = self._post_messages(
            {
                "model": "chat_model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": {"name": "not-a-list"},
            }
        )
        self.assertEqual(response.status_code, 400)
        self._assert_anthropic_error(
            self._json_body(response),
            error_type="invalid_request_error",
            message_substring="tools must be a list",
        )

    def test_handle_anthropic_messages_with_query_string(self):
        response = self._post_messages(
            self._basic_request(),
            query="beta=true&anthropic-version=2023-06-01",
        )
        self.assertEqual(response.status_code, 200)

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
        fake = make_fake_generate(["Hello ", "STOP", "ignored"], stop_token_sequences=[[2]])

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
                        self._tool_request(stream=stream),
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

    def test_handle_anthropic_messages_tool_parse_error_non_stream_best_effort(self):
        def bad_parser(text, tools):
            raise ValueError("bad tool call json")

        fake_generate = make_fake_tool_generate(
            ["<tool_call>", "not-json", "</tool_call>"],
            bad_parser,
        )

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            with self.assertLogs("mlx_lm.server_anthropic", level="WARNING") as logs:
                response = self._post_messages(self._tool_request())

        self.assertEqual(response.status_code, 200)
        body = self._json_body(response)
        self.assertEqual(body["type"], "message")
        self.assertNotEqual(body["stop_reason"], "tool_use")
        self.assertEqual(body["content"], [{"type": "text", "text": ""}])
        self.assertTrue(any("Tool parser failed" in message for message in logs.output))

    def test_handle_anthropic_messages_tool_parse_error_stream_best_effort(self):
        def bad_parser(text, tools):
            raise ValueError("bad tool call json")

        fake_generate = make_fake_tool_generate(
            ["<tool_call>", "not-json", "</tool_call>", "Hello"],
            bad_parser,
        )

        with patch.object(self.response_generator, "generate", side_effect=fake_generate):
            response = self._post_messages(self._tool_request(stream=True), stream=True)

        self.assertEqual(response.status_code, 200)
        events = collect_sse_events(response)
        self.assertFalse(any(event == "error" for event, _ in events))
        self.assertIn("Hello", "".join(text_deltas(events)))
        self.assertEqual(visible_event_names(events)[-1], "message_stop")

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
                with patch.object(self.response_generator, "generate", side_effect=fake_generate):
                    response = self._post_messages(
                        self._tool_request(stream=stream),
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
                    self.assertEqual(message_delta(events)["delta"]["stop_reason"], "tool_use")
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
                with patch.object(self.response_generator, "generate", side_effect=fake_generate):
                    response = self._post_messages(
                        self._tool_request(stream=stream),
                        stream=stream,
                    )
                self.assertEqual(response.status_code, 200)

                if stream:
                    events = collect_sse_events(response)
                    self._assert_stream_tool_inputs(events, [{"location": "sf"}, {"location": "nyc"}])
                    starts = tool_use_start_payloads(events)
                    self.assertNotEqual(
                        starts[0]["content_block"]["id"],
                        starts[1]["content_block"]["id"],
                    )
                    self.assertEqual(message_delta(events)["delta"]["stop_reason"], "tool_use")
                else:
                    body = self._json_body(response)
                    self._assert_non_stream_tool_inputs(body, [{"location": "sf"}, {"location": "nyc"}])
                    tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
                    self.assertNotEqual(tool_blocks[0]["id"], tool_blocks[1]["id"])
                    self.assertEqual(body["stop_reason"], "tool_use")

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

                with patch.object(self.response_generator, "generate", side_effect=fake):
                    response = self._post_messages(self._basic_request(stream=stream), stream=stream)
                    self.assertEqual(response.status_code, 200)

                if stream:
                    visible_text = "".join(text_deltas(collect_sse_events(response)))
                else:
                    visible_text = text_from_content_blocks(self._json_body(response)["content"])

                self.assertEqual(visible_text, "Hello")
                self.assertNotIn("internal reasoning", visible_text)

    def test_handle_anthropic_messages_ignores_unknown_content_blocks(self):
        fake = make_fake_generate(["Hello"])

        with patch.object(self.response_generator, "generate", side_effect=fake):
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
                                    "source": {"type": "url", "url": "https://example.com/a.png"},
                                },
                                {"type": "text", "text": "Hello!"},
                            ],
                        }
                    ],
                }
            )

        self.assertEqual(response.status_code, 200)


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

    def test_convert_anthropic_messages_skips_invalid_blocks(self):
        converted = convert_anthropic_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            "not-a-block",
                            {"type": "image", "url": "x"},
                            {"type": "text", "text": 42},
                        ],
                    }
                ]
            }
        )
        self.assertEqual(converted, [{"role": "user", "content": "42"}])

    def test_convert_anthropic_tools_ignores_invalid_entries(self):
        with self.assertLogs("mlx_lm.server_anthropic", level="WARNING") as logs:
            converted = convert_anthropic_tools(
                [
                    "not-a-tool",
                    {"type": "server"},
                    {
                        "name": "get_weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                ]
            )
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["function"]["name"], "get_weather")
        self.assertTrue(
            any("Ignoring invalid tool definition" in message for message in logs.output)
        )

    def test_convert_anthropic_tools_none(self):
        self.assertIsNone(convert_anthropic_tools(None))

    def test_convert_anthropic_tools_non_list_raises(self):
        with self.assertRaisesRegex(ValueError, "tools must be a list"):
            convert_anthropic_tools({"name": "not-a-list"})


if __name__ == "__main__":
    unittest.main()
