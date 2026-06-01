# Copyright © 2024 Apple Inc.

import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx

from mlx_lm.evaluate import MLXLM


class TestMLXLM(unittest.TestCase):
    def setUp(self):
        # Mock the load function to avoid loading actual models
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.model_max_length = 2048
        self.mock_tokenizer.chat_template = None
        self.mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4])

        with patch("mlx_lm.evaluate.load") as mock_load:
            mock_load.return_value = (self.mock_model, self.mock_tokenizer)
            self.mlx_lm = MLXLM("test_model", max_tokens=128)

    def test_loglikelihood_rolling_processes_all_inputs(self):
        """Test that loglikelihood_rolling processes all inputs correctly when batching."""
        # Create 5 mock requests to test batching with batch_size=2
        mock_requests = [MagicMock(args=(f"text {i}",)) for i in range(5)]

        # Mock inputs
        test_inputs = [(i, i + 1, i + 2) for i in range(5)]
        self.mlx_lm._tokenize = MagicMock(return_value=test_inputs)

        # Mock _score_fn to return different scores for each batch
        def mock_score_fn(batch):
            batch_size = len(batch)
            scores = mx.array([[0.1] * 3 for _ in range(batch_size)])
            lengths = mx.array([3] * batch_size)
            return scores, lengths, None

        self.mlx_lm._score_fn = MagicMock(side_effect=mock_score_fn)
        self.mlx_lm._batch_size = 2

        result = self.mlx_lm.loglikelihood_rolling(mock_requests)

        # Should return 5 results (one per request)
        self.assertEqual(len(result), 5)

        # Should have called _score_fn 3 times (batches of 2, 2, 1)
        self.assertEqual(self.mlx_lm._score_fn.call_count, 3)

        # Verify the batches were correct sizes
        call_args_list = self.mlx_lm._score_fn.call_args_list
        self.assertEqual(len(call_args_list[0][0][0]), 2)  # First batch: 2 items
        self.assertEqual(len(call_args_list[1][0][0]), 2)  # Second batch: 2 items
        self.assertEqual(len(call_args_list[2][0][0]), 1)  # Third batch: 1 item

    def test_loglikelihood_moves_context_trailing_space_to_continuation(self):
        encodings = {
            "hello": [1],
            "hello world": [1, 2],
            "hello there": [1, 3],
        }
        self.mock_tokenizer.encode.side_effect = lambda text, **_: encodings[text]
        self.mlx_lm._process_prompt = MagicMock(
            return_value=(mx.array([[0.0, 0.1, 0.9, 0.2]]), None)
        )
        self.mlx_lm._score_fn = MagicMock()
        group = MagicMock()
        group.rank.return_value = 0
        group.size.return_value = 1
        requests = [
            MagicMock(args=("hello ", "world")),
            MagicMock(args=("hello ", "there")),
        ]

        with (
            patch("mlx_lm.evaluate.mx.distributed.init", return_value=group),
            patch("mlx_lm.evaluate.mx.distributed.all_max", return_value=mx.array(2)),
            patch(
                "mlx_lm.evaluate.mx.distributed.all_gather",
                side_effect=lambda x, stream=None: x,
            ),
        ):
            result = self.mlx_lm.loglikelihood(requests)

        self.assertAlmostEqual(result[0][0], 0.9)
        self.assertTrue(result[0][1])
        self.assertAlmostEqual(result[1][0], 0.2)
        self.assertFalse(result[1][1])
        self.mlx_lm._score_fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
