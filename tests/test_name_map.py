import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.name_map import build_name_report, build_repo_name_report


class TestBuildNameReport(unittest.TestCase):

    def test_reports_unchanged_renamed_unmapped_and_created_targets(self):
        shared = mx.zeros((2, 2))
        original = {
            "same.weight": shared,
            "old.weight": mx.ones((2, 2)),
            "dropped.weight": mx.array([1.0]),
        }
        sanitized = {
            "same.weight": shared,
            "new.weight": original["old.weight"],
            "created.weight": mx.array([2.0]),
        }

        report = build_name_report(original, sanitized)

        self.assertEqual(report["summary"]["source_weights"], 3)
        self.assertEqual(report["summary"]["target_weights"], 3)
        self.assertEqual(report["summary"]["mapped_sources"], 2)
        self.assertEqual(report["summary"]["unmapped_sources"], 1)
        self.assertEqual(report["summary"]["created_targets"], 1)

        rows = {row["source_key"]: row for row in report["source_weights"]}
        self.assertEqual(rows["same.weight"]["kind"], "unchanged")
        self.assertEqual(rows["same.weight"]["target_keys"], ["same.weight"])
        self.assertEqual(rows["old.weight"]["kind"], "renamed")
        self.assertEqual(rows["old.weight"]["target_keys"], ["new.weight"])
        self.assertEqual(rows["dropped.weight"]["kind"], "unmapped")
        self.assertEqual(rows["dropped.weight"]["target_keys"], [])

        self.assertEqual(
            report["created_targets"],
            [
                {
                    "target_dtype": str(sanitized["created.weight"].dtype),
                    "target_key": "created.weight",
                    "target_module": "created",
                    "target_shape": [1],
                }
            ],
        )


class _FakeArgs:
    @classmethod
    def from_dict(cls, config):
        return config


class _FakeModel(nn.Module):
    def __init__(self, _args):
        super().__init__()

    def sanitize(self, weights):
        weights["renamed.weight"] = weights.pop("source.weight")
        return weights


class TestBuildRepoNameReport(unittest.TestCase):

    def test_includes_model_and_sanitize_metadata(self):
        weights = {"source.weight": mx.zeros((2, 2))}
        config = {"model_type": "fake"}

        with tempfile.TemporaryDirectory() as test_dir:
            model_path = Path(test_dir)
            with patch("mlx_lm.name_map._download", return_value=model_path):
                with patch("mlx_lm.name_map.load_config", return_value=config):
                    with patch("mlx_lm.name_map.glob.glob", return_value=["dummy"]):
                        with patch("mlx_lm.name_map.mx.load", return_value=weights):
                            with patch(
                                "mlx_lm.name_map._load_model_classes",
                                return_value=(_FakeModel, _FakeArgs),
                            ):
                                report = build_repo_name_report("stub/repo")

        self.assertEqual(report["model"]["repo"], "stub/repo")
        self.assertEqual(report["model"]["model_type"], "fake")
        self.assertEqual(report["summary"]["mapped_sources"], 1)
        self.assertTrue(report["sanitize"]["defined"])
        self.assertTrue(report["sanitize"]["file"].endswith("test_name_map.py"))


if __name__ == "__main__":
    unittest.main()
