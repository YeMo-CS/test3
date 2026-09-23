# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

MODULE_PATH = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "ppo" / "rollout_telemetry.py"
SPEC = importlib.util.spec_from_file_location("rollout_telemetry_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
compute_entropy_concentration = MODULE.compute_entropy_concentration
dump_rollout_telemetry = MODULE.dump_rollout_telemetry
dump_rollout_telemetry_safely = MODULE.dump_rollout_telemetry_safely


class _Tokenizer:
    all_special_ids = []
    added_tokens_decoder = {}
    bos_token_id = eos_token_id = pad_token_id = unk_token_id = mask_token_id = None
    pieces = {1: "A", 2: ",B", 3: "\nC", 4: "D", 10: "prompt-0", 11: "prompt-1"}

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row.tolist(), skip_special_tokens=skip_special_tokens) for row in rows]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.pieces.get(int(token_id), f"<{int(token_id)}>") for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        return [self.pieces.get(int(token_id), f"<{int(token_id)}>") for token_id in token_ids]


class TestRolloutEntropyConcentration(unittest.TestCase):
    def test_safe_dump_reports_error_without_raising(self):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = dump_rollout_telemetry_safely(
                batch=SimpleNamespace(batch={}, non_tensor_batch={}), tokenizer=_Tokenizer(), step=20,
                output_dir="unused", config={"plot": False},
            )
        self.assertIsNone(result)
        self.assertIn("training will continue without this dump", output.getvalue())
        self.assertIn("Cannot dump rollout telemetry", output.getvalue())

    def test_safe_dump_contains_ranking_writer_failure(self):
        batch = SimpleNamespace(
            batch={
                "prompts": torch.tensor([[10]]), "responses": torch.tensor([[1, 2, 3]]),
                "response_mask": torch.ones((1, 3), dtype=torch.bool), "old_log_probs": torch.full((1, 3), -0.5),
            },
            non_tensor_batch={"uid": np.asarray(["p0"], dtype=object)},
        )
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            MODULE, "write_entropy_concentration_ranking", side_effect=OSError("injected ranking write failure")
        ), redirect_stdout(output), redirect_stderr(output):
            result = dump_rollout_telemetry_safely(
                batch=batch, tokenizer=_Tokenizer(), step=20, output_dir=tmp_dir,
                config={"compression": "none", "plot": False}, entropys=torch.ones((1, 3)), expected_rollouts=1,
            )
        self.assertIsNone(result)
        self.assertIn("injected ranking write failure", output.getvalue())

    def test_entropy_concentration_uniform_and_spike(self):
        self.assertAlmostEqual(compute_entropy_concentration(np.ones(4)), 0.0)
        self.assertAlmostEqual(compute_entropy_concentration(np.asarray([4.0, 0.0, 0.0, 0.0])), 1.0)
        self.assertAlmostEqual(compute_entropy_concentration(np.asarray([1.0, -1e-6])), 1.0)
        with self.assertRaisesRegex(ValueError, "below -1e-4"):
            compute_entropy_concentration(np.asarray([1.0, -0.01]))

    def test_dump_writes_one_ranked_json_object_per_batch_sequence(self):
        batch = SimpleNamespace(
            batch={
                "prompts": torch.tensor([[10], [10], [11], [11]]),
                "responses": torch.tensor([[1, 2, 3, 0], [1, 4, 0, 0], [2, 4, 0, 0], [1, 2, 3, 4]]),
                "response_mask": torch.tensor(
                    [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool
                ),
                "old_log_probs": torch.full((4, 4), -0.5),
            },
            non_tensor_batch={"uid": np.asarray(["p0", "p0", "p1", "p1"], dtype=object)},
        )
        entropies = torch.tensor(
            [[4.0, 0.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            step_dir = dump_rollout_telemetry(
                batch=batch, tokenizer=_Tokenizer(), step=7, output_dir=tmp_dir,
                config={"compression": "none", "plot": False, "sampled_rollout_count": 0},
                entropys=entropies, expected_rollouts=2,
            )
            rank_path = step_dir / "entropy_concentration_ranked.jsonl"
            physical_lines = rank_path.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in physical_lines]
            self.assertEqual(len(physical_lines), 4)
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["rank"] for row in rows], [1, 2, 3, 4])
            concentrations = [row["entropy_concentration"] for row in rows]
            self.assertEqual(concentrations, sorted(concentrations, reverse=True))
            self.assertEqual(rows[0]["response"], "A,B\nC")
            self.assertTrue(all(set(row) == {"rank", "entropy_concentration", "response"} for row in rows))
            summary = json.loads((step_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertNotIn("algorithm", summary)
            self.assertEqual(summary["num_prompts"], 2)
            self.assertEqual(summary["num_sequences"], 4)
            self.assertEqual(summary["expected_rollouts_per_prompt"], 2)
            self.assertEqual(summary["entropy_concentration_ranking"]["file"], rank_path.name)
            self.assertEqual(summary["entropy_concentration_ranking"]["format"], "jsonl")
            rollout_lines = (step_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
            rollout_records = [json.loads(line) for line in rollout_lines]
            self.assertTrue(all("entropy_concentration" in record for record in rollout_records))

    def test_dump_rejects_wrong_prompt_rollout_group_size(self):
        batch = SimpleNamespace(
            batch={
                "prompts": torch.tensor([[10], [10], [11]]), "responses": torch.tensor([[1], [2], [3]]),
                "response_mask": torch.ones((3, 1), dtype=torch.bool), "old_log_probs": torch.full((3, 1), -0.5),
            },
            non_tensor_batch={"uid": np.asarray(["p0", "p0", "p1"], dtype=object)},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "Expected 2 rollouts per prompt"):
                dump_rollout_telemetry(
                    batch=batch, tokenizer=_Tokenizer(), step=1, output_dir=tmp_dir,
                    config={"compression": "none", "plot": False}, entropys=torch.ones((3, 1)), expected_rollouts=2,
                )
            self.assertFalse((Path(tmp_dir) / "step_000001" / "rollouts.jsonl").exists())

    def test_dump_writes_bidirectional_token_entropy_weight_details(self):
        batch = SimpleNamespace(
            batch={
                "prompts": torch.tensor([[10]]), "responses": torch.tensor([[1, 2, 3]]),
                "response_mask": torch.ones((1, 3), dtype=torch.bool), "old_log_probs": torch.full((1, 3), -0.5),
            },
            non_tensor_batch={"uid": np.asarray(["p0"], dtype=object)},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            step_dir = dump_rollout_telemetry(
                batch=batch, tokenizer=_Tokenizer(), step=3, output_dir=tmp_dir,
                config={"compression": "none", "plot": False, "sampled_rollout_count": 1},
                entropys=torch.tensor([[1.0, 4.0, 1.0]]), token_entropy_weighting_enable=True,
                token_entropy_weighting_window_size=2, token_entropy_weighting_epsilon=0.5, expected_rollouts=1,
            )
            record = json.loads((step_dir / "token_entropy_weight_samples.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["left_entropy_difference"], [0.0, 3.0, -1.5])
            self.assertEqual(record["right_entropy_difference"], [-1.5, 3.0, 0.0])
            self.assertEqual(record["bidirectional_soft_score"], [0.0, 3.0, 0.0])
            self.assertAlmostEqual(sum(record["token_normalized_weight"]), 1.0)
            self.assertAlmostEqual(sum(record["token_loss_multiplier"]) / 3, 1.0, places=5)
            summary = json.loads((step_dir / "summary.json").read_text(encoding="utf-8"))
            weighting_summary = summary["token_entropy_weight_samples"]
            self.assertEqual(weighting_summary["window_size"], 2)
            self.assertEqual(weighting_summary["g"], "identity")
            self.assertNotIn("temperature", weighting_summary)


if __name__ == "__main__":
    unittest.main()
