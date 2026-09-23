# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import gzip
import json

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.rollout_level_weighting import (
    compute_rollout_level_weights,
    dump_rollout_level_weighting_jsonl,
    pool_response_hidden_states,
)


def test_pool_response_hidden_states_uses_relative_blocks_and_normalizes():
    hidden = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8, 1).requires_grad_()
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]])
    embeddings = pool_response_hidden_states(hidden, mask, num_blocks=8)
    assert embeddings.shape == (2, 8)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(embeddings[0], hidden[0, :, 0] / hidden[0, :, 0].norm())
    assert torch.count_nonzero(embeddings[1]).item() == 4
    assert not embeddings.requires_grad


def test_rollout_weights_use_global_correct_centroid_and_normalize_incorrect_per_prompt():
    embeddings = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0],
        [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0],
    ], requires_grad=True)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], requires_grad=True)
    uids = np.array(["mixed"] * 4 + ["all-wrong"] * 2 + ["all-correct"] * 2, dtype=object)
    weights, metrics, _ = compute_rollout_level_weights(embeddings, rewards, uids, temperature=0.5)

    torch.testing.assert_close(weights[:2], torch.ones(2))
    torch.testing.assert_close(weights[2:4].sum(), torch.tensor(2.0))
    assert weights[2] > weights[3]
    torch.testing.assert_close(weights[4:6].sum(), torch.tensor(2.0))
    assert weights[4] > weights[5]
    torch.testing.assert_close(weights[6:], torch.ones(2))
    assert metrics["rollout_level_weighting/global_correct_centroid_count"] == 4
    assert metrics["rollout_level_weighting/weighted_prompt_count"] == 2
    assert metrics["rollout_level_weighting/incorrect_weight_mean"] == pytest.approx(1.0)
    assert not weights.requires_grad


def test_rollout_weights_exclude_prompt_filtered_rollouts_from_centroid_and_normalization():
    embeddings = torch.tensor([
        [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0], [0.0, -1.0],
    ])
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    uids = np.array(["kept"] * 3 + ["dropped"] * 3, dtype=object)
    eligible = torch.tensor([True, True, True, False, False, False])
    weights, metrics, details = compute_rollout_level_weights(
        embeddings, rewards, uids, temperature=0.5, expected_rollouts=3, eligible_mask=eligible
    )

    assert weights[1] > weights[2]
    torch.testing.assert_close(weights[1:3].sum(), torch.tensor(2.0))
    torch.testing.assert_close(weights[3:], torch.ones(3))
    assert details[0]["global_correct_centroid_distances"][1] == pytest.approx(0.0)
    assert details[0]["global_correct_centroid_distances"][2] == pytest.approx(2.0)
    assert metrics["rollout_level_weighting/global_correct_centroid_count"] == 1
    assert metrics["rollout_level_weighting/filtered_rollout_count"] == 3
    assert details[1]["eligible_incorrect_count"] == 0


def test_rollout_weights_fall_back_to_uniform_without_eligible_correct_rollout():
    weights, metrics, details = compute_rollout_level_weights(
        torch.eye(3), torch.tensor([0.0, 0.0, 1.0]), np.array(["p", "p", "q"]), temperature=1.0,
        eligible_mask=torch.tensor([True, True, False]),
    )
    torch.testing.assert_close(weights, torch.ones(3))
    assert metrics["rollout_level_weighting/fallback_to_uniform"] == 1.0
    assert all(detail["fallback_to_uniform"] for detail in details)


def test_rollout_weights_reject_non_binary_outcomes():
    with pytest.raises(ValueError, match="binary outcome rewards"):
        compute_rollout_level_weights(torch.eye(2), torch.tensor([1.0, 0.5]), np.array(["p", "p"]), temperature=1.0)


def test_rollout_weight_is_applied_after_sequence_token_mean():
    loss_mat = torch.tensor([[1.0, 3.0, 99.0], [2.0, 4.0, 6.0]], requires_grad=True)
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    rollout_weight = torch.tensor([0.5, 1.5], requires_grad=True)
    loss = agg_loss(loss_mat, loss_mask, "seq-mean-token-mean", sequence_weights=rollout_weight)
    torch.testing.assert_close(loss, torch.tensor((0.5 * 2.0 + 1.5 * 4.0) / 2))
    loss.backward()
    assert rollout_weight.grad is None


def test_rollout_weight_stays_aligned_during_batch_slicing():
    tensors = {"rollout_id": torch.arange(6), "rollout_weight": torch.arange(6).float() + 0.25}
    data = DataProto.from_dict(tensors=tensors)
    shuffled = data[torch.tensor([4, 1, 5, 0, 3, 2])]
    for minibatch in shuffled.split(2):
        torch.testing.assert_close(minibatch.batch["rollout_weight"], minibatch.batch["rollout_id"].float() + 0.25)


def test_rollout_weight_dump_contains_text_distances_and_weights(tmp_path):
    class Tokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(str(token_id) for token_id in token_ids if token_id)

    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0]])
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    uids = np.array(["prompt"] * 4, dtype=object)
    weights, _, details = compute_rollout_level_weights(embeddings, rewards, uids, temperature=0.5)
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[0, 10, 11]] * 4),
            "responses": torch.tensor([[20, 21, 0], [22, 0, 0], [23, 24, 0], [25, 26, 27]]),
            "attention_mask": torch.tensor([
                [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1]
            ]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]),
        },
        non_tensors={"uid": uids, "rollout_id": np.arange(4), "index": np.array([7] * 4)},
    )
    filename = dump_rollout_level_weighting_jsonl(
        batch=batch, tokenizer=Tokenizer(), sequence_rewards=rewards, rollout_weights=weights, step=20,
        output_dir=str(tmp_path), details=details, temperature=0.5,
    )
    with gzip.open(filename, "rt", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    assert len(records) == 4
    assert records[0]["prompt"] == "10 11"
    assert records[0]["response"] == "20 21"
    assert records[0]["weight_reason"] == "correct_fixed_one"
    assert records[0]["rollout_weight"] == 1.0
    assert records[2]["global_correct_centroid_cosine_distance"] is not None
    assert records[2]["raw_affinity"] is not None
    assert sum(record["rollout_weight"] for record in records[2:]) == pytest.approx(2.0)
    assert records[2]["incorrect_weight_sum"] == pytest.approx(2.0)
    assert records[2]["normalization_scope"] == "eligible_incorrect_rollouts_within_prompt"
