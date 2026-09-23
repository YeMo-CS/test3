import json

import numpy as np
import torch

from verl.trainer.ppo.prompt_filtering import (
    compute_prompt_variance_filter_weights,
    dump_prompt_filtering_jsonl,
    reduce_binary_outcome_rewards,
    reduce_sequence_rewards,
)


def test_prompt_variance_filter_drops_all_rollouts_of_bottom_twenty_percent():
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0])
    weights, metrics, details = compute_prompt_variance_filter_weights(
        sequence_rewards=rewards, prompt_uids=np.repeat(np.array(["p0", "p1", "p2", "p3", "p4"], dtype=object), 2),
        minibatch_ids=np.zeros(10, dtype=np.int64), drop_fraction=0.2, expected_rollouts=2, expected_prompts=5,
    )
    assert torch.equal(weights, torch.tensor([0.0, 0.0] + [1.25] * 8, dtype=torch.float64))
    assert [detail["kept"] for detail in details] == [False, True, True, True, True]
    assert metrics["prompt_filtering/dropped_prompt_count"] == 1.0
    assert metrics["prompt_filtering/relative_sequence_weight_mean"] == 1.0


def test_prompt_variance_filter_uses_stable_rank_for_ties():
    weights, _, details = compute_prompt_variance_filter_weights(
        sequence_rewards=torch.zeros(10), prompt_uids=np.repeat(np.array(["p0", "p1", "p2", "p3", "p4"], dtype=object), 2),
        minibatch_ids=np.zeros(10, dtype=np.int64), drop_fraction=0.2, expected_rollouts=2, expected_prompts=5,
    )
    assert torch.equal(weights[:2], torch.zeros(2, dtype=torch.float64))
    assert sum(not detail["kept"] for detail in details) == 1


def test_prompt_filter_dump_records_decision_and_rank(tmp_path):
    sequence_rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    weights, _, details = compute_prompt_variance_filter_weights(
        sequence_rewards=sequence_rewards, prompt_uids=np.array(["p0", "p0", "p1", "p1"], dtype=object),
        minibatch_ids=np.zeros(4, dtype=np.int64), drop_fraction=0.2, expected_rollouts=2, expected_prompts=2,
    )
    filename = dump_prompt_filtering_jsonl(
        output_dir=str(tmp_path), step=4, prompt_details=details, sequence_rewards=sequence_rewards,
        relative_weights=weights, rollout_ids=np.array([0, 1, 0, 1]),
    )
    with open(filename, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    assert rows[0]["filtering_method"] == "reward_variance_bottom_filter"
    assert rows[0]["kept"] is False
    assert rows[0]["variance_rank_ascending"] == 1.0
    assert rows[1]["kept"] is True


def test_grpo_reward_uses_masked_sequence_sum_without_length_normalization():
    rewards = torch.tensor([[0.0, 1.0, 9.0], [1.0, 0.0, 9.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    assert torch.equal(reduce_sequence_rewards(rewards, mask, reward_mode="grpo"), torch.tensor([1.0, 1.0]))


def test_opd_reward_uses_dense_token_reward_mean_and_top_k_sum():
    rewards = torch.tensor([[-3.0, -5.0, 99.0], [-7.0, -9.0, 99.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    assert torch.equal(reduce_sequence_rewards(rewards, mask, reward_mode="opd"), torch.tensor([-4.0, -8.0]))
    top_k_rewards = torch.tensor([[[-1.0, -2.0], [-3.0, -4.0]]])
    assert torch.equal(reduce_sequence_rewards(top_k_rewards, torch.ones(1, 2), reward_mode="opd"), torch.tensor([-5.0]))


def test_prompt_filter_requires_binary_outcomes():
    mask = torch.ones(2, 2)
    assert torch.equal(reduce_binary_outcome_rewards(torch.tensor([[0.0, 1.0], [0.0, 0.0]]), mask), torch.tensor([1.0, 0.0]))
    with np.testing.assert_raises_regex(ValueError, "binary 0/1"):
        reduce_binary_outcome_rewards(torch.tensor([[0.2, 0.3], [0.0, 0.0]]), mask)
