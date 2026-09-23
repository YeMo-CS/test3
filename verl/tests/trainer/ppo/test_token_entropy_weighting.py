import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.ppo.token_entropy_weighting import compute_token_entropy_loss_weights, compute_token_entropy_weighting
from verl.workers.config.actor import ActorConfig


def test_signed_differences_and_bilateral_peak_score_use_available_neighbors():
    entropies = torch.tensor([[1.0, 2.0, 5.0, 3.0, 1.0]])
    result = compute_token_entropy_weighting(entropies, torch.ones_like(entropies), window_size=2, epsilon=0.5)
    assert torch.allclose(result["left_difference"], torch.tensor([[0.0, 1.0, 3.5, -0.5, -3.0]]))
    assert torch.allclose(result["right_difference"], torch.tensor([[-2.5, -2.0, 3.0, 2.0, 0.0]]))
    assert torch.allclose(result["soft_score"], torch.tensor([[0.0, 0.0, (10.5) ** 0.5, 0.0, 0.0]]))


def test_only_tokens_above_both_neighborhoods_get_positive_scores():
    entropies = torch.tensor([[3.0, 2.0, 4.0, 5.0, 1.0]])
    result = compute_token_entropy_weighting(entropies, torch.ones_like(entropies), window_size=1)
    assert torch.equal(result["soft_score"] > 0, torch.tensor([[False, False, False, True, False]]))


def test_normalized_weights_sum_to_one_and_loss_multipliers_have_mean_one():
    entropies = torch.tensor([[1.0, 4.0, 1.0, 0.0], [2.0, 2.0, 9.0, 9.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    result = compute_token_entropy_weighting(entropies, mask, window_size=2)
    assert torch.allclose(result["normalized_weight"].sum(dim=-1), torch.ones(2))
    assert torch.allclose(result["loss_multiplier"].sum(dim=-1), mask.sum(dim=-1).float())
    assert torch.equal(result["normalized_weight"][~mask.bool()], torch.zeros(3))
    assert result["normalized_weight"][0, 1] > result["normalized_weight"][0, 0]
    assert torch.allclose(result["normalized_weight"][1, :2], torch.tensor([0.5, 0.5]))


def test_token_mean_with_loss_multipliers_equals_normalized_weighted_sum():
    entropies = torch.tensor([[0.0, 3.0, 1.0]])
    objectives = torch.tensor([[4.0, -2.0, 3.0]])
    mask = torch.ones_like(entropies)
    result = compute_token_entropy_weighting(entropies, mask)
    aggregated = (result["loss_multiplier"] * objectives).sum(dim=-1) / mask.sum(dim=-1)
    expected = (result["normalized_weight"] * objectives).sum(dim=-1)
    assert torch.allclose(aggregated, expected)


def test_weights_are_detached_and_fully_masked_sequence_is_zero():
    entropies = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[1, 1], [0, 0]])
    weights = compute_token_entropy_loss_weights(entropies, mask)
    assert not weights.requires_grad
    assert torch.equal(weights[1], torch.zeros(2))


@pytest.mark.parametrize("window_size", [0, -1, 1.5, True])
def test_window_size_must_be_a_positive_integer(window_size):
    with pytest.raises(ValueError, match="positive integer"):
        compute_token_entropy_loss_weights(torch.ones(1, 2), torch.ones(1, 2), window_size=window_size)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_epsilon_must_be_positive_and_finite(epsilon):
    with pytest.raises(ValueError, match="positive and finite"):
        compute_token_entropy_loss_weights(torch.ones(1, 2), torch.ones(1, 2), epsilon=epsilon)


def test_entropy_and_mask_shapes_must_match():
    with pytest.raises(ValueError, match="Expected entropy/mask shape"):
        compute_token_entropy_loss_weights(torch.ones(1, 2), torch.ones(1, 3))


def test_valid_entropies_must_be_finite_but_padding_may_be_nonfinite():
    entropies = torch.tensor([[1.0, float("nan")], [float("inf"), 2.0]])
    mask = torch.tensor([[1, 0], [0, 1]])
    weights = compute_token_entropy_loss_weights(entropies, mask)
    assert torch.equal(weights, mask.float())
    with pytest.raises(ValueError, match="must be finite"):
        compute_token_entropy_loss_weights(torch.tensor([[float("nan")]]), torch.ones(1, 1))


@pytest.mark.parametrize("top_k", [None, 2])
def test_token_weights_compose_with_lp_and_lr_for_grpo_and_opd(top_k):
    entropies = torch.tensor([[1.0, 4.0, 1.0], [2.0, 5.0, 2.0]])
    response_mask = torch.ones_like(entropies)
    token_multipliers = compute_token_entropy_loss_weights(entropies, response_mask)
    lp_weights = torch.tensor([1.0, 0.0])
    lr_weights = torch.tensor([0.5, 1.5])
    policy_token_weights = token_multipliers * lp_weights.unsqueeze(-1)
    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    if top_k is not None:
        advantages = torch.stack((advantages, advantages * 0.25), dim=-1)
    old_log_prob = torch.zeros_like(advantages)
    log_prob = torch.zeros_like(advantages)
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size=2, clip_ratio=0.2)
    loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
        response_mask=response_mask, loss_agg_mode="seq-mean-token-mean", config=config,
        rollout_is_weights=policy_token_weights, sequence_weights=lr_weights,
    )
    token_objective = -advantages.sum(dim=-1) if top_k is not None else -advantages
    expected = ((token_objective * policy_token_weights).mean(dim=-1) * lr_weights).mean()
    assert torch.allclose(loss, expected)
