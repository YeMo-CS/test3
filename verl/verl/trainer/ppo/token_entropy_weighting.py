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

import math

import torch


def compute_token_entropy_weighting(token_entropies: torch.Tensor, response_mask: torch.Tensor, window_size: int = 2, epsilon: float = 1e-6) -> dict[str, torch.Tensor]:
    """Compute detached bidirectional local-entropy-peak weights inside each response.

    Signed left/right differences average all available valid neighbors up to ``window_size``. The soft
    peak score is ``sqrt(relu(D_left) * relu(D_right))``. With ``g(s) = s``, valid-token weights are
    ``(epsilon + score) / sum(epsilon + score)`` and therefore sum to one within every non-empty response.
    ``loss_multiplier = valid_token_count * normalized_weight`` preserves the scale of token-mean losses.
    """
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
        raise ValueError(f"token entropy weighting window_size must be a positive integer, got {window_size}")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"token entropy weighting epsilon must be positive and finite, got {epsilon}")
    if token_entropies.dim() != 2 or response_mask.dim() != 2 or token_entropies.shape != response_mask.shape:
        raise ValueError(f"Expected entropy/mask shape [batch, response], got {token_entropies.shape=} and {response_mask.shape=}")

    entropies = token_entropies.detach().to(dtype=torch.float32)
    valid_mask = response_mask.detach().bool()
    if not torch.isfinite(entropies[valid_mask]).all():
        raise ValueError("Valid token entropies must be finite")

    left_sum, right_sum = torch.zeros_like(entropies), torch.zeros_like(entropies)
    left_count, right_count = torch.zeros_like(entropies), torch.zeros_like(entropies)
    response_length = entropies.shape[-1]
    for offset in range(1, min(window_size, max(response_length - 1, 0)) + 1):
        left_pairs = valid_mask[:, offset:] & valid_mask[:, :-offset]
        right_pairs = valid_mask[:, :-offset] & valid_mask[:, offset:]
        left_sum[:, offset:] += torch.where(left_pairs, entropies[:, offset:] - entropies[:, :-offset], 0.0)
        right_sum[:, :-offset] += torch.where(right_pairs, entropies[:, :-offset] - entropies[:, offset:], 0.0)
        left_count[:, offset:] += left_pairs
        right_count[:, :-offset] += right_pairs

    left_difference = torch.where(left_count > 0, left_sum / left_count.clamp_min(1), torch.zeros_like(left_sum))
    right_difference = torch.where(right_count > 0, right_sum / right_count.clamp_min(1), torch.zeros_like(right_sum))
    bilateral_mask = valid_mask & (left_count > 0) & (right_count > 0)
    soft_score = torch.where(bilateral_mask, torch.sqrt(left_difference.clamp_min(0) * right_difference.clamp_min(0)), torch.zeros_like(entropies))
    unnormalized_weight = torch.where(valid_mask, soft_score + epsilon, torch.zeros_like(soft_score))
    weight_sum = unnormalized_weight.sum(dim=-1, keepdim=True)
    normalized_weight = torch.where(weight_sum > 0, unnormalized_weight / weight_sum.clamp_min(torch.finfo(unnormalized_weight.dtype).tiny), torch.zeros_like(unnormalized_weight))
    valid_count = valid_mask.sum(dim=-1, keepdim=True).to(dtype=normalized_weight.dtype)
    loss_multiplier = normalized_weight * valid_count
    return {
        "left_difference": left_difference, "right_difference": right_difference, "soft_score": soft_score,
        "unnormalized_weight": unnormalized_weight, "normalized_weight": normalized_weight,
        "loss_multiplier": loss_multiplier, "left_neighbor_count": left_count, "right_neighbor_count": right_count,
    }


def compute_token_entropy_loss_weights(token_entropies: torch.Tensor, response_mask: torch.Tensor, window_size: int = 2, epsilon: float = 1e-6) -> torch.Tensor:
    """Return mean-one token loss multipliers derived from bidirectional local entropy peaks."""
    return compute_token_entropy_weighting(token_entropies, response_mask, window_size=window_size, epsilon=epsilon)["loss_multiplier"]
