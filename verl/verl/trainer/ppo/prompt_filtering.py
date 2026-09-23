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

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch


def reduce_sequence_rewards(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, reward_mode: str) -> torch.Tensor:
    """Reduce GRPO outcome or OPD dense token rewards to one value per rollout sequence."""
    if reward_mode not in {"grpo", "opd"}:
        raise ValueError(f"reward_mode must be 'grpo' or 'opd', got {reward_mode!r}")
    rewards = token_level_rewards.detach()
    rewards = rewards.sum(dim=-1, dtype=torch.float32) if rewards.dim() == 3 else rewards.float()
    mask = response_mask.detach().to(device=rewards.device, dtype=torch.float32)
    if rewards.dim() != 2 or mask.dim() != 2 or rewards.shape != mask.shape:
        raise ValueError(f"Expected reward/mask shape [batch, response], got {rewards.shape=} and {mask.shape=}")
    reward_sum = (rewards * mask).sum(dim=-1)
    return reward_sum / mask.sum(dim=-1).clamp_min(1) if reward_mode == "opd" else reward_sum


def reduce_binary_outcome_rewards(token_level_rewards: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Reduce sparse outcome rewards and require one binary 0/1 result per rollout."""
    sequence_rewards = reduce_sequence_rewards(token_level_rewards, response_mask, reward_mode="grpo")
    if not torch.all((sequence_rewards == 0) | (sequence_rewards == 1)):
        raise ValueError("Prompt filtering requires binary 0/1 outcome rewards for every rollout")
    return sequence_rewards


def compute_prompt_variance_filter_weights(sequence_rewards: torch.Tensor, prompt_uids: np.ndarray, minibatch_ids: np.ndarray, drop_fraction: float = 0.2, expected_rollouts: int | None = None, expected_prompts: int | None = None) -> tuple[torch.Tensor, dict[str, float], list[dict[str, Any]]]:
    """Drop the lowest-variance prompts and return zero-or-renormalized sequence loss multipliers."""
    if not np.isfinite(drop_fraction) or not 0 <= drop_fraction < 1:
        raise ValueError(f"prompt filtering drop_fraction must be finite and in [0, 1), got {drop_fraction}")
    rewards = sequence_rewards.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    uids = np.asarray(prompt_uids, dtype=object).reshape(-1)
    batch_ids = np.asarray(minibatch_ids).reshape(-1)
    if not (len(rewards) == len(uids) == len(batch_ids)):
        raise ValueError(f"Mismatched prompt-filtering inputs: {len(rewards)=}, {len(uids)=}, {len(batch_ids)=}")
    if not torch.isfinite(rewards).all():
        raise ValueError("Prompt filtering received non-finite sequence rewards")

    relative_weights = torch.ones_like(rewards)
    variance_values, cutoff_values, dropped_counts, prompt_details = [], [], [], []
    for minibatch_id in dict.fromkeys(batch_ids.tolist()):
        sequence_indices = np.flatnonzero(batch_ids == minibatch_id).tolist()
        prompt_to_indices: dict[Any, list[int]] = defaultdict(list)
        for index in sequence_indices:
            prompt_to_indices[uids[index]].append(index)
        prompt_items = list(prompt_to_indices.items())
        prompt_count = len(prompt_items)
        if expected_prompts is not None and prompt_count != expected_prompts:
            raise ValueError(f"PPO mini-batch {minibatch_id!r} has {prompt_count} prompts, expected {expected_prompts}")
        if expected_rollouts is not None:
            invalid = {uid: len(indices) for uid, indices in prompt_items if len(indices) != expected_rollouts}
            if invalid:
                raise ValueError(f"PPO mini-batch {minibatch_id!r} does not contain exactly {expected_rollouts} rollouts per prompt: {invalid}")
        if prompt_count == 0:
            raise ValueError(f"PPO mini-batch {minibatch_id!r} contains no prompts")

        prompt_variances = torch.stack([rewards[indices].var(unbiased=False) for _, indices in prompt_items])
        drop_count = min(prompt_count - 1, int(np.ceil(prompt_count * drop_fraction))) if drop_fraction > 0 else 0
        ranked_prompt_indices = sorted(range(prompt_count), key=lambda index: (prompt_variances[index].item(), index))
        dropped_prompt_indices = set(ranked_prompt_indices[:drop_count])
        retained_prompt_count = prompt_count - drop_count
        retained_weight = prompt_count / retained_prompt_count
        cutoff = prompt_variances[ranked_prompt_indices[drop_count - 1]].item() if drop_count else float("nan")
        ranks = {prompt_index: rank for rank, prompt_index in enumerate(ranked_prompt_indices, start=1)}
        for prompt_index, ((uid, indices), variance) in enumerate(zip(prompt_items, prompt_variances, strict=True)):
            kept = prompt_index not in dropped_prompt_indices
            prompt_weight = retained_weight if kept else 0.0
            relative_weights[indices] = prompt_weight
            prompt_details.append({"minibatch_id": minibatch_id, "uid": uid, "sequence_indices": indices, "filtering_method": "reward_variance_bottom_filter", "reward_variance": variance.item(), "variance_rank_ascending": ranks[prompt_index], "drop_fraction": float(drop_fraction), "kept": kept, "prompt_weight": prompt_weight})
        variance_values.extend(prompt_variances.tolist())
        dropped_counts.append(drop_count)
        if drop_count:
            cutoff_values.append(cutoff)

    variances = torch.tensor(variance_values, dtype=torch.float64)
    prompt_count_total = len(variance_values)
    dropped_prompt_count = sum(dropped_counts)
    stats = {
        "prompt_filtering/reward_variance_mean": variances.mean().item(), "prompt_filtering/reward_variance_std": variances.std(unbiased=False).item(),
        "prompt_filtering/dropped_prompt_count": float(dropped_prompt_count), "prompt_filtering/retained_prompt_count": float(prompt_count_total - dropped_prompt_count),
        "prompt_filtering/dropped_prompt_fraction": dropped_prompt_count / prompt_count_total,
        "prompt_filtering/variance_cutoff_mean": float(np.mean(cutoff_values)) if cutoff_values else float("nan"),
        "prompt_filtering/relative_sequence_weight_mean": relative_weights.mean().item(), "prompt_filtering/relative_sequence_weight_min": relative_weights.min().item(),
        "prompt_filtering/relative_sequence_weight_max": relative_weights.max().item(),
    }
    return relative_weights, stats, prompt_details


def dump_prompt_filtering_jsonl(output_dir: str, step: int, prompt_details: list[dict[str, Any]], sequence_rewards: torch.Tensor, relative_weights: torch.Tensor, rollout_ids: np.ndarray, sample_ids: np.ndarray | None = None) -> str:
    """Atomically save one JSONL row per prompt for a training step."""
    os.makedirs(output_dir, exist_ok=True)
    rewards = sequence_rewards.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    weights = relative_weights.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    rollout_ids = np.asarray(rollout_ids).reshape(-1)
    sample_ids = None if sample_ids is None else np.asarray(sample_ids, dtype=object).reshape(-1)
    if not (len(rewards) == len(weights) == len(rollout_ids)):
        raise ValueError("Mismatched sequence arrays while dumping prompt filtering telemetry")

    lines = []
    for detail in prompt_details:
        indices = detail["sequence_indices"]
        ordered_indices = sorted(indices, key=lambda index: int(rollout_ids[index]))
        first_index = ordered_indices[0]
        prompt_id = detail["uid"] if sample_ids is None else sample_ids[first_index]
        record = {
            "step": int(step), "ppo_minibatch_id": int(detail["minibatch_id"]), "prompt_id": prompt_id.item() if isinstance(prompt_id, np.generic) else prompt_id,
            "uid": str(detail["uid"]), "rollout_count": len(ordered_indices), "filtering_method": "reward_variance_bottom_filter",
            "prompt_weight": float(detail["prompt_weight"]), "reward_variance": float(detail["reward_variance"]),
            "variance_rank_ascending": float(detail["variance_rank_ascending"]), "drop_fraction": float(detail["drop_fraction"]), "kept": bool(detail["kept"]),
            "rollouts": [{"rollout_id": int(rollout_ids[index]), "sequence_reward": float(rewards[index]), "sequence_relative_weight": float(weights[index])} for index in ordered_indices],
        }
        lines.append(json.dumps(record, ensure_ascii=False, allow_nan=False))

    filename = os.path.join(output_dir, f"step_{step:06d}.jsonl")
    temporary_filename = f"{filename}.{os.getpid()}.tmp"
    with open(temporary_filename, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    os.replace(temporary_filename, filename)
    return filename
