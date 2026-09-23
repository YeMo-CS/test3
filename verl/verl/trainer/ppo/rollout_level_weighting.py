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
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from verl import DataProto


def pool_response_hidden_states(
    response_hidden_states: torch.Tensor, response_mask: torch.Tensor, num_blocks: int = 8
) -> torch.Tensor:
    """Mean-pool valid response-token states in relative-position blocks and L2-normalize."""
    if response_hidden_states.ndim != 3 or response_mask.ndim != 2:
        raise ValueError("response_hidden_states and response_mask must have shapes [B, T, H] and [B, T]")
    if response_hidden_states.shape[:2] != response_mask.shape:
        raise ValueError("response_hidden_states and response_mask must agree on batch and response dimensions")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    mask = response_mask.bool()
    lengths = mask.sum(dim=-1)
    if torch.any(lengths == 0):
        raise ValueError("Every rollout must contain at least one valid response token")
    relative_positions = mask.long().cumsum(dim=-1) - 1
    block_ids = torch.div(relative_positions * num_blocks, lengths.unsqueeze(-1), rounding_mode="floor")
    block_ids = block_ids.clamp_(0, num_blocks - 1)
    hidden = response_hidden_states.float()
    pooled = hidden.new_zeros((hidden.shape[0], num_blocks, hidden.shape[-1]))
    counts = hidden.new_zeros((hidden.shape[0], num_blocks, 1))
    pooled.scatter_add_(1, block_ids.unsqueeze(-1).expand_as(hidden), hidden * mask.unsqueeze(-1))
    counts.scatter_add_(1, block_ids.unsqueeze(-1), mask.unsqueeze(-1).to(hidden.dtype))
    embeddings = (pooled / counts.clamp_min(1)).flatten(start_dim=1)
    return F.normalize(embeddings, p=2, dim=-1).detach()


def compute_rollout_level_weights(
    rollout_embeddings: torch.Tensor, sequence_rewards: torch.Tensor, prompt_uids: np.ndarray, temperature: float,
    expected_rollouts: int | None = None, eligible_mask: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, float], list[dict[str, Any]]]:
    """Weight incorrect rollouts by distance to the eligible batch-global correct centroid, normalized per prompt."""
    if rollout_embeddings.ndim != 2 or sequence_rewards.ndim != 1:
        raise ValueError("rollout_embeddings and sequence_rewards must have shapes [B, D] and [B]")
    if len(rollout_embeddings) != len(sequence_rewards) or len(prompt_uids) != len(sequence_rewards):
        raise ValueError("rollout_embeddings, sequence_rewards, and prompt_uids must have the same batch size")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    with torch.no_grad():
        embeddings = F.normalize(rollout_embeddings.detach().float(), p=2, dim=-1)
        rewards = sequence_rewards.detach().to(embeddings.device)
        is_correct, is_incorrect = rewards == 1, rewards == 0
        if not torch.all(is_correct | is_incorrect):
            invalid = rewards[~(is_correct | is_incorrect)].detach().cpu().tolist()
            raise ValueError(f"Rollout-level weighting requires binary outcome rewards (0 or 1), got {invalid[:8]}")
        if eligible_mask is None:
            eligible = torch.ones(len(rewards), device=embeddings.device, dtype=torch.bool)
        else:
            eligible = torch.as_tensor(eligible_mask, device=embeddings.device)
            if eligible.ndim != 1 or len(eligible) != len(rewards):
                raise ValueError(f"eligible_mask must have shape [{len(rewards)}], got {tuple(eligible.shape)}")
            eligible = eligible.bool()

        groups: dict[Any, list[int]] = defaultdict(list)
        for index, uid in enumerate(prompt_uids.tolist()):
            groups[uid].append(index)
        if expected_rollouts is not None:
            invalid_groups = {uid: len(indices) for uid, indices in groups.items() if len(indices) != expected_rollouts}
            if invalid_groups:
                raise ValueError(f"Expected {expected_rollouts} rollouts per prompt, got {invalid_groups}")

        weights = torch.ones(len(rewards), device=embeddings.device, dtype=torch.float32)
        distances = torch.full_like(weights, float("nan"))
        eligible_correct = eligible & is_correct
        correct_centroid_count = int(eligible_correct.sum().item())
        centroid_pre_normalization_l2_norm = float("nan")
        unit_centroid = None
        if correct_centroid_count:
            centroid = embeddings[eligible_correct].mean(dim=0)
            centroid_pre_normalization_l2_norm = centroid.norm().item()
            if math.isfinite(centroid_pre_normalization_l2_norm) and centroid_pre_normalization_l2_norm > 0:
                unit_centroid = F.normalize(centroid, p=2, dim=0)

        weighted_prompt_count = 0
        details: list[dict[str, Any]] = []
        for uid, indices_list in groups.items():
            indices = torch.as_tensor(indices_list, device=embeddings.device, dtype=torch.long)
            correct_indices, incorrect_indices = indices[is_correct[indices]], indices[is_incorrect[indices]]
            eligible_indices = indices[eligible[indices]]
            eligible_correct_indices = indices[eligible[indices] & is_correct[indices]]
            eligible_incorrect_indices = indices[eligible[indices] & is_incorrect[indices]]
            if unit_centroid is not None and len(eligible_incorrect_indices) > 0:
                weighted_prompt_count += 1
                cosine_distance = (1 - embeddings[eligible_incorrect_indices] @ unit_centroid).clamp_(0, 2)
                incorrect_weights = len(eligible_incorrect_indices) * torch.softmax(-cosine_distance / temperature, dim=0)
                weights[eligible_incorrect_indices], distances[eligible_incorrect_indices] = incorrect_weights, cosine_distance
            details.append({
                "uid": uid, "indices": indices_list, "correct_count": len(correct_indices),
                "incorrect_count": len(incorrect_indices), "eligible_count": len(eligible_indices),
                "eligible_correct_count": len(eligible_correct_indices),
                "eligible_incorrect_count": len(eligible_incorrect_indices),
            })

        eligible_incorrect = eligible & is_incorrect
        incorrect_weight_values = weights[eligible_incorrect]
        metrics = {
            "rollout_level_weighting/global_correct_centroid_count": float(correct_centroid_count),
            "rollout_level_weighting/centroid_pre_normalization_l2_norm": centroid_pre_normalization_l2_norm,
            "rollout_level_weighting/weighted_prompt_count": float(weighted_prompt_count),
            "rollout_level_weighting/eligible_rollout_count": float(eligible.sum().item()),
            "rollout_level_weighting/filtered_rollout_count": float((~eligible).sum().item()),
            "rollout_level_weighting/fallback_to_uniform": float(unit_centroid is None),
            "rollout_level_weighting/weight_mean": weights[eligible].mean().item() if eligible.any() else 1.0,
            "rollout_level_weighting/weight_min": weights[eligible].min().item() if eligible.any() else 1.0,
            "rollout_level_weighting/weight_max": weights[eligible].max().item() if eligible.any() else 1.0,
            "rollout_level_weighting/incorrect_weight_mean": (
                incorrect_weight_values.mean().item() if len(incorrect_weight_values) else 1.0
            ),
        }
        for detail in details:
            detail["weights"] = weights[detail["indices"]].cpu().tolist()
            detail["global_correct_centroid_distances"] = distances[detail["indices"]].cpu().tolist()
            detail["eligible"] = eligible[detail["indices"]].cpu().tolist()
            detail["global_correct_centroid_count"] = correct_centroid_count
            detail["centroid_pre_normalization_l2_norm"] = centroid_pre_normalization_l2_norm
            detail["fallback_to_uniform"] = unit_centroid is None
        return weights.detach(), metrics, details


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def dump_rollout_level_weighting_jsonl(
    batch: "DataProto", tokenizer: Any, sequence_rewards: torch.Tensor, rollout_weights: torch.Tensor, step: int,
    output_dir: str, details: list[dict[str, Any]], temperature: float, num_blocks: int = 8, digits: int = 8,
) -> str:
    """Atomically save every rollout text, outcome, distance, raw affinity, and final weight."""
    required_batch = {"prompts", "responses", "attention_mask", "response_mask"}
    missing_batch = required_batch - set(batch.batch.keys())
    if missing_batch:
        raise KeyError(f"Cannot dump rollout-level weighting telemetry; batch is missing {sorted(missing_batch)}")
    required_non_tensor = {"uid", "rollout_id"}
    missing_non_tensor = required_non_tensor - set(batch.non_tensor_batch)
    if missing_non_tensor:
        raise KeyError(f"Cannot dump rollout-level weighting telemetry; batch is missing {sorted(missing_non_tensor)}")

    prompts, responses = batch.batch["prompts"].detach().cpu(), batch.batch["responses"].detach().cpu()
    attention_mask = batch.batch["attention_mask"].detach().cpu().bool()
    response_mask = batch.batch["response_mask"].detach().cpu().bool()
    rewards = sequence_rewards.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    weights = rollout_weights.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object).reshape(-1)
    rollout_ids = np.asarray(batch.non_tensor_batch["rollout_id"]).reshape(-1)
    sample_ids = batch.non_tensor_batch.get("index", None)
    sample_ids = None if sample_ids is None else np.asarray(sample_ids, dtype=object).reshape(-1)
    if not (len(batch) == len(rewards) == len(weights) == len(uids) == len(rollout_ids)):
        raise ValueError("Rollout-level weighting telemetry inputs must contain one entry per sequence")
    if sample_ids is not None and len(sample_ids) != len(batch):
        raise ValueError("Rollout-level weighting sample IDs must contain one entry per sequence")

    records = []
    covered_indices = set()
    prompt_length = prompts.shape[-1]
    for detail in details:
        indices = [int(index) for index in detail["indices"]]
        ordered_offsets = sorted(range(len(indices)), key=lambda offset: int(rollout_ids[indices[offset]]))
        incorrect_weight_sum = sum(float(weights[index]) for offset, index in enumerate(indices) if rewards[index].item() == 0 and detail["eligible"][offset])
        for offset in ordered_offsets:
            row = indices[offset]
            if row in covered_indices or not 0 <= row < len(batch):
                raise ValueError(f"Invalid or duplicate rollout-level weighting detail index: {row}")
            covered_indices.add(row)
            reward, weight = float(rewards[row]), float(weights[row])
            eligible = bool(detail["eligible"][offset])
            raw_distance = float(detail["global_correct_centroid_distances"][offset])
            raw_affinity = round(math.exp(-raw_distance / temperature), digits) if math.isfinite(raw_distance) else None
            distance = round(raw_distance, digits) if math.isfinite(raw_distance) else None
            prompt_ids = prompts[row][attention_mask[row, :prompt_length]].tolist()
            response_ids = responses[row][response_mask[row]].tolist()
            prompt_id = detail["uid"] if sample_ids is None else sample_ids[row]
            if not eligible:
                weight_reason = "prompt_filtered_fixed_one"
            elif detail["fallback_to_uniform"]:
                weight_reason = "no_global_correct_centroid_fixed_one"
            elif reward == 0:
                weight_reason = "incorrect_global_centroid_distance_weighted"
            else:
                weight_reason = "correct_fixed_one"
            records.append({
                "step": int(step), "prompt_id": _json_safe(prompt_id), "uid": str(detail["uid"]),
                "rollout_id": int(rollout_ids[row]), "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                "response": tokenizer.decode(response_ids, skip_special_tokens=True), "outcome_reward": int(reward),
                "is_correct": reward == 1, "eligible_by_prompt_filter": eligible,
                "weighting_applied": eligible and reward == 0 and not detail["fallback_to_uniform"], "weight_reason": weight_reason,
                "correct_count": int(detail["correct_count"]), "incorrect_count": int(detail["incorrect_count"]),
                "eligible_correct_count": int(detail["eligible_correct_count"]),
                "eligible_incorrect_count": int(detail["eligible_incorrect_count"]),
                "global_correct_centroid_count": int(detail["global_correct_centroid_count"]),
                "centroid_pre_normalization_l2_norm": round(float(detail["centroid_pre_normalization_l2_norm"]), digits) if math.isfinite(float(detail["centroid_pre_normalization_l2_norm"])) else None,
                "global_correct_centroid_cosine_distance": distance, "raw_affinity": raw_affinity,
                "rollout_weight": round(weight, digits), "incorrect_weight_sum": round(incorrect_weight_sum, digits),
                "incorrect_weight_sum_target": int(detail["eligible_incorrect_count"]), "temperature": float(temperature),
                "num_blocks": int(num_blocks), "centroid_scope": "eligible_batch_global_correct_rollouts",
                "normalization_scope": "eligible_incorrect_rollouts_within_prompt",
            })
    if covered_indices != set(range(len(batch))):
        raise ValueError("Rollout-level weighting telemetry details do not cover the full rollout batch")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"step_{step:06d}.jsonl.gz"
    temporary_filename = filename.with_name(f".{filename.name}.{os.getpid()}.tmp")
    with gzip.open(temporary_filename, "wt", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    os.replace(temporary_filename, filename)
    return str(filename)
