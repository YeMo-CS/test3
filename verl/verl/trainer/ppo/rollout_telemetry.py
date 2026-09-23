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
"""Persist per-rollout uncertainty metrics and render per-step distributions."""

from __future__ import annotations

import gzip
import json
import math
import os
import traceback
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from verl import DataProto

_MAX_EXPONENT = math.log(np.finfo(np.float64).max)
_NEGATIVE_ENTROPY_TOLERANCE = 1e-4


def _round_finite(value: float, digits: int) -> float | None:
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _ppl(value: float, digits: int | None = None) -> float | None:
    value = float(value)
    if not math.isfinite(value) or value > _MAX_EXPONENT:
        return None
    result = math.exp(value)
    return round(result, digits) if digits is not None else result


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0, "mean": None, "std": None, "min": None, "p50": None,
            "p90": None, "p95": None, "p99": None, "max": None,
        }
    percentiles = np.percentile(values, [50, 90, 95, 99])
    result = {
        "count": int(values.size), "mean": values.mean(), "std": values.std(), "min": values.min(),
        "p50": percentiles[0], "p90": percentiles[1], "p95": percentiles[2], "p99": percentiles[3],
        "max": values.max(),
    }
    return {key: int(value) if key == "count" else _round_finite(value, 12) for key, value in result.items()}


def _ppl_stats(nll_values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(nll_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {**_stats(values), "overflow_count": 0}
    overflow_count = int(np.count_nonzero(values > _MAX_EXPONENT))
    representable = values[values <= _MAX_EXPONENT]
    result = _stats(np.exp(representable))
    result["count"] = int(values.size)
    result["finite_count"] = int(representable.size)
    result["overflow_count"] = overflow_count
    return result


def _sample(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if limit > 0 and values.size > limit:
        return values[rng.choice(values.size, size=limit, replace=False)]
    return values


def compute_entropy_concentration(token_entropies: np.ndarray) -> float:
    """Return normalized positional concentration of one response's token-entropy mass."""
    values = np.asarray(token_entropies, dtype=np.float64)
    invalid = values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
    if invalid or (values < -_NEGATIVE_ENTROPY_TOLERANCE).any():
        raise ValueError("token_entropies must be a non-empty finite vector with no values below -1e-4")
    values = np.maximum(values, 0.0)
    total = float(values.sum(dtype=np.float64))
    if values.size == 1 or total <= np.finfo(np.float64).tiny:
        return 0.0
    probabilities = values / total
    positive = probabilities[probabilities > 0]
    positional_entropy = float(-(positive * np.log(positive)).sum(dtype=np.float64))
    return float(np.clip(1.0 - positional_entropy / math.log(values.size), 0.0, 1.0))


def write_entropy_concentration_ranking(
    path: str | Path, concentrations: list[float] | np.ndarray, responses: list[str], digits: int = 12
) -> Path:
    """Atomically write rank, concentration, and response as one JSON object per physical line."""
    path = Path(path)
    values = np.asarray(concentrations, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(responses) or not np.isfinite(values).all():
        raise ValueError("concentrations and responses must be finite one-dimensional arrays of equal length")
    ranked_rows = sorted(range(len(responses)), key=lambda row: (-float(values[row]), row))
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for rank, row in enumerate(ranked_rows, start=1):
            record = {"rank": rank, "entropy_concentration": _round_finite(values[row], digits), "response": str(responses[row])}
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    os.replace(tmp_path, path)
    return path


def _plot_sorted_values(values: np.ndarray, output_path: Path, title: str, ylabel: str, log_y: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if log_y:
        values = values[values > 0]
    if values.size == 0:
        return
    values.sort()
    sorted_indices = np.arange(values.size)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sorted_indices, values, color="#4C78A8" if log_y else "#F58518", linewidth=1, rasterized=True)
    if log_y:
        ax.set_yscale("log")
    stats = _stats(values)
    note = f"n={stats['count']:,}  median={stats['p50']:.4g}  p90={stats['p90']:.4g}  p99={stats['p99']:.4g}"
    ax.text(0.01, 0.99, note, transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Sorted index (ascending)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_token_line(values: np.ndarray, output_path: Path, title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return
    positions = np.arange(values.size)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(positions, values, color="#4C78A8", linewidth=0.9)
    if finite_values.min() < 0 < finite_values.max():
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Token position")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _token_label(tokenizer: Any, token_id: int, cache: dict[int, tuple[str, bool]]) -> tuple[str, bool]:
    if token_id in cache:
        return cache[token_id]
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    has_newline = "\n" in text or "\r" in text
    text = text.replace("\r\n", "↵").replace("\r", "↵").replace("\n", "↵").replace("\t", "⇥")
    text = text.strip()
    if not text:
        text = "␠"
    text = text.replace("$", r"\$")
    cache[token_id] = (text, has_newline)
    return cache[token_id]


def _decode_response(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def _display_width(text: str) -> int:
    return max(1, sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text))


def _contains_chinese(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        or "\U00020000" <= char <= "\U0003134f"
        for char in text
    )


@lru_cache(maxsize=1)
def _cjk_font_properties() -> Any:
    from matplotlib import font_manager
    candidates = (
        "Arial Unicode MS", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei",
        "PingFang SC", "Hiragino Sans GB", "Heiti SC", "SimHei",
    )
    for family in candidates:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        if path and Path(path).is_file():
            return font_manager.FontProperties(fname=path)
    return None


def _special_token_ids(tokenizer: Any) -> set[int]:
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None}
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "mask_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    for token_id, added_token in getattr(tokenizer, "added_tokens_decoder", {}).items():
        if getattr(added_token, "special", False):
            special_ids.add(int(token_id))
    return special_ids


def _is_special_token(tokenizer: Any, token_id: int, special_ids: set[int]) -> bool:
    if token_id in special_ids:
        return True
    try:
        token = tokenizer.convert_ids_to_tokens(token_id)
    except (AttributeError, TypeError):
        return False
    if isinstance(token, list):
        token = token[0] if token else ""
    token = str(token)
    return (token.startswith("<|") and token.endswith("|>")) or token in {
        "<s>", "</s>", "<pad>", "<unk>", "<mask>", "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    }


def _plot_sequence_tokens(
    tokenizer: Any,
    token_ids: list[int],
    values: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
    log_scale: bool,
    token_label_cache: dict[int, tuple[str, bool]],
    max_columns: int,
    font_size: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize

    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if log_scale:
        finite_values = finite_values[finite_values > 0]
    if finite_values.size == 0:
        return
    vmin, vmax = float(finite_values.min()), float(finite_values.max())
    if vmin == vmax:
        if log_scale:
            vmin, vmax = max(vmin / 1.01, np.finfo(np.float64).tiny), vmax * 1.01
        else:
            delta = max(abs(vmin) * 0.01, 1e-6)
            vmin, vmax = vmin - delta, vmax + delta
    norm = (
        LogNorm(vmin=max(vmin, np.finfo(np.float64).tiny), vmax=vmax, clip=True)
        if log_scale
        else Normalize(vmin=vmin, vmax=vmax, clip=True)
    )
    cmap = plt.get_cmap("coolwarm")
    placements = []
    line, column = 0, 0
    for token_id, value in zip(token_ids, values, strict=True):
        label, has_newline = _token_label(tokenizer, int(token_id), token_label_cache)
        width = _display_width(label)
        if column and column + width > max_columns:
            line, column = line + 1, 0
        placements.append((column, line, label, float(value)))
        column += width + 1
        if has_newline:
            line, column = line + 1, 0
    line_count = line + 1
    figure_width = 18
    figure_height = max(2.5, 1.2 + line_count * max(0.16, font_size * 0.022))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    for column, line, label, value in placements:
        rgba = cmap(norm(value)) if math.isfinite(value) and (not log_scale or value > 0) else (0.75, 0.75, 0.75, 1)
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        needs_unicode_font = any(ord(char) > 127 for char in label)
        cjk_font = _cjk_font_properties() if needs_unicode_font else None
        needs_cjk_font = _contains_chinese(label) or any(
            unicodedata.east_asian_width(char) in {"W", "F"} for char in label
        )
        if needs_cjk_font and cjk_font is None:
            raise RuntimeError(
                "Chinese token text was detected, but no CJK font is installed. Install Noto Sans CJK SC "
                "(for example fonts-noto-cjk on Debian/Ubuntu) or provide another supported CJK font."
            )
        font_kwargs = {"fontproperties": cjk_font} if cjk_font is not None else {"family": "DejaVu Sans"}
        ax.text(
            column, line_count - line - 1, label, fontsize=font_size, **font_kwargs,
            color="black" if luminance > 0.55 else "white", va="center", ha="left",
            bbox={"facecolor": rgba, "edgecolor": "none", "pad": 0.12},
        )
    ax.set_xlim(-1, max_columns + 1)
    ax.set_ylim(-1, line_count)
    ax.set_title(title, fontsize=max(10, font_size + 2), loc="left")
    ax.axis("off")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01, fraction=0.025)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")
    os.replace(tmp_path, path)


def _write_jsonl(path: Path, records: list[dict[str, Any]], compression: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    opener = gzip.open if compression == "gzip" else open
    with opener(tmp_path, "wt", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    os.replace(tmp_path, path)


def dump_rollout_telemetry(
    batch: DataProto,
    tokenizer: Any,
    step: int,
    output_dir: str,
    config: Any,
    entropys: Any | None = None,
    token_kl: Any | None = None,
    kl_estimator: str | None = None,
    token_entropy_weighting_enable: bool = False,
    token_entropy_weighting_window_size: int = 2,
    token_entropy_weighting_epsilon: float = 1e-6,
    expected_rollouts: int | None = None,
) -> Path:
    """Save rollout uncertainty distributions and sampled bidirectional token-entropy weights for one step."""
    required = {"prompts", "responses", "response_mask", "old_log_probs"}
    if entropys is None:
        required.add("entropys")
    missing = required - set(batch.batch.keys())
    if missing:
        raise KeyError(f"Cannot dump rollout telemetry; batch is missing {sorted(missing)}")

    digits = int(config.get("float_precision", 6))
    compression = str(config.get("compression", "gzip")).lower()
    if compression not in {"gzip", "none"}:
        raise ValueError(f"Unsupported rollout telemetry compression: {compression}")
    save_token_text = bool(config.get("save_token_text", False))
    plot_enabled = bool(config.get("plot", True))
    max_plot_tokens = int(config.get("max_plot_tokens", 2_000_000))
    plot_prompt_sequences = bool(config.get("plot_prompt_sequences", True))
    selected_prompt_count = int(config.get("selected_prompt_count", 5))
    sequence_plot_max_columns = int(config.get("sequence_plot_max_columns", 180))
    sequence_plot_font_size = float(config.get("sequence_plot_font_size", 8))
    sequence_plot_dpi = int(config.get("sequence_plot_dpi", 120))
    skip_chinese_sequences = bool(config.get("skip_chinese_sequences", True))
    sampled_rollout_count = int(config.get("sampled_rollout_count", 5))
    if sampled_rollout_count < 0:
        raise ValueError("trainer.rollout_telemetry.sampled_rollout_count must be non-negative")
    step_dir = Path(output_dir).expanduser() / f"step_{step:06d}"
    plots_dir = step_dir / "plots"
    step_dir.mkdir(parents=True, exist_ok=True)
    if plot_enabled:
        plots_dir.mkdir(parents=True, exist_ok=True)

    prompts = batch.batch["prompts"].detach().cpu()
    responses = batch.batch["responses"].detach().cpu()
    response_mask = batch.batch["response_mask"].detach().cpu().bool()
    log_probs = batch.batch["old_log_probs"].detach().cpu().float()
    entropys = (entropys if entropys is not None else batch.batch["entropys"]).detach().cpu().float()
    token_kls = token_kl.detach().cpu().float() if token_kl is not None else None
    shapes_match = responses.shape == response_mask.shape == log_probs.shape == entropys.shape
    if not shapes_match:
        raise ValueError(
            f"Rollout telemetry tensor shape mismatch: responses={tuple(responses.shape)}, "
            f"mask={tuple(response_mask.shape)}, log_probs={tuple(log_probs.shape)}, entropys={tuple(entropys.shape)}"
        )
    if token_kls is not None and token_kls.shape != responses.shape:
        raise ValueError(
            f"Rollout telemetry token KL shape mismatch: responses={tuple(responses.shape)}, "
            f"token_kls={tuple(token_kls.shape)}"
        )
    token_weighting = None
    if token_entropy_weighting_enable:
        from verl.trainer.ppo.token_entropy_weighting import compute_token_entropy_weighting

        token_weighting = compute_token_entropy_weighting(
            token_entropies=entropys, response_mask=response_mask, window_size=token_entropy_weighting_window_size,
            epsilon=token_entropy_weighting_epsilon,
        )

    special_ids = _special_token_ids(tokenizer)
    visualization_mask = response_mask.clone()
    for special_id in special_ids:
        visualization_mask &= responses.ne(special_id)

    batch_size = responses.shape[0]
    sample_rng = np.random.default_rng(step)
    sampled_rows = set(
        int(row) for row in np.atleast_1d(
            sample_rng.choice(batch_size, size=min(sampled_rollout_count, batch_size), replace=False)
        )
    ) if sampled_rollout_count > 0 else set()
    token_weight_sample_records = []
    decoded_prompts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
    raw_uids = batch.non_tensor_batch.get("uid")
    uids = [str(raw_uids[i]) if raw_uids is not None else f"step-{step}-prompt-{i}" for i in range(batch_size)]
    rollout_counts: defaultdict[str, int] = defaultdict(int)
    sequence_nlls = np.empty(batch_size, dtype=np.float64)
    sequence_entropys = np.empty(batch_size, dtype=np.float64)
    entropy_concentrations = np.empty(batch_size, dtype=np.float64)
    sequence_lengths = np.empty(batch_size, dtype=np.int64)
    sequence_kls = np.empty(batch_size, dtype=np.float64) if token_kls is not None else None
    token_nll_chunks: list[np.ndarray] = []
    token_entropy_chunks: list[np.ndarray] = []
    token_kl_chunks: list[np.ndarray] = []
    prompt_rows: defaultdict[str, list[int]] = defaultdict(list)
    prompt_texts: dict[str, str] = {}
    response_texts: list[str] = [""] * batch_size
    extension = ".jsonl.gz" if compression == "gzip" else ".jsonl"
    rollout_path = step_dir / f"rollouts{extension}"
    rollout_tmp_path = rollout_path.with_name(f".{rollout_path.name}.tmp")
    opener = gzip.open if compression == "gzip" else open

    with opener(rollout_tmp_path, "wt", encoding="utf-8") as rollout_file:
        for row in range(batch_size):
            mask = response_mask[row]
            token_ids = responses[row][mask].tolist()
            token_log_probs = log_probs[row][mask].numpy().astype(np.float64, copy=False)
            token_entropys = entropys[row][mask].numpy().astype(np.float64, copy=False)
            token_kls_for_row = (
                token_kls[row][mask].numpy().astype(np.float64, copy=False) if token_kls is not None else None
            )
            token_nlls = -token_log_probs
            length = len(token_ids)
            if length == 0:
                raise ValueError(f"Rollout telemetry found an empty response at row {row}")
            sequence_nll = float(token_nlls.mean())
            sequence_entropy = float(token_entropys.mean())
            token_entropy_variance = float(token_entropys.var(ddof=0))
            entropy_concentration = compute_entropy_concentration(token_entropys)
            uid = uids[row]
            rollout_index = rollout_counts[uid]
            rollout_counts[uid] += 1
            prompt_rows[uid].append(row)
            prompt_texts.setdefault(uid, decoded_prompts[row])
            sequence_nlls[row] = sequence_nll
            sequence_entropys[row] = sequence_entropy
            entropy_concentrations[row] = entropy_concentration
            sequence_lengths[row] = length
            token_nll_chunks.append(token_nlls.astype(np.float32, copy=False))
            token_entropy_chunks.append(token_entropys.astype(np.float32, copy=False))
            if token_kls_for_row is not None:
                sequence_kl = float(token_kls_for_row.mean())
                sequence_kls[row] = sequence_kl
                token_kl_chunks.append(token_kls_for_row.astype(np.float32, copy=False))
            response_text = _decode_response(tokenizer, token_ids)
            response_texts[row] = response_text
            record = {
                "step": step, "prompt_id": uid, "rollout_index": rollout_index,
                "prompt": decoded_prompts[row], "response": response_text,
                "response_token_ids": [int(token_id) for token_id in token_ids],
                "token_logprobs": [_round_finite(value, digits) for value in token_log_probs],
                "token_nll": [_round_finite(value, digits) for value in token_nlls],
                "token_ppl": [_ppl(value, digits) for value in token_nlls],
                "token_entropy": [_round_finite(value, digits) for value in token_entropys],
                "sequence_nll": _round_finite(sequence_nll, digits), "sequence_ppl": _ppl(sequence_nll, digits),
                "sequence_entropy": _round_finite(sequence_entropy, digits), "response_length": length,
                "token_entropy_variance": _round_finite(token_entropy_variance, digits),
                "entropy_concentration": _round_finite(entropy_concentration, digits),
            }
            if save_token_text:
                record["response_tokens"] = tokenizer.convert_ids_to_tokens(token_ids)
            if token_kls_for_row is not None:
                record["token_kl"] = [_round_finite(value, digits) for value in token_kls_for_row]
                record["sequence_kl"] = _round_finite(sequence_kl, digits)
                record["kl_estimator"] = kl_estimator
            if token_weighting is not None and row in sampled_rows:
                left_differences = token_weighting["left_difference"][row][mask].numpy().astype(np.float64, copy=False)
                right_differences = token_weighting["right_difference"][row][mask].numpy().astype(np.float64, copy=False)
                soft_scores = token_weighting["soft_score"][row][mask].numpy().astype(np.float64, copy=False)
                unnormalized_weights = token_weighting["unnormalized_weight"][row][mask].numpy().astype(np.float64, copy=False)
                normalized_weights = token_weighting["normalized_weight"][row][mask].numpy().astype(np.float64, copy=False)
                token_multipliers = token_weighting["loss_multiplier"][row][mask].numpy().astype(np.float64, copy=False)
                token_weight_sample_records.append({
                    "step": step, "selection_seed": step, "batch_row": row, "prompt_id": uid,
                    "rollout_index": rollout_index, "prompt": decoded_prompts[row], "response": response_text,
                    "response_token_ids": [int(token_id) for token_id in token_ids],
                    "response_tokens": tokenizer.convert_ids_to_tokens(token_ids),
                    "token_entropy": [_round_finite(value, digits) for value in token_entropys],
                    "left_entropy_difference": [_round_finite(value, digits) for value in left_differences],
                    "right_entropy_difference": [_round_finite(value, digits) for value in right_differences],
                    "bidirectional_soft_score": [_round_finite(value, digits) for value in soft_scores],
                    "token_unnormalized_weight": [_round_finite(value, digits) for value in unnormalized_weights],
                    "token_normalized_weight": [_round_finite(value, digits) for value in normalized_weights],
                    "token_loss_multiplier": [_round_finite(value, digits) for value in token_multipliers],
                    "window_size": token_entropy_weighting_window_size, "epsilon": token_entropy_weighting_epsilon,
                    "response_length": length, "normalized_weight_sum": _round_finite(normalized_weights.sum(), digits),
                    "loss_multiplier_mean": _round_finite(token_multipliers.mean(), digits),
                    "loss_multiplier_min": _round_finite(token_multipliers.min(), digits),
                    "loss_multiplier_max": _round_finite(token_multipliers.max(), digits),
                    "positive_peak_fraction": _round_finite(np.count_nonzero(soft_scores > 0) / length, digits),
                    "ess": _round_finite(1.0 / np.square(normalized_weights).sum(), digits),
                    "ess_fraction": _round_finite(1.0 / (length * np.square(normalized_weights).sum()), digits),
                })
            if "token_level_scores" in batch.batch:
                scores = batch.batch["token_level_scores"][row].detach().cpu()
                record["reward"] = _round_finite(float(scores[mask].sum()), digits)
            rollout_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
            )
    if expected_rollouts is not None:
        if expected_rollouts <= 0:
            rollout_tmp_path.unlink(missing_ok=True)
            raise ValueError("expected_rollouts must be positive")
        unexpected = {uid: len(rows) for uid, rows in prompt_rows.items() if len(rows) != expected_rollouts}
        if unexpected:
            rollout_tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Expected {expected_rollouts} rollouts per prompt, got {unexpected}")
        expected_sequence_count = len(prompt_rows) * expected_rollouts
        if batch_size != expected_sequence_count:
            rollout_tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Expected {expected_sequence_count} batch sequences, got {batch_size}")
    os.replace(rollout_tmp_path, rollout_path)
    concentration_path = write_entropy_concentration_ranking(
        step_dir / "entropy_concentration_ranked.jsonl", entropy_concentrations, response_texts
    )
    if token_weighting is not None:
        token_weight_sample_records.sort(key=lambda record: record["batch_row"])
        _write_jsonl(step_dir / "token_entropy_weight_samples.jsonl", token_weight_sample_records, "none")

    prompt_records = []
    prompt_nlls = []
    prompt_entropys = []
    for uid, rows in prompt_rows.items():
        row_indices = np.asarray(rows, dtype=np.int64)
        prompt_nll = float(sequence_nlls[row_indices].mean())
        prompt_entropy = float(sequence_entropys[row_indices].mean())
        token_weighted_nll = float(np.average(sequence_nlls[row_indices], weights=sequence_lengths[row_indices]))
        token_weighted_entropy = float(
            np.average(sequence_entropys[row_indices], weights=sequence_lengths[row_indices])
        )
        prompt_nlls.append(prompt_nll)
        prompt_entropys.append(prompt_entropy)
        prompt_records.append({
            "step": step, "prompt_id": uid, "prompt": prompt_texts[uid], "num_rollouts": len(rows),
            "num_tokens": int(sequence_lengths[row_indices].sum()),
            "prompt_nll": _round_finite(prompt_nll, digits), "prompt_ppl": _ppl(prompt_nll, digits),
            "prompt_entropy": _round_finite(prompt_entropy, digits),
            "token_weighted_prompt_nll": _round_finite(token_weighted_nll, digits),
            "token_weighted_prompt_ppl": _ppl(token_weighted_nll, digits),
            "token_weighted_prompt_entropy": _round_finite(token_weighted_entropy, digits),
        })

    _write_jsonl(step_dir / "prompt_metrics.jsonl", prompt_records, "none")
    token_nlls = np.concatenate(token_nll_chunks)
    token_entropys = np.concatenate(token_entropy_chunks)
    prompt_nlls_array = np.asarray(prompt_nlls, dtype=np.float64)
    prompt_entropys_array = np.asarray(prompt_entropys, dtype=np.float64)
    summary = {
        "step": step,
        "num_prompts": len(prompt_records),
        "num_sequences": batch_size,
        "expected_rollouts_per_prompt": expected_rollouts,
        "num_tokens": int(token_nlls.size),
        "definitions": {
            "token_ppl": "exp(-log p(sampled_token))", "token_entropy": "-sum_v p(v) log p(v)",
            "token_entropy_variance": "mean((token_entropy - sequence_entropy)^2) within a sequence",
            "entropy_concentration": (
                "1 - entropy(normalized token-entropy mass) / log(response_length); larger means entropy is "
                "concentrated in fewer token positions"
            ),
            "sequence_ppl": "exp(mean token NLL within a sequence)",
            "sequence_entropy": "mean token entropy within a sequence",
            "prompt_ppl": "exp(mean sequence NLL across rollouts for a prompt)",
            "prompt_entropy": "mean sequence entropy across rollouts for a prompt",
        },
        "distributions": {
            "token_ppl": _ppl_stats(token_nlls), "token_entropy": _stats(token_entropys),
            "sequence_ppl": _ppl_stats(sequence_nlls), "sequence_entropy": _stats(sequence_entropys),
            "entropy_concentration": _stats(entropy_concentrations),
            "prompt_ppl": _ppl_stats(prompt_nlls_array), "prompt_entropy": _stats(prompt_entropys_array),
        },
    }
    summary["entropy_concentration_ranking"] = {
        "file": concentration_path.name, "format": "jsonl", "order": "descending",
        "fields": ["rank", "entropy_concentration", "response"],
        "response_encoding": "JSON string escaping keeps every sequence on one physical line",
    }
    if token_weighting is not None:
        summary["token_entropy_weight_samples"] = {
            "file": "token_entropy_weight_samples.jsonl", "count": len(token_weight_sample_records),
            "selection_seed": step, "window_size": token_entropy_weighting_window_size,
            "epsilon": token_entropy_weighting_epsilon, "g": "identity",
            "definitions": {
                "left_entropy_difference": "mean(H_t - H_{t-i}) over available valid left neighbors",
                "right_entropy_difference": "mean(H_t - H_{t+i}) over available valid right neighbors",
                "bidirectional_soft_score": "sqrt(relu(left_entropy_difference) * relu(right_entropy_difference))",
                "token_unnormalized_weight": "epsilon + bidirectional_soft_score",
                "token_normalized_weight": "token_unnormalized_weight normalized to sum to one within the response",
                "token_loss_multiplier": "response_length * token_normalized_weight; mean is one over valid tokens",
                "ess": "1 / sum(token_normalized_weight^2)", "ess_fraction": "ess / response_length",
            },
        }
    if sequence_kls is not None:
        token_kls_flat = np.concatenate(token_kl_chunks)
        summary["kl_estimator"] = kl_estimator
        summary["definitions"]["token_kl"] = "sampled-token KL estimate between policy and reference"
        summary["definitions"]["sequence_kl"] = "mean token KL estimate within a sequence"
        summary["distributions"]["token_kl"] = _stats(token_kls_flat)
        summary["distributions"]["sequence_kl"] = _stats(sequence_kls)
    _atomic_json(step_dir / "summary.json", summary)

    if plot_enabled:
        rng = np.random.default_rng(step)
        visual_token_nlls = -log_probs[visualization_mask].numpy().astype(np.float64, copy=False)
        visual_token_entropys = entropys[visualization_mask].numpy().astype(np.float64, copy=False)
        sampled_token_nlls = _sample(visual_token_nlls, max_plot_tokens, rng)
        sampled_token_entropys = _sample(visual_token_entropys, max_plot_tokens, rng)
        _plot_sorted_values(
            np.exp(np.minimum(sampled_token_nlls, _MAX_EXPONENT)), plots_dir / "token_ppl.png",
            f"Sorted Token Perplexity — Step {step}", "Token perplexity", True,
        )
        _plot_sorted_values(
            sampled_token_entropys, plots_dir / "token_entropy.png",
            f"Sorted Token Entropy — Step {step}", "Token entropy (nats)", False,
        )
        _plot_sorted_values(
            np.exp(np.minimum(sequence_nlls, _MAX_EXPONENT)), plots_dir / "sequence_ppl.png",
            f"Sorted Sequence Perplexity — Step {step}", "Sequence perplexity", True,
        )
        _plot_sorted_values(
            sequence_entropys, plots_dir / "sequence_entropy.png",
            f"Sorted Sequence Entropy — Step {step}", "Mean token entropy (nats)", False,
        )
        _plot_sorted_values(
            np.exp(np.minimum(prompt_nlls_array, _MAX_EXPONENT)), plots_dir / "prompt_ppl.png",
            f"Sorted Prompt Perplexity — Step {step}", "Prompt perplexity", True,
        )
        _plot_sorted_values(
            prompt_entropys_array, plots_dir / "prompt_entropy.png",
            f"Sorted Prompt Entropy — Step {step}", "Mean sequence entropy (nats)", False,
        )
        if sequence_kls is not None:
            visual_token_kls = token_kls[visualization_mask].numpy().astype(np.float64, copy=False)
            _plot_sorted_values(
                _sample(visual_token_kls, max_plot_tokens, rng), plots_dir / "token_kl.png",
                f"Sorted Token KL Estimate — Step {step} ({kl_estimator})", "Token KL estimate", False,
            )
            _plot_sorted_values(
                sequence_kls, plots_dir / "sequence_kl.png",
                f"Sorted Sequence KL Estimate — Step {step} ({kl_estimator})", "Sequence KL estimate", False,
            )
        if plot_prompt_sequences and selected_prompt_count > 0:
            prompt_sequence_dir = plots_dir / "prompt_sequences"
            prompt_sequence_dir.mkdir(parents=True, exist_ok=True)
            available_uids = list(prompt_rows.keys())
            skipped_chinese_uids = []
            eligible_uids = []
            for uid in available_uids:
                contains_chinese = any(_contains_chinese(response_texts[row]) for row in prompt_rows[uid])
                if skip_chinese_sequences and contains_chinese:
                    skipped_chinese_uids.append(uid)
                else:
                    eligible_uids.append(uid)
            selection_count = min(selected_prompt_count, len(eligible_uids))
            if selection_count:
                selected_indices = rng.choice(len(eligible_uids), size=selection_count, replace=False)
                selected_uids = [eligible_uids[int(index)] for index in np.atleast_1d(selected_indices)]
            else:
                selected_uids = []
            if selection_count < selected_prompt_count:
                print(
                    f"Rollout telemetry step {step}: requested {selected_prompt_count} prompt sequence plots, "
                    f"but only {selection_count} prompts remain after Chinese filtering."
                )
            token_label_cache: dict[int, tuple[str, bool]] = {}
            selected_prompt_records = []
            for prompt_index, uid in enumerate(selected_uids, start=1):
                prompt_dir = prompt_sequence_dir / f"prompt_{prompt_index:02d}"
                prompt_dir.mkdir(parents=True, exist_ok=True)
                rollout_files = []
                for rollout_index, row in enumerate(prompt_rows[uid]):
                    mask = response_mask[row]
                    all_token_ids = [int(token_id) for token_id in responses[row][mask].tolist()]
                    keep_positions = np.asarray(
                        [not _is_special_token(tokenizer, token_id, special_ids) for token_id in all_token_ids],
                        dtype=bool,
                    )
                    token_ids = [token_id for token_id, keep in zip(all_token_ids, keep_positions, strict=True) if keep]
                    excluded_special_token_count = len(all_token_ids) - len(token_ids)
                    if not token_ids:
                        print(
                            f"Rollout telemetry step {step}: prompt {prompt_index} rollout {rollout_index} "
                            "contains no normal tokens after special-token filtering; skipping its token plots."
                        )
                        continue
                    token_entropys_for_row = (
                        entropys[row][mask].numpy().astype(np.float64, copy=False)[keep_positions]
                    )
                    token_nlls_for_row = (
                        -log_probs[row][mask].numpy().astype(np.float64, copy=False)[keep_positions]
                    )
                    token_ppls_for_row = np.exp(np.minimum(token_nlls_for_row, _MAX_EXPONENT))
                    token_kls_for_visualization = (
                        token_kls[row][mask].numpy().astype(np.float64, copy=False)[keep_positions]
                        if token_kls is not None
                        else None
                    )
                    entropy_name = f"rollout_{rollout_index:02d}_entropy.png"
                    ppl_name = f"rollout_{rollout_index:02d}_ppl.png"
                    _plot_sequence_tokens(
                        tokenizer, token_ids, token_entropys_for_row, prompt_dir / entropy_name,
                        f"Step {step} · Prompt {prompt_index} · Rollout {rollout_index} · Token Entropy",
                        "Token entropy (nats)", False, token_label_cache, sequence_plot_max_columns,
                        sequence_plot_font_size, sequence_plot_dpi,
                    )
                    _plot_sequence_tokens(
                        tokenizer, token_ids, token_ppls_for_row, prompt_dir / ppl_name,
                        f"Step {step} · Prompt {prompt_index} · Rollout {rollout_index} · Token Perplexity",
                        "Token perplexity", True, token_label_cache, sequence_plot_max_columns,
                        sequence_plot_font_size, sequence_plot_dpi,
                    )
                    kl_name = None
                    kl_line_name = None
                    if token_kls_for_visualization is not None:
                        kl_name = f"rollout_{rollout_index:02d}_kl.png"
                        kl_line_name = f"rollout_{rollout_index:02d}_kl_line.png"
                        _plot_sequence_tokens(
                            tokenizer, token_ids, token_kls_for_visualization, prompt_dir / kl_name,
                            f"Step {step} · Prompt {prompt_index} · Rollout {rollout_index} · Token KL",
                            f"Token KL estimate ({kl_estimator})", False, token_label_cache,
                            sequence_plot_max_columns, sequence_plot_font_size, sequence_plot_dpi,
                        )
                        _plot_token_line(
                            token_kls_for_visualization, prompt_dir / kl_line_name,
                            f"Step {step} · Prompt {prompt_index} · Rollout {rollout_index} · Token KL",
                            f"Token KL estimate ({kl_estimator})",
                        )
                    rollout_files.append({
                        "rollout_index": rollout_index, "response_length": len(all_token_ids),
                        "visualized_token_count": len(token_ids),
                        "excluded_special_token_count": excluded_special_token_count,
                        "entropy_plot": entropy_name, "ppl_plot": ppl_name,
                        "entropy_range": [float(token_entropys_for_row.min()), float(token_entropys_for_row.max())],
                        "ppl_range": [float(token_ppls_for_row.min()), float(token_ppls_for_row.max())],
                        "kl_plot": kl_name, "kl_line_plot": kl_line_name,
                        "sequence_kl": float(sequence_kls[row]) if sequence_kls is not None else None,
                        "visualized_sequence_kl": (
                            float(token_kls_for_visualization.mean())
                            if token_kls_for_visualization is not None
                            else None
                        ),
                    })
                selected_prompt_records.append({
                    "prompt_index": prompt_index, "prompt_id": uid, "prompt": prompt_texts[uid],
                    "num_rollouts": len(prompt_rows[uid]), "directory": prompt_dir.name,
                    "rollouts": rollout_files,
                })
            _atomic_json(
                prompt_sequence_dir / "selected_prompts.json",
                {
                    "step": step, "selection_seed": step, "skip_chinese_sequences": skip_chinese_sequences,
                    "kl_estimator": kl_estimator,
                    "eligible_prompt_count": len(eligible_uids),
                    "skipped_chinese_prompt_count": len(skipped_chinese_uids),
                    "skipped_chinese_prompt_ids": skipped_chinese_uids, "prompts": selected_prompt_records,
                },
            )

    print(f"Dumped rollout telemetry for step {step} to {step_dir}")
    return step_dir


def dump_rollout_telemetry_safely(**kwargs: Any) -> Path | None:
    """Best-effort telemetry boundary: report every failure but never interrupt training."""
    step = kwargs.get("step", "unknown")
    try:
        return dump_rollout_telemetry(**kwargs)
    except Exception as error:
        print(f"WARNING: rollout telemetry failed at step {step}; training will continue without this dump: {error}", flush=True)
        traceback.print_exc()
        return None
