from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DegradationConfig:
    # Cases (no normal case):
    # 0 missing peaks
    # 1 overlap peaks
    # 2 baseline drift
    # 3 baseline drift + missing peaks + overlap peaks
    p_case: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    # Tuned defaults for this dataset (small sample size + GC-MS overlap characteristics).
    p_random_zero: float = 0.12
    p_overlap: float = 0.18
    overlap_min_len: int = 2
    overlap_max_len: int = 3
    # Baseline drift magnitude expressed as a fraction of the dataset-global intensity scale.
    baseline_shift_min: float = 0.02
    baseline_shift_max: float = 0.08
    p_baseline_increase: float = 0.50
    overlap_min_keep_ratio: float = 0.30
    # Keep zero-intensity non-overlap features in input and let status encode degradation.
    keep_zero_tokens: bool = True


def _build_base_tokens(
    feature_indices: np.ndarray,
    intensities: np.ndarray,
    mz_values: np.ndarray,
    status_values: np.ndarray | None = None,
    drift_values: np.ndarray | None = None,
) -> List[Dict]:
    """Build one token per feature before any overlap merge is applied."""
    if status_values is None:
        status_values = np.zeros_like(intensities, dtype=np.int64)
    if drift_values is None:
        drift_values = np.zeros_like(intensities, dtype=np.int64)

    tokens: List[Dict] = []
    for i, feature_index in enumerate(feature_indices.astype(int).tolist()):
        tokens.append(
            {
                "feature_index": feature_index,
                "intensity": float(intensities[i]),
                "base_intensity": float(intensities[i]),
                "start_index": -1,
                "end_index": -1,
                "status": int(status_values[i]),
                "mz": float(mz_values[i]),
                "drift_direction": int(drift_values[i]),
                "covered_indices": [feature_index],
                "is_overlap": False,
            }
        )
    return tokens


def _sample_random_overlap_groups(
    tokens: List[Dict],
    p_overlap: float,
    min_len: int,
    max_len: int,
    rng: np.random.Generator,
    force_one_group: bool = False,
) -> List[List[int]]:
    """Randomly pick non-missing peaks and partition them into overlap groups."""
    min_len = max(2, int(min_len))
    max_len = max(min_len, int(max_len))

    eligible = [
        int(token["feature_index"])
        for token in tokens
        if float(token["intensity"]) > 0.0 and int(token["status"]) != 2
    ]
    if len(eligible) < min_len:
        return []

    selected = [idx for idx in eligible if rng.random() < p_overlap]
    if force_one_group and len(selected) < min_len:
        selected_set = set(selected)
        remaining = [idx for idx in eligible if idx not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: min_len - len(selected)])

    if len(selected) < min_len:
        return []

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)

    groups: List[List[int]] = []
    pos = 0
    while (selected.size - pos) >= min_len:
        remaining = int(selected.size - pos)
        if remaining <= max_len:
            group_size = remaining
        else:
            group_size = int(rng.integers(min_len, max_len + 1))
            if remaining - group_size == 1:
                if group_size > min_len:
                    group_size -= 1
                else:
                    group_size += 1

        group = sorted(selected[pos : pos + group_size].astype(int).tolist())
        groups.append(group)
        pos += group_size

    return groups


def _merge_tokens_by_groups(tokens: List[Dict], groups: List[List[int]]) -> List[Dict]:
    """
    Merge randomly selected peaks into one token kept at the smallest feature index.

    start_index/end_index record the min/max covered feature indices even when the
    merged group is not contiguous.
    """
    if len(groups) == 0:
        return tokens

    token_by_index = {int(token["feature_index"]): token for token in tokens}
    group_by_anchor: Dict[int, List[int]] = {}
    merged_members: set[int] = set()
    for group in groups:
        if len(group) < 2:
            continue
        group_sorted = sorted(int(idx) for idx in group)
        anchor = group_sorted[0]
        group_by_anchor[anchor] = group_sorted
        merged_members.update(group_sorted[1:])

    merged_tokens: List[Dict] = []
    for token in tokens:
        feature_index = int(token["feature_index"])
        if feature_index in merged_members:
            continue
        if feature_index not in group_by_anchor:
            merged_tokens.append(token)
            continue

        covered = group_by_anchor[feature_index]
        member_tokens = [token_by_index[idx] for idx in covered]
        merged_intensity = float(sum(float(member["intensity"]) for member in member_tokens))
        merged_tokens.append(
            {
                "feature_index": feature_index,
                "intensity": merged_intensity,
                "base_intensity": merged_intensity,
                "start_index": covered[0],
                "end_index": covered[-1],
                "status": 1,
                "mz": float(token_by_index[feature_index]["mz"]),
                "drift_direction": int(token_by_index[feature_index]["drift_direction"]),
                "covered_indices": covered,
                "is_overlap": True,
            }
        )

    return merged_tokens


def _apply_random_zero_type1(
    tokens: List[Dict],
    p_random_zero: float,
    protected_indices: set[int],
    rng: np.random.Generator,
    degraded_mask_full: np.ndarray,
) -> int:
    """Randomly set non-overlap token intensities to zero, marked as status=2."""
    changed = 0
    for token in tokens:
        if token["is_overlap"]:
            continue
        if any(int(idx) in protected_indices for idx in token["covered_indices"]):
            continue
        if rng.random() < p_random_zero:
            token["intensity"] = 0.0
            token["status"] = 2
            for idx in token["covered_indices"]:
                degraded_mask_full[idx - 1] = True
            changed += 1
    return changed


def apply_baseline_drift(
    full_intensity: np.ndarray,
    shift_min: float,
    shift_max: float,
    rng: np.random.Generator,
    p_increase: float = 0.5,
    atol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a uniform baseline drift to all non-zero peaks in one sample:
        intensity <- clip(intensity +/- shift, min=0)

    Returns:
        drifted_intensity: shifted 1D intensity array
        changed_mask: bool mask where the drift changed the value
        drift_values: int array, 0=no drift, 1=shift up, 2=shift down
    """
    values = np.asarray(full_intensity, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("full_intensity must be a 1D array")

    lo = max(0.0, float(min(shift_min, shift_max)))
    hi = max(lo, float(max(shift_min, shift_max)))
    pos_mask = values > float(max(0.0, atol))
    drift_values = np.zeros(values.shape[0], dtype=np.int64)
    if hi <= 0.0 or not np.any(pos_mask):
        return values.copy(), np.zeros(values.shape[0], dtype=bool), drift_values

    shift = float(rng.uniform(lo, hi)) if hi > lo else hi
    increase = bool(rng.random() < float(np.clip(p_increase, 0.0, 1.0)))
    signed_shift = shift if increase else -shift

    drifted = values.copy()
    drifted[pos_mask] = np.clip(drifted[pos_mask] + signed_shift, a_min=0.0, a_max=None)
    changed_mask = np.abs(drifted - values) > float(max(0.0, atol))
    drift_values[pos_mask] = 1 if increase else 2
    return drifted.astype(np.float32), changed_mask.astype(bool), drift_values


def degrade_sample(
    full_intensity: np.ndarray,
    mz_values: np.ndarray,
    cfg: DegradationConfig,
    rng: np.random.Generator | None = None,
    intensity_scale: float = 1.0,
    forced_case: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Degrade one full sample.

    Args:
        full_intensity: [F]
        mz_values: [F]
        cfg: degradation config
        rng: numpy random generator
        intensity_scale: dataset-global scale used to convert baseline shift ratio to raw units
        forced_case: optional fixed case in [0, len(cfg.p_case)-1]

    Returns:
        degraded_tokens: [N_remain, 7]
            columns = [feature_index, intensity, start_index, end_index, status, mz, drift_direction]
        degraded_mask_full: [F] bool, True where degradation occurred
        overlap_copy_source_full: [F] int64, 0 means no copy target. Positive values
            are 1-based anchor feature indices whose overlap token should be copied in.
        case_id: int in [0, len(cfg.p_case)-1]
    """
    if rng is None:
        rng = np.random.default_rng()

    full_intensity = np.asarray(full_intensity, dtype=np.float32)
    mz_values = np.asarray(mz_values, dtype=np.float32)
    full_feature_num = full_intensity.shape[0]
    intensity_scale = float(max(1e-8, intensity_scale))

    feature_indices = np.arange(1, full_feature_num + 1, dtype=np.int64)
    degraded_mask_full = np.zeros(full_feature_num, dtype=bool)
    overlap_copy_source_full = np.zeros(full_feature_num, dtype=np.int64)

    if forced_case is not None:
        if forced_case < 0 or forced_case >= len(cfg.p_case):
            raise ValueError(f"forced_case must be in [0, {len(cfg.p_case)-1}]")
        case_id = int(forced_case)
    else:
        p = np.asarray(cfg.p_case, dtype=np.float64)
        if p.ndim != 1 or p.size < 1:
            raise ValueError("cfg.p_case must be a non-empty 1D tuple/list")
        if np.any(p < 0):
            raise ValueError("cfg.p_case must be non-negative")
        s = float(p.sum())
        if s <= 0:
            raise ValueError("cfg.p_case sum must be > 0")
        p = p / s
        case_id = int(rng.choice(p.size, p=p))

    shifted_intensity = full_intensity.copy()
    status_values = np.zeros(full_feature_num, dtype=np.int64)
    drift_values = np.zeros(full_feature_num, dtype=np.int64)

    if case_id in (2, 3):
        shifted_intensity, drift_mask, drift_values = apply_baseline_drift(
            full_intensity=full_intensity,
            shift_min=cfg.baseline_shift_min * intensity_scale,
            shift_max=cfg.baseline_shift_max * intensity_scale,
            rng=rng,
            p_increase=cfg.p_baseline_increase,
        )
        degraded_mask_full |= drift_mask

    tokens = _build_base_tokens(
        feature_indices,
        shifted_intensity,
        mz_values,
        status_values=status_values,
        drift_values=drift_values,
    )

    # Case logic:
    # 0 missing peaks
    # 1 overlap peaks
    # 2 baseline drift
    # 3 baseline drift -> missing peaks -> overlap peaks
    if case_id in (0, 3):
        p_zero = cfg.p_random_zero
        changed = _apply_random_zero_type1(tokens, p_zero, set(), rng, degraded_mask_full)
        if changed == 0:
            # Enforce at least one missing feature for cases that include missing peaks.
            non_overlap_pos = [i for i, t in enumerate(tokens) if not t["is_overlap"]]
            if len(non_overlap_pos) > 0:
                k = int(rng.choice(non_overlap_pos))
                token = tokens[k]
                token["intensity"] = 0.0
                token["status"] = 2
                for idx in token["covered_indices"]:
                    degraded_mask_full[idx - 1] = True

    if case_id in (1, 3):
        overlap_groups = _sample_random_overlap_groups(
            tokens=tokens,
            p_overlap=cfg.p_overlap,
            min_len=cfg.overlap_min_len,
            max_len=cfg.overlap_max_len,
            rng=rng,
            force_one_group=(case_id in (1, 3)),
        )
        for group in overlap_groups:
            anchor = int(group[0])
            for idx in group:
                degraded_mask_full[idx - 1] = True
            for idx in group[1:]:
                overlap_copy_source_full[idx - 1] = anchor
        tokens = _merge_tokens_by_groups(tokens, overlap_groups)

    # Keep only remaining features in input sequence.
    if cfg.keep_zero_tokens:
        remain_tokens = tokens
    else:
        remain_tokens = [t for t in tokens if (t["is_overlap"] or t["intensity"] > 0.0)]

    # Safety fallback: avoid empty sequence.
    if len(remain_tokens) == 0:
        keep_idx = int(np.argmax(full_intensity))
        remain_tokens = [
            {
                "feature_index": keep_idx + 1,
                "intensity": 0.0,
                "start_index": -1,
                "end_index": -1,
                "status": 2,
                "mz": float(mz_values[keep_idx]),
                "drift_direction": int(drift_values[keep_idx]),
                "covered_indices": [keep_idx + 1],
                "is_overlap": False,
            }
        ]
        degraded_mask_full[keep_idx] = True

    degraded_tokens = np.asarray(
        [
            [
                float(t["feature_index"]),
                float(t["intensity"]),
                float(t["start_index"]),
                float(t["end_index"]),
                float(t["status"]),
                float(t["mz"]),
                float(t["drift_direction"]),
            ]
            for t in remain_tokens
        ],
        dtype=np.float32,
    )

    return degraded_tokens, degraded_mask_full, overlap_copy_source_full, case_id
