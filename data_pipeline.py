from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GCMSRestorationDataset, DatasetConfig, collate_fn
from degradation import DegradationConfig
from data_sources import DataBundle


@dataclass
class SplitBundle:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    all_loader: DataLoader
    splits: SplitBundle


def normalize_intensity(full_intensity: np.ndarray) -> tuple[np.ndarray, float]:
    """Dataset-global max normalization shared across all samples/features."""
    scale = float(np.max(full_intensity))
    if scale <= 0:
        scale = 1.0
    normalized = (full_intensity / scale).astype(np.float32)
    return normalized, scale


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> SplitBundle:
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)

    train_idx_list = []
    val_idx_list = []
    test_idx_list = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx_list.append(idx[:n_train])
        val_idx_list.append(idx[n_train : n_train + n_val])
        test_idx_list.append(idx[n_train + n_val :])

    train_idx = np.concatenate(train_idx_list) if train_idx_list else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_idx_list) if val_idx_list else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_idx_list) if test_idx_list else np.array([], dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return SplitBundle(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def _make_loader(
    full_intensity: np.ndarray,
    mz_values: np.ndarray,
    intensity_scale: float,
    class_ids: np.ndarray,
    indices: np.ndarray,
    degradation_cfg: DegradationConfig,
    deterministic: bool,
    seed: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    is_real_flags: np.ndarray | None = None,
    forced_case: int | None = None,
) -> DataLoader:
    dataset = GCMSRestorationDataset(
        full_intensity_matrix=full_intensity[indices],
        mz_values=mz_values,
        intensity_scale=intensity_scale,
        class_ids=class_ids[indices],
        is_real_flags=(is_real_flags[indices] if is_real_flags is not None else None),
        degradation_cfg=degradation_cfg,
        dataset_cfg=DatasetConfig(deterministic=deterministic, seed=seed, forced_case=forced_case),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def build_loaders_from_bundle(
    bundle: DataBundle,
    class_ids: np.ndarray,
    degradation_cfg: DegradationConfig,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    is_real_flags: np.ndarray | None = None,
) -> tuple[LoaderBundle, np.ndarray, float]:
    full_intensity_raw = np.asarray(bundle.full_intensity, dtype=np.float32)
    full_intensity_norm, intensity_scale = normalize_intensity(full_intensity_raw)

    splits = stratified_split_indices(
        y=class_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_loader = _make_loader(
        full_intensity=full_intensity_raw,
        intensity_scale=intensity_scale,
        mz_values=bundle.mz_values,
        class_ids=class_ids,
        is_real_flags=is_real_flags,
        indices=splits.train_idx,
        degradation_cfg=degradation_cfg,
        deterministic=False,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = _make_loader(
        full_intensity=full_intensity_raw,
        intensity_scale=intensity_scale,
        mz_values=bundle.mz_values,
        class_ids=class_ids,
        is_real_flags=is_real_flags,
        indices=splits.val_idx,
        degradation_cfg=degradation_cfg,
        deterministic=True,
        seed=seed + 1000,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = _make_loader(
        full_intensity=full_intensity_raw,
        intensity_scale=intensity_scale,
        mz_values=bundle.mz_values,
        class_ids=class_ids,
        is_real_flags=is_real_flags,
        indices=splits.test_idx,
        degradation_cfg=degradation_cfg,
        deterministic=True,
        seed=seed + 2000,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    all_indices = np.arange(full_intensity_norm.shape[0], dtype=np.int64)
    all_loader = _make_loader(
        full_intensity=full_intensity_raw,
        intensity_scale=intensity_scale,
        mz_values=bundle.mz_values,
        class_ids=class_ids,
        is_real_flags=is_real_flags,
        indices=all_indices,
        degradation_cfg=degradation_cfg,
        deterministic=True,
        seed=seed + 3000,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return (
        LoaderBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            all_loader=all_loader,
            splits=splits,
        ),
        full_intensity_norm,
        intensity_scale,
    )


def build_case_loader(
    full_intensity_raw: np.ndarray,
    mz_values: np.ndarray,
    intensity_scale: float,
    class_ids: np.ndarray,
    degradation_cfg: DegradationConfig,
    case_id: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    is_real_flags: np.ndarray | None = None,
) -> DataLoader:
    indices = np.arange(full_intensity_raw.shape[0], dtype=np.int64)
    return _make_loader(
        full_intensity=full_intensity_raw,
        intensity_scale=intensity_scale,
        mz_values=mz_values,
        class_ids=class_ids,
        is_real_flags=is_real_flags,
        indices=indices,
        degradation_cfg=degradation_cfg,
        deterministic=True,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        forced_case=case_id,
    )
