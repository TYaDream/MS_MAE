from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from degradation import DegradationConfig, degrade_sample


@dataclass
class DatasetConfig:
    deterministic: bool = False
    seed: int = 42
    forced_case: int | None = None


class GCMSRestorationDataset(Dataset):
    """Dataset with on-the-fly degradation for GC-MS restoration."""

    def __init__(
        self,
        full_intensity_matrix: np.ndarray,
        mz_values: np.ndarray,
        intensity_scale: float = 1.0,
        class_ids: np.ndarray | None = None,
        is_real_flags: np.ndarray | None = None,
        degradation_cfg: Optional[DegradationConfig] = None,
        dataset_cfg: Optional[DatasetConfig] = None,
    ) -> None:
        super().__init__()
        self.full_intensity_matrix = np.asarray(full_intensity_matrix, dtype=np.float32)
        self.mz_values = np.asarray(mz_values, dtype=np.float32)

        if self.full_intensity_matrix.ndim != 2:
            raise ValueError("full_intensity_matrix must be shape [num_samples, full_feature_num]")

        if self.mz_values.ndim != 1:
            raise ValueError("mz_values must be shape [full_feature_num]")

        if self.full_intensity_matrix.shape[1] != self.mz_values.shape[0]:
            raise ValueError("feature dimension mismatch between intensity matrix and mz_values")
        self.intensity_scale = float(max(1e-8, intensity_scale))

        if class_ids is None:
            class_ids = np.zeros(self.full_intensity_matrix.shape[0], dtype=np.int64)
        self.class_ids = np.asarray(class_ids, dtype=np.int64)
        if self.class_ids.shape[0] != self.full_intensity_matrix.shape[0]:
            raise ValueError("class_ids length mismatch with sample count")

        if is_real_flags is None:
            is_real_flags = np.ones(self.full_intensity_matrix.shape[0], dtype=bool)
        self.is_real_flags = np.asarray(is_real_flags).astype(bool)
        if self.is_real_flags.shape[0] != self.full_intensity_matrix.shape[0]:
            raise ValueError("is_real_flags length mismatch with sample count")

        self.degradation_cfg = degradation_cfg or DegradationConfig()
        self.dataset_cfg = dataset_cfg or DatasetConfig()

        self._rng = np.random.default_rng(self.dataset_cfg.seed)

    def __len__(self) -> int:
        return self.full_intensity_matrix.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target_raw = self.full_intensity_matrix[idx]
        class_id = int(self.class_ids[idx])

        if self.dataset_cfg.deterministic:
            rng = np.random.default_rng(self.dataset_cfg.seed + int(idx))
        else:
            rng = self._rng

        degraded_input, degraded_mask_full, overlap_copy_source_full, case_id = degrade_sample(
            full_intensity=target_raw,
            mz_values=self.mz_values,
            cfg=self.degradation_cfg,
            rng=rng,
            intensity_scale=self.intensity_scale,
            forced_case=self.dataset_cfg.forced_case,
        )
        target = (target_raw / self.intensity_scale).astype(np.float32)
        degraded_input = degraded_input.copy()
        degraded_input[:, 1] = degraded_input[:, 1] / self.intensity_scale

        sample = {
            "degraded_input": torch.from_numpy(degraded_input).float(),
            "target_full_intensity": torch.from_numpy(target).float(),
            "degraded_mask_full": torch.from_numpy(degraded_mask_full),
            "overlap_copy_source_full": torch.from_numpy(overlap_copy_source_full).long(),
            "nonzero_mask_full": torch.from_numpy(target_raw > 0),
            "case_id": torch.tensor(case_id, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "is_real": torch.tensor(bool(self.is_real_flags[idx]), dtype=torch.bool),
        }
        return sample


def collate_fn(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length degraded input [N_remain, 7] to [B, N_max, 7]."""
    batch_size = len(batch)
    max_len = max(int(item["degraded_input"].shape[0]) for item in batch)
    max_len = max(1, max_len)

    padded_input = torch.zeros(batch_size, max_len, 7, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # status column uses 3 for PAD token, distinct from real statuses 0..2
    # 0 normal, 1 overlap, 2 missing
    padded_input[:, :, 4] = 3.0
    # drift column uses 3 for PAD token, distinct from real values 0..2
    # 0 no drift, 1 drift up, 2 drift down
    padded_input[:, :, 6] = 3.0

    targets = torch.stack([item["target_full_intensity"] for item in batch], dim=0)
    degraded_masks = torch.stack([item["degraded_mask_full"] for item in batch], dim=0).bool()
    overlap_copy_source_full = torch.stack([item["overlap_copy_source_full"] for item in batch], dim=0).long()
    nonzero_masks = torch.stack([item["nonzero_mask_full"] for item in batch], dim=0).bool()
    case_ids = torch.stack([item["case_id"] for item in batch], dim=0)
    class_ids = torch.stack([item["class_id"] for item in batch], dim=0)
    is_real = torch.stack([item["is_real"] for item in batch], dim=0).bool()

    for i, item in enumerate(batch):
        n_i = int(item["degraded_input"].shape[0])
        padded_input[i, :n_i, :] = item["degraded_input"]
        token_mask[i, :n_i] = True

    return {
        "degraded_input": padded_input,
        "token_mask": token_mask,
        "target_full_intensity": targets,
        "degraded_mask_full": degraded_masks,
        "overlap_copy_source_full": overlap_copy_source_full,
        "nonzero_mask_full": nonzero_masks,
        "case_id": case_ids,
        "class_id": class_ids,
        "is_real": is_real,
    }
