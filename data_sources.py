from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    full_intensity: np.ndarray
    mz_values: np.ndarray
    source_name: str
    feature_names: Optional[list[str]] = None
    sample_ids: Optional[np.ndarray] = None
    labels_str: Optional[np.ndarray] = None
    is_synthetic: Optional[np.ndarray] = None

    def validate(self) -> None:
        self.full_intensity = np.asarray(self.full_intensity, dtype=np.float32)
        self.mz_values = np.asarray(self.mz_values, dtype=np.float32)

        if self.full_intensity.ndim != 2:
            raise ValueError("full_intensity must be [num_samples, full_feature_num]")
        if self.mz_values.ndim != 1:
            raise ValueError("mz_values must be [full_feature_num]")
        if self.full_intensity.shape[1] != self.mz_values.shape[0]:
            raise ValueError(
                f"feature mismatch: intensity={self.full_intensity.shape[1]}, mz={self.mz_values.shape[0]}"
            )

        n = self.full_intensity.shape[0]
        if self.labels_str is not None and len(self.labels_str) != n:
            raise ValueError("labels_str length mismatch with sample count")
        if self.is_synthetic is not None and len(self.is_synthetic) != n:
            raise ValueError("is_synthetic length mismatch with sample count")


def _feature_col_key(name: str) -> int:
    return int(name.replace("feature", ""))


class BaseDataSource(ABC):
    @abstractmethod
    def load(self) -> DataBundle:
        raise NotImplementedError


class ExcelDataSource(BaseDataSource):
    """Load from gcms_preprocessed_samples.xlsx style file."""

    def __init__(
        self,
        excel_path: str,
        raw_sheet: str = "raw_samples",
        mapping_sheet: str = "feature_mapping",
    ) -> None:
        self.excel_path = excel_path
        self.raw_sheet = raw_sheet
        self.mapping_sheet = mapping_sheet

    def load(self) -> DataBundle:
        raw_df = pd.read_excel(self.excel_path, sheet_name=self.raw_sheet)
        feature_cols = sorted(
            [c for c in raw_df.columns if c.startswith("feature")],
            key=_feature_col_key,
        )
        if not feature_cols:
            raise ValueError(f"No feature columns found in sheet: {self.raw_sheet}")

        full_intensity = raw_df[feature_cols].to_numpy(dtype=np.float32)

        map_df = pd.read_excel(self.excel_path, sheet_name=self.mapping_sheet)
        if "feature_index" in map_df.columns:
            map_df = map_df.sort_values("feature_index")

        if "m_z" in map_df.columns:
            mz_values = map_df["m_z"].to_numpy(dtype=np.float32)
        elif "mz" in map_df.columns:
            mz_values = map_df["mz"].to_numpy(dtype=np.float32)
        else:
            raise ValueError(f"{self.mapping_sheet} must contain m_z (or mz) column")

        sample_ids = None
        if "sample_id" in raw_df.columns:
            sample_ids = raw_df["sample_id"].to_numpy()

        labels_str = None
        if "class" in raw_df.columns:
            labels_str = raw_df["class"].astype(str).to_numpy()

        is_synthetic = None
        if "is_synthetic" in raw_df.columns:
            is_synthetic = raw_df["is_synthetic"].to_numpy().astype(bool)

        bundle = DataBundle(
            full_intensity=full_intensity,
            mz_values=mz_values,
            source_name=f"excel:{self.excel_path}",
            feature_names=feature_cols,
            sample_ids=sample_ids,
            labels_str=labels_str,
            is_synthetic=is_synthetic,
        )
        bundle.validate()
        return bundle


class SyntheticDataSource(BaseDataSource):
    """Small runnable demo data source."""

    def __init__(
        self,
        num_samples: int = 512,
        full_feature_num: int = 128,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.full_feature_num = full_feature_num
        self.seed = seed

    def load(self) -> DataBundle:
        rng = np.random.default_rng(self.seed)

        mz_values = np.sort(rng.uniform(50.0, 600.0, size=self.full_feature_num)).astype(np.float32)
        full_intensity = np.zeros((self.num_samples, self.full_feature_num), dtype=np.float32)

        class_names = np.array(["Class0", "Class1", "Class2"], dtype=object)
        labels_str = class_names[rng.integers(0, len(class_names), size=self.num_samples)]

        for i in range(self.num_samples):
            n_peaks = int(
                rng.integers(
                    max(6, self.full_feature_num // 12),
                    max(12, self.full_feature_num // 4),
                )
            )
            peak_idx = rng.choice(self.full_feature_num, size=n_peaks, replace=False)
            amps = rng.lognormal(mean=1.2 + 0.15 * (i % len(class_names)), sigma=0.8, size=n_peaks).astype(np.float32)

            full_intensity[i, peak_idx] = amps

            for p, a in zip(peak_idx, amps):
                if p - 1 >= 0:
                    full_intensity[i, p - 1] += a * float(rng.uniform(0.05, 0.20))
                if p + 1 < self.full_feature_num:
                    full_intensity[i, p + 1] += a * float(rng.uniform(0.05, 0.20))

            full_intensity[i] += rng.uniform(0.0, 0.01, size=self.full_feature_num).astype(np.float32)

            sparse_mask = rng.random(self.full_feature_num) < 0.60
            full_intensity[i, sparse_mask] *= rng.uniform(0.0, 0.08)

        full_intensity[full_intensity < 1e-3] = 0.0

        feature_names = [f"feature{i}" for i in range(1, self.full_feature_num + 1)]
        bundle = DataBundle(
            full_intensity=full_intensity,
            mz_values=mz_values,
            source_name="synthetic",
            feature_names=feature_names,
            labels_str=labels_str,
            is_synthetic=np.ones(self.num_samples, dtype=bool),
        )
        bundle.validate()
        return bundle


class NPZDataSource(BaseDataSource):
    """
    Load from NPZ with keys:
    - full_intensity: [num_samples, full_feature_num]
    - mz_values: [full_feature_num]
    Optional:
    - feature_names
    - labels_str or labels_int+label_names
    - is_synthetic
    """

    def __init__(self, npz_path: str) -> None:
        self.npz_path = npz_path

    def load(self) -> DataBundle:
        data = np.load(self.npz_path, allow_pickle=True)
        if "full_intensity" not in data or "mz_values" not in data:
            raise ValueError("NPZ must contain keys: full_intensity and mz_values")

        feature_names = None
        if "feature_names" in data:
            feature_names = [str(x) for x in data["feature_names"].tolist()]

        labels_str = None
        if "labels_str" in data:
            labels_str = np.asarray(data["labels_str"], dtype=object)
        elif "labels_int" in data and "label_names" in data:
            li = np.asarray(data["labels_int"], dtype=np.int64)
            ln = data["label_names"].tolist()
            labels_str = np.asarray([str(ln[i]) for i in li], dtype=object)

        is_synthetic = None
        if "is_synthetic" in data:
            is_synthetic = np.asarray(data["is_synthetic"]).astype(bool)

        bundle = DataBundle(
            full_intensity=data["full_intensity"],
            mz_values=data["mz_values"],
            source_name=f"npz:{self.npz_path}",
            feature_names=feature_names,
            labels_str=labels_str,
            is_synthetic=is_synthetic,
        )
        bundle.validate()
        return bundle


DATA_SOURCE_REGISTRY = {
    "excel": ExcelDataSource,
    "synthetic": SyntheticDataSource,
    "npz": NPZDataSource,
}


def build_data_source(kind: str, **kwargs) -> BaseDataSource:
    kind = kind.lower()
    if kind not in DATA_SOURCE_REGISTRY:
        raise ValueError(f"Unsupported dataset type: {kind}. Supported: {list(DATA_SOURCE_REGISTRY.keys())}")
    return DATA_SOURCE_REGISTRY[kind](**kwargs)


def resolve_dataset_type(
    dataset_type: str,
    use_synthetic: bool,
    data_excel: str,
    data_npz: str,
) -> str:
    if dataset_type != "auto":
        return dataset_type
    if use_synthetic:
        return "synthetic"
    if data_excel and os.path.exists(data_excel):
        return "excel"
    if data_npz and os.path.exists(data_npz):
        return "npz"
    return "synthetic"


def load_data_bundle(
    dataset_type: str,
    *,
    data_excel: str = "",
    data_npz: str = "",
    demo_num_samples: int = 512,
    demo_full_feature_num: int = 128,
    seed: int = 42,
) -> DataBundle:
    if dataset_type == "excel":
        source = build_data_source("excel", excel_path=data_excel)
    elif dataset_type == "npz":
        source = build_data_source("npz", npz_path=data_npz)
    elif dataset_type == "synthetic":
        source = build_data_source(
            "synthetic",
            num_samples=demo_num_samples,
            full_feature_num=demo_full_feature_num,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    return source.load()


def merge_data_bundles(
    bundles: list[DataBundle],
    source_name: str = "merged",
) -> DataBundle:
    if not bundles:
        raise ValueError("bundles cannot be empty")

    base = bundles[0]
    f = base.full_intensity.shape[1]
    mz = base.mz_values.astype(np.float32)

    xs = []
    ys = []
    synth_flags = []
    sample_ids = []

    for b in bundles:
        if b.full_intensity.shape[1] != f:
            raise ValueError("Cannot merge bundles with different feature dimensions")
        if b.mz_values.shape[0] != mz.shape[0]:
            raise ValueError("Cannot merge bundles with different m/z length")
        if not np.allclose(b.mz_values, mz, atol=1e-6, rtol=1e-6):
            raise ValueError("Cannot merge bundles with different m/z values/order")

        xs.append(b.full_intensity.astype(np.float32))

        if b.labels_str is None:
            ys.append(np.asarray(["Class0"] * b.full_intensity.shape[0], dtype=object))
        else:
            ys.append(np.asarray(b.labels_str, dtype=object))

        if b.is_synthetic is None:
            synth_flags.append(np.zeros(b.full_intensity.shape[0], dtype=bool))
        else:
            synth_flags.append(np.asarray(b.is_synthetic).astype(bool))

        if b.sample_ids is not None:
            sample_ids.append(np.asarray(b.sample_ids))

    out = DataBundle(
        full_intensity=np.concatenate(xs, axis=0),
        mz_values=mz,
        source_name=source_name,
        feature_names=base.feature_names,
        sample_ids=np.concatenate(sample_ids, axis=0) if sample_ids else None,
        labels_str=np.concatenate(ys, axis=0),
        is_synthetic=np.concatenate(synth_flags, axis=0),
    )
    out.validate()
    return out


def drop_all_zero_features(bundle: DataBundle, atol: float = 0.0) -> DataBundle:
    """Remove features that are zero for every sample in the bundle."""
    x = np.asarray(bundle.full_intensity, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("bundle.full_intensity must be 2D")

    keep_mask = np.any(np.abs(x) > float(atol), axis=0)
    if not np.any(keep_mask):
        raise ValueError("All features are zero across all samples; nothing to keep")
    if bool(np.all(keep_mask)):
        bundle.validate()
        return bundle

    feature_names = None
    if bundle.feature_names is not None:
        feature_names = [name for name, keep in zip(bundle.feature_names, keep_mask.tolist()) if keep]

    out = DataBundle(
        full_intensity=x[:, keep_mask],
        mz_values=np.asarray(bundle.mz_values, dtype=np.float32)[keep_mask],
        source_name=f"{bundle.source_name}[nonzero_features]",
        feature_names=feature_names,
        sample_ids=None if bundle.sample_ids is None else np.asarray(bundle.sample_ids),
        labels_str=None if bundle.labels_str is None else np.asarray(bundle.labels_str, dtype=object),
        is_synthetic=None if bundle.is_synthetic is None else np.asarray(bundle.is_synthetic).astype(bool),
    )
    out.validate()
    return out


def build_label_encoder(labels: np.ndarray) -> tuple[np.ndarray, dict[str, int], list[str]]:
    labels = np.asarray(labels, dtype=object)
    classes = sorted(np.unique(labels).tolist())
    cls_to_id = {str(c): i for i, c in enumerate(classes)}
    y = np.asarray([cls_to_id[str(x)] for x in labels], dtype=np.int64)
    return y, cls_to_id, classes
