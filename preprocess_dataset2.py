from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def discover_study_dir(study_name: str) -> Path:
    root = Path("massspec_data")
    if not root.exists():
        raise FileNotFoundError("massspec_data directory not found")

    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("dataset"):
            continue
        candidate = child / study_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find study directory for {study_name!r} under massspec_data")


def load_sample_metadata(
    study_dir: Path,
    *,
    class_column: str,
) -> pd.DataFrame:
    sample_file = next(study_dir.glob("s_*.txt"))
    sample_df = pd.read_csv(sample_file, sep="\t").copy()

    required = [
        "Sample Name",
        class_column,
        "Factor Value[Stage]",
        "Factor Value[Biological replicate]",
        "Factor Value[Technical replicate]",
    ]
    missing = [col for col in required if col not in sample_df.columns]
    if missing:
        raise ValueError(f"Missing columns in {sample_file.name}: {missing}")

    sample_df["sample_id"] = sample_df["Sample Name"].astype(str)
    sample_df["class"] = sample_df[class_column].astype(str)
    sample_df["cultivar"] = sample_df["Factor Value[Cultivar]"].astype(str)
    sample_df["stage"] = sample_df["Factor Value[Stage]"].astype(str)
    sample_df["form"] = sample_df["sample_id"].str.rsplit("-", n=1).str[-1]
    sample_df["biological_replicate"] = sample_df["Factor Value[Biological replicate]"].astype(str)
    sample_df["technical_replicate"] = sample_df["Factor Value[Technical replicate]"].astype(str)
    return sample_df[
        [
            "sample_id",
            "class",
            "cultivar",
            "stage",
            "form",
            "biological_replicate",
            "technical_replicate",
        ]
    ].drop_duplicates().reset_index(drop=True)


def load_maf_table(study_dir: Path) -> pd.DataFrame:
    maf_file = next(study_dir.glob("m_*_maf.tsv"))
    return pd.read_csv(maf_file, sep="\t").copy()


def build_raw_samples(
    sample_meta: pd.DataFrame,
    maf_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    sample_ids = sample_meta["sample_id"].tolist()
    missing = [sid for sid in sample_ids if sid not in maf_df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} sample columns in MAF, first few: {missing[:5]}")

    values = maf_df[sample_ids].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32).T
    keep_mask = np.any(np.abs(values) > 0.0, axis=0)
    if not np.any(keep_mask):
        raise ValueError("All MTBLS968 metabolite features are zero across all samples")

    values = values[:, keep_mask]
    feature_cols = [f"feature{i}" for i in range(1, values.shape[1] + 1)]

    raw_df = sample_meta.copy()
    for idx, col in enumerate(feature_cols):
        raw_df[col] = values[:, idx]

    return raw_df, feature_cols, keep_mask


def build_normalized_samples(raw_df: pd.DataFrame, feature_cols: list[str], zero_floor_ratio: float) -> pd.DataFrame:
    norm_df = raw_df.copy()
    feat = norm_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    feat_max = np.max(feat, axis=0)
    safe_max = np.where(feat_max > 0.0, feat_max, 1.0)
    feat = feat / safe_max
    if zero_floor_ratio > 0.0:
        feat[feat < zero_floor_ratio] = 0.0
    norm_df.loc[:, feature_cols] = feat.astype(np.float32)
    return norm_df


def build_feature_mapping(maf_df: pd.DataFrame, keep_mask: np.ndarray, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    kept = maf_df.loc[keep_mask].reset_index(drop=True).copy()
    mapping = pd.DataFrame(
        {
            "feature_index": np.arange(1, len(feature_cols) + 1, dtype=np.int64),
            "feature_name": feature_cols,
            "database_identifier": kept["database_identifier"].astype(str).to_numpy(),
            "metabolite_identification": kept["metabolite_identification"].astype(str).to_numpy(),
            "chemical_formula": kept["chemical_formula"].astype(str).to_numpy(),
            "m_z": pd.to_numeric(kept["mass_to_charge"], errors="coerce").to_numpy(dtype=np.float32),
            "retention_time": pd.to_numeric(kept["retention_time"], errors="coerce").to_numpy(dtype=np.float32),
            "fragmentation": kept["fragmentation"].astype(str).to_numpy(),
            "reliability": kept["reliability"].astype(str).to_numpy(),
            "source_study": "MTBLS968",
        }
    )
    mz_values = mapping["m_z"].to_numpy(dtype=np.float32)
    return mapping, mz_values


def save_npz(
    output_npz: Path,
    raw_df: pd.DataFrame,
    feature_cols: list[str],
    mz_values: np.ndarray,
) -> None:
    np.savez_compressed(
        output_npz,
        full_intensity=raw_df[feature_cols].to_numpy(dtype=np.float32),
        mz_values=mz_values.astype(np.float32),
        labels_str=raw_df["class"].astype(str).to_numpy(dtype=object),
        sample_ids=raw_df["sample_id"].astype(str).to_numpy(dtype=object),
        is_synthetic=np.zeros(raw_df.shape[0], dtype=bool),
        feature_names=np.asarray(feature_cols, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MTBLS968 MAF table into dataset2 workbook/npz format.")
    parser.add_argument("--study_name", type=str, default="MTBLS968")
    parser.add_argument("--study_dir", type=Path, default=None)
    parser.add_argument("--class_column", type=str, default="Factor Value[Cultivar]")
    parser.add_argument("--normalized_zero_floor", type=float, default=1e-4)
    parser.add_argument("--output_excel", type=Path, default=Path("massspec_data/dataset2/dataset2_preprocessed.xlsx"))
    parser.add_argument("--output_npz", type=Path, default=Path("massspec_data/dataset2/dataset2_preprocessed.npz"))
    args = parser.parse_args()

    study_dir = args.study_dir if args.study_dir is not None else discover_study_dir(args.study_name)
    if not study_dir.exists():
        raise FileNotFoundError(f"study_dir not found: {study_dir}")

    sample_meta = load_sample_metadata(study_dir, class_column=args.class_column)
    maf_df = load_maf_table(study_dir)
    raw_df, feature_cols, keep_mask = build_raw_samples(sample_meta=sample_meta, maf_df=maf_df)
    normalized_df = build_normalized_samples(
        raw_df=raw_df,
        feature_cols=feature_cols,
        zero_floor_ratio=float(args.normalized_zero_floor),
    )
    feature_map, mz_values = build_feature_mapping(maf_df=maf_df, keep_mask=keep_mask, feature_cols=feature_cols)

    args.output_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_excel) as writer:
        raw_df.to_excel(writer, sheet_name="raw_samples", index=False)
        normalized_df.to_excel(writer, sheet_name="normalized_samples", index=False)
        feature_map.to_excel(writer, sheet_name="feature_mapping", index=False)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_npz(args.output_npz, raw_df=raw_df, feature_cols=feature_cols, mz_values=mz_values)

    print(f"Saved Excel: {args.output_excel.resolve()}")
    print(f"Saved NPZ: {args.output_npz.resolve()}")
    print(
        f"Samples: {raw_df.shape[0]} | Features: {len(feature_cols)} | "
        f"Classes: {raw_df['class'].value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()
