from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data_pipeline import stratified_split_indices
from data_sources import build_label_encoder, drop_all_zero_features, load_data_bundle
from dataset import DatasetConfig, GCMSRestorationDataset, collate_fn
from degradation import DegradationConfig
from model import UnconditionalMAERestorationModel
from torch.utils.data import DataLoader
from train_unconditional_mae import (
    FrozenMLPClassifier,
    Log1pZScoreTransform,
    evaluate_degraded_mlp_classification,
    evaluate_recon_mlp_classification,
)
from utils import compute_metrics, load_checkpoint


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    excel_path: Path


@dataclass(frozen=True)
class VariantSpec:
    name: str
    mae_ckpt: Path
    frozen_mlp_ckpt: Path
    extra_args: tuple[str, ...]
    reuse_existing: bool


def _normalize_case_weights(args_dict: dict) -> tuple[float, ...]:
    raw = np.asarray(
        [
            args_dict.get("p_case_0", 0.25),
            args_dict.get("p_case_1", 0.25),
            args_dict.get("p_case_2", 0.25),
            args_dict.get("p_case_3", 0.25),
        ],
        dtype=np.float64,
    )
    raw = np.clip(raw, a_min=0.0, a_max=None)
    if float(raw.sum()) <= 0.0:
        return (0.25, 0.25, 0.25, 0.25)
    return tuple((raw / raw.sum()).tolist())


def _build_degradation_cfg(ckpt: dict, args_dict: dict) -> DegradationConfig:
    cfg_dict = ckpt.get("degradation_cfg")
    if cfg_dict is not None:
        return DegradationConfig(**cfg_dict)
    return DegradationConfig(
        p_case=_normalize_case_weights(args_dict),
        p_random_zero=float(args_dict.get("p_random_zero", 0.2)),
        p_overlap=float(args_dict.get("p_overlap", 0.2)),
        overlap_min_len=int(args_dict.get("overlap_min_len", 2)),
        overlap_max_len=int(args_dict.get("overlap_max_len", 6)),
        baseline_shift_min=float(args_dict.get("baseline_shift_min", 0.02)),
        baseline_shift_max=float(args_dict.get("baseline_shift_max", 0.02)),
        p_baseline_increase=float(args_dict.get("p_baseline_increase", 0.5)),
        keep_zero_tokens=not bool(args_dict.get("drop_zero_tokens", False)),
    )


def _build_mae_model(ckpt: dict, mz_values: np.ndarray, device: torch.device) -> UnconditionalMAERestorationModel:
    args_dict = ckpt.get("args", {})
    model = UnconditionalMAERestorationModel(
        full_feature_num=int(ckpt["full_feature_num"]),
        embedding_dim=int(args_dict.get("embedding_dim", 96)),
        encoder_layers=int(args_dict.get("encoder_layers", 3)),
        decoder_layers=int(args_dict.get("decoder_layers", 3)),
        num_heads=int(args_dict.get("num_heads", 4)),
        dropout=float(args_dict.get("dropout", 0.2)),
        full_mz_values=torch.from_numpy(np.asarray(mz_values, dtype=np.float32)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _build_frozen_mlp(ckpt: dict, device: torch.device) -> tuple[FrozenMLPClassifier, Log1pZScoreTransform, list[str]]:
    hidden_dims = tuple(int(v) for v in ckpt.get("hidden_dims", [64]))
    model = FrozenMLPClassifier(
        input_dim=int(np.asarray(ckpt["transform_mean"]).shape[-1]),
        num_classes=len(ckpt["classes"]),
        hidden_dims=hidden_dims,
        dropout=float(ckpt.get("dropout", 0.10)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    transform = Log1pZScoreTransform(
        mean=np.asarray(ckpt["transform_mean"], dtype=np.float32),
        std=np.asarray(ckpt["transform_std"], dtype=np.float32),
        intensity_scale=float(ckpt["intensity_scale"]),
    )
    return model, transform, list(ckpt["classes"])


def _build_case3_test_loader(
    bundle,
    class_ids: np.ndarray,
    is_real: np.ndarray,
    ckpt: dict,
    batch_size: int,
) -> tuple[DataLoader, np.ndarray]:
    args_dict = ckpt.get("args", {})
    splits = stratified_split_indices(
        y=class_ids,
        train_ratio=float(args_dict.get("train_ratio", 0.6)),
        val_ratio=float(args_dict.get("val_ratio", 0.2)),
        seed=int(args_dict.get("seed", 42)),
    )
    test_idx = np.asarray(splits.test_idx, dtype=np.int64)
    test_real_idx = test_idx[is_real[test_idx]]
    eval_idx = test_real_idx if test_real_idx.size > 0 else test_idx

    dataset = GCMSRestorationDataset(
        full_intensity_matrix=np.asarray(bundle.full_intensity[eval_idx], dtype=np.float32),
        mz_values=np.asarray(bundle.mz_values, dtype=np.float32),
        intensity_scale=float(ckpt["intensity_scale"]),
        class_ids=np.asarray(class_ids[eval_idx], dtype=np.int64),
        is_real_flags=np.ones(eval_idx.shape[0], dtype=bool),
        degradation_cfg=_build_degradation_cfg(ckpt, args_dict),
        dataset_cfg=DatasetConfig(
            deterministic=True,
            seed=int(args_dict.get("seed", 42)) + 4000 + 3,
            forced_case=3,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    return loader, eval_idx


@torch.no_grad()
def _evaluate_case3_reconstruction(
    mae_model: UnconditionalMAERestorationModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    preds = []
    targets = []
    degraded_masks = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = mae_model(
            batch["degraded_input"],
            batch["token_mask"],
            batch["overlap_copy_source_full"],
        )
        preds.append(outputs["pred_intensity"].detach().cpu())
        targets.append(batch["target_full_intensity"].detach().cpu())
        degraded_masks.append(batch["degraded_mask_full"].detach().cpu())
    return compute_metrics(
        pred=torch.cat(preds, dim=0),
        target=torch.cat(targets, dim=0),
        degraded_mask=torch.cat(degraded_masks, dim=0),
    )


def _ensure_variant_trained(
    dataset: DatasetSpec,
    variant: VariantSpec,
    *,
    epochs: int,
    mlp_epochs: int,
    batch_size: int,
    mlp_batch_size: int,
) -> None:
    if variant.reuse_existing and variant.mae_ckpt.exists() and variant.frozen_mlp_ckpt.exists():
        print(f"[Reuse] {dataset.name} / {variant.name}: {variant.mae_ckpt.name}")
        return

    cmd = [
        sys.executable,
        "train_unconditional_mae.py",
        "--dataset_mode",
        "single",
        "--single_dataset_type",
        "excel",
        "--single_excel",
        str(dataset.excel_path),
        "--save_dir",
        str(variant.mae_ckpt.parent),
        "--mae_save_name",
        variant.mae_ckpt.name,
        "--frozen_mlp_save_name",
        variant.frozen_mlp_ckpt.name,
        "--epochs",
        str(int(epochs)),
        "--mlp_epochs",
        str(int(mlp_epochs)),
        "--batch_size",
        str(int(batch_size)),
        "--mlp_batch_size",
        str(int(mlp_batch_size)),
        *variant.extra_args,
    ]
    print(f"[Train] {dataset.name} / {variant.name}")
    print("        " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parent))


def _evaluate_variant(dataset: DatasetSpec, variant: VariantSpec, device: torch.device, batch_size: int) -> dict[str, object]:
    bundle = load_data_bundle(dataset_type="excel", data_excel=str(dataset.excel_path))
    bundle = drop_all_zero_features(bundle)
    labels = np.asarray(bundle.labels_str if bundle.labels_str is not None else ["Class0"] * bundle.full_intensity.shape[0], dtype=object)
    y, _, classes = build_label_encoder(labels)
    if bundle.is_synthetic is None:
        is_real = np.ones(labels.shape[0], dtype=bool)
    else:
        is_real = ~np.asarray(bundle.is_synthetic).astype(bool)

    mae_ckpt = load_checkpoint(str(variant.mae_ckpt), map_location=device)
    mlp_ckpt = load_checkpoint(str(variant.frozen_mlp_ckpt), map_location=device)
    mae_model = _build_mae_model(mae_ckpt, bundle.mz_values, device)
    frozen_mlp, mlp_transform, class_names = _build_frozen_mlp(mlp_ckpt, device)
    case3_loader, eval_idx = _build_case3_test_loader(bundle, y, is_real, mae_ckpt, batch_size=batch_size)

    recon_metrics = _evaluate_case3_reconstruction(mae_model, case3_loader, device)
    degraded_cls = evaluate_degraded_mlp_classification(
        frozen_mlp=frozen_mlp,
        mlp_transform=mlp_transform,
        loader=case3_loader,
        device=device,
        full_feature_num=int(mae_ckpt["full_feature_num"]),
        class_names=class_names,
        return_distribution=False,
    )
    recon_cls = evaluate_recon_mlp_classification(
        mae_model=mae_model,
        frozen_mlp=frozen_mlp,
        mlp_transform=mlp_transform,
        loader=case3_loader,
        device=device,
        class_names=class_names,
        return_distribution=False,
    )
    return {
        "dataset": dataset.name,
        "variant": variant.name,
        "n_case3_test": int(eval_idx.shape[0]),
        "case3_mse": float(recon_metrics["mse"]),
        "case3_mae": float(recon_metrics["mae"]),
        "case3_pearson": float(recon_metrics["pearson"]),
        "case3_nonzero_mse": float(recon_metrics["nonzero_mse"]),
        "case3_degraded_mse": float(recon_metrics["degraded_mse"]),
        "case3_degraded_acc": float(degraded_cls["acc"]),
        "case3_recon_acc": float(recon_cls["acc"]),
        "case3_acc_gain": float(recon_cls["acc"] - degraded_cls["acc"]),
        "best_val_mse": float(mae_ckpt.get("best_val_mse") or float("nan")),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused case3 ablations for dataset1 and dataset2.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--mlp_epochs", type=int, default=220)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mlp_batch_size", type=int, default=16)
    parser.add_argument("--output_csv", type=Path, default=Path("analysis_outputs/case3_ablation_results.csv"))
    args = parser.parse_args()

    datasets = [
        DatasetSpec("dataset1", Path("massspec_data/dataset1/dataset1_preprocessed.xlsx")),
        DatasetSpec("dataset2", Path("massspec_data/dataset2/dataset2_preprocessed.xlsx")),
    ]
    variants_by_dataset = {
        "dataset1": [
            VariantSpec("baseline", Path("checkpoints/dataset1_mae.pt"), Path("checkpoints/dataset1_frozen_mlp.pt"), (), True),
            VariantSpec(
                "no_aux_cls",
                Path("checkpoints/ablate_dataset1_noaux_mae.pt"),
                Path("checkpoints/ablate_dataset1_noaux_frozen_mlp.pt"),
                ("--recon_cls_weight", "0.0"),
                False,
            ),
            VariantSpec(
                "unweighted_recon",
                Path("checkpoints/ablate_dataset1_unweighted_mae.pt"),
                Path("checkpoints/ablate_dataset1_unweighted_frozen_mlp.pt"),
                ("--nonzero_weight", "1.0", "--degraded_weight", "1.0", "--peak_weight", "1.0"),
                False,
            ),
        ],
        "dataset2": [
            VariantSpec("baseline", Path("checkpoints/dataset2_mae.pt"), Path("checkpoints/dataset2_frozen_mlp.pt"), (), True),
            VariantSpec(
                "no_aux_cls",
                Path("checkpoints/ablate_dataset2_noaux_mae.pt"),
                Path("checkpoints/ablate_dataset2_noaux_frozen_mlp.pt"),
                ("--recon_cls_weight", "0.0"),
                False,
            ),
            VariantSpec(
                "unweighted_recon",
                Path("checkpoints/ablate_dataset2_unweighted_mae.pt"),
                Path("checkpoints/ablate_dataset2_unweighted_frozen_mlp.pt"),
                ("--nonzero_weight", "1.0", "--degraded_weight", "1.0", "--peak_weight", "1.0"),
                False,
            ),
        ],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []

    for dataset in datasets:
        for variant in variants_by_dataset[dataset.name]:
            _ensure_variant_trained(
                dataset,
                variant,
                epochs=args.epochs,
                mlp_epochs=args.mlp_epochs,
                batch_size=args.batch_size,
                mlp_batch_size=args.mlp_batch_size,
            )
            row = _evaluate_variant(dataset, variant, device=device, batch_size=args.batch_size)
            rows.append(row)
            print(row)

    _write_csv(args.output_csv, rows)
    print(f"Saved ablation table: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
