from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from typing import Dict, Optional

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_pipeline import build_loaders_from_bundle, stratified_split_indices
from data_sources import DataBundle, build_label_encoder, drop_all_zero_features, load_data_bundle
from dataset import GCMSRestorationDataset, DatasetConfig, collate_fn
from degradation import DegradationConfig
from model import UnconditionalMAERestorationModel
from utils import (
    WeightedReconstructionLoss,
    compute_metrics,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def format_metrics(title: str, metrics: Dict[str, float]) -> str:
    keys = ["loss", "mse", "mae", "pearson", "nonzero_mse", "degraded_mse"]
    parts = [f"{k}={metrics[k]:.6f}" for k in keys if k in metrics]
    return f"{title}: " + ", ".join(parts)


def write_metrics_csv(csv_path: str, rows: list[Dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _overlap_count(a: np.ndarray, b: np.ndarray) -> int:
    if a.size == 0 or b.size == 0:
        return 0
    return int(np.intersect1d(a, b).size)


def degraded_tokens_to_proxy_full(
    degraded_input: torch.Tensor,
    token_mask: torch.Tensor,
    full_feature_num: int,
) -> torch.Tensor:
    """
    Scatter degraded tokens back onto the fixed full feature axis.
    This is a coarse proxy spectrum used to classify the degraded sample directly.
    """
    bsz = degraded_input.shape[0]
    device = degraded_input.device
    proxy = torch.zeros(bsz, full_feature_num, device=device, dtype=degraded_input.dtype)

    feat = degraded_input[..., 0].long() - 1
    intensity = degraded_input[..., 1].clamp_min(0.0)
    valid = token_mask & (feat >= 0) & (feat < full_feature_num)
    feat_safe = feat.clamp(min=0, max=max(0, full_feature_num - 1))
    proxy.scatter_add_(1, feat_safe, intensity * valid.float())
    return proxy


def build_subset_eval_loader(
    full_intensity_raw: np.ndarray,
    mz_values: np.ndarray,
    intensity_scale: float,
    class_ids: np.ndarray,
    indices: np.ndarray,
    degradation_cfg: DegradationConfig,
    batch_size: int,
    num_workers: int,
    seed: int,
    forced_case: Optional[int] = None,
) -> DataLoader:
    dataset = GCMSRestorationDataset(
        full_intensity_matrix=full_intensity_raw[indices],
        mz_values=mz_values,
        intensity_scale=intensity_scale,
        class_ids=class_ids[indices],
        is_real_flags=np.ones(indices.shape[0], dtype=bool),
        degradation_cfg=degradation_cfg,
        dataset_cfg=DatasetConfig(deterministic=True, seed=seed, forced_case=forced_case),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def load_training_bundle(args: argparse.Namespace) -> DataBundle:
    if args.dataset_mode == "single":
        if args.single_dataset_type == "excel":
            if not args.single_excel or not os.path.exists(args.single_excel):
                raise FileNotFoundError(f"Single-dataset excel not found: {args.single_excel}")
            out = load_data_bundle(dataset_type="excel", data_excel=args.single_excel)
        elif args.single_dataset_type == "npz":
            if not args.single_npz or not os.path.exists(args.single_npz):
                raise FileNotFoundError(f"Single-dataset npz not found: {args.single_npz}")
            out = load_data_bundle(dataset_type="npz", data_npz=args.single_npz)
        elif args.single_dataset_type == "synthetic":
            out = load_data_bundle(
                dataset_type="synthetic",
                demo_num_samples=args.single_demo_num_samples,
                demo_full_feature_num=args.single_demo_full_feature_num,
                seed=args.seed,
            )
        else:
            raise ValueError(f"Unsupported single_dataset_type: {args.single_dataset_type}")

        out = drop_all_zero_features(out)
        if out.labels_str is None:
            out.labels_str = np.asarray(["Class0"] * out.full_intensity.shape[0], dtype=object)
        out.validate()
        return out

    if not os.path.exists(args.orig_excel):
        raise FileNotFoundError(f"Original excel not found: {args.orig_excel}")
    out = load_data_bundle(dataset_type="excel", data_excel=args.orig_excel)
    out = drop_all_zero_features(out)
    if out.labels_str is None:
        out.labels_str = np.asarray(["Class0"] * out.full_intensity.shape[0], dtype=object)
    out.validate()
    return out


class FrozenMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Log1pZScoreTransform:
    def __init__(self, mean: np.ndarray, std: np.ndarray, intensity_scale: float) -> None:
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, -1)
        self.intensity_scale = float(max(1e-8, intensity_scale))

    @classmethod
    def fit(cls, x_train_raw: np.ndarray, intensity_scale: float) -> "Log1pZScoreTransform":
        x = np.log1p(np.clip(np.asarray(x_train_raw, dtype=np.float32), a_min=0.0, a_max=None))
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std > 1e-6, std, 1.0)
        return cls(mean=mean, std=std, intensity_scale=intensity_scale)

    def transform_numpy(self, x_raw: np.ndarray) -> np.ndarray:
        x = np.log1p(np.clip(np.asarray(x_raw, dtype=np.float32), a_min=0.0, a_max=None))
        x = (x - self.mean) / self.std
        return x.astype(np.float32)

    def transform_tensor(self, x_norm: torch.Tensor, device: torch.device) -> torch.Tensor:
        mean = torch.from_numpy(self.mean).to(device=device, dtype=x_norm.dtype)
        std = torch.from_numpy(self.std).to(device=device, dtype=x_norm.dtype)
        x_raw = x_norm * self.intensity_scale
        x_log = torch.log1p(torch.clamp(x_raw, min=0.0))
        return (x_log - mean) / std


def _make_mlp_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_frozen_mlp(
    model: FrozenMLPClassifier,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()
    loader = _make_mlp_loader(x, y, batch_size=batch_size, shuffle=False)
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        total_loss += float(loss.item()) * xb.shape[0]
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_count += int(xb.shape[0])

    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
    }


def train_frozen_mlp_classifier(
    *,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_val_raw: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    intensity_scale: float,
    hidden_dims: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    class_weight: bool,
    device: torch.device,
) -> tuple[FrozenMLPClassifier, Log1pZScoreTransform, Dict[str, float]]:
    transform = Log1pZScoreTransform.fit(x_train_raw=x_train_raw, intensity_scale=intensity_scale)
    x_train = transform.transform_numpy(x_train_raw)
    x_val = transform.transform_numpy(x_val_raw)

    model = FrozenMLPClassifier(
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    weight = None
    if class_weight:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        counts = np.where(counts > 0, counts, 1.0)
        weight = torch.from_numpy(counts.sum() / (num_classes * counts)).float().to(device)

    ce = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = _make_mlp_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _make_mlp_loader(x_val, y_val, batch_size=max(batch_size, 64), shuffle=False)

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: list[Dict[str, float | int | bool]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_correct += int((logits.argmax(dim=1) == yb).sum().item())
            train_count += int(xb.shape[0])

        train_loss = train_loss_sum / max(1, train_count)
        train_acc = train_correct / max(1, train_count)
        val_eval = evaluate_frozen_mlp(model, x_val, y_val, device=device, batch_size=max(batch_size, 64))

        if epoch == 1 or epoch % 10 == 0:
            print(f"[FrozenMLP] epoch={epoch:03d} train_acc={train_acc:.4f} val_acc={val_eval['acc']:.4f}")

        improved = (val_eval["acc"] > best_val_acc + 1e-6) or (
            abs(val_eval["acc"] - best_val_acc) <= 1e-6 and val_eval["loss"] < best_val_loss
        )
        best_val_acc_next = best_val_acc
        best_val_loss_next = best_val_loss
        if improved:
            best_val_acc = val_eval["acc"]
            best_val_loss = val_eval["loss"]
            best_val_acc_next = best_val_acc
            best_val_loss_next = best_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_eval["loss"]),
                "val_acc": float(val_eval["acc"]),
                "best_val_acc": float(best_val_acc_next),
                "best_val_loss": float(best_val_loss_next),
                "is_best": bool(improved),
            }
        )

        if bad_epochs >= patience:
            print(
                f"[FrozenMLP] early stop at epoch {epoch}, "
                f"best_val_acc={best_val_acc:.4f}, best_val_loss={best_val_loss:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    train_eval = evaluate_frozen_mlp(model, x_train, y_train, device=device, batch_size=max(batch_size, 64))
    val_eval = evaluate_frozen_mlp(model, x_val, y_val, device=device, batch_size=max(batch_size, 64))
    info = {
        "best_val_acc": float(best_val_acc),
        "train_acc": float(train_eval["acc"]),
        "val_acc": float(val_eval["acc"]),
        "history": history_rows,
    }
    return model, transform, info


def train_one_epoch_mae(
    model: UnconditionalMAERestorationModel,
    loader,
    criterion: WeightedReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    frozen_mlp: FrozenMLPClassifier | None = None,
    mlp_transform: Log1pZScoreTransform | None = None,
    recon_cls_weight: float = 0.0,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_cls_loss = 0.0
    total_cls_batches = 0
    total_cls_correct = 0
    total_cls_count = 0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(
                batch["degraded_input"],
                batch["token_mask"],
                batch["overlap_copy_source_full"],
            )
            pred = outputs["pred_intensity"]

            recon_loss, _ = criterion(
                pred=pred,
                target=batch["target_full_intensity"],
                nonzero_mask=batch["nonzero_mask_full"],
                degraded_mask=batch["degraded_mask_full"],
                pred_logvar=outputs.get("pred_logvar"),
            )
            loss = recon_loss
            cls_loss = torch.zeros((), device=device)
            if recon_cls_weight > 0 and frozen_mlp is not None and mlp_transform is not None:
                real_mask = batch["is_real"].bool()
                if torch.any(real_mask):
                    mlp_input = mlp_transform.transform_tensor(pred[real_mask], device=device)
                    mlp_logits = frozen_mlp(mlp_input)
                    cls_loss = nn.functional.cross_entropy(mlp_logits, batch["class_id"][real_mask])
                    cls_pred = torch.argmax(mlp_logits.detach(), dim=1)
                    total_cls_correct += int((cls_pred == batch["class_id"][real_mask]).sum().item())
                    total_cls_count += int(real_mask.sum().item())
                    loss = loss + recon_cls_weight * cls_loss

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_recon_loss += float(recon_loss.detach().cpu())
        if float(cls_loss.detach().cpu()) > 0:
            total_cls_loss += float(cls_loss.detach().cpu())
            total_cls_batches += 1
        total_batches += 1

    return {
        "train_loss": total_loss / max(1, total_batches),
        "train_recon_loss": total_recon_loss / max(1, total_batches),
        "train_cls_loss_real": total_cls_loss / max(1, total_cls_batches),
        "train_cls_acc_real": total_cls_correct / max(1, total_cls_count),
    }


@torch.no_grad()
def evaluate_mae(
    model: UnconditionalMAERestorationModel,
    loader,
    criterion: WeightedReconstructionLoss,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_batches = 0
    preds = []
    targets = []
    degraded_masks = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(
                batch["degraded_input"],
                batch["token_mask"],
                batch["overlap_copy_source_full"],
            )
            pred = outputs["pred_intensity"]
            loss, _ = criterion(
                pred=pred,
                target=batch["target_full_intensity"],
                nonzero_mask=batch["nonzero_mask_full"],
                degraded_mask=batch["degraded_mask_full"],
                pred_logvar=outputs.get("pred_logvar"),
            )

        total_loss += float(loss.detach().cpu())
        total_batches += 1

        preds.append(pred.detach().cpu())
        targets.append(batch["target_full_intensity"].detach().cpu())
        degraded_masks.append(batch["degraded_mask_full"].detach().cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    degraded_mask_all = torch.cat(degraded_masks, dim=0)

    metrics = compute_metrics(pred_all, target_all, degraded_mask_all)
    metrics["loss"] = total_loss / max(1, total_batches)
    return metrics


@torch.no_grad()
def evaluate_recon_mlp_classification(
    mae_model: UnconditionalMAERestorationModel,
    frozen_mlp: FrozenMLPClassifier,
    mlp_transform: Log1pZScoreTransform,
    loader,
    device: torch.device,
    class_names: Optional[list[str]] = None,
    return_distribution: bool = False,
    use_amp: bool = False,
) -> Dict[str, object]:
    mae_model.eval()

    correct = 0
    total = 0
    pred_list = []
    tgt_list = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = mae_model(
                batch["degraded_input"],
                batch["token_mask"],
                batch["overlap_copy_source_full"],
            )
            recon = outputs["pred_intensity"]
            mlp_input = mlp_transform.transform_tensor(recon, device=device)
            logits = frozen_mlp(mlp_input)

        pred = torch.argmax(logits, dim=1)
        tgt = batch["class_id"]
        correct += int((pred == tgt).sum().item())
        total += int(tgt.numel())
        if return_distribution:
            pred_list.append(pred.detach().cpu())
            tgt_list.append(tgt.detach().cpu())

    acc = correct / max(1, total)
    out: Dict[str, object] = {"acc": float(acc)}

    if return_distribution:
        if pred_list:
            pred_all = torch.cat(pred_list, dim=0)
            tgt_all = torch.cat(tgt_list, dim=0)
            n_cls = len(class_names) if class_names is not None else int(
                max(int(pred_all.max().item()), int(tgt_all.max().item())) + 1
            )
            pred_cnt = torch.bincount(pred_all, minlength=n_cls).tolist()
            tgt_cnt = torch.bincount(tgt_all, minlength=n_cls).tolist()
        else:
            pred_cnt = []
            tgt_cnt = []

        if class_names is None:
            pred_dist = {str(i): int(v) for i, v in enumerate(pred_cnt) if int(v) > 0}
            tgt_dist = {str(i): int(v) for i, v in enumerate(tgt_cnt) if int(v) > 0}
        else:
            pred_dist = {class_names[i]: int(v) for i, v in enumerate(pred_cnt) if int(v) > 0}
            tgt_dist = {class_names[i]: int(v) for i, v in enumerate(tgt_cnt) if int(v) > 0}
        out["pred_dist"] = pred_dist
        out["target_dist"] = tgt_dist
    return out


@torch.no_grad()
def evaluate_degraded_mlp_classification(
    frozen_mlp: FrozenMLPClassifier,
    mlp_transform: Log1pZScoreTransform,
    loader,
    device: torch.device,
    full_feature_num: int,
    class_names: Optional[list[str]] = None,
    return_distribution: bool = False,
) -> Dict[str, object]:
    correct = 0
    total = 0
    pred_list = []
    tgt_list = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        proxy = degraded_tokens_to_proxy_full(
            degraded_input=batch["degraded_input"],
            token_mask=batch["token_mask"],
            full_feature_num=full_feature_num,
        )
        mlp_input = mlp_transform.transform_tensor(proxy, device=device)
        logits = frozen_mlp(mlp_input)

        pred = torch.argmax(logits, dim=1)
        tgt = batch["class_id"]
        correct += int((pred == tgt).sum().item())
        total += int(tgt.numel())
        if return_distribution:
            pred_list.append(pred.detach().cpu())
            tgt_list.append(tgt.detach().cpu())

    acc = correct / max(1, total)
    out: Dict[str, object] = {"acc": float(acc)}

    if return_distribution:
        if pred_list:
            pred_all = torch.cat(pred_list, dim=0)
            tgt_all = torch.cat(tgt_list, dim=0)
            n_cls = len(class_names) if class_names is not None else int(
                max(int(pred_all.max().item()), int(tgt_all.max().item())) + 1
            )
            pred_cnt = torch.bincount(pred_all, minlength=n_cls).tolist()
            tgt_cnt = torch.bincount(tgt_all, minlength=n_cls).tolist()
        else:
            pred_cnt = []
            tgt_cnt = []

        if class_names is None:
            pred_dist = {str(i): int(v) for i, v in enumerate(pred_cnt) if int(v) > 0}
            tgt_dist = {str(i): int(v) for i, v in enumerate(tgt_cnt) if int(v) > 0}
        else:
            pred_dist = {class_names[i]: int(v) for i, v in enumerate(pred_cnt) if int(v) > 0}
            tgt_dist = {class_names[i]: int(v) for i, v in enumerate(tgt_cnt) if int(v) > 0}
        out["pred_dist"] = pred_dist
        out["target_dist"] = tgt_dist
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Unconditional MAE training with a frozen log1p_zscore MLP")

    parser.add_argument(
        "--dataset_mode",
        type=str,
        choices=["builtin", "single"],
        default="builtin",
        help="builtin=run the default original excel dataset, single=run one arbitrary excel/npz/synthetic dataset",
    )
    parser.add_argument(
        "--single_dataset_type",
        type=str,
        choices=["excel", "npz", "synthetic"],
        default="excel",
        help="Used when dataset_mode=single",
    )
    parser.add_argument("--single_excel", type=str, default="")
    parser.add_argument("--single_npz", type=str, default="")
    parser.add_argument("--single_demo_num_samples", type=int, default=512)
    parser.add_argument("--single_demo_full_feature_num", type=int, default=128)
    parser.add_argument("--orig_excel", type=str, default="massspec_data/dataset1/gcms_preprocessed_samples.xlsx")

    parser.add_argument("--embedding_dim", type=int, default=96)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.20)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--mae_save_name", type=str, default="best_unconditional_mae.pt")
    parser.add_argument("--frozen_mlp_save_name", type=str, default="best_frozen_mlp_for_unconditional_mae.pt")

    parser.add_argument("--nonzero_weight", type=float, default=4.0)
    parser.add_argument("--degraded_weight", type=float, default=3.0)
    parser.add_argument("--peak_weight", type=float, default=2.0)
    parser.add_argument("--recon_cls_weight", type=float, default=0.03)
    parser.add_argument("--mlp_hidden_dims", type=int, nargs="+", default=[64])
    parser.add_argument("--mlp_dropout", type=float, default=0.10)
    parser.add_argument("--mlp_lr", type=float, default=8e-4)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_batch_size", type=int, default=16)
    parser.add_argument("--mlp_epochs", type=int, default=220)
    parser.add_argument("--mlp_patience", type=int, default=35)
    parser.add_argument("--mlp_class_weight", action="store_true")
    parser.add_argument("--p_case_0", type=float, default=0.25)
    parser.add_argument("--p_case_1", type=float, default=0.25)
    parser.add_argument("--p_case_2", type=float, default=0.25)
    parser.add_argument("--p_case_3", type=float, default=0.25)

    parser.add_argument("--p_random_zero", type=float, default=0.2)
    parser.add_argument("--p_overlap", type=float, default=0.2)
    parser.add_argument("--overlap_min_len", type=int, default=2)
    parser.add_argument("--overlap_max_len", type=int, default=6)
    parser.add_argument("--baseline_shift_min", type=float, default=0.02)
    parser.add_argument("--baseline_shift_max", type=float, default=0.02)
    parser.add_argument("--p_baseline_increase", type=float, default=0.5)
    parser.add_argument("--drop_zero_tokens", action="store_true")

    parser.add_argument("--use_amp", action="store_true")

    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: train_ratio={args.train_ratio}, val_ratio={args.val_ratio}. "
            "Require: train_ratio>0, val_ratio>0, and train_ratio+val_ratio<1."
        )

    p_case_raw = np.asarray(
        [args.p_case_0, args.p_case_1, args.p_case_2, args.p_case_3],
        dtype=np.float64,
    )
    if np.any(p_case_raw < 0):
        raise ValueError(f"Invalid p_case values (must be >=0): {p_case_raw.tolist()}")
    if float(p_case_raw.sum()) <= 0:
        raise ValueError(f"Invalid p_case values (sum must be >0): {p_case_raw.tolist()}")
    if args.baseline_shift_min < 0 or args.baseline_shift_max < 0:
        raise ValueError(
            f"Invalid baseline shift range: min={args.baseline_shift_min}, max={args.baseline_shift_max}. "
            "Require both >= 0."
        )
    if not (0.0 <= args.p_baseline_increase <= 1.0):
        raise ValueError(
            f"Invalid p_baseline_increase={args.p_baseline_increase}. Require 0 <= p_baseline_increase <= 1."
        )
    p_case = tuple((p_case_raw / p_case_raw.sum()).tolist())

    set_seed(args.seed)

    bundle = load_training_bundle(args)
    y, cls_to_id, classes = build_label_encoder(bundle.labels_str)
    if bundle.is_synthetic is None:
        is_real = np.ones(len(y), dtype=bool)
    else:
        is_real = ~np.asarray(bundle.is_synthetic).astype(bool)

    keep_zero_tokens = not args.drop_zero_tokens
    degradation_cfg = DegradationConfig(
        p_case=p_case,
        p_random_zero=args.p_random_zero,
        p_overlap=args.p_overlap,
        overlap_min_len=args.overlap_min_len,
        overlap_max_len=args.overlap_max_len,
        baseline_shift_min=args.baseline_shift_min,
        baseline_shift_max=args.baseline_shift_max,
        p_baseline_increase=args.p_baseline_increase,
        keep_zero_tokens=keep_zero_tokens,
    )

    loader_bundle, full_intensity_norm, intensity_scale = build_loaders_from_bundle(
        bundle=bundle,
        class_ids=y,
        is_real_flags=is_real,
        degradation_cfg=degradation_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_idx = loader_bundle.splits.train_idx
    val_idx = loader_bundle.splits.val_idx
    test_idx = loader_bundle.splits.test_idx

    val_real_idx = val_idx[is_real[val_idx]]
    test_real_idx = test_idx[is_real[test_idx]]
    val_eval_idx = val_real_idx if len(val_real_idx) > 0 else val_idx
    test_eval_idx = test_real_idx if len(test_real_idx) > 0 else test_idx
    eval_scope = "real-only" if len(val_real_idx) > 0 and len(test_real_idx) > 0 else "all-samples"

    val_eval_loader = build_subset_eval_loader(
        full_intensity_raw=bundle.full_intensity,
        mz_values=bundle.mz_values,
        intensity_scale=intensity_scale,
        class_ids=y,
        indices=val_eval_idx,
        degradation_cfg=degradation_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 1500,
    )
    test_eval_loader = build_subset_eval_loader(
        full_intensity_raw=bundle.full_intensity,
        mz_values=bundle.mz_values,
        intensity_scale=intensity_scale,
        class_ids=y,
        indices=test_eval_idx,
        degradation_cfg=degradation_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 2500,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("Data Summary")
    print(f"Dataset mode: {args.dataset_mode}")
    print(f"Source: {bundle.source_name}")
    print(
        f"Split ratios: train={args.train_ratio:.3f}, val={args.val_ratio:.3f}, "
        f"test={1.0 - args.train_ratio - args.val_ratio:.3f} (seed={args.seed})"
    )
    print("Split method: stratified by class with per-class shuffle")
    print(f"Samples: total={len(y)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Recon->FrozenMLP evaluation scope: {eval_scope} | val={len(val_eval_idx)}, test={len(test_eval_idx)}")
    print(f"Features: {full_intensity_norm.shape[1]} | Classes: {classes}")
    print(f"Intensity scale: {intensity_scale:.6f} | Device: {device}")
    print(
        "Degradation: "
        f"p_case={tuple(round(x, 3) for x in degradation_cfg.p_case)}, "
        "(0=missing,1=overlap,2=baseline-drift,3=baseline-drift+missing+overlap), "
        f"p_random_zero={degradation_cfg.p_random_zero:.3f}, "
        f"p_overlap={degradation_cfg.p_overlap:.3f}, "
        f"overlap_len=[{degradation_cfg.overlap_min_len},{degradation_cfg.overlap_max_len}], "
        f"baseline_shift_ratio=[{degradation_cfg.baseline_shift_min:.3f},{degradation_cfg.baseline_shift_max:.3f}], "
        f"p_baseline_increase={degradation_cfg.p_baseline_increase:.2f}"
    )
    print("Pipeline: apply case0-3 on raw intensities, then normalize degraded input/target by dataset-global scale")
    print(
        "Model: unconditional MAE"
        + f" | frozen_mlp_cls_weight={args.recon_cls_weight:.3f}"
    )
    if bundle.is_synthetic is not None:
        print(
            f"Synthetic flag: synthetic={int(bundle.is_synthetic.sum())}, real={int((~bundle.is_synthetic).sum())}"
        )
    print("=" * 100)

    clf_splits = stratified_split_indices(y=y, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    ov_tv = _overlap_count(clf_splits.train_idx, clf_splits.val_idx)
    ov_tt = _overlap_count(clf_splits.train_idx, clf_splits.test_idx)
    ov_vt = _overlap_count(clf_splits.val_idx, clf_splits.test_idx)
    if ov_tv or ov_tt or ov_vt:
        raise RuntimeError(
            f"Frozen MLP split overlap detected: train-val={ov_tv}, train-test={ov_tt}, val-test={ov_vt}"
        )
    print(
        f"[FrozenMLP Data] source={bundle.source_name}, total={len(y)}, "
        f"train={len(clf_splits.train_idx)}, val={len(clf_splits.val_idx)}, test={len(clf_splits.test_idx)}"
    )
    print(f"[FrozenMLP Split Check] train-val={ov_tv}, train-test={ov_tt}, val-test={ov_vt}")

    frozen_mlp, mlp_transform, mlp_info = train_frozen_mlp_classifier(
        x_train_raw=bundle.full_intensity[clf_splits.train_idx],
        y_train=y[clf_splits.train_idx],
        x_val_raw=bundle.full_intensity[clf_splits.val_idx],
        y_val=y[clf_splits.val_idx],
        num_classes=len(classes),
        intensity_scale=intensity_scale,
        hidden_dims=tuple(int(v) for v in args.mlp_hidden_dims),
        dropout=args.mlp_dropout,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        batch_size=args.mlp_batch_size,
        epochs=args.mlp_epochs,
        patience=args.mlp_patience,
        class_weight=args.mlp_class_weight,
        device=device,
    )
    mlp_val_eval_idx = val_real_idx if len(val_real_idx) > 0 else clf_splits.val_idx
    mlp_test_eval_idx = test_real_idx if len(test_real_idx) > 0 else clf_splits.test_idx
    mlp_val_eval = evaluate_frozen_mlp(
        frozen_mlp,
        x=mlp_transform.transform_numpy(bundle.full_intensity[mlp_val_eval_idx]),
        y=y[mlp_val_eval_idx],
        device=device,
        batch_size=max(args.mlp_batch_size, 64),
    )
    mlp_test_eval = evaluate_frozen_mlp(
        frozen_mlp,
        x=mlp_transform.transform_numpy(bundle.full_intensity[mlp_test_eval_idx]),
        y=y[mlp_test_eval_idx],
        device=device,
        batch_size=max(args.mlp_batch_size, 64),
    )
    print(
        f"[FrozenMLP:log1p_zscore] train_acc={mlp_info['train_acc']:.4f} | "
        f"val_acc({eval_scope})={mlp_val_eval['acc']:.4f} (n={len(mlp_val_eval_idx)}) | "
        f"test_acc({eval_scope})={mlp_test_eval['acc']:.4f} (n={len(mlp_test_eval_idx)})"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    clf_path = os.path.join(args.save_dir, args.frozen_mlp_save_name)
    frozen_mlp_history_path = os.path.join(
        args.save_dir,
        f"{os.path.splitext(args.frozen_mlp_save_name)[0]}_history.csv",
    )
    torch.save(
        {
            "model_state_dict": frozen_mlp.state_dict(),
            "model_type": "frozen_mlp_log1p_zscore",
            "hidden_dims": tuple(int(v) for v in args.mlp_hidden_dims),
            "dropout": args.mlp_dropout,
            "classes": classes,
            "label_to_id": cls_to_id,
            "transform_mean": mlp_transform.mean,
            "transform_std": mlp_transform.std,
            "intensity_scale": mlp_transform.intensity_scale,
            "args": vars(args),
        },
        clf_path,
    )
    print(f"Saved frozen MLP checkpoint: {clf_path}")
    write_metrics_csv(frozen_mlp_history_path, mlp_info.get("history", []))
    print(f"Saved frozen MLP history: {frozen_mlp_history_path}")

    mae_model = UnconditionalMAERestorationModel(
        full_feature_num=full_intensity_norm.shape[1],
        embedding_dim=args.embedding_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        full_mz_values=torch.from_numpy(bundle.mz_values),
    ).to(device)

    criterion = WeightedReconstructionLoss(
        nonzero_weight=args.nonzero_weight,
        degraded_weight=args.degraded_weight,
        peak_weight=args.peak_weight,
    )

    optimizer = AdamW(mae_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    mae_path = os.path.join(args.save_dir, args.mae_save_name)
    mae_history_path = os.path.join(
        args.save_dir,
        f"{os.path.splitext(args.mae_save_name)[0]}_history.csv",
    )
    best_val_mse = float("inf")
    mae_history_rows: list[Dict[str, float | int | bool | str]] = []

    for epoch in range(1, args.epochs + 1):
        train_info = train_one_epoch_mae(
            model=mae_model,
            loader=loader_bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            frozen_mlp=frozen_mlp,
            mlp_transform=mlp_transform,
            recon_cls_weight=args.recon_cls_weight,
            scaler=scaler,
            use_amp=args.use_amp,
        )

        val_info = evaluate_mae(mae_model, loader_bundle.val_loader, criterion, device, use_amp=args.use_amp)
        val_degraded_cls = evaluate_degraded_mlp_classification(
            frozen_mlp=frozen_mlp,
            mlp_transform=mlp_transform,
            loader=val_eval_loader,
            device=device,
            full_feature_num=full_intensity_norm.shape[1],
            class_names=classes,
            return_distribution=False,
        )
        val_recon_cls = evaluate_recon_mlp_classification(
            mae_model=mae_model,
            frozen_mlp=frozen_mlp,
            mlp_transform=mlp_transform,
            loader=val_eval_loader,
            device=device,
            class_names=classes,
            return_distribution=True,
            use_amp=args.use_amp,
        )

        scheduler.step()

        msg = (
            f"[MAE] Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_info['train_loss']:.6f} "
            f"(recon={train_info['train_recon_loss']:.6f}, "
            f"cls_real={train_info['train_cls_loss_real']:.6f}, "
            f"cls_acc_real={train_info['train_cls_acc_real']:.4f}) | "
            + format_metrics("val", val_info)
            + f" | val_degraded_mlp_acc({eval_scope})={float(val_degraded_cls['acc']):.4f}"
            + f" | val_recon_mlp_acc({eval_scope})={float(val_recon_cls['acc']):.4f}"
        )
        print(msg)
        print(
            f"      val_recon_pred_dist({eval_scope})={val_recon_cls.get('pred_dist')} | "
            f"target_dist={val_recon_cls.get('target_dist')}"
        )

        is_best = val_info["mse"] < best_val_mse
        best_val_mse_next = best_val_mse
        if is_best:
            best_val_mse = val_info["mse"]
            best_val_mse_next = best_val_mse
            ckpt = {
                "model_state_dict": mae_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_mse": best_val_mse,
                "full_feature_num": full_intensity_norm.shape[1],
                "classes": classes,
                "intensity_scale": intensity_scale,
                "mz_values": bundle.mz_values,
                "args": vars(args),
                "degradation_cfg": asdict(degradation_cfg),
                "model_type": "unconditional_mae",
            }
            save_checkpoint(mae_path, ckpt)
            print(f"  -> Saved best unconditional MAE checkpoint: {mae_path}")

        mae_history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_info["train_loss"]),
                "train_recon_loss": float(train_info["train_recon_loss"]),
                "train_cls_loss_real": float(train_info["train_cls_loss_real"]),
                "train_cls_acc_real": float(train_info["train_cls_acc_real"]),
                "val_loss": float(val_info["loss"]),
                "val_mse": float(val_info["mse"]),
                "val_mae": float(val_info["mae"]),
                "val_pearson": float(val_info["pearson"]),
                "val_nonzero_mse": float(val_info["nonzero_mse"]),
                "val_degraded_mse": float(val_info["degraded_mse"]),
                "val_degraded_mlp_acc": float(val_degraded_cls["acc"]),
                "val_recon_mlp_acc": float(val_recon_cls["acc"]),
                "best_val_mse": float(best_val_mse_next),
                "is_best": bool(is_best),
                "eval_scope": eval_scope,
            }
        )

    write_metrics_csv(mae_history_path, mae_history_rows)
    print(f"Saved MAE history: {mae_history_path}")

    print("\nLoading best unconditional MAE checkpoint for test...")
    best_ckpt = load_checkpoint(mae_path, map_location=device)
    mae_model.load_state_dict(best_ckpt["model_state_dict"])

    test_info = evaluate_mae(mae_model, loader_bundle.test_loader, criterion, device, use_amp=args.use_amp)
    print(format_metrics("MAE test", test_info))

    degraded_test_cls = evaluate_degraded_mlp_classification(
        frozen_mlp=frozen_mlp,
        mlp_transform=mlp_transform,
        loader=test_eval_loader,
        device=device,
        full_feature_num=full_intensity_norm.shape[1],
        class_names=classes,
        return_distribution=True,
    )
    print(f"[Degraded->FrozenMLP] test_acc({eval_scope})={float(degraded_test_cls['acc']):.4f}")
    print(
        f"                 pred_dist={degraded_test_cls.get('pred_dist')} | "
        f"target_dist={degraded_test_cls.get('target_dist')}"
    )

    recon_test_cls = evaluate_recon_mlp_classification(
        mae_model=mae_model,
        frozen_mlp=frozen_mlp,
        mlp_transform=mlp_transform,
        loader=test_eval_loader,
        device=device,
        class_names=classes,
        return_distribution=True,
        use_amp=args.use_amp,
    )
    print(f"[Recon->FrozenMLP] test_acc({eval_scope})={float(recon_test_cls['acc']):.4f}")
    print(
        f"              pred_dist={recon_test_cls.get('pred_dist')} | "
        f"target_dist={recon_test_cls.get('target_dist')}"
    )

    eval_indices = np.where(is_real)[0]
    eval_scope_case = "real-only"
    if eval_indices.size == 0:
        eval_indices = np.arange(len(y), dtype=np.int64)
        eval_scope_case = "all-samples"

    n_cases = len(degradation_cfg.p_case)
    print(f"\n[{eval_scope_case} {n_cases}-case evaluation via frozen MLP on MAE reconstructions]")
    for case_id in range(n_cases):
        case_loader = build_subset_eval_loader(
            full_intensity_raw=bundle.full_intensity,
            mz_values=bundle.mz_values,
            intensity_scale=intensity_scale,
            class_ids=y,
            indices=eval_indices,
            degradation_cfg=degradation_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed + 4000 + case_id,
            forced_case=case_id,
        )
        case_degraded_cls = evaluate_degraded_mlp_classification(
            frozen_mlp=frozen_mlp,
            mlp_transform=mlp_transform,
            loader=case_loader,
            device=device,
            full_feature_num=full_intensity_norm.shape[1],
        )
        case_cls = evaluate_recon_mlp_classification(
            mae_model=mae_model,
            frozen_mlp=frozen_mlp,
            mlp_transform=mlp_transform,
            loader=case_loader,
            device=device,
            use_amp=args.use_amp,
        )
        print(
            f"  case={case_id} -> degraded_mlp_acc({eval_scope_case})={float(case_degraded_cls['acc']):.4f}, "
            f"recon_mlp_acc({eval_scope_case})={float(case_cls['acc']):.4f}"
        )


if __name__ == "__main__":
    main()
