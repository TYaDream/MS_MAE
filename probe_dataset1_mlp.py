from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline import stratified_split_indices
from data_sources import build_label_encoder, drop_all_zero_features, load_data_bundle


@dataclass(frozen=True)
class TransformConfig:
    sample_max_norm: bool = False
    sample_l1_norm: bool = False
    log1p: bool = False
    zscore: bool = False


@dataclass(frozen=True)
class CaseConfig:
    name: str
    description: str
    hidden_dims: tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    transform: TransformConfig
    class_weight: bool = False
    train_subset_per_class: int = 0


CASES: dict[str, CaseConfig] = {
    "baseline_raw": CaseConfig(
        name="baseline_raw",
        description="Raw intensity into a small MLP. This is the baseline.",
        hidden_dims=(64,),
        dropout=0.10,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        epochs=200,
        patience=30,
        transform=TransformConfig(),
    ),
    "zscore_only": CaseConfig(
        name="zscore_only",
        description="Feature-wise z-score only. Checks whether scale mismatch is the main issue.",
        hidden_dims=(64,),
        dropout=0.10,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        epochs=200,
        patience=30,
        transform=TransformConfig(zscore=True),
    ),
    "log1p_zscore": CaseConfig(
        name="log1p_zscore",
        description="log1p plus z-score. Often much more stable for mass-spec inputs.",
        hidden_dims=(64,),
        dropout=0.10,
        lr=8e-4,
        weight_decay=1e-4,
        batch_size=16,
        epochs=220,
        patience=35,
        transform=TransformConfig(log1p=True, zscore=True),
    ),
    "samplemax_log1p_zscore": CaseConfig(
        name="samplemax_log1p_zscore",
        description="Per-sample max norm, then log1p and z-score. Tests TIC/intensity-scale drift.",
        hidden_dims=(64, 32),
        dropout=0.10,
        lr=8e-4,
        weight_decay=1e-4,
        batch_size=16,
        epochs=220,
        patience=35,
        transform=TransformConfig(sample_max_norm=True, log1p=True, zscore=True),
    ),
    "wider_deeper": CaseConfig(
        name="wider_deeper",
        description="A wider and deeper MLP. Checks whether the baseline is underfitting.",
        hidden_dims=(128, 64, 32),
        dropout=0.15,
        lr=8e-4,
        weight_decay=2e-4,
        batch_size=16,
        epochs=260,
        patience=40,
        transform=TransformConfig(log1p=True, zscore=True),
    ),
    "class_weighted": CaseConfig(
        name="class_weighted",
        description="Weighted CE. Useful if minority classes are being ignored.",
        hidden_dims=(64, 32),
        dropout=0.10,
        lr=8e-4,
        weight_decay=1e-4,
        batch_size=16,
        epochs=220,
        patience=35,
        transform=TransformConfig(log1p=True, zscore=True),
        class_weight=True,
    ),
    "overfit_probe": CaseConfig(
        name="overfit_probe",
        description="Train on a tiny subset and try to overfit. If this fails, optimization/modeling is the problem.",
        hidden_dims=(128, 64),
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=8,
        epochs=300,
        patience=80,
        transform=TransformConfig(log1p=True, zscore=True),
        train_subset_per_class=8,
    ),
}


class SimpleMLP(nn.Module):
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


def _np_counts(y: np.ndarray, classes: list[str]) -> str:
    parts = []
    for cls_id, cls_name in enumerate(classes):
        parts.append(f"{cls_name}={int((y == cls_id).sum())}")
    return ", ".join(parts)


def _apply_sample_max_norm(x: np.ndarray) -> np.ndarray:
    scale = np.max(x, axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    return x / scale


def _apply_sample_l1_norm(x: np.ndarray) -> np.ndarray:
    scale = np.sum(np.abs(x), axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    return x / scale


def transform_with_train_stats(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    cfg: TransformConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float32).copy()
    x_val = np.asarray(x_val, dtype=np.float32).copy()
    x_test = np.asarray(x_test, dtype=np.float32).copy()

    if cfg.sample_max_norm:
        x_train = _apply_sample_max_norm(x_train)
        x_val = _apply_sample_max_norm(x_val)
        x_test = _apply_sample_max_norm(x_test)

    if cfg.sample_l1_norm:
        x_train = _apply_sample_l1_norm(x_train)
        x_val = _apply_sample_l1_norm(x_val)
        x_test = _apply_sample_l1_norm(x_test)

    if cfg.log1p:
        x_train = np.log1p(np.clip(x_train, a_min=0.0, a_max=None))
        x_val = np.log1p(np.clip(x_val, a_min=0.0, a_max=None))
        x_test = np.log1p(np.clip(x_test, a_min=0.0, a_max=None))

    if cfg.zscore:
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
        std = np.where(std > 1e-6, std, 1.0)
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std

    return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    ce = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        pred = logits.argmax(dim=1)
        total_loss += float(loss.item()) * xb.shape[0]
        total_correct += int((pred == yb).sum().item())
        total_count += int(xb.shape[0])

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def subset_train_indices_per_class(indices: np.ndarray, y: np.ndarray, n_per_class: int) -> np.ndarray:
    if n_per_class <= 0:
        return indices
    chosen = []
    for cls in np.unique(y[indices]):
        cls_idx = indices[y[indices] == cls]
        chosen.append(cls_idx[: min(n_per_class, len(cls_idx))])
    return np.concatenate(chosen, axis=0)


def make_stratified_kfold_splits(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0].astype(np.int64)
        rng.shuffle(cls_idx)
        cls_parts = np.array_split(cls_idx, n_splits)
        for fold_id, part in enumerate(cls_parts):
            folds[fold_id].extend(part.tolist())

    split_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(y), dtype=np.int64)
    for fold_id in range(n_splits):
        test_idx = np.asarray(sorted(folds[fold_id]), dtype=np.int64)
        test_mask = np.zeros(len(y), dtype=bool)
        test_mask[test_idx] = True
        train_idx = all_idx[~test_mask]
        split_pairs.append((train_idx, test_idx))
    return split_pairs


def fit_case(
    x: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    case: CaseConfig,
    device: torch.device,
    seed: int,
    epochs_override: int = 0,
    patience_override: int = 0,
) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx_used = subset_train_indices_per_class(train_idx, y, case.train_subset_per_class)
    x_train, x_val, x_test = x[train_idx_used], x[val_idx], x[test_idx]
    y_train, y_val, y_test = y[train_idx_used], y[val_idx], y[test_idx]
    x_train, x_val, x_test = transform_with_train_stats(x_train, x_val, x_test, case.transform)

    model = SimpleMLP(
        input_dim=x.shape[1],
        num_classes=len(classes),
        hidden_dims=case.hidden_dims,
        dropout=case.dropout,
    ).to(device)

    class_weight = None
    if case.class_weight:
        counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
        counts = np.where(counts > 0, counts, 1.0)
        class_weight = torch.from_numpy(counts.sum() / (len(classes) * counts)).float().to(device)

    ce = nn.CrossEntropyLoss(weight=class_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=case.lr, weight_decay=case.weight_decay)

    train_loader = make_loader(x_train, y_train, batch_size=case.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=max(case.batch_size, 32), shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size=max(case.batch_size, 32), shuffle=False)

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    bad_epochs = 0

    max_epochs = epochs_override if epochs_override > 0 else case.epochs
    patience = patience_override if patience_override > 0 else case.patience

    for epoch in range(1, max_epochs + 1):
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
        val_loss, val_acc = evaluate(model, val_loader, device)

        improved = (val_acc > best_val_acc + 1e-6) or (
            abs(val_acc - best_val_acc) <= 1e-6 and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[{case.name}] epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if bad_epochs >= patience:
            print(
                f"[{case.name}] early stop at epoch {epoch}, "
                f"best_val_acc={best_val_acc:.4f}, best_val_loss={best_val_loss:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss, train_acc = evaluate(model=model, loader=train_loader, device=device)
    val_loss, val_acc = evaluate(model=model, loader=val_loader, device=device)
    test_loss, test_acc = evaluate(model=model, loader=test_loader, device=device)

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "train_size": float(len(train_idx_used)),
        "val_size": float(len(val_idx)),
        "test_size": float(len(test_idx)),
    }


def summarize_cv_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = ["train_acc", "val_acc", "test_acc", "train_loss", "val_loss", "test_loss"]
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([m[key] for m in fold_metrics], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def run_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    case: CaseConfig,
    device: torch.device,
    seed: int,
    cv_folds: int,
    inner_val_ratio: float,
    epochs_override: int = 0,
    patience_override: int = 0,
) -> dict[str, float]:
    fold_metrics: list[dict[str, float]] = []
    fold_pairs = make_stratified_kfold_splits(y=y, n_splits=cv_folds, seed=seed)

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(fold_pairs, start=1):
        inner_y = y[outer_train_idx]
        inner_splits = stratified_split_indices(
            inner_y,
            train_ratio=max(0.0, 1.0 - inner_val_ratio),
            val_ratio=inner_val_ratio,
            seed=seed + fold_id,
        )
        train_idx = outer_train_idx[inner_splits.train_idx]
        val_idx = outer_train_idx[inner_splits.val_idx]
        test_idx = outer_test_idx

        metrics = fit_case(
            x=x,
            y=y,
            classes=classes,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            case=case,
            device=device,
            seed=seed + fold_id,
            epochs_override=epochs_override,
            patience_override=patience_override,
        )
        fold_metrics.append(metrics)
        print(
            f"[{case.name}] fold={fold_id:02d}/{cv_folds} "
            f"train_acc={metrics['train_acc']:.4f} "
            f"val_acc={metrics['val_acc']:.4f} "
            f"test_acc={metrics['test_acc']:.4f}"
        )

    summary = summarize_cv_metrics(fold_metrics)
    print(
        f"[{case.name}] cv_summary "
        f"train_acc={summary['train_acc_mean']:.4f}±{summary['train_acc_std']:.4f} "
        f"val_acc={summary['val_acc_mean']:.4f}±{summary['val_acc_std']:.4f} "
        f"test_acc={summary['test_acc_mean']:.4f}±{summary['test_acc_std']:.4f}"
    )
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe why dataset1 MLP underperforms random forest by testing several controlled cases."
    )
    parser.add_argument(
        "--data_excel",
        type=str,
        default="massspec_data/dataset1/dataset1_preprocessed.xlsx",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        help="Case name to run, or 'all'. Use --list_cases to inspect available cases.",
    )
    parser.add_argument("--list_cases", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs_override", type=int, default=0)
    parser.add_argument("--patience_override", type=int, default=0)
    parser.add_argument("--cv_folds", type=int, default=1)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.list_cases:
        print("Available cases:")
        for case in CASES.values():
            print(f"- {case.name}: {case.description}")
        return

    bundle = load_data_bundle("excel", data_excel=args.data_excel)
    bundle = drop_all_zero_features(bundle)
    if bundle.labels_str is None:
        raise ValueError("Dataset must contain labels in raw_samples.class")

    y, _, classes = build_label_encoder(bundle.labels_str)
    device = resolve_device(args.device)

    print(f"Dataset: {args.data_excel}")
    print(f"Shape: samples={bundle.full_intensity.shape[0]}, features={bundle.full_intensity.shape[1]}")
    print(f"Classes: {classes}")
    print(f"Device: {device}")
    if args.cv_folds > 1:
        print(f"Cross-validation: {args.cv_folds}-fold | inner_val_ratio={args.val_ratio}")
    else:
        splits = stratified_split_indices(y, args.train_ratio, args.val_ratio, args.seed)
        print(f"Split train: {len(splits.train_idx)} | {_np_counts(y[splits.train_idx], classes)}")
        print(f"Split val:   {len(splits.val_idx)} | {_np_counts(y[splits.val_idx], classes)}")
        print(f"Split test:  {len(splits.test_idx)} | {_np_counts(y[splits.test_idx], classes)}")

    if args.case == "all":
        cases = list(CASES.values())
    else:
        if args.case not in CASES:
            raise ValueError(f"Unknown case: {args.case}. Use --list_cases.")
        cases = [CASES[args.case]]

    results = []
    for case in cases:
        print("")
        print(f"=== Case: {case.name} ===")
        print(case.description)
        if args.cv_folds > 1:
            metrics = run_cross_validation(
                x=bundle.full_intensity.astype(np.float32),
                y=y,
                classes=classes,
                case=case,
                device=device,
                seed=args.seed,
                cv_folds=args.cv_folds,
                inner_val_ratio=args.val_ratio,
                epochs_override=args.epochs_override,
                patience_override=args.patience_override,
            )
        else:
            metrics = fit_case(
                x=bundle.full_intensity.astype(np.float32),
                y=y,
                classes=classes,
                train_idx=splits.train_idx,
                val_idx=splits.val_idx,
                test_idx=splits.test_idx,
                case=case,
                device=device,
                seed=args.seed,
                epochs_override=args.epochs_override,
                patience_override=args.patience_override,
            )
        results.append((case.name, metrics))
        if args.cv_folds > 1:
            print(
                f"[{case.name}] final "
                f"train_acc={metrics['train_acc_mean']:.4f}±{metrics['train_acc_std']:.4f} "
                f"val_acc={metrics['val_acc_mean']:.4f}±{metrics['val_acc_std']:.4f} "
                f"test_acc={metrics['test_acc_mean']:.4f}±{metrics['test_acc_std']:.4f}"
            )
        else:
            print(
                f"[{case.name}] final "
                f"train_acc={metrics['train_acc']:.4f} "
                f"val_acc={metrics['val_acc']:.4f} "
                f"test_acc={metrics['test_acc']:.4f}"
            )

    print("")
    print("=== Summary ===")
    for name, metrics in results:
        if args.cv_folds > 1:
            print(
                f"{name:>24s} | "
                f"train_acc={metrics['train_acc_mean']:.4f}±{metrics['train_acc_std']:.4f} "
                f"val_acc={metrics['val_acc_mean']:.4f}±{metrics['val_acc_std']:.4f} "
                f"test_acc={metrics['test_acc_mean']:.4f}±{metrics['test_acc_std']:.4f}"
            )
        else:
            print(
                f"{name:>24s} | "
                f"train_acc={metrics['train_acc']:.4f} "
                f"val_acc={metrics['val_acc']:.4f} "
                f"test_acc={metrics['test_acc']:.4f}"
            )


if __name__ == "__main__":
    main()
