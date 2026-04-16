from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


@dataclass
class ClassifierTrainConfig:
    model_type: str = "svm"
    random_state: int = 42
    svm_c: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    rf_n_estimators: int = 400
    rf_max_depth: int = 0
    rf_min_samples_leaf: int = 1
    xgb_n_estimators: int = 300
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9


@dataclass
class AuxClassifierTrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    hidden_dim: int = 32
    dropout: float = 0.15
    num_layers: int = 3


class AuxMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 32,
        dropout: float = 0.15,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@torch.no_grad()
def classifier_metrics(scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    pred = scores.argmax(dim=1)
    acc = (pred == targets).float().mean().item()
    return {"acc": float(acc)}


def _build_estimator(cfg: ClassifierTrainConfig, num_classes: int):
    if cfg.model_type == "svm":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=cfg.svm_c,
                        kernel=cfg.svm_kernel,
                        gamma=cfg.svm_gamma,
                        probability=True,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )

    if cfg.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=None if cfg.rf_max_depth <= 0 else cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            random_state=cfg.random_state,
            n_jobs=1,
        )

    if cfg.model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Please install xgboost first, or switch --clf_model_type "
                "to svm or random_forest."
            )
        return XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            learning_rate=cfg.xgb_learning_rate,
            max_depth=cfg.xgb_max_depth,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=cfg.random_state,
            n_jobs=1,
        )

    raise ValueError(f"Unsupported classifier model_type: {cfg.model_type}")


def _to_numpy_2d(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={arr.shape}")
    return arr


def _predict_scores_numpy(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x)
    else:
        pred = np.asarray(model.predict(x), dtype=np.int64)
        n_classes = int(pred.max()) + 1 if pred.size > 0 else 1
        scores = np.zeros((pred.shape[0], n_classes), dtype=np.float32)
        scores[np.arange(pred.shape[0]), pred] = 1.0

    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return scores


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int,
    cfg: ClassifierTrainConfig,
    device: torch.device,
    ) -> tuple[object, Dict[str, float]]:
    del input_dim, device
    model = _build_estimator(cfg, num_classes=num_classes)
    model.fit(_to_numpy_2d(x_train), np.asarray(y_train, dtype=np.int64))

    train_scores = torch.from_numpy(_predict_scores_numpy(model, _to_numpy_2d(x_train)))
    train_targets = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    val_scores = torch.from_numpy(_predict_scores_numpy(model, _to_numpy_2d(x_val)))
    val_targets = torch.from_numpy(np.asarray(y_val, dtype=np.int64))

    train_acc = classifier_metrics(train_scores, train_targets)["acc"]
    val_acc = classifier_metrics(val_scores, val_targets)["acc"]
    print(f"[Classifier:{cfg.model_type}] train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return model, {"best_val_acc": float(val_acc)}


def train_aux_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int,
    cfg: AuxClassifierTrainConfig,
    device: torch.device,
) -> tuple[AuxMLPClassifier, Dict[str, float]]:
    model = AuxMLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        num_layers=cfg.num_layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    ce = nn.CrossEntropyLoss()

    train_loader = _make_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = _make_loader(x_val, y_val, cfg.batch_size, shuffle=False)

    best = None
    best_val = -1.0
    bad = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            pred = torch.argmax(logits.detach(), dim=1)
            train_correct += int((pred == yb).sum().item())
            train_total += int(yb.numel())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sch.step()

        model.eval()
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                all_logits.append(model(xb).cpu())
                all_targets.append(yb.cpu())

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        train_acc = train_correct / max(1, train_total)
        val_acc = classifier_metrics(logits, targets)["acc"]

        if epoch == 1 or epoch % 10 == 0:
            print(f"[AuxClassifier] epoch={epoch:03d} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[AuxClassifier] early stop at epoch {epoch}, best_val_acc={best_val:.4f}")
                break

    if best is not None:
        model.load_state_dict(best)

    return model, {"best_val_acc": float(best_val)}


@torch.no_grad()
def evaluate_aux_classifier(
    model: AuxMLPClassifier,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    model.eval()
    loader = _make_loader(x, y, batch_size=batch_size, shuffle=False)
    all_logits = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device)
        all_logits.append(model(xb).cpu())
        all_targets.append(yb.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return classifier_metrics(logits, targets)


@torch.no_grad()
def evaluate_classifier(
    model,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    del batch_size
    scores = predict_classifier_logits(model, _to_numpy_2d(x), device=device)
    targets = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)
    return classifier_metrics(scores, targets)


@torch.no_grad()
def predict_classifier_logits(
    model,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    scores = _predict_scores_numpy(model, _to_numpy_2d(x))
    return torch.from_numpy(scores).to(device)


train_mlp_classifier = train_classifier
evaluate_mlp_classifier = evaluate_classifier
predict_mlp_logits = predict_classifier_logits
