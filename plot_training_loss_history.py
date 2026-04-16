from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class LossHistory:
    name: str
    df: pd.DataFrame
    best_epoch: int
    best_val_mse: float


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values.astype(np.float64, copy=False)
    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.size):
        lo = max(0, i - window + 1)
        out[i] = np.mean(values[lo : i + 1])
    return out


def _load_history(name: str, csv_path: Path) -> LossHistory:
    df = pd.read_csv(csv_path)
    required = ["epoch", "train_recon_loss", "train_cls_loss_real", "val_mse"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    best_idx = int(df["val_mse"].idxmin())
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val_mse = float(df.loc[best_idx, "val_mse"])
    return LossHistory(name=name, df=df, best_epoch=best_epoch, best_val_mse=best_val_mse)


def _plot_one(ax, history: LossHistory, smooth_window: int) -> None:
    df = history.df
    epochs = df["epoch"].to_numpy(dtype=np.int64)
    train_recon = _moving_average(df["train_recon_loss"].to_numpy(dtype=np.float64), smooth_window)
    val_mse = _moving_average(df["val_mse"].to_numpy(dtype=np.float64), smooth_window)
    cls_loss = _moving_average(df["train_cls_loss_real"].to_numpy(dtype=np.float64), smooth_window)

    line_train = ax.plot(epochs, train_recon, color="#0072B2", linewidth=2.2, label="Train recon loss")[0]
    line_val = ax.plot(epochs, val_mse, color="#D55E00", linewidth=2.2, label="Validation MSE")[0]
    best_line = ax.axvline(
        history.best_epoch,
        color="#7A7A7A",
        linestyle="--",
        linewidth=1.4,
        label=f"Best val epoch ({history.best_epoch})",
    )

    ax.set_title(history.name, fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.7)

    ax2 = ax.twinx()
    line_cls = ax2.plot(
        epochs,
        cls_loss,
        color="#009E73",
        linewidth=1.9,
        linestyle="-.",
        label="Auxiliary cls loss",
    )[0]
    ax2.set_ylabel("Auxiliary cls loss")
    ax2.set_yscale("log")

    handles = [line_train, line_val, line_cls, best_line]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.96, fontsize=9)

    summary = f"best val MSE={history.best_val_mse:.4g}"
    ax.text(
        0.02,
        0.03,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )


def build_figure(dataset1_csv: Path, dataset2_csv: Path, output_path: Path, smooth_window: int) -> Path:
    d1 = _load_history("Dataset 1", dataset1_csv)
    d2 = _load_history("Dataset 2", dataset2_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    _plot_one(axes[0], d1, smooth_window=smooth_window)
    _plot_one(axes[1], d2, smooth_window=smooth_window)

    fig.suptitle("Training Dynamics Of The Unconditional MAE", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAE training-loss histories for dataset1 and dataset2.")
    parser.add_argument("--dataset1_csv", type=Path, default=Path("checkpoints/dataset1_mae_history.csv"))
    parser.add_argument("--dataset2_csv", type=Path, default=Path("checkpoints/dataset2_mae_history.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/mae_training_dynamics.png"))
    parser.add_argument("--smooth_window", type=int, default=5)
    args = parser.parse_args()

    out = build_figure(
        dataset1_csv=args.dataset1_csv,
        dataset2_csv=args.dataset2_csv,
        output_path=args.output,
        smooth_window=max(1, int(args.smooth_window)),
    )
    print(f"Saved figure: {out.resolve()}")


if __name__ == "__main__":
    main()
