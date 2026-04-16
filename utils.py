from __future__ import annotations

import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)

    num = (pred_c * target_c).sum(dim=1)
    den = torch.sqrt((pred_c.pow(2).sum(dim=1) + 1e-8) * (target_c.pow(2).sum(dim=1) + 1e-8))
    corr = num / den
    corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
    return corr


class WeightedReconstructionLoss(nn.Module):
    """
    Weighted restoration objective with frontier-oriented additions:
    - weighted MSE + SmoothL1
    - peak-preserving loss
    - heteroscedastic NLL (predicted uncertainty)
    """

    def __init__(
        self,
        nonzero_weight: float = 4.0,
        degraded_weight: float = 3.0,
        peak_weight: float = 2.0,
        peak_quantile: float = 0.90,
        alpha_mse: float = 0.45,
        beta_smoothl1: float = 0.25,
        gamma_peak: float = 0.10,
        lambda_nll: float = 0.15,
    ) -> None:
        super().__init__()
        self.nonzero_weight = nonzero_weight
        self.degraded_weight = degraded_weight
        self.peak_weight = peak_weight
        self.peak_quantile = peak_quantile

        self.alpha_mse = alpha_mse
        self.beta_smoothl1 = beta_smoothl1
        self.gamma_peak = gamma_peak
        self.lambda_nll = lambda_nll

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        nonzero_mask: torch.Tensor,
        degraded_mask: torch.Tensor,
        pred_logvar: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        weights = torch.ones_like(target)
        weights = weights + (self.nonzero_weight - 1.0) * nonzero_mask.float()
        weights = weights + (self.degraded_weight - 1.0) * degraded_mask.float()

        q = torch.quantile(target, self.peak_quantile, dim=1, keepdim=True)
        peak_mask = target >= q
        weights = weights + (self.peak_weight - 1.0) * peak_mask.float()

        w_sum = weights.sum().clamp_min(1.0)
        err = pred - target

        mse = (err.pow(2) * weights).sum() / w_sum
        smooth = (F.smooth_l1_loss(pred, target, reduction="none") * weights).sum() / w_sum

        if torch.any(peak_mask):
            peak_loss = (err.abs() * peak_mask.float()).sum() / peak_mask.float().sum().clamp_min(1.0)
        else:
            peak_loss = torch.zeros((), device=pred.device)

        if pred_logvar is not None:
            logvar = pred_logvar.clamp(min=-6.0, max=4.0)
            nll_map = 0.5 * torch.exp(-logvar) * err.pow(2) + 0.5 * logvar
            nll = (nll_map * weights).sum() / w_sum
            nll = torch.clamp(nll, min=0.0)
        else:
            nll = torch.zeros((), device=pred.device)

        total = (
            self.alpha_mse * mse
            + self.beta_smoothl1 * smooth
            + self.gamma_peak * peak_loss
            + self.lambda_nll * nll
        )

        detail = {
            "loss_total": float(total.detach().cpu()),
            "loss_mse": float(mse.detach().cpu()),
            "loss_smoothl1": float(smooth.detach().cpu()),
            "loss_peak": float(peak_loss.detach().cpu()),
            "loss_nll": float(nll.detach().cpu()),
        }
        return total, detail


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    degraded_mask: torch.Tensor,
) -> Dict[str, float]:
    diff = pred - target

    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()
    pearson = _batch_pearson(pred, target).mean().item()

    nonzero_mask = target > 0
    nonzero_den = nonzero_mask.float().sum().clamp_min(1.0)
    nonzero_mse = (diff.pow(2) * nonzero_mask.float()).sum() / nonzero_den

    degraded_den = degraded_mask.float().sum().clamp_min(1.0)
    degraded_mse = (diff.pow(2) * degraded_mask.float()).sum() / degraded_den

    return {
        "mse": float(mse),
        "mae": float(mae),
        "pearson": float(pearson),
        "nonzero_mse": float(nonzero_mse.item()),
        "degraded_mse": float(degraded_mse.item()),
    }


def save_checkpoint(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict:
    # PyTorch >=2.6 defaults to weights_only=True; we need full dict state here.
    return torch.load(path, map_location=map_location, weights_only=False)
