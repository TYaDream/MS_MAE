from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarFourierEncoder(nn.Module):
    """Fourier features for scalar values."""

    def __init__(self, num_frequencies: int = 8) -> None:
        super().__init__()
        freqs = (2.0 ** torch.arange(num_frequencies).float()) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return 1 + 2 * int(self.freqs.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 1]
        x = torch.nan_to_num(x)
        view_shape = [1] * (x.ndim - 1) + [self.freqs.numel()]
        xb = x * self.freqs.view(*view_shape)
        return torch.cat([x, torch.sin(xb), torch.cos(xb)], dim=-1)


class FeatureAwareEmbedding(nn.Module):
    """
    Encode token fields:
    [feature_index, intensity, start_index, end_index, status, mz, drift_direction]

    Design:
    - discrete embedding for feature index, status, and drift direction
    - Fourier scalar encoding for intensity/mz/start/end
    - gated fusion for robust mixed-type representation
    """

    def __init__(
        self,
        full_feature_num: int,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.full_feature_num = full_feature_num

        d_idx = embedding_dim // 4
        d_status = embedding_dim // 8
        d_drift = embedding_dim // 8
        d_cont = embedding_dim - d_idx - d_status - d_drift

        self.feature_embed = nn.Embedding(full_feature_num + 1, d_idx, padding_idx=0)
        # status: 0 normal, 1 overlap, 2 missing, 3 PAD
        self.status_embed = nn.Embedding(4, d_status, padding_idx=3)
        # drift_direction: 0 no drift, 1 drift up, 2 drift down, 3 PAD
        self.drift_embed = nn.Embedding(4, d_drift, padding_idx=3)

        self.intensity_fourier = ScalarFourierEncoder(num_frequencies=8)
        self.mz_fourier = ScalarFourierEncoder(num_frequencies=8)
        self.start_fourier = ScalarFourierEncoder(num_frequencies=6)
        self.end_fourier = ScalarFourierEncoder(num_frequencies=6)

        cont_in_dim = (
            self.intensity_fourier.out_dim
            + self.mz_fourier.out_dim
            + self.start_fourier.out_dim
            + self.end_fourier.out_dim
        )

        self.cont_proj = nn.Sequential(
            nn.Linear(cont_in_dim, d_cont),
            nn.LayerNorm(d_cont),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fuse_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 7]
        feature_index = x[..., 0].long().clamp(min=0, max=self.full_feature_num)
        intensity = torch.log1p(x[..., 1:2].clamp_min(0.0))

        start_index = x[..., 2:3]
        start_index = torch.where(start_index < 0, torch.zeros_like(start_index), start_index)
        start_index = start_index / float(max(1, self.full_feature_num))

        end_index = x[..., 3:4]
        end_index = torch.where(end_index < 0, torch.zeros_like(end_index), end_index)
        end_index = end_index / float(max(1, self.full_feature_num))

        status = x[..., 4].long().clamp(min=0, max=3)

        # M/Z normalization: robust fixed-scale transform for GC-MS range.
        mz = torch.log1p(x[..., 5:6].clamp_min(0.0)) / math.log1p(1000.0)
        drift_direction = x[..., 6].long().clamp(min=0, max=3)

        cont = torch.cat(
            [
                self.intensity_fourier(intensity),
                self.mz_fourier(mz),
                self.start_fourier(start_index),
                self.end_fourier(end_index),
            ],
            dim=-1,
        )
        cont_emb = self.cont_proj(cont)

        disc_emb = torch.cat(
            [
                self.feature_embed(feature_index),
                self.status_embed(status),
                self.drift_embed(drift_direction),
            ],
            dim=-1,
        )
        token = torch.cat([disc_emb, cont_emb], dim=-1)

        fused = self.fuse_proj(token)
        gate = self.fuse_gate(token)
        return token + gate * fused


class UnconditionalMAERestorationModel(nn.Module):
    """Unconditional MAE-style restoration for degraded GC-MS samples."""

    def __init__(
        self,
        full_feature_num: int,
        embedding_dim: int = 96,
        encoder_layers: int = 3,
        decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        full_mz_values: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.full_feature_num = full_feature_num
        self.embedding_dim = embedding_dim

        self.input_embedding = FeatureAwareEmbedding(
            full_feature_num=full_feature_num,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.full_pos_embed = nn.Embedding(full_feature_num + 1, embedding_dim, padding_idx=0)

        self.mz_prior_fourier = ScalarFourierEncoder(num_frequencies=8)
        self.mz_prior_proj = nn.Sequential(
            nn.Linear(self.mz_prior_fourier.out_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        if full_mz_values is None:
            full_mz_values = torch.linspace(0.05, 0.60, full_feature_num)
        else:
            full_mz_values = full_mz_values.float().view(-1)
            if full_mz_values.numel() != full_feature_num:
                raise ValueError("full_mz_values length must equal full_feature_num")
            mmin = torch.min(full_mz_values)
            mmax = torch.max(full_mz_values)
            if float(mmax - mmin) < 1e-8:
                full_mz_values = torch.zeros_like(full_mz_values)
            else:
                full_mz_values = (full_mz_values - mmin) / (mmax - mmin)
        self.register_buffer("full_mz_values", full_mz_values.view(1, full_feature_num, 1), persistent=False)

        self.observed_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)

        self.local_refine = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.reg_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2),
        )

    def _scatter_to_full_axis(
        self,
        encoded_tokens: torch.Tensor,
        degraded_input: torch.Tensor,
        token_mask: torch.Tensor,
        overlap_copy_source_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, dim = encoded_tokens.shape
        device = encoded_tokens.device

        observed_sum = torch.zeros(bsz, self.full_feature_num, dim, device=device, dtype=encoded_tokens.dtype)
        observed_cnt = torch.zeros(bsz, self.full_feature_num, 1, device=device, dtype=encoded_tokens.dtype)
        observed_mask = torch.zeros(bsz, self.full_feature_num, device=device, dtype=torch.bool)

        feat_index = degraded_input[..., 0].long()

        for b in range(bsz):
            valid = token_mask[b] & (feat_index[b] > 0) & (feat_index[b] <= self.full_feature_num)
            if not torch.any(valid):
                continue

            fi = feat_index[b, valid]
            anchor = fi.clamp(min=1, max=self.full_feature_num) - 1

            src = self.observed_proj(encoded_tokens[b, valid])
            idx_expand = anchor.unsqueeze(-1).expand(-1, dim)
            observed_sum[b].scatter_add_(0, idx_expand, src)

            ones = torch.ones(anchor.shape[0], 1, device=device, dtype=encoded_tokens.dtype)
            observed_cnt[b].scatter_add_(0, anchor.unsqueeze(-1), ones)
            observed_mask[b, anchor] = True

        observed_avg = observed_sum / observed_cnt.clamp_min(1.0)
        overlap_copy_source_full = overlap_copy_source_full.long()
        for b in range(bsz):
            copy_src = overlap_copy_source_full[b]
            copy_mask = (copy_src > 0) & (copy_src <= self.full_feature_num)
            if not torch.any(copy_mask):
                continue
            target_idx = torch.nonzero(copy_mask, as_tuple=False).squeeze(-1)
            source_idx = copy_src[target_idx] - 1
            observed_avg[b, target_idx] = observed_avg[b, source_idx]
            observed_mask[b, target_idx] = True
        return observed_avg, observed_mask

    def forward(
        self,
        degraded_input: torch.Tensor,
        token_mask: torch.Tensor,
        overlap_copy_source_full: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        token_embed = self.input_embedding(degraded_input)
        encoded = self.encoder(token_embed, src_key_padding_mask=~token_mask)

        bsz = token_embed.shape[0]
        device = token_embed.device

        full_idx = torch.arange(1, self.full_feature_num + 1, device=device).unsqueeze(0).expand(bsz, -1)
        pos = self.full_pos_embed(full_idx)

        mz_prior = self.full_mz_values.to(device).expand(bsz, -1, -1)
        mz_emb = self.mz_prior_proj(self.mz_prior_fourier(mz_prior))

        feature_queries = self.mask_token.expand(bsz, self.full_feature_num, self.embedding_dim) + pos + mz_emb

        if overlap_copy_source_full is None:
            overlap_copy_source_full = torch.zeros(
                bsz,
                self.full_feature_num,
                device=device,
                dtype=torch.long,
            )
        observed_tokens, observed_full_mask = self._scatter_to_full_axis(
            encoded,
            degraded_input,
            token_mask,
            overlap_copy_source_full,
        )
        feature_queries = feature_queries + observed_tokens

        decoded = self.decoder(
            tgt=feature_queries,
            memory=encoded,
            memory_key_padding_mask=~token_mask,
        )

        local = self.local_refine(decoded.transpose(1, 2)).transpose(1, 2)
        decoded = decoded + local

        reg_out = self.reg_head(decoded)
        mean_raw = reg_out[..., 0]
        logvar = reg_out[..., 1].clamp(min=-6.0, max=4.0)

        pred_intensity = F.softplus(mean_raw)

        return {
            "pred_intensity": pred_intensity,
            "pred_logvar": logvar,
            "observed_full_mask": observed_full_mask,
        }
