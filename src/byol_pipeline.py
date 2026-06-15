"""
BYOL Peak-Token Pipeline for Ion Concentration Level Classification.

Stage 1: BYOL self-supervised pre-training on peak tokens.
  - Extracts peaks from each spectrum as (position, intensity, width) tokens.
  - Online: augmented peak tokens -> PeakTokenEncoder -> Projector -> Predictor
  - Target: clean peak tokens -> PeakTokenEncoder -> Projector (EMA)
  - Symmetric cosine distance loss.

Stage 2: Multi-class concentration-level classification.
  - 3-fold OOF group-stratified CV.
  - Phase 1: freeze encoder, train MultiClassHead.
  - Phase 2: unfreeze all, fine-tune.
  - 5 concentration levels per analyte (Cu/Fe/Zn).

Adapted from Transfer-Learning-Assisted-SERS (dino_model.py).
X.T.Liu 20260615
"""

import copy
import csv
import math
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

from src.utils import read_data, preprocess_data
from src.plsr_unmixing import (
    _group_folds, _make_mixture_label, _make_group_id,
    _filter_mix_conc,
    ANALYTES, VALID_MIXTURES, N_OUTER, RANDOM_STATE,
)

# ===========================================================================
# Configuration
# ===========================================================================

CU_LEVELS = [50, 100, 500, 1000, 5000]       # nM
FE_LEVELS = [50, 100, 500, 1000, 5000]       # nM
ZN_LEVELS = [500, 1000, 5000, 10000, 50000]  # nM
LEVELS_MAP = {"Cu": CU_LEVELS, "Fe": FE_LEVELS, "Zn": ZN_LEVELS}

BYOL_CONFIG = {
    # Peak-token encoder (Transformer)
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 1024,
    "transformer_dropout": 0.2,
    # BYOL projector + predictor
    "proj_hidden": 1024,
    "proj_out": 256,
    # EMA
    "ema_momentum_base": 0.996,
    # Peak detection
    "min_height_sigma": 3.0,
    "noise_region": (2000, 2500),
    "norm_peak_center": 250,
    "norm_peak_half_window": 20,
    # Peak masking / augmentation
    "weak_fraction": 0.30,
    "mask_ratio_min": 0.05,
    "mask_ratio_max": 0.10,
    # Stage 1 training
    "stage1_epochs": 20,
    "stage1_batch_size": 64,
    "stage1_lr": 1e-3,
    # Stage 2 training
    "stage2_frozen_epochs": 20,
    "stage2_full_epochs": 50,
    "stage2_batch_size": 32,
    "stage2_lr": 1e-3,
    # Classification
    "num_classes": 5,
}

STAGE1_FILENAME = "byol_stage1_encoder.pt"
STAGE2_FILENAME = "byol_stage2_classifier.pt"
CSV_FILENAME = "byol_classification_predictions.csv"


# ===========================================================================
# Peak Extraction
# ===========================================================================

class PeakExtractor:
    """Extract peaks from spectra as (position, intensity, width) tokens.

    Pipeline per spectrum:
      1. Normalize by max intensity at `norm_center ± norm_window` cm-1.
      2. Compute sigma = std of the noise region (e.g. 2000-2500 cm-1).
      3. Detect peaks via first-derivative zero-crossing, height > min_height_sigma * sigma.
      4. Build token: (position/2500, intensity, width/max_width).

    Args:
        raman_shift: 1D np.ndarray of Raman shift values.
        noise_region: (min, max) cm-1 for sigma calculation.
        norm_center: Raman shift of the reference peak for normalization.
        norm_half_window: half-width around norm_center.
        min_height_sigma: peak height threshold multiplier.
    """

    def __init__(self, raman_shift, noise_region=(2000, 2500),
                 norm_center=250, norm_half_window=20,
                 min_height_sigma=3.0):
        self.raman_shift = np.asarray(raman_shift)
        self.noise_region = noise_region
        self.norm_center = norm_center
        self.norm_half_window = norm_half_window
        self.min_height_sigma = min_height_sigma

        # Pre-compute index ranges
        self._norm_indices = np.where(
            (self.raman_shift >= norm_center - norm_half_window) &
            (self.raman_shift <= norm_center + norm_half_window)
        )[0]
        if len(self._norm_indices) == 0:
            raise ValueError(
                f"No Raman shift points near {norm_center} cm-1 "
                f"for normalization."
            )

        self._noise_indices = np.where(
            (self.raman_shift >= noise_region[0]) &
            (self.raman_shift <= noise_region[1])
        )[0]
        if len(self._noise_indices) == 0:
            print(f"Warning: no points in noise region {noise_region}. "
                  f"Using full-range std as fallback.")
            self._noise_indices = np.arange(len(self.raman_shift))

        self._max_shift = self.raman_shift[-1]

    def normalize_spectrum(self, spectrum):
        """Divide spectrum by max intensity at reference peak."""
        peak_val = np.max(spectrum[self._norm_indices])
        if peak_val > 0:
            return spectrum / peak_val
        return spectrum.copy()

    def compute_sigma(self, norm_spec):
        """Sigma = standard deviation of the noise region."""
        return float(np.std(norm_spec[self._noise_indices]))

    def detect_peaks(self, spec, min_height):
        """First-derivative zero-crossing peak detection.

        Each peak = {center, height, left, right, width}.

        Args:
            spec: 1D numpy array (normalized spectrum).
            min_height: minimum peak height to accept.

        Returns:
            list of dicts, sorted by height descending.
        """
        L = len(spec)
        deriv = np.diff(spec)

        # Find positive->negative zero-crossings as peak centres
        peak_centers = []
        for j in range(1, L - 1):
            if (deriv[j - 1] > 0 and deriv[j] < 0
                    and spec[j] > min_height):
                peak_centers.append(j)

        if not peak_centers:
            return []

        peaks = []
        for center in peak_centers:
            height = spec[center]
            half_max = 0.5 * height

            left = center
            for j in range(center - 1, 0, -1):
                if spec[j] <= half_max or deriv[j - 1] <= 0:
                    left = j
                    break

            right = center
            for j in range(center, L - 1):
                if spec[j] <= half_max or deriv[j] >= 0:
                    right = j
                    break

            peaks.append({
                'center': center,
                'height': height,
                'left': left,
                'right': right,
                'width': right - left,
            })

        peaks.sort(key=lambda p: p['height'], reverse=True)
        return peaks

    def __call__(self, spectra):
        """Extract peak tokens from a batch of spectra.

        Args:
            spectra: 2D np.ndarray (n_samples, n_features).

        Returns:
            peaks_tensor: (B, N_max, 3) float32 tensor.
            mask: (B, N_max) bool tensor, True where padded.
            n_peaks: list of int, actual peak counts per spectrum.
            positions: list of np.ndarray (actual Raman shift per peak).
        """
        all_tokens = []
        n_peaks_list = []
        max_width = 0

        for i in range(spectra.shape[0]):
            norm_spec = self.normalize_spectrum(spectra[i])
            sigma = self.compute_sigma(norm_spec)
            min_height = self.min_height_sigma * sigma
            peaks = self.detect_peaks(norm_spec, min_height)

            tokens = []
            for p in peaks:
                pos_norm = self.raman_shift[p['center']] / self._max_shift
                tokens.append([pos_norm, p['height'], p['width']])
                if p['width'] > max_width:
                    max_width = p['width']

            all_tokens.append(np.array(tokens, dtype=np.float32))
            n_peaks_list.append(len(tokens))

        # Normalize widths
        if max_width > 0:
            for tokens in all_tokens:
                tokens[:, 2] /= max_width

        # Pad to N_max
        N_max = max(len(t) for t in all_tokens) if all_tokens else 1
        B = len(all_tokens)
        padded = np.zeros((B, N_max, 3), dtype=np.float32)
        mask = np.ones((B, N_max), dtype=bool)

        positions_raw = []
        for i, tokens in enumerate(all_tokens):
            n = len(tokens)
            padded[i, :n, :] = tokens
            mask[i, :n] = False
            # Store actual Raman shift positions
            pos_shift = tokens[:, 0] * self._max_shift
            positions_raw.append(pos_shift)

        peaks_tensor = torch.from_numpy(padded)
        mask_tensor = torch.from_numpy(mask)

        return peaks_tensor, mask_tensor, n_peaks_list, positions_raw


# ===========================================================================
# Sinusoidal Positional Encoding
# ===========================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for peak token sequences."""

    def __init__(self, d_model=256, max_len=4096):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ===========================================================================
# PeakTokenEncoder — Transformer on peak tokens
# ===========================================================================

class PeakTokenEncoder(nn.Module):
    """Pure Transformer encoder operating on peak tokens.

    Input:  (B, N, 3)  — normalized (position, intensity, width)
            mask: (B, N) — True where PADDED
    Output: (B, d_model) feature vector.
    """

    def __init__(self, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.2):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        # x: (B, N, 3), mask: (B, N) bool, True = padded
        x = self.token_proj(x)              # (B, N, d_model)
        x = self.pos_encoder(x)             # (B, N, d_model)

        src_key_mask = mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_mask)

        # Masked mean pooling
        if mask is not None:
            keep = (~mask).float().unsqueeze(-1)   # (B, N, 1)
            x = x * keep
            lengths = keep.sum(dim=1).clamp(min=1)  # (B, 1)
            return x.sum(dim=1) / lengths           # (B, d_model)
        return x.mean(dim=1)                        # (B, d_model)


# ===========================================================================
# BYOL Projector, Predictor, Model
# ===========================================================================

class BYOLProjector(nn.Module):
    """MLP: 256 -> 1024 -> BN+ReLU -> 256 -> BN+ReLU -> 256, L2-norm."""

    def __init__(self, in_dim=256, hidden=1024, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class BYOLPredictor(nn.Module):
    """MLP: 256 -> 1024 -> BN+ReLU -> 256, L2-norm."""

    def __init__(self, in_dim=256, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class BYOLPeakModel(nn.Module):
    """PeakTokenEncoder + BYOLProjector + optional BYOLPredictor.

    - has_predictor=True for the online network.
    - has_predictor=False for the target network (EMA, no predictor).
    """

    def __init__(self, config=None, has_predictor=False):
        super().__init__()
        cfg = config or BYOL_CONFIG
        self.encoder = PeakTokenEncoder(
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["transformer_dropout"],
        )
        self.projector = BYOLProjector(
            in_dim=cfg["d_model"],
            hidden=cfg["proj_hidden"],
            out_dim=cfg["proj_out"],
        )
        if has_predictor:
            self.predictor = BYOLPredictor(
                in_dim=cfg["proj_out"],
                hidden=cfg["proj_hidden"],
            )
        else:
            self.predictor = None

    def forward(self, x, mask=None):
        feat = self.encoder(x, mask)
        z = self.projector(feat)
        if self.predictor is not None:
            return self.predictor(z)
        return z


# ===========================================================================
# Peak Augmentation (for online view)
# ===========================================================================

class PeakAugmentation:
    """Augment peak tokens for the BYOL online view.

    Operations (in order):
      1. Jitter positions by ±shift_max pixels.
      2. Scale intensities by 0.7-1.3.
      3. Add 0-3 random noise peaks.
      4. Drop 5-10% of the weakest 30% of peaks.
    """

    def __init__(self, shift_max=1, scale_range=(0.5, 1.5),
                 noise_peaks=3, noise_height=0.02,
                 weak_fraction=0.30,
                 mask_ratio_min=0.05, mask_ratio_max=0.10):
        self.shift_max = shift_max
        self.scale_range = scale_range
        self.noise_peaks = noise_peaks
        self.noise_height = noise_height
        self.weak_fraction = weak_fraction
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max

    def __call__(self, peaks, mask):
        """Augment a batch of peak tokens.

        Args:
            peaks: (B, N, 3) float tensor.
            mask: (B, N) bool tensor, True = padded.

        Returns:
            aug_peaks: (B, N, 3) tensor, same N_max.
            aug_mask: (B, N) bool, updated mask.
        """
        device = peaks.device
        B, N, _ = peaks.shape
        p = peaks.clone()
        m = mask.clone()

        for i in range(B):
            n_valid = (~m[i]).sum().item()
            if n_valid < 2:
                continue

            valid = p[i, :int(n_valid), :]  # (n_valid, 3)

            # 1. Jitter positions
            shift = (torch.rand(1, device=device).item() - 0.5) * 2 * self.shift_max
            valid[:, 0] = (valid[:, 0] + shift / 2500.0).clamp(0.0, 1.0)

            # 2. Scale intensities
            scale = (torch.rand(1, device=device).item()
                     * (self.scale_range[1] - self.scale_range[0])
                     + self.scale_range[0])
            valid[:, 1] *= scale

            # 3. Add random noise peaks
            n_noise = np.random.randint(0, self.noise_peaks + 1)
            for _ in range(n_noise):
                pos = np.random.uniform(0.0, 1.0)
                h = np.random.uniform(0.0, self.noise_height) * valid[:, 1].max().item()
                w = np.random.uniform(0.01, 0.05)
                noise_token = torch.tensor(
                    [[pos, h, w]], device=device, dtype=torch.float32)
                valid = torch.cat([valid, noise_token], dim=0)

            # 4. Drop weak peaks
            n_total = valid.shape[0]
            valid_sorted = valid[valid[:, 1].argsort(descending=True)]
            n_weak = max(1, int(n_total * self.weak_fraction))
            weak_start = n_total - n_weak
            ratio = np.random.uniform(self.mask_ratio_min, self.mask_ratio_max)
            n_drop = max(1, int(n_weak * ratio))
            drop_indices = np.random.choice(
                n_weak, size=min(n_drop, n_weak), replace=False)
            keep_mask = torch.ones(n_total, dtype=torch.bool, device=device)
            for idx in drop_indices:
                keep_mask[weak_start + idx] = False
            valid = valid[keep_mask]

            # Pad back to N
            n_kept = valid.shape[0]
            if n_kept > N:
                valid = valid[:N, :]
                n_kept = N
            p[i, :n_kept, :] = valid
            p[i, n_kept:, :] = 0.0
            m[i, :n_kept] = False
            m[i, n_kept:] = True

        return p, m


# ===========================================================================
# EMA helpers & BYOL loss
# ===========================================================================

@torch.no_grad()
def _ema_update(student, teacher, momentum):
    """teacher = momentum * teacher + (1 - momentum) * student."""
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


def _cosine_momentum(step, total_steps, base=0.996):
    """Cosine schedule for EMA momentum: base -> 1.0."""
    if total_steps <= 1:
        return 1.0
    return 1.0 - (1.0 - base) * (1.0 + math.cos(
        math.pi * step / total_steps)) / 2.0


# ===========================================================================
# Label encoding
# ===========================================================================

def _conc_to_label(conc_nm, levels):
    """Map a concentration value in nM to its class index (0-4)."""
    for idx, lv in enumerate(levels):
        if conc_nm == lv:
            return idx
    raise ValueError(f"Concentration {conc_nm} nM not in levels {levels}")


def _encode_labels(concentrations_arr):
    """Convert (N, 3) concentration array to (N, 3) class labels.

    Args:
        concentrations_arr: (N, 3) np.ndarray, [Cu, Fe, Zn] in nM.

    Returns:
        labels: (N, 3) np.ndarray of int class indices.
    """
    N = concentrations_arr.shape[0]
    labels = np.zeros((N, 3), dtype=np.int64)
    labels[:, 0] = [_conc_to_label(c, CU_LEVELS) for c in concentrations_arr[:, 0]]
    labels[:, 1] = [_conc_to_label(c, FE_LEVELS) for c in concentrations_arr[:, 1]]
    labels[:, 2] = [_conc_to_label(c, ZN_LEVELS) for c in concentrations_arr[:, 2]]
    return labels


# ===========================================================================
# Classification Head
# ===========================================================================

class MultiClassHead(nn.Module):
    """3 independent multi-class heads, 5 classes each (Cu/Fe/Zn).

    Shared trunk: Linear(256->128) + ReLU + Dropout(0.3).
    Then 3 separate Linear(128->5) heads.
    """

    def __init__(self, in_dim=256, num_classes=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head_cu = nn.Linear(128, num_classes)
        self.head_fe = nn.Linear(128, num_classes)
        self.head_zn = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.shared(x)
        return self.head_cu(h), self.head_fe(h), self.head_zn(h)


# ===========================================================================
# Stage 1: BYOL Pre-training
# ===========================================================================

def _stage1_byol_pretrain(peaks_tensor, peak_mask, config, model_dir,
                          device, re_training=False):
    """BYOL self-supervised pre-training on peak tokens.

    Args:
        peaks_tensor: (N, N_max, 3) float tensor.
        peak_mask: (N, N_max) bool tensor, True = padded.
        config: BYOL_CONFIG dict.
        model_dir: path to save encoder weights.
        device: torch device.
        re_training: if True, train from scratch.

    Returns:
        encoder_path: path to saved Stage 1 encoder weights.
    """
    cfg = config
    proj_hidden = cfg["proj_hidden"]
    proj_out = cfg["proj_out"]

    print("\n" + "=" * 60)
    print("Stage 1: BYOL Self-Supervised Pre-training on Peak Tokens")
    print("=" * 60)
    print(f"  Online: augmented peaks -> encoder -> proj({proj_hidden}/{proj_out}) -> predictor")
    print(f"  Target: clean peaks -> encoder -> proj (EMA, no predictor)")
    print(f"  Loss: symmetric cosine distance")
    print(f"  Epochs: {cfg['stage1_epochs']}, "
          f"Batch: {cfg['stage1_batch_size']}, LR: {cfg['stage1_lr']}")

    # Build models
    online = BYOLPeakModel(config, has_predictor=True).to(device)
    target = BYOLPeakModel(config, has_predictor=False).to(device)
    target.load_state_dict(online.state_dict(), strict=False)
    for p in target.parameters():
        p.requires_grad = False
    target.eval()

    n_params = sum(p.numel() for p in online.parameters())
    print(f"  Online params: {n_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg["stage1_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["stage1_epochs"])

    # Peak augmentation
    augmenter = PeakAugmentation(
        scale_range=(0.7, 1.3),
        mask_ratio_min=cfg["mask_ratio_min"],
        mask_ratio_max=cfg["mask_ratio_max"],
        weak_fraction=cfg["weak_fraction"],
    )

    # Checkpoint
    ckpt_path = os.path.join(model_dir, "byol_stage1_checkpoint.pt")
    start_epoch = 0
    global_step = 0
    if not re_training and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        online.load_state_dict(ckpt['online_state_dict'])
        target.load_state_dict(ckpt['target_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        print(f"  Resumed from checkpoint at epoch {start_epoch}")

    # Data (move to device)
    n_samples = peaks_tensor.shape[0]
    batch_size = cfg["stage1_batch_size"]
    total_steps = cfg["stage1_epochs"] * max(1, n_samples // batch_size)
    ema_base = cfg["ema_momentum_base"]

    # Move all data to device once
    X_clean = peaks_tensor.to(device)
    M_clean = peak_mask.to(device)

    # Logging
    os.makedirs("log", exist_ok=True)
    log_name = f"log/byol_stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_f = open(log_name, 'w', newline='')
    log_w = csv.writer(log_f)
    log_w.writerow(['epoch', 'loss', 'cos_sim', 'proj_std', 'lr',
                     'elapsed_min', 'ETA_total_min'])
    t_start = time.time()

    print(f"\n  Training {cfg['stage1_epochs']} epochs ...")
    for epoch in range(start_epoch, cfg["stage1_epochs"]):
        online.train()
        epoch_loss, epoch_cos, n = 0.0, 0.0, 0
        perm = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start:start + batch_size]
            clean_batch = X_clean[batch_idx]
            clean_mask = M_clean[batch_idx]

            # View 1 (online): augmented peak tokens
            aug_batch, aug_mask = augmenter(clean_batch, clean_mask)

            # View 2 (target): clean peak tokens
            tgt_batch = clean_batch
            tgt_mask = clean_mask

            # Symmetric BYOL loss
            q1 = online(aug_batch, aug_mask)
            q2 = online(tgt_batch, tgt_mask)
            with torch.no_grad():
                z1 = target(aug_batch, aug_mask)
                z2 = target(tgt_batch, tgt_mask)

            loss = (2.0 - F.cosine_similarity(q1, z2).mean()
                      - F.cosine_similarity(q2, z1).mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update target
            m = _cosine_momentum(global_step, total_steps, base=ema_base)
            _ema_update(online, target, m)
            global_step += 1

            epoch_loss += loss.item() * batch_idx.size(0)
            epoch_cos += (2.0 - loss.item()) * batch_idx.size(0)
            n += batch_idx.size(0)

        scheduler.step()
        loss_avg = epoch_loss / n
        cos_avg = epoch_cos / n

        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                z_sample = target(X_clean[:min(64, n_samples)],
                                  M_clean[:min(64, n_samples)])
                proj_std = z_sample.std(dim=0).mean().item()
            elapsed = (time.time() - t_start) / 60
            progress = (epoch + 1 - start_epoch) / max(
                cfg['stage1_epochs'] - start_epoch, 1)
            eta = elapsed / max(progress, 0.001)
            print(f"  Epoch {epoch + 1:3d}/{cfg['stage1_epochs']}: "
                  f"loss={loss_avg:.4f}, cos={cos_avg:.4f}, "
                  f"proj_std={proj_std:.4f}, "
                  f"elapsed={elapsed:.1f}m, ETA={eta:.1f}m")

        # Log
        elapsed = (time.time() - t_start) / 60
        progress = (epoch + 1 - start_epoch) / max(
            cfg['stage1_epochs'] - start_epoch, 1)
        eta = elapsed / max(progress, 0.001)
        proj_std_val = proj_std if (epoch + 1) % 5 == 0 or epoch == 0 else 0.0
        log_w.writerow([epoch + 1, f"{loss_avg:.6f}", f"{cos_avg:.6f}",
                         f"{proj_std_val:.6f}",
                         f"{scheduler.get_last_lr()[0]:.2e}",
                         f"{elapsed:.2f}", f"{eta:.2f}"])
        log_f.flush()

        # Checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({'epoch': epoch, 'global_step': global_step,
                         'online_state_dict': online.state_dict(),
                         'target_state_dict': target.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         }, ckpt_path)
            print(f"    Checkpoint saved to {ckpt_path}")

    log_f.close()
    print(f"  Log saved to {log_name}")

    # Save online encoder
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, STAGE1_FILENAME)
    torch.save({'encoder_state_dict': online.encoder.state_dict(),
                'config': {k: v for k, v in cfg.items()
                          if not callable(v)}},
               save_path)
    print(f"\n  Online encoder saved to {save_path}")

    return save_path


# ===========================================================================
# Stage 2: Multi-Class Classification Fine-tuning
# ===========================================================================

def _stage2_finetune(peaks_tensor, peak_mask, labels, df_all,
                     encoder_path, config, model_dir, device):
    """Multi-class concentration-level classification fine-tuning.

    3-fold OOF CV. Each fold:
      Phase 1: freeze encoder, train MultiClassHead.
      Phase 2: unfreeze all, fine-tune end-to-end with lower LR.

    Args:
        peaks_tensor: (N, N_max, 3) float tensor.
        peak_mask: (N, N_max) bool tensor.
        labels: (N, 3) int64 array, class indices [0-4].
        df_all: DataFrame with group_id, mixture, group_number, outer_fold.
        encoder_path: path to Stage 1 encoder weights.
        config: BYOL_CONFIG dict.
        model_dir: path to save classifier.
        device: torch device.

    Returns:
        all_preds: (N, 3) int array of predicted class indices.
        all_probs: (N, 3, 5) float array of class probabilities.
    """
    cfg = config
    num_classes = cfg["num_classes"]

    print("\n" + "=" * 60)
    print("Stage 2: Multi-Class Classification Fine-tuning")
    print("=" * 60)
    print(f"  Frozen phase: {cfg['stage2_frozen_epochs']} epochs")
    print(f"  Full phase: {cfg['stage2_full_epochs']} epochs")
    print(f"  Task: 3 x 5-class classification (Cu/Fe/Zn levels)")

    N = peaks_tensor.shape[0]
    all_preds = np.zeros((N, 3), dtype=int)
    all_probs = np.zeros((N, 3, num_classes), dtype=np.float32)

    for fold in range(N_OUTER):
        # Reload clean encoder for each fold (prevents OOF leakage)
        encoder = PeakTokenEncoder(
            d_model=cfg["d_model"], nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["transformer_dropout"],
        ).to(device)
        ckpt = torch.load(encoder_path, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

        test_mask = df_all["outer_fold"].to_numpy() == fold
        train_mask = ~test_mask

        X_tr_p = peaks_tensor[train_mask].to(device)
        M_tr_p = peak_mask[train_mask].to(device)
        Y_tr = torch.from_numpy(labels[train_mask]).long().to(device)
        X_te_p = peaks_tensor[test_mask].to(device)
        M_te_p = peak_mask[test_mask].to(device)

        print(f"\n  --- Fold {fold + 1}/{N_OUTER} ---")
        print(f"  Train: {train_mask.sum()} spectra, "
              f"Test: {test_mask.sum()} spectra")

        # Build head
        head = MultiClassHead(in_dim=cfg["d_model"],
                              num_classes=num_classes).to(device)

        # --- Phase 1: frozen encoder ---
        print(f"  Phase 1: frozen encoder ({cfg['stage2_frozen_epochs']} ep)")
        optimizer = torch.optim.Adam(head.parameters(),
                                     lr=cfg["stage2_lr"])

        for epoch in range(cfg["stage2_frozen_epochs"]):
            head.train()
            epoch_loss, n = 0.0, 0
            perm = torch.randperm(X_tr_p.shape[0])
            for start in range(0, X_tr_p.shape[0], cfg["stage2_batch_size"]):
                idx = perm[start:start + cfg["stage2_batch_size"]]
                bx, bm, by = X_tr_p[idx], M_tr_p[idx], Y_tr[idx]
                with torch.no_grad():
                    feats = encoder(bx, bm)
                logits_cu, logits_fe, logits_zn = head(feats)
                loss = (F.cross_entropy(logits_cu, by[:, 0]) +
                        F.cross_entropy(logits_fe, by[:, 1]) +
                        F.cross_entropy(logits_zn, by[:, 2]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
                n += bx.size(0)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:2d}: loss={epoch_loss / n:.4f}")

        # --- Phase 2: unfreeze all ---
        print(f"  Phase 2: unfreeze all ({cfg['stage2_full_epochs']} ep)")
        for p in encoder.parameters():
            p.requires_grad = True
        encoder.train()
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()),
            lr=cfg["stage2_lr"] * 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["stage2_full_epochs"])

        for epoch in range(cfg["stage2_full_epochs"]):
            epoch_loss, n = 0.0, 0
            perm = torch.randperm(X_tr_p.shape[0])
            for start in range(0, X_tr_p.shape[0], cfg["stage2_batch_size"]):
                idx = perm[start:start + cfg["stage2_batch_size"]]
                bx, bm, by = X_tr_p[idx], M_tr_p[idx], Y_tr[idx]
                feats = encoder(bx, bm)
                logits_cu, logits_fe, logits_zn = head(feats)
                loss = (F.cross_entropy(logits_cu, by[:, 0]) +
                        F.cross_entropy(logits_fe, by[:, 1]) +
                        F.cross_entropy(logits_zn, by[:, 2]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
                n += bx.size(0)
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:2d}: loss={epoch_loss / n:.4f}")

        # Evaluate
        head.eval()
        encoder.eval()
        with torch.no_grad():
            feats_te = encoder(X_te_p, M_te_p)
            logits_cu, logits_fe, logits_zn = head(feats_te)
            probs_cu = F.softmax(logits_cu, dim=-1)
            probs_fe = F.softmax(logits_fe, dim=-1)
            probs_zn = F.softmax(logits_zn, dim=-1)
            preds_cu = logits_cu.argmax(dim=-1)
            preds_fe = logits_fe.argmax(dim=-1)
            preds_zn = logits_zn.argmax(dim=-1)

        all_preds[test_mask, 0] = preds_cu.cpu().numpy()
        all_preds[test_mask, 1] = preds_fe.cpu().numpy()
        all_preds[test_mask, 2] = preds_zn.cpu().numpy()
        all_probs[test_mask, 0] = probs_cu.cpu().numpy()
        all_probs[test_mask, 1] = probs_fe.cpu().numpy()
        all_probs[test_mask, 2] = probs_zn.cpu().numpy()

    # --- Final metrics ---
    print(f"\n  --- OOF Classification Results ---")
    for j, a in enumerate(ANALYTES):
        acc = accuracy_score(labels[:, j], all_preds[:, j])
        print(f"  {a}: Accuracy={acc:.4f}")

    exact = (all_preds == labels).all(axis=1).mean()
    print(f"  Exact-match (all 3): {exact:.4f}")

    # Save classifier
    os.makedirs(model_dir, exist_ok=True)
    torch.save({'encoder_state_dict': encoder.state_dict(),
                'head_state_dict': head.state_dict()},
               os.path.join(model_dir, STAGE2_FILENAME))
    print(f"  Classifier saved to {os.path.join(model_dir, STAGE2_FILENAME)}")

    return all_preds, all_probs


# ===========================================================================
# CSV Export & Plotting
# ===========================================================================

def _export_results(df_all, labels, all_preds, all_probs, levels_map):
    """Export per-spectrum classification results to CSV.

    Args:
        df_all: DataFrame with group_number, group_id, mixture, etc.
        labels: (N, 3) true class indices.
        all_preds: (N, 3) predicted class indices.
        all_probs: (N, 3, 5) class probabilities.
        levels_map: {"Cu": CU_LEVELS, "Fe": FE_LEVELS, "Zn": ZN_LEVELS}.
    """
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    rows = []
    for i in range(len(labels)):
        cu_true = CU_LEVELS[labels[i, 0]]
        cu_pred = CU_LEVELS[all_preds[i, 0]]
        fe_true = FE_LEVELS[labels[i, 1]]
        fe_pred = FE_LEVELS[all_preds[i, 1]]
        zn_true = ZN_LEVELS[labels[i, 2]]
        zn_pred = ZN_LEVELS[all_preds[i, 2]]

        rows.append({
            "group_number": df_all["group_number"].iloc[i],
            "mixture": df_all["mixture"].iloc[i],
            "Cu_true(nM)": cu_true,
            "Cu_pred(nM)": cu_pred,
            "Cu_correct": cu_true == cu_pred,
            "Cu_confidence": all_probs[i, 0, all_preds[i, 0]],
            "Fe_true(nM)": fe_true,
            "Fe_pred(nM)": fe_pred,
            "Fe_correct": fe_true == fe_pred,
            "Fe_confidence": all_probs[i, 1, all_preds[i, 1]],
            "Zn_true(nM)": zn_true,
            "Zn_pred(nM)": zn_pred,
            "Zn_correct": zn_true == zn_pred,
            "Zn_confidence": all_probs[i, 2, all_preds[i, 2]],
        })

    csv_df = pd.DataFrame(rows)
    csv_path = os.path.join(reports_dir, CSV_FILENAME)
    csv_df.to_csv(csv_path, index=False)
    print(f"\nPer-spectrum predictions saved to {csv_path}")

    return csv_df


def _plot_stage2_confusion(labels, all_preds, all_probs, df_all):
    """Generate confusion matrices for Stage 2 classification."""
    os.makedirs("visualization", exist_ok=True)

    # ---- Per-analyte 5x5 confusion matrices ----
    levels_names = {
        "Cu": [f"{lv}" for lv in CU_LEVELS],
        "Fe": [f"{lv}" for lv in FE_LEVELS],
        "Zn": [f"{lv / 1000:.1f}uM" if lv >= 1000 else f"{lv}"
               for lv in ZN_LEVELS],
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for j, a in enumerate(ANALYTES):
        lvls = LEVELS_MAP[a]
        lvl_names = levels_names[a]

        # Spectra-level confusion
        cm_s = confusion_matrix(labels[:, j], all_preds[:, j],
                                 labels=list(range(len(lvls))))
        im = axes[0, j].imshow(cm_s, cmap='Blues', aspect='auto')
        axes[0, j].set_xticks(range(len(lvl_names)))
        axes[0, j].set_yticks(range(len(lvl_names)))
        axes[0, j].set_xticklabels(lvl_names, rotation=45, fontsize=7)
        axes[0, j].set_yticklabels(lvl_names, fontsize=7)
        axes[0, j].set_xlabel("Pred (nM)"); axes[0, j].set_ylabel("True (nM)")
        for r in range(len(lvls)):
            for c in range(len(lvls)):
                axes[0, j].text(c, r, str(cm_s[r, c]),
                               ha='center', va='center', fontsize=6)
        acc = accuracy_score(labels[:, j], all_preds[:, j])
        axes[0, j].set_title(f"{a} spectra (acc={acc:.3f})")

        # Group-level confusion
        grp = df_all.copy()
        grp[f"pred_{a}"] = all_preds[:, j]
        grp[f"true_{a}"] = labels[:, j]
        ga = grp.groupby("group_id").agg(
            **{f"true_{a}": (f"true_{a}", "first"),
               f"pred_{a}": (f"pred_{a}", lambda x: pd.Series.mode(x).iloc[0])}
        ).reset_index()

        cm_g = confusion_matrix(
            ga[f"true_{a}"], ga[f"pred_{a}"],
            labels=list(range(len(lvls))))
        axes[1, j].imshow(cm_g, cmap='Blues', aspect='auto')
        axes[1, j].set_xticks(range(len(lvl_names)))
        axes[1, j].set_yticks(range(len(lvl_names)))
        axes[1, j].set_xticklabels(lvl_names, rotation=45, fontsize=7)
        axes[1, j].set_yticklabels(lvl_names, fontsize=7)
        axes[1, j].set_xlabel("Pred (nM)"); axes[1, j].set_ylabel("True (nM)")
        for r in range(len(lvls)):
            for c in range(len(lvls)):
                axes[1, j].text(c, r, str(cm_g[r, c]),
                               ha='center', va='center', fontsize=6)
        gacc = accuracy_score(ga[f"true_{a}"], ga[f"pred_{a}"])
        axes[1, j].set_title(f"{a} group (acc={gacc:.3f})")

    axes[0, 0].set_ylabel("Spectra-level")
    axes[1, 0].set_ylabel("Group-level")
    fig.suptitle("BYOL Stage 2: Concentration Level Classification (OOF)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig("visualization/BYOL_Stage2_Confusion.png", dpi=300)
    plt.close(fig)

    # ---- Group-level predicted vs true concentration ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    for j, (ax, a) in enumerate(zip(axes2, ANALYTES)):
        lvls = LEVELS_MAP[a]
        grp = df_all.copy()
        grp[f"pred_{a}"] = [lvls[p] for p in all_preds[:, j]]
        grp[f"true_{a}"] = [lvls[t] for t in labels[:, j]]
        ga = grp.groupby("group_id").agg(
            true=(f"true_{a}", "first"),
            mean=(f"pred_{a}", "mean"),
            sd=(f"pred_{a}", "std"),
        ).reset_index()
        ga["sd"] = ga["sd"].fillna(0)

        t = ga["true"].to_numpy(); m = ga["mean"].to_numpy()
        s = ga["sd"].to_numpy()
        ax.errorbar(t, m, yerr=s, fmt='o', color=colors[j],
                    capsize=3, markersize=6, markeredgecolor='k',
                    markeredgewidth=0.5)
        mx = max(t.max(), (m + s).max()) * 1.1
        if mx <= 0:
            mx = 1
        ax.plot([0, mx], [0, mx], 'r--', lw=1)
        ax.set_xlim(-mx * 0.05, mx); ax.set_ylim(-mx * 0.05, mx)
        ax.set_xlabel(f"True {a} (nM)")
        ax.set_ylabel(f"Predicted {a} (nM)")
        gacc = accuracy_score(
            [lvls.index(v) for v in ga["true"]],
            [np.round(v / min(np.diff(lvls))) * min(np.diff(lvls))
             if min(np.diff(lvls)) > 0 else v
             for v in ga["mean"]])  # approximation
        # Use simple accuracy from confusion matrix
        ga_pred_labels = []
        for v in ga["mean"]:
            best = min(lvls, key=lambda x: abs(x - v))
            ga_pred_labels.append(lvls.index(best))
        ga_true_labels = [lvls.index(v) for v in ga["true"]]
        gacc = accuracy_score(ga_true_labels, ga_pred_labels)
        ax.set_title(f"{a}: group acc={gacc:.3f}")
        ax.grid(alpha=0.25)

    fig2.suptitle("BYOL Stage 2: Group-Level Predicted vs True Concentration",
                  fontsize=14)
    fig2.tight_layout()
    fig2.savefig("visualization/BYOL_Stage2_Group_Predictions.png", dpi=300)
    plt.close(fig2)


# ===========================================================================
# Main entry point
# ===========================================================================

def run_byol_pipeline(data_dir, model_dir, plot=True,
                      mix_only=False, present_conc_range=None,
                      stage1=True, stage2=True, re_training=False,
                      config=None,
                      cut_range=(0, 2500),
                      airpls_lambda=1e7, airpls_polyorder=3,
                      airpls_max_iters=150):
    """Full BYOL peak-token pre-training + multi-class classification pipeline.

    Args:
        data_dir: path to data directory with txt files.
        model_dir: directory for saving models.
        plot: generate diagnostic plots.
        mix_only: keep only binary+ternary mixtures.
        present_conc_range: (min, max) nM for present component concentrations.
        stage1: run Stage 1 BYOL pre-training.
        stage2: run Stage 2 classification fine-tuning.
        re_training: if True, train from scratch.
        config: override default BYOL_CONFIG.
        cut_range: Raman shift range for preprocessing.
        airpls_lambda, airpls_polyorder, airpls_max_iters: airPLS params.

    Returns:
        dict with predictions, labels, and metrics.
    """
    cfg = BYOL_CONFIG.copy()
    if config:
        cfg.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. Read data ------------------------------------------------------
    print("\n" + "=" * 60)
    print("BYOL Peak-Token Pipeline")
    print("=" * 60)
    print("\n[1/7] Reading data ...")
    group_numbers, concentrations, raman_shift, intensities = read_data(
        data_dir)

    # airPLS baseline correction (full range, no min-max)
    cut_raman, cut_intensities = preprocess_data(
        raman_shift, intensities,
        cut_range=cut_range,
        lamb=airpls_lambda,
        polyorder=airpls_polyorder,
        max_iters=airpls_max_iters,
        plot=False,
        minmax_normalize=False
    )

    X_preprocessed = np.array(cut_intensities)

    # Mixtures and filtering
    mixtures = np.array([
        _make_mixture_label(c[0], c[1], c[2])
        for c in concentrations
    ])

    cut_intensities_list, concentrations_list, group_numbers_list, mixtures_list = \
        _filter_mix_conc(
            [row for row in X_preprocessed],
            concentrations, group_numbers, mixtures,
            mix_only=mix_only, present_conc_range=present_conc_range)

    X_filtered = np.array(cut_intensities_list)
    concentrations_arr = np.array(concentrations_list)
    group_numbers_arr = np.array(group_numbers_list)
    mixtures_arr = np.array(mixtures_list)

    print(f"  Spectra: {X_filtered.shape[0]}, "
          f"Raman: {cut_raman[0]:.0f}-{cut_raman[-1]:.0f} cm-1 "
          f"({X_filtered.shape[1]} points)")

    # ---- 2. Extract peak tokens --------------------------------------------
    print("\n[2/7] Extracting peak tokens ...")
    extractor = PeakExtractor(
        cut_raman,
        noise_region=cfg["noise_region"],
        norm_center=cfg["norm_peak_center"],
        norm_half_window=cfg["norm_peak_half_window"],
        min_height_sigma=cfg["min_height_sigma"],
    )

    peaks_tensor, peak_mask, n_peaks_list, _ = extractor(X_filtered)
    n_avg = np.mean(n_peaks_list)
    n_max = peaks_tensor.shape[1]
    print(f"  Avg peaks/spectrum: {n_avg:.0f}, max: {n_max}")

    # ---- 3. Encode labels --------------------------------------------------
    print("\n[3/7] Encoding concentration labels ...")
    labels = _encode_labels(concentrations_arr)
    for j, a in enumerate(ANALYTES):
        unique_lbl = np.unique(labels[:, j])
        print(f"  {a}: {len(unique_lbl)} classes, "
              f"labels={sorted(unique_lbl)}")

    # ---- 4. Build group table & assign folds -------------------------------
    print("\n[4/7] Assigning 3-fold splits ...")
    group_ids = np.array([
        _make_group_id(group_numbers_arr[i],
                       concentrations_arr[i, 0],
                       concentrations_arr[i, 1],
                       concentrations_arr[i, 2])
        for i in range(len(group_numbers_arr))
    ])

    group_table = pd.DataFrame({
        "group_id": group_ids,
        "mixture": mixtures_arr,
        "conc_Cu": concentrations_arr[:, 0],
        "conc_Fe": concentrations_arr[:, 1],
        "conc_Zn": concentrations_arr[:, 2],
    }).drop_duplicates("group_id").reset_index(drop=True)

    for mix in VALID_MIXTURES:
        n = (mixtures_arr == mix).sum()
        if n > 0:
            ng = (group_table["mixture"] == mix).sum()
            print(f"    {mix:12s}: {n:5d} spectra, {ng:3d} groups")

    outer_folds = _group_folds(
        group_table, n_splits=N_OUTER, random_state=RANDOM_STATE)
    fold_lookup = dict(zip(outer_folds["group_id"], outer_folds["fold"]))

    df_all = pd.DataFrame({
        "group_id": group_ids, "mixture": mixtures_arr,
        "group_number": group_numbers_arr,
        "conc_Cu": concentrations_arr[:, 0],
        "conc_Fe": concentrations_arr[:, 1],
        "conc_Zn": concentrations_arr[:, 2],
    })
    df_all["outer_fold"] = (
        df_all["group_id"].map(fold_lookup).astype(int))
    for f in range(N_OUTER):
        n_s = (df_all["outer_fold"] == f).sum()
        print(f"  Fold {f}: {n_s} spectra")

    # ---- 5. Stage 1: BYOL pre-training -------------------------------------
    encoder_path = os.path.join(model_dir, STAGE1_FILENAME)
    if stage1:
        print("\n[5/7] Stage 1: BYOL pre-training ...")
        encoder_path = _stage1_byol_pretrain(
            peaks_tensor, peak_mask, cfg, model_dir, device,
            re_training=re_training)
    elif not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Stage 1 encoder not found at {encoder_path}. "
            f"Run with --stage1 first or provide a pre-trained encoder."
        )
    else:
        print(f"\n[5/7] Using existing encoder: {encoder_path}")

    # ---- 6. Stage 2: Classification fine-tuning ----------------------------
    if stage2:
        print("\n[6/7] Stage 2: Multi-class classification ...")
        all_preds, all_probs = _stage2_finetune(
            peaks_tensor, peak_mask, labels, df_all,
            encoder_path, cfg, model_dir, device)
    else:
        print("\n[6/7] Stage 2 skipped.")
        all_preds = None
        all_probs = None

    # ---- 7. Export & Plot --------------------------------------------------
    print("\n[7/7] Exporting results ...")
    if all_preds is not None:
        _export_results(df_all, labels, all_preds, all_probs, LEVELS_MAP)
        if plot:
            print("  Generating plots ...")
            _plot_stage2_confusion(labels, all_preds, all_probs, df_all)

    print("\n" + "=" * 60)
    print("BYOL Peak-Token Pipeline completed.")
    print("=" * 60)

    return {
        "all_preds": all_preds,
        "all_probs": all_probs,
        "labels": labels,
        "df_all": df_all,
        "encoder_path": encoder_path,
    }
