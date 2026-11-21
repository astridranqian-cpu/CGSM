from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


def compute_physics_swe_and_season(df: pd.DataFrame, ddf: float) -> pd.DataFrame:
    out = df.copy()
    t_mean = (
        out["tmax_c"].astype(np.float64) + out["tmin_c"].astype(np.float64)
    ) / 2.0
    prcp = out["prcp_mm"].astype(np.float64).values
    T = t_mean.values
    n = len(out)
    swe = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        P = prcp[t]
        Tt = T[t]
        p_snow = P if Tt < 0 else 0.0
        m_melt = min(swe[t - 1], ddf * Tt) if Tt > 0 else 0.0
        swe[t] = max(0.0, swe[t - 1] + p_snow - m_melt)
    out["T_mean_c"] = t_mean.astype(np.float32)
    out["SWE_mm"] = swe.astype(np.float32)
    doy = out["date"].dt.dayofyear.astype(np.float32)
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    out["prcp_7d"] = out["prcp_mm"].rolling(7, min_periods=1).sum().astype(np.float32)
    out["prcp_30d"] = out["prcp_mm"].rolling(30, min_periods=1).sum().astype(np.float32)
    return out


def make_swe_refiner(dyn_dim):
    hidden = max(16, dyn_dim * 2)
    return nn.Sequential(
        nn.Conv1d(dyn_dim, hidden, kernel_size=5, padding=2),
        nn.GELU(),
        nn.Conv1d(hidden, 1, kernel_size=5, padding=2),
    )


def residual_correction(model, dyn, dyn_e, film_cond):
    B, L, _ = dyn.shape
    if model.disable_swe_refiner:
        delta = torch.zeros((B, L), device=dyn.device, dtype=dyn.dtype)
        return delta, delta
    delta = model.swe_refiner(dyn_e.transpose(1, 2)).transpose(1, 2).squeeze(-1)
    rg = model.swe_residual_gate(film_cond)
    correction = torch.tanh(model.ddpm_alpha) * delta * rg
    if model._t_freeze_norm.numel() > 0 and model.t_mean_idx is not None:
        warm = (
            dyn[:, :, model.t_mean_idx] >= model._t_freeze_norm[0]
        ).to(correction.dtype)
        correction = correction * warm
    return delta, correction
