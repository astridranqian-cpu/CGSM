from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch


def compute_imte_proxy_weights(
    train_raw: pd.DataFrame,
    dyn_cols: Sequence[str],
    q_col: str = "Q_mm_day",
    lag: int = 3,
) -> np.ndarray:
    lag = int(lag)
    acc = np.zeros(len(dyn_cols), dtype=np.float64)
    cnt = np.zeros(len(dyn_cols), dtype=np.int64)
    for _gid, grp in train_raw.groupby("gauge_id"):
        grp = grp.sort_values("date")
        if len(grp) < lag + 30:
            continue
        q = grp[q_col].values.astype(np.float64)
        for j, c in enumerate(dyn_cols):
            f = grp[c].values.astype(np.float64)
            a = f[:-lag]
            b = q[lag:]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 25:
                continue
            r = np.corrcoef(a[m], b[m])[0, 1]
            if np.isfinite(r):
                acc[j] += abs(float(r))
                cnt[j] += 1
    mean_abs = np.zeros(len(dyn_cols), dtype=np.float64)
    for j in range(len(dyn_cols)):
        if cnt[j] > 0:
            mean_abs[j] = acc[j] / cnt[j]
    idx = np.flatnonzero(cnt > 0)
    if idx.size == 0:
        w = np.ones(len(dyn_cols), dtype=np.float64) / max(len(dyn_cols), 1)
    else:
        sub = np.maximum(mean_abs[idx], 1e-8)
        sub = sub / sub.sum()
        w = np.zeros(len(dyn_cols), dtype=np.float64)
        w[idx] = sub
    return w.astype(np.float32)


compute_ric_weights = compute_imte_proxy_weights


def weight_inputs(dyn, weights, gain, disabled=False):
    if disabled:
        return dyn
    g = torch.sigmoid(gain) * 0.5
    scale = 1.0 + g * (dyn.shape[-1] * weights.view(1, 1, -1) - 1.0)
    return dyn * scale
