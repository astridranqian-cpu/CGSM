from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .losses import total_loss


def _ensure_device(d) -> torch.device:
    if isinstance(d, str):
        return torch.device(d)
    return d


def set_reproducible(seed: int, fast_cudnn: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = not fast_cudnn
        torch.backends.cudnn.benchmark = fast_cudnn


def compute_peak_threshold(loader, quantile=0.90):
    ts = []
    for batch in loader:
        ts.append(batch[2])
        if len(ts) > 200:
            break
    th = torch.quantile(torch.cat(ts), quantile).item()
    return th


def calc_metrics_physical(preds_norm, targets_norm, stats_flow: dict):
    mu = stats_flow["QObs_mm"]["mean"]
    sd = stats_flow["QObs_mm"]["std"]
    preds = preds_norm * sd + mu
    targets = targets_norm * sd + mu
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    nse = 1 - ss_res / (ss_tot + 1e-8)
    rmse = np.sqrt(np.mean((targets - preds) ** 2))
    r = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0
    alpha = np.std(preds) / (np.std(targets) + 1e-8)
    beta = np.mean(preds) / (np.mean(targets) + 1e-8)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    p90 = np.percentile(targets, 90)
    peak_mask = targets >= p90
    if peak_mask.sum() > 10:
        ss_res_p = np.sum((targets[peak_mask] - preds[peak_mask]) ** 2)
        ss_tot_p = np.sum((targets[peak_mask] - np.mean(targets[peak_mask])) ** 2)
        peak_nse = 1 - ss_res_p / (ss_tot_p + 1e-8)
    else:
        peak_nse = np.nan
    return {
        "NSE": float(nse),
        "KGE": float(kge),
        "KGE_r": float(r),
        "KGE_beta": float(beta),
        "KGE_alpha": float(alpha),
        "RMSE": float(rmse),
        "Peak_NSE": float(peak_nse) if np.isfinite(peak_nse) else float("nan"),
    }


@torch.no_grad()
def eval_validation_regional(
    model, val_loader, criterion, device, stats_flow: dict
) -> dict:
    device = _ensure_device(device)
    model.eval()
    total, n = 0.0, 0
    preds_chunks, tgt_chunks, gids_all = [], [], []
    non_block = device.type == "cuda"
    for batch in tqdm(val_loader, desc="Val", leave=False, unit="batch"):
        dyn = batch[0].to(device, non_blocking=non_block)
        stat = batch[1].to(device, non_blocking=non_block)
        tgt = batch[2].to(device, non_blocking=non_block)
        pred = model(dyn, stat)
        total += criterion(pred, tgt).item()
        n += 1
        preds_chunks.append(pred.cpu().numpy().ravel())
        tgt_chunks.append(tgt.cpu().numpy().ravel())
        gids_all.extend(batch[3])

    preds_norm = np.concatenate(preds_chunks)
    targets_norm = np.concatenate(tgt_chunks)
    mu = stats_flow["QObs_mm"]["mean"]
    sd = stats_flow["QObs_mm"]["std"]
    preds_phy = preds_norm * sd + mu
    targets_phy = targets_norm * sd + mu

    by_g: DefaultDict[str, Tuple[List[float], List[float]]] = defaultdict(
        lambda: ([], [])
    )
    for g, p, t in zip(gids_all, preds_phy, targets_phy):
        by_g[str(g)][0].append(float(p))
        by_g[str(g)][1].append(float(t))

    nse_list = []
    for _g, (pl, tl) in by_g.items():
        pl_arr = np.asarray(pl, dtype=np.float64)
        tl_arr = np.asarray(tl, dtype=np.float64)
        if len(tl_arr) < 3:
            continue
        ss_res = np.sum((tl_arr - pl_arr) ** 2)
        ss_tot = np.sum((tl_arr - np.mean(tl_arr)) ** 2)
        if ss_tot < 1e-12:
            continue
        nse_list.append(float(1.0 - ss_res / ss_tot))

    macro_mean = float(np.mean(nse_list)) if nse_list else float("nan")
    macro_std = float(np.std(nse_list)) if len(nse_list) > 1 else 0.0

    micro = calc_metrics_physical(preds_norm, targets_norm, stats_flow)
    micro["loss"] = total / max(n, 1)
    micro["NSE_macro_mean"] = macro_mean
    micro["NSE_macro_std"] = macro_std
    micro["n_basins_val"] = len(nse_list)
    return micro


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_settings,
    *,
    use_amp=False,
    scaler=None,
):
    model.train()
    device = _ensure_device(device)
    total, n = 0.0, 0
    for dyn, stat, target, *_ in loader:
        dyn = dyn.to(device, non_blocking=True)
        stat = stat.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            prediction, delta, dyn_work = model(dyn, stat, return_aux=True)
            loss = total_loss(model, prediction, target, delta, dyn_work, loss_settings)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)
