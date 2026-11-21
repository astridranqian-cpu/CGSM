from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PeakWeightedHuberLoss(nn.Module):
    def __init__(self, threshold, peak_weight=3.0, delta=1.0):
        super().__init__()
        self.threshold = threshold
        self.peak_weight = peak_weight
        self.huber = nn.SmoothL1Loss(reduction="none", beta=delta)
        self._current_weight = 1.0

    def set_epoch(self, epoch, warmup_epochs=5):
        self._current_weight = 1.0 if epoch <= warmup_epochs else self.peak_weight

    def forward(self, pred, target):
        loss = self.huber(pred, target)
        w = torch.ones_like(target)
        w[target >= self.threshold] = self._current_weight
        return (w * loss).mean()


def loss_terms(model, prediction, target, delta, dyn_work, settings):
    data = F.smooth_l1_loss(prediction, target, beta=settings["huber_beta"])
    residual = settings["swe_residual_reg"] * (delta**2).mean()
    melt = prediction.new_zeros(())
    if settings["melt_monotone_reg"] > 0 and model.t_mean_idx is not None:
        swe = dyn_work[:, :, model.swe_idx]
        temperature = dyn_work[:, :, model.t_mean_idx]
        increment = swe[:, 1:] - swe[:, :-1]
        t_mid = 0.5 * (temperature[:, 1:] + temperature[:, :-1])
        melt = (
            settings["melt_monotone_reg"]
            * (F.relu(increment) * F.relu(t_mid)).mean()
        )
    pred_mm = prediction * model.q_mm_sd + model.q_mm_mu
    nonnegative_q = settings["q_nonneg_mm_reg"] * F.relu(-pred_mm).mean()
    return {
        "data": data,
        "residual": residual,
        "snow": melt,
        "q_nonnegative": nonnegative_q,
    }


def total_loss(model, prediction, target, delta, dyn_work, settings):
    return sum(
        loss_terms(model, prediction, target, delta, dyn_work, settings).values()
    )
