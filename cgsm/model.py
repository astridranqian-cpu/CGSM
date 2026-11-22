from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .modules.tcn import FiLMResBlock
from .modules.ric import weight_inputs
from .modules.swe import make_swe_refiner, residual_correction


def _dyn_col_idx(dyn_cols: Sequence[str], name: str) -> int:
    try:
        return dyn_cols.index(name)
    except ValueError:
        return -1


class CausalGuidedSnowmeltModel(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        static_dim: int,
        dyn_cols_list: List[str],
        imte_w_np: np.ndarray,
        tcn_ch=(64, 64, 128, 128),
        ks=7,
        drop=0.3,
        imte_gain_init: float = 0.35,
        ddpm_alpha_init: float = 0.12,
        q_mm_mu: float = 0.0,
        q_mm_sd: float = 1.0,
        phys_fusion_dim: int = 48,
        t_freeze_norm: Optional[float] = None,
        prcp_bypass_gain_init: float = 1.5,
        prcp_skip_gain_init: float = 2.0,
        disable_imte: bool = False,
        disable_film: bool = False,
        disable_swe_refiner: bool = False,
        disable_phys_fusion: bool = False,
        tcn_only: bool = False,
    ):
        super().__init__()
        self.disable_imte = bool(disable_imte or tcn_only)
        self.disable_film = bool(disable_film or tcn_only)
        self.disable_swe_refiner = bool(disable_swe_refiner or tcn_only)
        self.disable_phys_fusion = bool(disable_phys_fusion or tcn_only)
        self.tcn_only = bool(tcn_only)
        w = np.asarray(imte_w_np, dtype=np.float32).reshape(-1)
        assert w.shape[0] == dyn_dim
        self.register_buffer("imte_w", torch.from_numpy(w))
        self.register_buffer(
            "q_mm_mu", torch.tensor([float(q_mm_mu)], dtype=torch.float32)
        )
        self.register_buffer(
            "q_mm_sd", torch.tensor([float(q_mm_sd)], dtype=torch.float32)
        )
        if "SWE_mm" in dyn_cols_list:
            self.swe_idx = int(dyn_cols_list.index("SWE_mm"))
        else:
            self.swe_idx = 0
        if "T_mean_c" in dyn_cols_list:
            self.t_mean_idx: Optional[int] = int(dyn_cols_list.index("T_mean_c"))
        else:
            self.t_mean_idx = None

        prcp_idx = [
            dyn_cols_list.index(n)
            for n in ("prcp_mm", "prcp_7d", "prcp_30d")
            if n in dyn_cols_list
        ]
        self.n_prcp = len(prcp_idx)
        if prcp_idx:
            self.register_buffer(
                "prcp_channel_idx", torch.tensor(prcp_idx, dtype=torch.long)
            )
            self.prcp_bypass_gain = nn.Parameter(
                torch.tensor(float(prcp_bypass_gain_init))
            )
            self.prcp_skip_gain = nn.Parameter(
                torch.tensor(float(prcp_skip_gain_init))
            )
        else:
            self.register_buffer("prcp_channel_idx", torch.empty(0, dtype=torch.long))
            self.prcp_bypass_gain = None
            self.prcp_skip_gain = None

        self.sin_doy_idx = _dyn_col_idx(dyn_cols_list, "sin_doy")
        self.cos_doy_idx = _dyn_col_idx(dyn_cols_list, "cos_doy")
        self.season_dim = (1 if self.sin_doy_idx >= 0 else 0) + (
            1 if self.cos_doy_idx >= 0 else 0
        )

        if t_freeze_norm is not None:
            self.register_buffer(
                "_t_freeze_norm", torch.tensor([float(t_freeze_norm)])
            )
        else:
            self.register_buffer("_t_freeze_norm", torch.empty(0))

        self.imte_gain = nn.Parameter(torch.tensor(float(imte_gain_init)))
        self.ddpm_alpha = nn.Parameter(torch.tensor(float(ddpm_alpha_init)))

        stat_aug_dim = static_dim if self.disable_imte else static_dim + dyn_dim
        dpf = int(phys_fusion_dim)
        self.phys_fusion_dim = dpf
        film_stat_dim = stat_aug_dim + dpf + self.season_dim

        hid_dyn = max(16, dyn_dim * 2)
        self.dyn_phys_encoder = nn.Sequential(
            nn.Conv1d(dyn_dim, hid_dyn, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hid_dyn, dpf, kernel_size=1),
        )
        self.static_phys_encoder = nn.Sequential(
            nn.Linear(stat_aug_dim, dpf * 2),
            nn.GELU(),
            nn.Linear(dpf * 2, dpf),
        )
        self.phys_fusion_gate = nn.Linear(dpf * 2, dpf)

        gate_hid = 28
        self.swe_residual_gate = nn.Sequential(
            nn.Linear(film_stat_dim, gate_hid),
            nn.GELU(),
            nn.Linear(gate_hid, 1),
            nn.Sigmoid(),
        )
        tcn_in_dim = dyn_dim if self.tcn_only else dyn_dim + self.n_prcp
        blocks = []
        for i, oc in enumerate(tcn_ch):
            ic = tcn_in_dim if i == 0 else tcn_ch[i - 1]
            dil = 2**i
            blocks.append(FiLMResBlock(ic, oc, ks, dil, film_stat_dim, drop))
        self.tcn = nn.ModuleList(blocks)
        last_ch = tcn_ch[-1]
        self.head = nn.Sequential(
            nn.Linear(last_ch * 2 + 1 + dpf, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1),
        )
        self.swe_refiner = make_swe_refiner(dyn_dim)

    def forward(
        self,
        dyn,
        stat,
        return_delta: bool = False,
        return_aux: bool = False,
    ):
        B, L, Fdim = dyn.shape
        dyn_e = weight_inputs(dyn, self.imte_w, self.imte_gain, self.disable_imte)
        if self.n_prcp > 0 and self.prcp_bypass_gain is not None and not self.tcn_only:
            idx = self.prcp_channel_idx
            dyn_e = dyn_e.clone()
            dyn_e[:, :, idx] = dyn[:, :, idx] * self.prcp_bypass_gain

        if self.disable_imte:
            stat_aug = stat
        else:
            stat_aug = torch.cat(
                [stat, self.imte_w.unsqueeze(0).expand(B, -1)], dim=1
            )
        d_enc = self.dyn_phys_encoder(dyn_e.transpose(1, 2))
        d_phys = d_enc.mean(dim=2)
        s_phys = self.static_phys_encoder(stat_aug)
        if self.disable_phys_fusion:
            phys_vec = torch.zeros_like(s_phys)
        else:
            gate = torch.sigmoid(
                self.phys_fusion_gate(torch.cat([s_phys, d_phys], dim=1))
            )
            phys_vec = gate * s_phys + (1.0 - gate) * d_phys

        season_parts = []
        if self.sin_doy_idx >= 0:
            season_parts.append(dyn[:, -1, self.sin_doy_idx : self.sin_doy_idx + 1])
        if self.cos_doy_idx >= 0:
            season_parts.append(dyn[:, -1, self.cos_doy_idx : self.cos_doy_idx + 1])
        if season_parts:
            season_vec = torch.cat(season_parts, dim=1)
            film_cond = torch.cat([stat_aug, phys_vec, season_vec], dim=1)
        else:
            film_cond = torch.cat([stat_aug, phys_vec], dim=1)

        delta, swe_corr = residual_correction(self, dyn, dyn_e, film_cond)
        dyn_work = dyn_e.clone()
        dyn_work[:, :, self.swe_idx] = dyn_work[:, :, self.swe_idx] + swe_corr

        x = dyn_work.transpose(1, 2)
        if self.n_prcp > 0 and self.prcp_skip_gain is not None and not self.tcn_only:
            prcp_raw = dyn.index_select(2, self.prcp_channel_idx).transpose(1, 2)
            x = torch.cat([x, prcp_raw * self.prcp_skip_gain], dim=1)
        for blk in self.tcn:
            x = blk(x, film_cond, disable_film=self.disable_film)
        last = x[:, :, -1]
        avg = x.mean(dim=2)
        delta_bar = delta.mean(dim=1, keepdim=True)
        out = self.head(
            torch.cat([last, avg, delta_bar, phys_vec], dim=1)
        ).squeeze(-1)

        if return_aux:
            return out, delta, dyn_work
        if return_delta:
            return out, delta
        return out


CGSM = CausalGuidedSnowmeltModel


def build_model(config, preprocessing):
    p = preprocessing
    m = config["model"]
    tmu, tsd = p["norm_stats"]["T_mean_c"]
    qmu, qsd = p["norm_stats"]["Q_mm_day"]
    return CGSM(
        len(p["dyn_cols"]),
        len(p["static_columns"]),
        p["dyn_cols"],
        np.asarray(p["ric"], dtype=np.float32),
        tcn_ch=tuple(m["tcn_channels"]),
        ks=m["kernel_size"],
        drop=m["dropout"],
        imte_gain_init=m["ric_gain_init"],
        ddpm_alpha_init=m["swe_alpha_init"],
        q_mm_mu=qmu,
        q_mm_sd=qsd,
        phys_fusion_dim=m["fusion_dim"],
        t_freeze_norm=(m["temperature_threshold_c"] - tmu) / max(tsd, 1e-8),
        prcp_bypass_gain_init=m["prcp_bypass_gain_init"],
        prcp_skip_gain_init=m["prcp_skip_gain_init"],
    )
