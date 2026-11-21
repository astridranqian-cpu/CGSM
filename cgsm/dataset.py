from __future__ import annotations
import glob, os, re
from collections import defaultdict, deque
from typing import DefaultDict, Dict, Iterator, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from .modules.swe import compute_physics_swe_and_season
from .modules.ric import compute_ric_weights

CFS_TO_M3S = 0.028316846592
STREAMFLOW_COLS = ['gauge_id_raw','Year','Month','Day','QObs','QC_flag']


RUNOFF_DERIVED_STATIC_FEATURES = frozenset({
    "q_mean",
    "runoff_ratio",
    "stream_elas",
    "slope_fdc",
    "baseflow_index",
    "hfd_mean",
    "q5",
    "q95",
    "high_q_freq",
    "high_q_dur",
    "low_q_freq",
    "low_q_dur",
    "zero_q_freq",
})


def select_static_feature_names(candidate_names: Sequence[str]) -> List[str]:


    names = list(candidate_names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate static attributes")

    excluded = [
        name for name in names
        if name.strip().casefold() in RUNOFF_DERIVED_STATIC_FEATURES
    ]
    selected = [
        name for name in names
        if name.strip().casefold() not in RUNOFF_DERIVED_STATIC_FEATURES
    ]

    if len(names) not in (40, 40 + len(RUNOFF_DERIVED_STATIC_FEATURES)) or len(selected) != 40:
        raise ValueError(
            f"Invalid static attributes: {len(names)} supplied, "
            f"{len(selected)} retained; expected 40"
        )
    if len(names) == 40 + len(RUNOFF_DERIVED_STATIC_FEATURES) and len(excluded) != 13:
        raise ValueError(
            f"Expected 13 runoff-derived attributes, got {len(excluded)}"
        )
    return selected

def _to_float32_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=[np.number]).columns:
        out[c] = out[c].astype(np.float32)
    return out


def load_merged_attributes_camels_us(attr_root: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(attr_root, "**", "*.txt"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"Attribute files not found: {attr_root}")
    merged: Optional[pd.DataFrame] = None
    for p in paths:
        try:
            part = pd.read_csv(p, sep=";", low_memory=False)
        except Exception:
            continue
        if "gauge_id" not in part.columns:
            continue
        part = part.copy()
        part["gauge_id"] = part["gauge_id"].astype(str).str.strip().str.zfill(8)
        part = _to_float32_numeric(part)
        if merged is None:
            merged = part
            continue
        new_cols = [c for c in part.columns if c != "gauge_id" and c not in merged.columns]
        if not new_cols:
            continue
        merged = merged.merge(part[["gauge_id"] + new_cols], on="gauge_id", how="outer")
    if merged is None or merged.empty:
        raise RuntimeError("No attributes loaded")
    return merged


def _normalize_forcing_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c: re.sub(r"\s+", "", str(c).strip()).lower() for c in df.columns}
    rev = {v: k for k, v in colmap.items()}

    def pick(*aliases):
        for a in aliases:
            if a in rev:
                return rev[a]
        return None

    y, m, d = pick("year"), pick("mnth", "month"), pick("day")
    if y is None or m is None or d is None:
        raise ValueError(f"Missing date columns: {list(df.columns)}")
    prcp_c = pick("prcp(mm/day)", "prcp", "p_mm_day", "prcpmmday")
    tmax_c = pick("tmax(c)", "tmax", "tmax_c")
    tmin_c = pick("tmin(c)", "tmin", "tmin_c")
    vp_c = pick("vp(pa)", "vp", "vap", "vp_pa")
    srad_c = pick("srad(w/m2)", "srad", "swrad", "sradwm2")
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.to_numeric(df[y], errors="coerce").astype("Int64"),
                "month": pd.to_numeric(df[m], errors="coerce").astype("Int64"),
                "day": pd.to_numeric(df[d], errors="coerce").astype("Int64"),
            }
        )
    )
    if not prcp_c:
        raise ValueError("Missing precipitation column")
    out["prcp_mm"] = pd.to_numeric(df[prcp_c], errors="coerce").astype(np.float32)
    if not (tmax_c and tmin_c):
        raise ValueError("Missing tmax or tmin column")
    out["tmax_c"] = pd.to_numeric(df[tmax_c], errors="coerce").astype(np.float32)
    out["tmin_c"] = pd.to_numeric(df[tmin_c], errors="coerce").astype(np.float32)
    out["vp_pa"] = (
        pd.to_numeric(df[vp_c], errors="coerce").astype(np.float32)
        if vp_c
        else np.float32(np.nan)
    )
    out["srad_wm2"] = (
        pd.to_numeric(df[srad_c], errors="coerce").astype(np.float32)
        if srad_c
        else np.float32(np.nan)
    )
    return out.sort_values("date").reset_index(drop=True)


def read_nldas_forcing(gauge_id: str, cfg: dict) -> Optional[pd.DataFrame]:
    root = os.path.join(cfg["root"], cfg["forcing_subdir"])
    pattern = os.path.join(root, "**", f"{gauge_id}_*nldas*.txt")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return None
    raw = pd.read_csv(files[0], sep=r"\s+", skiprows=3, engine="python")
    raw.columns = [str(c).strip() for c in raw.columns]
    return _normalize_forcing_columns(raw)


def _resolve_area_km2(row: pd.Series) -> float:
    for name in ("area_gages2", "area_geospa_fabric", "area", "basin_area"):
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    raise KeyError("Missing basin area")


def read_usgs_streamflow_mm_day(
    gauge_id: str, area_km2: float, cfg: dict
) -> Optional[pd.DataFrame]:
    root = os.path.join(cfg["root"], cfg["streamflow_subdir"])
    pattern = os.path.join(root, "**", f"{gauge_id}_*streamflow*.txt")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return None
    raw = pd.read_csv(
        files[0], sep=r"\s+", header=None, names=STREAMFLOW_COLS, engine="python"
    )
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.to_numeric(raw["Year"], errors="coerce").astype("Int64"),
                "month": pd.to_numeric(raw["Month"], errors="coerce").astype("Int64"),
                "day": pd.to_numeric(raw["Day"], errors="coerce").astype("Int64"),
            }
        )
    )
    q = pd.to_numeric(raw["QObs"], errors="coerce").astype(np.float32)
    bad = (q <= 0) | (np.abs(q - cfg["streamflow_missing_token"]) < 1e-3)
    q = q.mask(bad, np.nan)
    area_m2 = area_km2 * 1e6
    q_mm_day = (q.astype(np.float64) * CFS_TO_M3S * 86400.0 / area_m2) * 1000.0
    df["Q_mm_day"] = q_mm_day.astype(np.float32)
    return df.sort_values("date").reset_index(drop=True)


def assemble_snow_cluster_timeseries(
    selected_gauge_ids: Sequence[str], merged_attrs: pd.DataFrame, cfg: dict
) -> Tuple[pd.DataFrame, List[str]]:
    attr_index = merged_attrs.set_index("gauge_id")
    rows = []
    lb = cfg["look_back"]
    dyn_cols: Optional[List[str]] = None
    for gid in selected_gauge_ids:
        gid = str(gid)
        if gid not in attr_index.index:
            continue
        area = _resolve_area_km2(attr_index.loc[gid])
        forcing = read_nldas_forcing(gid, cfg)
        flow = read_usgs_streamflow_mm_day(gid, area, cfg)
        if forcing is None or flow is None:
            continue
        m = forcing.merge(flow, on="date", how="inner")
        if len(m) < lb + 32:
            continue
        m = compute_physics_swe_and_season(m, cfg["ddf"])
        m["gauge_id"] = gid
        if dyn_cols is None:
            base_dyn = [
                "prcp_mm",
                "tmax_c",
                "tmin_c",
                "vp_pa",
                "srad_wm2",
                "T_mean_c",
                "SWE_mm",
                "sin_doy",
                "cos_doy",
                "prcp_7d",
                "prcp_30d",
            ]
            dyn_cols = [c for c in base_dyn if c in m.columns]
        rows.append(m)
    if not rows:
        raise RuntimeError("No basin time series available")
    full = pd.concat(rows, ignore_index=True)
    assert dyn_cols is not None
    return full, dyn_cols


def slice_by_split(
    full_df: pd.DataFrame,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    split_map: Dict[str, Tuple[int, int, int]],
    mode: str,
) -> pd.DataFrame:
    parts = []
    if mode == "target_all":
        for gid in target_ids:
            parts.append(full_df[full_df["gauge_id"] == gid].sort_values("date"))
        return pd.concat(parts, ignore_index=True) if parts else full_df.iloc[0:0]
    for gid in source_ids:
        g = full_df[full_df["gauge_id"] == gid].sort_values("date")
        t_end, v_end, _ = split_map[gid]
        if mode == "train":
            parts.append(g.iloc[:t_end])
        elif mode == "val":
            parts.append(g.iloc[t_end:v_end])
        elif mode == "local_test":
            parts.append(g.iloc[v_end:])
        else:
            raise ValueError(f"Invalid split: {mode}")
    return pd.concat(parts, ignore_index=True) if parts else full_df.iloc[0:0]


def compute_norm_stats_source_train(
    train_df: pd.DataFrame, dyn_cols: Sequence[str]
) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for c in list(dyn_cols) + ["Q_mm_day"]:
        s = train_df[c].astype(np.float64)
        stats[c] = (float(s.mean()), float(s.std()) + 1e-8)
    return stats


def apply_norm(
    df: pd.DataFrame, dyn_cols: Sequence[str], stats: Dict[str, Tuple[float, float]]
):
    out = df.copy()
    for c in list(dyn_cols) + ["Q_mm_day"]:
        mu, sig = stats[c]
        out[c] = (out[c].astype(np.float64) - mu) / sig
    return out


def fit_apply_static_scaler(
    static_table: pd.DataFrame,
    static_num_cols: Sequence[str],
    source_train_gauges: Sequence[str],
    all_gauges: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    sub_tr = static_table[static_table["gauge_id"].isin(source_train_gauges)][
        ["gauge_id"] + list(static_num_cols)
    ]
    X_tr = np.nan_to_num(sub_tr[static_num_cols].values.astype(np.float64))
    scaler = StandardScaler().fit(X_tr)
    sub_all = static_table[static_table["gauge_id"].isin(all_gauges)][
        ["gauge_id"] + list(static_num_cols)
    ]
    X_all = np.nan_to_num(sub_all[static_num_cols].values.astype(np.float64))
    Z = scaler.transform(X_all).astype(np.float32)
    stat_dict = {str(gid): Z[i] for i, gid in enumerate(sub_all["gauge_id"].tolist())}
    return stat_dict, scaler


class RegionalFusionWindowDataset(Dataset):


    def __init__(
        self,
        df_norm: pd.DataFrame,
        dyn_cols: Sequence[str],
        stat_dict: Dict[str, np.ndarray],
        look_back: int,
    ):
        self.dyn_cols = list(dyn_cols)
        self.stat_dict = stat_dict
        self.look_back = look_back
        self.samples: List[Tuple[np.ndarray, np.ndarray, float, str, int, int]] = []
        self.index_by_gauge: DefaultDict[str, List[int]] = defaultdict(list)

        for gid, grp in df_norm.groupby("gauge_id"):
            gid = str(gid)
            if gid not in self.stat_dict:
                continue
            grp = grp.sort_values("date").reset_index(drop=True)
            d = grp[self.dyn_cols].values.astype(np.float32)
            q = grp["Q_mm_day"].values.astype(np.float32)
            s = self.stat_dict[gid].astype(np.float32)
            dates = pd.to_datetime(grp["date"], errors="coerce")
            for i in range(look_back, len(grp)):
                if np.isnan(q[i]):
                    continue
                dt = dates.iloc[i]
                if pd.isna(dt):
                    yr, doy = 2000, 1
                else:
                    yr = int(dt.year)
                    doy = int(dt.dayofyear)
                idx = len(self.samples)
                self.samples.append((d[i - look_back : i], s, float(q[i]), gid, yr, doy))
                self.index_by_gauge[gid].append(idx)

        self.source_gauge_ids = sorted(self.index_by_gauge.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        d, s, t, gid, yr, doy = self.samples[idx]
        return (
            torch.from_numpy(d),
            torch.from_numpy(s),
            torch.tensor(t, dtype=torch.float32),
            gid,
            yr,
            doy,
        )

    def get_indices_by_gauge(self) -> Dict[str, List[int]]:
        return {g: list(idxs) for g, idxs in self.index_by_gauge.items()}


def collate_fn_verify(batch):
    ds, ss, ts, gids, yrs, doys = zip(*batch)
    return (
        torch.stack(ds, 0),
        torch.stack(ss, 0),
        torch.stack(ts, 0),
        list(gids),
        np.array(yrs, dtype=np.int32),
        np.array(doys, dtype=np.int32),
    )


class RegionalMixBatchSampler:


    def __init__(
        self,
        indices_by_gauge: Dict[str, List[int]],
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ):
        self.indices_by_gauge = {str(k): list(v) for k, v in indices_by_gauge.items()}
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_interleaved_order(self) -> List[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        queues = {
            g: deque(rng.permutation(idxs).tolist())
            for g, idxs in self.indices_by_gauge.items()
            if len(idxs) > 0
        }
        if not queues:
            return []
        gids = rng.permutation(list(queues.keys())).tolist()
        order: List[int] = []
        while any(len(q) > 0 for q in queues.values()):
            for g in gids:
                if len(queues[g]) > 0:
                    order.append(queues[g].popleft())
        return order

    def __iter__(self) -> Iterator[List[int]]:
        order = self._build_interleaved_order()
        bs = self.batch_size
        for i in range(0, len(order), bs):
            chunk = order[i : i + bs]
            if len(chunk) < bs and self.drop_last:
                break
            yield chunk

    def __len__(self) -> int:
        n = sum(len(v) for v in self.indices_by_gauge.values())
        bs = self.batch_size
        if self.drop_last:
            return n // bs
        return (n + bs - 1) // bs
