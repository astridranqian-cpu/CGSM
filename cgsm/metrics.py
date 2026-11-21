import numpy as np
import pandas as pd


def nse(obs, pred):
    o = np.asarray(obs, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if len(o) < 2:
        return float("nan")
    denominator = np.sum((o - o.mean()) ** 2)
    return (
        float(1 - np.sum((p - o) ** 2) / denominator)
        if denominator > 1e-12
        else float("nan")
    )


def score(obs, pred, lnse_mode="zero_clip", epsilon=0.01, peak_quantile=0.90):
    o = np.asarray(obs, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    if o.shape != p.shape:
        raise ValueError("Observation/prediction lengths differ")
    if epsilon <= 0 or not 0 < peak_quantile < 1:
        raise ValueError("Invalid metric settings")
    mask = np.isfinite(o) & (o > 0)
    o, p = o[mask], p[mask]
    if not len(o):
        raise ValueError("No valid observations")
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite predictions on valid observation dates")
    positive = (o > 0) & (p > 0)
    legacy = (
        nse(np.log(o[positive] + epsilon), np.log(p[positive] + epsilon))
        if positive.sum() >= 3
        else float("nan")
    )
    clipped = nse(np.log(o + epsilon), np.log(np.maximum(p, 0) + epsilon))
    if lnse_mode not in ("legacy_positive", "zero_clip"):
        raise ValueError(f"Unknown LNSE mode: {lnse_mode}")
    peak = o >= np.quantile(o, peak_quantile)
    r = (
        float(np.corrcoef(o, p)[0, 1])
        if o.std() > 1e-12 and p.std() > 1e-12
        else float("nan")
    )
    alpha = float(p.std() / (o.std() + 1e-12))
    beta = float(p.mean() / (o.mean() + 1e-12))
    return dict(
        NSE=nse(o, p),
        PkNSE=nse(o[peak], p[peak]) if peak.sum() >= 3 else float("nan"),
        KGE=float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)),
        KGE_r=r,
        KGE_alpha=alpha,
        KGE_beta=beta,
        LNSE=legacy if lnse_mode == "legacy_positive" else clipped,
        LNSE_legacy_positive=legacy,
        LNSE_zero_clip=clipped,
        lnse_mode=lnse_mode,
        RMSE=float(np.sqrt(np.mean((p - o) ** 2))),
        n_days=int(len(o)),
        n_negative_predictions=int((p < 0).sum()),
        n_legacy_positive_pairs=int(positive.sum()),
    )


def summarize(basin_ids, obs, pred, **kwargs):
    ids = np.asarray(basin_ids).astype(str)
    o = np.asarray(obs)
    p = np.asarray(pred)
    if len(ids) != len(o):
        raise ValueError("Basin identifiers are misaligned")
    rows = []
    for gid in sorted(set(ids)):
        m = ids == gid
        rows.append({"basin_id": gid, **score(o[m], p[m], **kwargs)})
    return score(o, p, **kwargs), pd.DataFrame(rows)


def read_prediction_bundle(path):
    with np.load(path, allow_pickle=True) as z:
        o = (
            np.asarray(z["tgt_phy"], dtype=np.float64)
            if "tgt_phy" in z
            else np.asarray(z["q_obs"], dtype=np.float64)
        )
        p = (
            np.asarray(z["pred_phy"], dtype=np.float64)
            if "pred_phy" in z
            else np.asarray(z["q_pred"], dtype=np.float64)
        )
        ids = z["gids"].astype(str) if "gids" in z else z["basin_id"].astype(str)
        if "date" in z:
            dates = z["date"].astype(str)
        else:
            dates = (
                pd.to_datetime(z["years"].astype(str), format="%Y")
                + pd.to_timedelta(z["doys"] - 1, unit="D")
            ).strftime("%Y-%m-%d").to_numpy()
    keys = pd.MultiIndex.from_arrays([ids, dates])
    if keys.has_duplicates:
        raise ValueError("Duplicate basin/date pairs")
    return ids, dates, o.ravel(), p.ravel()
