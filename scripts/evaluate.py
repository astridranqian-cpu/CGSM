import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from cgsm.io import load_config, load_checkpoint, write_json
from cgsm.metrics import read_prediction_bundle, summarize
from cgsm.pipeline import prepare_evaluation


def _valid_records(ids, dates, obs, pred):
    ids = np.asarray(ids, dtype=str).ravel()
    dates = np.asarray(dates, dtype=str).ravel()
    obs = np.asarray(obs).ravel()
    pred = np.asarray(pred).ravel()
    if not (len(ids) == len(dates) == len(obs) == len(pred)):
        raise ValueError("Prediction records are misaligned")
    if pd.MultiIndex.from_arrays([ids, dates]).has_duplicates:
        raise ValueError("Duplicate basin/date pairs")
    valid = np.isfinite(obs) & (obs > 0)
    return ids[valid], dates[valid], obs[valid], pred[valid]


def main():
    p = argparse.ArgumentParser(description="Evaluate CGSM")
    p.add_argument("--config")
    p.add_argument("--data-root")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint", help="Model checkpoint")
    inputs.add_argument("--predictions", help="Predictions NPZ")
    p.add_argument("--metadata", help="Checkpoint metadata JSON")
    p.add_argument(
        "--region",
        default="S0",
        choices=["S0", "R1", "R2", "R3", "R4", "source_local_test"],
    )
    p.add_argument("--lnse-mode", choices=["legacy_positive", "zero_clip"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="outputs/evaluation")
    a = p.parse_args()
    if a.metadata and a.predictions:
        p.error("--metadata requires --checkpoint")

    c = load_config(a.config)
    evaluation = dict(c["evaluation"])
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)

    if a.predictions:
        ids, dates, obs, pred = read_prediction_bundle(a.predictions)
    else:
        model, c, prep = load_checkpoint(a.checkpoint, c, a.metadata)
        loader = prepare_evaluation(c, prep, a.data_root, a.region)
        device = torch.device(a.device)
        model = model.to(device).eval()
        ids = []
        years = []
        doys = []
        ps = []
        os = []
        with torch.inference_mode():
            for dyn, stat, y, g, yr, dy in loader:
                ps.append(model(dyn.to(device), stat.to(device)).cpu().numpy())
                os.append(y.numpy())
                ids.extend(g)
                years.extend(yr)
                doys.extend(dy)
        mu, sd = prep["norm_stats"]["Q_mm_day"]
        pred = np.concatenate(ps) * sd + mu
        obs = np.concatenate(os) * sd + mu
        dates = (
            pd.to_datetime(np.asarray(years).astype(str), format="%Y")
            + pd.to_timedelta(np.asarray(doys) - 1, unit="D")
        ).strftime("%Y-%m-%d").to_numpy()
        ids = np.asarray(ids, dtype=str)

    ids, dates, obs, pred = _valid_records(ids, dates, obs, pred)
    mode = a.lnse_mode or evaluation["lnse_mode"]
    pooled, per = summarize(
        ids,
        obs,
        pred,
        lnse_mode=mode,
        epsilon=evaluation["epsilon"],
        peak_quantile=evaluation["peak_quantile"],
    )
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "predictions.npz",
        basin_id=np.asarray(ids, dtype=str),
        date=np.asarray(dates, dtype=str),
        q_obs=obs,
        q_pred=pred,
    )
    per.to_csv(out / "per_basin_metrics.csv", index=False)
    write_json(out / "pooled_metrics.json", pooled)
    write_json(
        out / "evaluation_protocol.json",
        {
            "lnse_mode": mode,
            "epsilon": evaluation["epsilon"],
            "peak_quantile": evaluation["peak_quantile"],
            "region": a.region,
            "source_predictions": a.predictions,
            "checkpoint": a.checkpoint,
            "metadata": a.metadata,
            "q_units": "mm/day",
            "negative_predictions_saved_unchanged": True,
            "observation_rule": "finite positive observed runoff",
        },
    )
    print(
        f"NSE {pooled['NSE']:.4f} | PkNSE {pooled['PkNSE']:.4f} | "
        f"KGE {pooled['KGE']:.4f} | LNSE {pooled['LNSE']:.4f}"
    )


if __name__ == "__main__":
    main()
