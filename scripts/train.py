import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from cgsm.io import load_config, write_json
from cgsm.model import build_model
from cgsm.pipeline import prepare_training
from cgsm.losses import PeakWeightedHuberLoss
from cgsm.training import (
    set_reproducible,
    train_one_epoch,
    eval_validation_regional,
    compute_peak_threshold,
)


def main():
    parser = argparse.ArgumentParser(description="Train CGSM")
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--output", default="outputs/cgsm")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Prepare data only"
    )
    args = parser.parse_args()
    c = load_config(args.config)
    t = c["training"]
    out = Path(args.output)

    out.mkdir(parents=True, exist_ok=False)
    set_reproducible(t["seed"], fast_cudnn=t["fast_cudnn"])
    loaders, prep = prepare_training(c, args.data_root)
    write_json(
        out / "checkpoint_metadata.json", {"config": c, "preprocessing": prep}
    )
    write_json(
        out / "sample_counts.json", {k: len(v.dataset) for k, v in loaders.items()}
    )
    if args.prepare_only:
        return

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    model = build_model(c, prep).to(device)
    threshold = compute_peak_threshold(
        loaders["train"], c["evaluation"]["peak_quantile"]
    )
    criterion = PeakWeightedHuberLoss(
        threshold, peak_weight=1.0, delta=c["loss"]["huber_beta"]
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )
    warmup = max(1, t["lr_warmup_epochs"])

    def lr_factor(epoch):
        return (
            (epoch + 1) / warmup
            if epoch < warmup
            else 0.5
            * (
                1
                + math.cos(
                    math.pi * (epoch - warmup) / max(t["epochs"] - warmup, 1)
                )
            )
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    amp = t["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if amp else None
    mu, sd = prep["norm_stats"]["Q_mm_day"]
    flow = {"QObs_mm": {"mean": mu, "std": sd}}
    best = -1e9
    bad = 0
    logs = []

    for epoch in range(1, t["epochs"] + 1):
        criterion.set_epoch(epoch, 10**9)
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            opt,
            device,
            c["loss"],
            use_amp=amp,
            scaler=scaler,
        )
        val = eval_validation_regional(
            model, loaders["val"], criterion, device, flow
        )
        scheduler.step()
        metric = val["NSE_macro_mean"]
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "NSE_macro_mean": metric,
                "NSE_pooled": val["NSE"],
                "Peak_NSE_val": val["Peak_NSE"],
                "KGE_val": val["KGE"],
            }
        )
        pd.DataFrame(logs).to_csv(out / "train_log.csv", index=False)
        print(f"Epoch {epoch} | Val macro NSE {metric:.6f}", flush=True)
        if math.isfinite(metric) and metric > best:
            best = metric
            bad = 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad += 1
            if bad >= t["patience"]:
                break

    if not (out / "model.pt").exists():
        raise RuntimeError("No checkpoint saved; see train_log.csv")
    write_json(
        out / "training_summary.json",
        {"best_validation_macro_NSE": best, "epochs_completed": len(logs)},
    )


if __name__ == "__main__":
    main()
