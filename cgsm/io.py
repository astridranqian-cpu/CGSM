from pathlib import Path
import json

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    return yaml.safe_load(
        Path(path or PROJECT / "configs/cgsm.yaml").read_text(encoding="utf-8")
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(path, config, metadata_path=None):
    meta_path = (
        Path(metadata_path)
        if metadata_path
        else Path(path).parent / "checkpoint_metadata.json"
    )
    if not meta_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
    meta = read_json(meta_path)
    config = meta.get("config", config)
    prep = meta.get("preprocessing", meta)
    state = torch.load(path, map_location="cpu", weights_only=True)
    from .model import build_model

    model = build_model(config, prep)
    model.load_state_dict(state, strict=True)
    return model, config, prep
