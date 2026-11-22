# CGSM

CGSM is a process-guided model for runoff prediction in snow-influenced ungauged catchments. It combines lagged-correlation input weighting (RIC), catchment conditioning, FiLM-modulated temporal convolutions, and snow-state refinement. Training uses a runoff-fitting loss and soft physical constraints.

The default setup uses **40 static attributes, 11 dynamic inputs, a 365-day input window, and a fixed split of 77 source basins and 20 held-out basins**. The SWE residual gate has 28 hidden units in `cgsm/model.py`. This project includes the implementation, configuration, basin lists, and a local test checkpoint in `cgsm40/`. Obtain the CAMELS-US data from the sources below for training or full evaluation.

## Installation

Use **Python 3.11** in a virtual environment. Run commands from the repository root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a specific CPU or CUDA build, follow the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for the version pinned in `requirements.txt`.

## Load the local test checkpoint

`cgsm40/model.pt` and `cgsm40/checkpoint_metadata.json` form a matched pair. Keep them together. The current checkpoint is for loading and inference tests; final training and evaluation will produce the checkpoint used for formal results.

Verify strict loading from the project root; this command requires neither CAMELS data nor a GPU:

```bash
python -c "from cgsm.io import load_config, load_checkpoint; m,c,p=load_checkpoint('cgsm40/model.pt',load_config()); print('Loaded:',len(p['static_columns']),'static inputs; gate width:',m.swe_residual_gate[0].out_features)"
```

Expected dimensions are **40 static inputs and a gate width of 28**. Loading uses the implementation in `cgsm/model.py`; `cgsm40/` stores checkpoint files only.

Evaluate the current checkpoint after providing the CAMELS-US files:

```bash
python scripts/evaluate.py --data-root data/CAMELS_FINAL --checkpoint cgsm40/model.pt --region S0 --device cpu --output outputs/S0_local_test
```

Use a new output directory for every evaluation. Full evaluation requires the source data; successful checkpoint loading alone does not establish prediction accuracy.

## Data preparation

### Sources and directory

Use CAMELS-US NLDAS basin-mean meteorological forcing, USGS daily streamflow, and catchment attributes:

- [Hydrometeorological time series (Newman et al.)](https://doi.org/10.5065/D6MW2F4D)
- [Catchment attributes (Addor et al.)](https://doi.org/10.5065/D6G73C3Q)

The default data root is `data/CAMELS_FINAL`. For training, the location is resolved from `--data-root`, the `CAMELS_ROOT` environment variable, then `data.root` in `configs/cgsm.yaml`. Subdirectory names are configured in the same YAML file. Checkpoint evaluation restores the saved data configuration; use `--data-root` or `CAMELS_ROOT` to relocate inputs.

| Data | Path relative to the data root | Format |
|---|---|---|
| Meteorological forcing | `basin_mean_forcing/nldas_extended/<huc>/<gauge_id>_*nldas*.txt` | Whitespace-separated; three metadata lines followed by a header |
| Streamflow | `usgs_streamflow/<huc>/<gauge_id>_*streamflow*.txt` | Whitespace-separated; no header; columns: basin ID, year, month, day, discharge, quality flag |
| Catchment attributes | `camels_us/camels_*.txt` | Semicolon-separated; includes `gauge_id` |

Preserve eight-digit basin IDs, including leading zeros. Forcing files provide `Year`, `Mnth`, `Day`, `prcp(mm/day)`, `tmax(C)`, `tmin(C)`, `vp(Pa)`, and `srad(W/m2)`. Streamflow is supplied in cfs and converted to mm/day using catchment area in km², with `area_gages2` preferred.

The loader matches forcing and streamflow by date. Keep daily date rows when runoff is missing and use the configured missing-value token (`-999`). Missing and nonpositive runoff observations are masked; their date rows remain available for input windows.

### Inputs, splits, and preprocessing

`configs/input_schema.json` defines input names and their order. Mean temperature, reconstructed SWE, seasonal encodings, and 7-day and 30-day precipitation sums are computed internally. SWE is expressed in mm; precipitation and runoff in mm/day.

| File | Purpose |
|---|---|
| `splits/source_basins.txt` | Fixed list of source basins |
| `splits/target_basins.txt` | Fixed list of held-out basins |
| `splits/regions_k5.csv` | Basin-to-region assignments, with `gauge_id` and `region` columns |

Keep one basin ID per line in the text lists. Training uses these lists directly. Each source basin is divided chronologically into 70% training, 15% validation, and 15% testing records, with windows constructed within each partition. Target and external-region evaluation use the available records for their selected basins.

For target row `i`, the input window is `[i - 365, i)`. Windows follow ordered, date-matched records; retain the daily time grid so that 365 records represent 365 days. Samples require a valid positive runoff observation at the target date.

Dynamic inputs and runoff are standardized using source-basin training records. Static missing values are filled with zero before fitting `StandardScaler` on source basins. Evaluation reuses the saved preprocessing statistics.

## Training

The commands below use `data/CAMELS_FINAL`; replace it with your data location if needed. Choose a new output directory for each run.

Prepare the data and save preprocessing metadata:

```bash
python scripts/train.py --data-root data/CAMELS_FINAL --prepare-only --output outputs/preflight
```

Train a new CGSM model. This command starts from fresh initialization; it does not automatically load or fine-tune the checkpoint in `cgsm40/`:

```bash
python scripts/train.py --data-root data/CAMELS_FINAL --output outputs/cgsm
```

Training selects CUDA when available and otherwise uses CPU. Set `--device cpu` or `--device cuda` explicitly when needed. Use `--config path/to/config.yaml` to supply another configuration; the default is `configs/cgsm.yaml`.

Checkpoints are selected using the mean NSE across source validation basins. Training produces:

| File | Contents |
|---|---|
| `model.pt` | Selected model weights |
| `checkpoint_metadata.json` | Configuration, input order, normalization statistics, and RIC weights |
| `sample_counts.json` | Sample counts for each data partition |
| `train_log.csv` | Training and validation metrics by epoch |
| `training_summary.json` | Selected validation score and completed epochs |

Keep `model.pt` and `checkpoint_metadata.json` together for inference. If metadata is stored elsewhere, pass its path with `--metadata` when evaluating.

## Evaluation

After the training command above finishes, evaluate its checkpoint on the fixed held-out basins. For the existing local test checkpoint, use the `cgsm40/model.pt` command shown earlier:

```bash
python scripts/evaluate.py --data-root data/CAMELS_FINAL --checkpoint outputs/cgsm/model.pt --region S0 --output outputs/S0
```

Evaluate an external region:

```bash
python scripts/evaluate.py --data-root data/CAMELS_FINAL --checkpoint outputs/cgsm/model.pt --region R4 --output outputs/R4
```

| Region option | Evaluation set |
|---|---|
| `S0` | The 20 fixed held-out basins |
| `R1`, `R2`, `R3`, `R4` | The corresponding external region |
| `source_local_test` | The temporal test partition of the source basins |

Evaluation uses CPU by default; add `--device cuda` for GPU inference. The loader constructs the model from the saved configuration and preprocessing metadata and checks all checkpoint keys and tensor shapes strictly. The current implementation uses a 28-unit SWE gate. Scoring settings come from the current evaluation configuration.

### Evaluate existing predictions

```bash
python scripts/evaluate.py --predictions outputs/S0/predictions.npz --output outputs/metrics
```

An input NPZ file must contain equally sized arrays named `basin_id`, `date`, `q_obs`, and `q_pred`. Use dates in `YYYY-MM-DD` format and runoff in mm/day. Each basin-date pair must be unique, with finite predictions on valid observation dates.

### Metrics and outputs

Evaluation retains dates with finite positive observed runoff. NSE, PkNSE, KGE, and RMSE use unmodified predictions. PkNSE uses observations at or above the configured peak quantile (90th percentile by default), calculated separately for each basin or for the pooled series.

The default LNSE mode is `zero_clip`, comparing `log(q_obs + 0.01)` with `log(max(q_pred, 0) + 0.01)`. Zero clipping applies only to LNSE; exported predictions retain their original values. The logarithmic offset and peak quantile are specified under `evaluation` in `configs/cgsm.yaml`.

Each evaluation writes `predictions.npz`, `per_basin_metrics.csv`, `pooled_metrics.json`, and `evaluation_protocol.json` to the selected output directory.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/cgsm.yaml` | Data, model, training, and evaluation settings |
| `configs/input_schema.json` | Input names and order |
| `cgsm/model.py` | The model implementation used by both training and evaluation |
| `cgsm40/model.pt`, `cgsm40/checkpoint_metadata.json` | Paired local test weights and preprocessing metadata |
| `cgsm/modules/` | RIC, FiLM, SWE refinement, and TCN components |
| `cgsm/dataset.py`, `cgsm/pipeline.py` | Data loading, preprocessing, and sample construction |
| `cgsm/losses.py`, `cgsm/training.py` | Loss functions, training, and validation |
| `cgsm/metrics.py`, `cgsm/io.py` | Metrics, configuration, and checkpoint handling |
| `scripts/train.py`, `scripts/evaluate.py` | Training and evaluation entry points |
| `splits/` | Fixed basin lists and regional assignments |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignore local data, environments, outputs, caches, and weight files in Git |
