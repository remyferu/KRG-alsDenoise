# Workflow development

The maintained notebook is `notebooks/Noise2Inverse360_HPC_v3.5.ipynb`.
Its action cells call `krg-ct-workflow`, a separate Python 3.8 package installed
in the ALS environment. The `denoise` backend runs PyTorch in n2i.

## Installation and updates

Follow [NERSC setup](SETUP_NERSC.md) for a new installation. In an existing
ALS environment with scientific dependencies installed:

```bash
python -m pip install --no-deps -e "$HOME/microct-n2i-source/KRG-alsDenoise/workflow"
krg-workflow doctor
```

The n2i backend does not require this package. The checkout-local configuration
example reads shared presets directly from `workflow/src`.

Stop jobs that use the checkout before updating it, then restart the notebook
kernel. Cell 1 reports the installed package version and source path. Earlier
notebooks retain their original code; v3.5 is maintained directly.

## Modules

| Module in `workflow/src/krg_workflow` | Responsibility |
|---|---|
| `context.py` | Run state, snapshots and stale-control checks |
| `api.py` | Notebook actions and workspace/training state |
| `presets.py` | Defaults for new reconstruction, training and inference runs |
| `runtime.py` | Allocation checks, thread settings and subprocess logging |
| `files.py` | Locks, atomic writes and path containment |
| `scans.py` | HDF5 metadata, fingerprints and projection views |
| `plans.py` | Frozen reconstruction identity and schema validation |
| `cor.py` | Vo centering, alignment and manual selection |
| `reconstruction.py` | ALS adapter, row chunks and stage validation |
| `queue.py` | Queue execution and repair previews |
| `denoising.py` | Configuration, checkpoint and output validation |
| `history.py` | Log parsing, loss-history exports and concurrent log snapshots |
| `cleanup.py` | Cleanup manifests and deletion |
| `notebook_ui.py` | Widgets, linked zoom and live log callbacks |

Configure NumExpr before importing scientific modules. Package initialization
and runtime helpers contain no scientific imports. Command execution accepts
an `on_line` callback; the notebook supplies `live_log_box()`. Complete logs
retain ANSI escapes, while the displayed tail strips them.

Display dependencies belong in UI code. Numeric and path settings work without
widgets, although the ALS backend itself may require ipywidgets.

`RunContext` stores the selected scan and reconstruction settings. On selection
changes, `inspect_run(..., previous=run)` invalidates the previous context.
Viewers check their captured snapshot before reading or updating data.
Rerun a configuration cell after editing its settings. Frozen source, backend
and settings identity cannot be changed under the same run label.

Backend configuration, checkpoint loading and dataset/loader construction are
in `denoise/configuration.py`, `checkpoints.py` and `training_setup.py`.
`denoise/validation.py` records disjoint axial partitions and fixed validation
crops; normalization uses training planes only. Evaluation shards do not pad
or repeat sample indices. `denoise/evaluation.py` forwards the plain module
rather than its DDP wrapper, so uneven and empty evaluation shards avoid DDP
forward collectives. Sample-weighted sums and counts are reduced across ranks.

`denoise/training_controls.py` manages plateau LR reduction, a refinement
interval and FP32 EMA weights. `denoise/early_stopping.py` monitors the raw
model's post-warmup total-loss mean. Training and evaluation share
`paired_objective()` from `denoise/loss.py`; reporting metrics do not control
stopping. Scheduler, monitor, EMA, validation plan and warmup state are stored
in version-2 resume checkpoints. LR changes do not reset loss patience.

## Compatibility

- Schema-4 reconstruction plans from v3.0–v3.5 retain their geometry, output names and identity.
- Queue schema 1, entry IDs, stage signatures, completion files and lock names are unchanged.
- v3.5 NAFNet and resume checkpoints are incompatible with previous releases.
  Start a new training label; existing reconstruction stacks need no conversion.
- Existing metric logs remain readable by history plotting and the snooper.
- Cleanup previews include the generated `auto_cor_vo` directory.

## Terminal commands

Run in the ALS environment:

```bash
krg-workflow doctor
krg-workflow history /absolute/path/logs/train.log --moving-window 10
krg-workflow inspect-queue /absolute/path/reconstruction_queue_v3.json
# Requires a compute allocation.
krg-workflow run-queue /absolute/path/reconstruction_queue_v3.json
```

Only `run-queue` performs reconstruction. These commands do not start training,
delete outputs or apply queue repairs.

## Training snooper

Open [Training_Snooper.ipynb](../notebooks/Training_Snooper.ipynb) in a separate
ALS kernel from the training notebook. Set `TRAIN_LOG` to the path printed by
the training launcher, then run the snapshot cell. Rerun that cell to refresh;
the optional last cell shows the captured log tail. PyTorch, checkpoints and
GPU access are not required.

`snapshot_training_history()` copies complete lines through the log size
observed at open, then calls `plot_training_history()` on that private copy.
It supports a growing log without acquiring training locks or modifying the
source. Each refresh writes PNG, SVG, CSV and `snapshot.json` under a unique
timestamp/UUID directory in `logs/training_snooper/` by default. `SNAPSHOT_ROOT`
selects an alternate destination. The training notebook's own exports are
unchanged. Missing logs or incomplete L1/LCL summaries produce a status message.

Snapshot metadata records capture time, observed log size and the last parsed
epoch. Total loss and the logged stopping curve appear when available. Plot
smoothing remains independent of the training early-stopping policy.
v3.5 logs add EMA evaluation total and learning rate. The snooper displays
the same optional EMA overlay and LR panel as notebook history exports.

## Notebook maintenance and checks

Keep implementations in helper modules and action calls in the notebook.
Clear outputs before release and retain sequential code-cell titles. The setup
section is copied from `docs/SETUP_NERSC.md`:

```bash
python scripts/check_notebook.py --refresh-setup
python scripts/check_notebook.py
PYTHONPATH=workflow/src MPLBACKEND=Agg PYTHONNOUSERSITE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  python -m unittest discover -s tests -v
```

The full suite requires backend dependencies. For workflow checks alone, use
`workflow/tests` as the discovery directory. Tests cover temporary scans and
TIFFs, log parsing, mocked subprocess/ALS calls and inference. Training tests
are disabled by default. Python 3.8 compatibility is checked statically.
Policy checks use scalar loss sequences, serialized state and manual EMA
weight snapshots. A three-process CPU/Gloo inference check includes an empty
evaluation shard. Local checks do not train a model or verify NERSC/NCCL
throughput, a fresh NERSC installation or ALS reconstruction.

Historical documentation and environment exports are under `docs/legacy/`
and `envs/legacy/`. See [license notices](../NOTICE.md) for attribution.
