# Workflow development and repository map

The supported entry point is `notebooks/Noise2Inverse360_HPC_v3.4.ipynb`.
It runs short actions from an independently installed, Python 3.8-compatible
`krg-ct-workflow` distribution. The existing `denoise` distribution runs PyTorch
in the n2i environment. Importing workflow helpers never imports Torch.

## Installation and updating

Follow [first-time NERSC setup](SETUP_NERSC.md) for both environments. To add the
workflow to an existing working ALS environment, activate that environment and run:

```bash
python -m pip install --no-deps -e "$HOME/microct-n2i-source/KRG-alsDenoise/workflow"
krg-workflow doctor
```

The ALS scientific dependencies must already be present for `--no-deps`.
The n2i backend does not require this distribution. The checkout-local config
example reads its presets from the same source directory without loading ALS.
Stop active jobs before updating the checkout; restart the notebook kernel after
pulling helper changes. Cell 1 reports the installed package version and source.
Preserve v3.3 and older notebooks as historical references; maintain v3.4 directly.

## Module boundaries

| Module in `workflow/src/krg_workflow` | Responsibility |
|---|---|
| `context.py` | Explicit `RunContext`, snapshots, stale-control checks |
| `api.py` | Short public actions and explicit workspace/training state |
| `presets.py` | Fresh-run reconstruction, training and inference defaults |
| `runtime.py` | Allocation/CPU/GPU checks and subprocess lifetime/logging |
| `files.py` | Locks, atomic files and path containment |
| `scans.py` | HDF5 metadata/fingerprints and independent projection views |
| `plans.py` | Frozen reconstruction identity and schema validation |
| `cor.py` | Native Vo centering, alignment and manual selection |
| `reconstruction.py` | ALS adapter, row chunks and stage validation/resume |
| `queue.py` | Queue execution and previewed repair |
| `denoising.py` | Training YAML and inference output validation |
| `history.py` | Old/new log parsing and loss-history exports |
| `cleanup.py` | Reviewed cleanup manifests and constrained deletion |
| `notebook_ui.py` | Optional widgets, linked zoom and live log callbacks |

Configure NumExpr before importing the scientific workflow modules. Keep runtime
and package initialization free of scientific imports. Core command execution
accepts an optional `on_line` callback; the notebook supplies `live_log_box()`.
Raw complete logs retain ANSI escapes; the live box strips them. Import optional
display dependencies only in UI code. Essential numeric/path settings work without
widgets; the ALS backend itself may still require ipywidgets.

`RunContext` replaces notebook `globals()` lookups. Call `inspect_run(...,
previous=run)` on selection changes, and configure through `configure_reconstruction`.
Old viewers retain their snapshot and check it before reading/updating data.
After modifying a notebook setting, rerun that setting's action cell before
continuing. Frozen source/backend/settings identity remains immutable on disk.

The backend has shared `configuration.py`, `checkpoints.py`, and `training_setup.py`
helpers. The epoch loop, L1/LCL objective, optional MSE/SSIM measurements, optimizer,
warmup, checkpoint selection and early-stopping behavior are unchanged. Further
algorithm/data-loader changes should be separate from repository organization.

## Compatibility

- Schema-4 plans from v3.0–v3.4 retain their geometry, output naming and identity.
- Queue schema 1, stable IDs, stage signatures, completion files and lock names remain unchanged.
- Existing training YAML, normalization, checkpoint payloads and reference metric history remain readable.
- v3.3 is retained unmodified. v3.4 uses new helpers and includes their installation instructions.
- New cleanup previews now include the existing `auto_cor_vo` output directory.

## Terminal helpers

Run these in the ALS environment. History and inspection do not train a model.

```bash
krg-workflow doctor
krg-workflow history /absolute/path/logs/train.log --moving-window 10
krg-workflow inspect-queue /absolute/path/reconstruction_queue_v3.json
# Explicit reconstruction action: requires the allocated compute environment.
krg-workflow run-queue /absolute/path/reconstruction_queue_v3.json
```

There is no automatic training, cleanup or queue repair in these terminal commands.

## Maintaining the notebook and checks

Notebook code calls helpers; do not copy implementations back into cells. Edit
the notebook normally, clear its outputs before release, and keep numbered titles.
The setup section is embedded from `docs/SETUP_NERSC.md`:

```bash
python scripts/check_notebook.py --refresh-setup
python scripts/check_notebook.py
PYTHONPATH=workflow/src MPLBACKEND=Agg PYTHONNOUSERSITE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  python -m unittest discover -s tests -v
```

Use an environment with backend dependencies for the full suite. To check just
the workflow, discover `workflow/tests` instead. Tests use temporary scans/TIFFs,
recorded-style logs, mocked subprocess/ALS calls and inference. Training tests
remain disabled by default; do not enable them during this cleanup workflow.
Python 3.8 syntax is checked statically; local verification does not establish a
fresh NERSC installation or real reconstruction success.

Original Sphinx documentation and the original environment export are under
`docs/legacy/` and `envs/legacy/`. They are historical material, not the current
installation guide. Attribution and original license files remain intact.
