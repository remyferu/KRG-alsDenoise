## First-time NERSC setup

Prerequisites: a NERSC account, access to the raw scan and clone access to
`remyferu/KRG-alsDenoise`. The steps below cover a new installation.

Terminal commands use Bash on NERSC. Notebook cells run in a Python kernel,
which must be selected separately from the Jupyter compute allocation.

### 1. Environment roles

| Environment | Tasks | Main software | Selection |
|---|---|---|---|
| `als832` | Notebook cells, projection processing, COR and reconstruction | ALS `microct` source, TomoPy, ASTRA, dxchange, NumPy, widgets | Select **ALS CT + N2I notebook** as the Jupyter kernel |
| `n2i` | Denoising training and inference | This repository, PyTorch with CUDA | Notebook launches its absolute `denoise` executable as a subprocess |

Create both environments once. Keep the notebook in the ALS kernel during the
whole workflow. Training runs in an n2i subprocess.
The source checkouts contain code, while the Conda prefixes contain Python and
installed packages. Cloning a repository alone does not install its dependencies.

Reconstruction uses ALS/TomoPy/ASTRA. The training configuration action runs
`denoise prepare` to create a YAML file. Its printed Tomocupy examples belong
to the upstream workflow; cells 9–11 produce and check the paired stacks
through ALS instead.

An existing ALS-provided `microCT` environment can serve as
the kernel after verifying the imports below. The
[ALS project](https://github.com/als-computing/microct#readme) describes its
beamline-specific service. The private installation below is an alternative for
standard Perlmutter Jupyter sessions. Their filesystem mounts may differ.

### 2. Create directories and download the source (login-node terminal)

Log in to NERSC through SSH, or use a login-node Jupyter terminal for lightweight
installation work. Use a compute allocation for expensive compilation or data
processing. These paths match the notebook's defaults. The Conda module follows
the [NERSC custom-environment procedure](https://docs.nersc.gov/development/languages/python/nersc-python/#option-2-custom-conda-environment):

```bash
module load conda
CT_SOURCE_DIR="$HOME/microct-n2i-source"
CT_SOFTWARE_DIR="$HOME/microct-n2i-envs"
mkdir -p "$CT_SOURCE_DIR" "$CT_SOFTWARE_DIR"
cd "$CT_SOURCE_DIR"
git clone https://github.com/als-computing/microct.git microct
git -C "$CT_SOURCE_DIR/microct" checkout e4116984003294af8dabba9d32db3fccb5ef2253
git clone https://github.com/remyferu/KRG-alsDenoise.git KRG-alsDenoise
```

The private KRG repository requires GitHub HTTPS authentication or
a configured SSH key with `git@github.com:remyferu/KRG-alsDenoise.git` as the
clone URL. If a directory already exists, inspect it and reuse the checkout;
do not rerun `git clone` into it or replace it. The ALS commit above is the pinned backend for this notebook.

### 3. Install the ALS reconstruction environment (login-node terminal)

For a fresh private installation, create `als832` from the pinned ALS source's
environment file, then add the notebook's supporting packages:

```bash
module load conda
CT_SOURCE_DIR="$HOME/microct-n2i-source"
CT_SOFTWARE_DIR="$HOME/microct-n2i-envs"
conda env create --prefix "$CT_SOFTWARE_DIR/als832" \
  -f "$CT_SOURCE_DIR/microct/environment.yml"
conda activate "$CT_SOFTWARE_DIR/als832"
conda install -c conda-forge h5py tifffile pyyaml pillow numexpr ipywidgets ipykernel
python -c 'import tomopy, dxchange, astra, h5py, tifffile, yaml, PIL, ipywidgets; print("ALS imports OK")'
python -m pip install --no-deps -e "$CT_SOURCE_DIR/KRG-alsDenoise/workflow"
krg-workflow doctor
python -m ipykernel install --user --name als832-n2i \
  --display-name "ALS CT + N2I notebook"
```

The [pinned ALS environment](https://github.com/als-computing/microct/blob/e4116984003294af8dabba9d32db3fccb5ef2253/environment.yml)
uses Python 3.8 and older Matplotlib/Jupyter pins. This is a source-based setup
recipe rather than a solver-verified NERSC lock file. If Conda cannot solve it, keep an
existing working ALS environment and resolve the solver error before proceeding;
do not install TomoPy into n2i as a substitute. `ipywidgets` is needed by the ALS
backend itself. Zoom displays do not require an interactive Matplotlib backend.

In JupyterLab choose **Kernel → Change Kernel → ALS CT + N2I notebook**. Reload
JupyterLab if the kernel is missing. Changes to an already running kernel's
environment require a kernel restart. The first code cell prints `sys.executable`;
for this installation it should end in `microct-n2i-envs/als832/bin/python`.
See the [NERSC custom-kernel procedure](https://docs.nersc.gov/services/jupyter/how-to-guides/#how-to-use-a-conda-environment-as-a-python-kernel).

The `krg-ct-workflow` package lives in `workflow/` and requires no Torch. Its
scientific dependencies are supplied by the ALS environment, so use `--no-deps`
to preserve those compatible versions. If pip/setuptools is too old for editable
installation, install `setuptools>=61,<76` and `pip<25.1` inside this Python 3.8
environment, then retry. Restart the notebook kernel after installing/updating it.

### 4. Install the denoising environment (login-node terminal)

For a new n2i prefix, install a CUDA-enabled PyTorch build and this package with
its dependencies. The example uses an official PyTorch 2.7.1 / CUDA 11.8 wheel;
the notebook requires PyTorch >= 2.0. Use a site-compatible build when necessary.
[PyTorch wheel choices](https://pytorch.org/get-started/previous-versions/#v271)
and [NERSC guidance](https://docs.nersc.gov/machinelearning/pytorch/) explain the
custom-environment option.

```bash
module load conda
CT_SOURCE_DIR="$HOME/microct-n2i-source"
CT_SOFTWARE_DIR="$HOME/microct-n2i-envs"
conda create --prefix "$CT_SOFTWARE_DIR/n2i" python=3.11 pip
conda activate "$CT_SOFTWARE_DIR/n2i"
python -m pip install 'numpy<2.0'
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e "$CT_SOURCE_DIR/KRG-alsDenoise"
python -m pip check
MPLBACKEND=Agg PYTHONNOUSERSITE=1 "$CT_SOFTWARE_DIR/n2i/bin/denoise" train -h
```

CLI help must include `--architecture` and `nafnet25d`. The adaptation is
included in this repository and has no BasicSR dependency. For an existing n2i prefix,
skip `conda create` and the dependency installations; activate that prefix and run
`python -m pip install --no-deps -e "$HOME/microct-n2i-source/KRG-alsDenoise"`,
then check imports and CLI help. `--no-deps` is only for an environment with the
dependencies already installed. The repository's original
`envs/legacy/denoise_environment.yml` is a machine-specific upstream export, not the
portable first-time recipe for this notebook.

### 5. Start Jupyter on a compute allocation

Open [NERSC JupyterHub](https://jupyter.nersc.gov/), start a **single-node GPU**
compute server and open
`microct-n2i-source/KRG-alsDenoise/notebooks/Noise2Inverse360_HPC_v3.4.ipynb`.
For the full-volume example, choose **Exclusive GPU Node**, or a Configurable job
with sufficient host RAM and time. NERSC documents four GPUs and full node memory
for the exclusive GPU configuration; a shared GPU server has one GPU and a
quarter of the node's host resources. [Allocation options](https://docs.nersc.gov/services/jupyter/reference/).

Select **ALS CT + N2I notebook** as the kernel. Activating Conda or starting an
allocation in another terminal does not relocate or change the running kernel.
The workflow uses one node: training uses all assigned GPUs, inference uses one.
In a terminal opened **inside this compute Jupyter session**, check:

```bash
hostname
echo "Slurm job: ${SLURM_JOB_ID:-unset}"
echo "GPU assignment: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
"$HOME/microct-n2i-envs/n2i/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print("Visible GPUs:", torch.cuda.device_count())'
```

This checks CUDA availability; it does not train a model. Keep the scheduler's
`CUDA_VISIBLE_DEVICES` assignment. If it is unset, resolve the allocation/kernel
setup before training instead of guessing identifiers from `nvidia-smi`. If either
environment needs site CUDA modules, apply them through a kernel/helper wrapper
following [NERSC's helper instructions](https://docs.nersc.gov/services/jupyter/how-to-guides/#how-to-customize-a-kernel-with-a-helper-shell-script).
CUDA modules must also be available to the subprocess environment.

### 6. Stage the scan and edit cell 2

The notebook does not download or copy scan data. Place the raw HDF5 scan in a
directory visible from the compute kernel. On standard Perlmutter, an example is
`$PSCRATCH/raw_microct/sample.h5`. Set the source filename and verify that
the file meets the projection/flat/dark/metadata requirements in the notebook’s Method and input requirements section.
Reconstructed TIFFs alone are insufficient for the notebook's projection splitting.

| Cell 2 input | Value |
|---|---|
| `INPUT_ROOT` | Folder containing raw `.h5`/`.hdf5` files; replace the example folder or use an explicit absolute `Path` |
| `SCAN_PATH` | Absolute raw scan path; confirm it in cell 3 |
| `RUN_LABEL` | A name for this scan's COR, settings and model combination; use a new label when those change |
| `ALS_MICROCT_DIR` | Checkout containing `backend/ALS_recon_helper.py`, normally `Path.home() / 'microct-n2i-source' / 'microct'` |
| `DENOISE_CMD` | `[str(Path.home() / 'microct-n2i-envs' / 'n2i' / 'bin' / 'denoise')]` |
| `OUTPUT_ROOT_OVERRIDE` | Leave `None` to use `$PSCRATCH`/`$SCRATCH`, or enter an absolute output scratch directory |

In Python, `Path.home()` expands the account home directory; `Path('$HOME/...')` does not
expand Bash variables. `/alsdata` and `/alsuser/pscratch` are service-specific
paths, not assumed mounts on standard Perlmutter. Stage to a stable input path:
moving the raw file later breaks the saved plan and virtual projection views.

### 7. Directory layout

Keep code, environments and notebooks in home (subject to quota); use scratch for
large input copies, TIFFs and active model outputs. Standard output layout:

```text
$HOME/microct-n2i-source/microct/backend/ALS_recon_helper.py
$HOME/microct-n2i-source/KRG-alsDenoise/notebooks/Noise2Inverse360_HPC_v3.4.ipynb
$HOME/microct-n2i-envs/als832/         # notebook Python
$HOME/microct-n2i-envs/n2i/           # denoising Python
$PSCRATCH/noise2inverse360/<scan>/<RUN_LABEL>/
  run_plan.json, scan_metadata.json
  projection_views/even.h5, odd.h5
  <scan>_rec/, <scan>_rec_0/, <scan>_rec_1/
  <scan>_rec_config.yaml
  TrainOutput/results/reference_metrics.csv
  TrainOutput/{resume,best_val_model,best_lcl_model,best_edge_model}.pth
  <scan>_denoised_slices_nafnet25d/
  <scan>_denoised_volume_2.5d_nafnet25d/
  logs/, training_history/
$PSCRATCH/noise2inverse360/reconstruction_queue_v3.json
```

The selected output root can differ from this example. Full/even/odd float32
stacks for 624 × 2560 × 2560 occupy about 46 GiB on disk. Training with four
processes holds about 122 GiB of paired input arrays in host RAM before overhead.
GPU VRAM is separate from host RAM. Read the volume section for its additional
memory needs. Scratch is working storage; preserve accepted data and small run
records in durable storage before cleanup.

### 8. Update an existing installation

After stopping jobs that use this checkout, update it in a terminal:

```bash
cd "$HOME/microct-n2i-source/KRG-alsDenoise"
git status --short
git pull --ff-only
```

For the first upgrade from v3.3, also activate the ALS environment and install
`python -m pip install --no-deps -e "$HOME/microct-n2i-source/KRG-alsDenoise/workflow"`.
Open the new v3.4 notebook; the v3.3 notebook remains available.

Preserve local notebook edits before pulling. Editable installation reads code
from this checkout, so routine code updates need no package reinstall; changed
dependencies or entry points do. Reopen the updated notebook, restart its kernel and rerun initialization.
Changing ALS backend files invalidates saved source identity, so keep the
pinned checkout unchanged during active runs.

### Setup troubleshooting

| Symptom | Next step |
|---|---|
| `ModuleNotFoundError: tomopy`, `astra` or `dxchange` in a cell | Check the printed kernel Python; select the ALS kernel and verify ALS imports |
| `ModuleNotFoundError: torch` from `denoise` | Check `DENOISE_CMD` and the n2i environment; PyTorch belongs there |
| CLI help lacks `--architecture` | Update/install KRG-alsDenoise in n2i; an older Noise2Inverse executable is being used |
| Invalid `matplotlib_inline` backend in the CLI | Confirm the subprocess uses `MPLBACKEND=Agg`; the notebook's launcher sets it |
| NumExpr thread-count error | Restart the kernel and run cell 1 first. If NumExpr is preloaded, set `NUMEXPR_MAX_THREADS` in the kernel launcher to at least the host's detected logical CPU count before Python starts |
| Kernel remains missing | Reload JupyterLab after registration; check the kernelspec points to an existing ALS Python |
| No accessible scratch or scan folder | Set cell 2's input and scratch paths to locations visible on the actual compute node |

If NumExpr loads before cell 1, inspect the host CPU count with
`getconf _NPROCESSORS_ONLN` in the compute-session terminal. In JupyterLab's text
editor open `$HOME/.local/share/jupyter/kernels/als832-n2i/kernel.json`. Add an
`env` field at the same level as `argv`, using this object as its value. Keep
the existing interpreter path:

```json
{
  "NUMEXPR_MAX_THREADS": "256",
  "NUMEXPR_NUM_THREADS": "8"
}
```

`256` is an example: use a maximum at least as large as that host's detected
logical CPU count. Keep valid JSON commas between the existing fields, then
restart the kernel. Cell 1 reduces the initial active count to the available CPU
limits. [NERSC kernel environment instructions](https://docs.nersc.gov/services/jupyter/how-to-guides/#how-to-set-environment-variables-for-a-python-kernel).

Initialize the notebook with cells 1–4, then follow Workflow and restart.
Training requires `RUN_TRAINING=True` in cell 13. A fresh NERSC installation
and reconstruction have not been verified for this setup. See
[denoising details](NAFNET_CT.md) for architecture and loss settings.
