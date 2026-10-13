# Environment specifications

The [NERSC setup guide](../docs/SETUP_NERSC.md) specifies separate Python 3.8
ALS and Python 3.11 denoising environments.

`n2i_environment.yml` supplies a base environment without machine-specific paths.
Run from the repository root, then install a site-compatible CUDA-enabled
PyTorch build and the backend:

```bash
module load conda
conda env create --prefix "$HOME/microct-n2i-envs/n2i" -f envs/n2i_environment.yml
conda activate "$HOME/microct-n2i-envs/n2i"
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
python -m pip check
```

The ALS environment uses the pinned `als-computing/microct/environment.yml`
specified in the setup guide. Install `workflow/` there with `--no-deps`.

These specifications are not solver-verified NERSC lock files. Retain working
environments and record successful installations with `conda list --explicit`
and `python -m pip freeze`. Store those records with run provenance; exclude
credentials and machine-specific paths from repository commits.

`legacy/denoise_environment.yml` is the upstream machine-specific export.
Root `requirements.txt` lists backend dependencies. Documentation tools are
available through the backend's `docs` extra.
