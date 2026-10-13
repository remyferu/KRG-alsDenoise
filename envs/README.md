# Environment specifications

Use [the complete NERSC guide](../docs/SETUP_NERSC.md). It distinguishes the
Python 3.8 ALS reconstruction/kernel environment from the Python 3.11 n2i backend.

`n2i_environment.yml` creates a portable starting environment without machine paths:

```bash
module load conda
conda env create --prefix "$HOME/microct-n2i-envs/n2i" -f envs/n2i_environment.yml
conda activate "$HOME/microct-n2i-envs/n2i"
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
python -m pip check
```

The ALS environment comes from the pinned `als-computing/microct/environment.yml`
in the setup guide. Install this repository's `workflow/` distribution into that
working environment with `--no-deps`; it does not install Torch.

These are installation specifications, not fresh NERSC solver-tested lock files.
Keep working environments and site-compatible CUDA builds. To record a successful
installation for your site, save `conda list --explicit` and `python -m pip freeze`
alongside your durable run provenance. Do not commit user paths or credentials.

`legacy/denoise_environment.yml` is the original upstream machine-specific export.
`requirements.txt` describes backend/runtime tools; documentation tools are available
through the backend's `docs` extra and are not needed for scan processing.
