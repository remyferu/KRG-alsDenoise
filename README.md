# KRG-alsDenoise
Tools to run on NERSC for denoising ALS 8.3.2 micro-ct reconstructions

This repository extends [Noise2Inverse360](https://github.com/tomography/Noise2Inverse360)
with an optional **NAFNet-based 2.5D CT model**, selected with
`train.architecture: nafnet25d` in the scan YAML. The original 2.5D and 3D U-Nets
remain available. NAFNet requires a fresh training run; existing U-Net weights are
incompatible.

Training retains the paired L1/LCL objective and also logs **L2/MSE** and
**SSIM loss (1 - SSIM)** for reference. The new metrics are saved in CSV and resume
history and shown by the v3.4 history plots; checkpoint criteria are unchanged.

The model uses learned neighboring-slice gates, NAFBlocks, a full-resolution
detail branch, and a residual prediction of the center slice. It has **989,745
parameters** at the supplied settings and supports 5, 7, or 9 slices. This is an
experimental adaptation: preservation of 2Ã¢â‚¬â€œ5 pixel features needs validation on
your scans.

Start with [First-time NERSC setup](docs/SETUP_NERSC.md), then the
[denoising usage guide](docs/NAFNET_CT.md), or open the
[v3.4 HPC notebook](notebooks/Noise2Inverse360_HPC_v3.4.ipynb).
Reconstructed full/even/odd TIFF stacks from the existing workflow can be reused.

```bash
# After cloning this repository and activating your existing n2i environment:
python -m pip install --no-deps -e .
export MPLBACKEND=Agg
python examples/configure_nafnet.py --input /path/scan_config.yaml --output /path/scan_nafnet.yaml

# Run yourself when ready, using a new training output directory:
denoise train --config /path/scan_nafnet.yaml --gpus 0,1,2,3 --no-search --output-dir /path/TrainOutput_NAF
denoise volume --config /path/scan_nafnet.yaml --model-dir /path/TrainOutput_NAF --checkpoint val
```

The example starts with five slices, 256-pixel patches, batch 4 per GPU, and BF16.
The supplied learning rate and patience are starting values, not tuned optima.
Each rank still loads both TIFF stacks into host RAM; the smaller network does
not eliminate that memory requirement.

Code and inference checks (training tests are skipped by default):

```bash
MPLBACKEND=Agg NO_ALBUMENTATIONS_UPDATE=1 python -m unittest discover -s tests -v
```

See [source attribution and licenses](NOTICE.md). Original upstream documentation
is kept in [the legacy upstream guide](docs/legacy/UPSTREAM_NOISE2INVERSE.md).

The v3.4 notebook contains short action cells backed by a separate Python 3.8
workflow package. Install `workflow/` into your working ALS environment with
`python -m pip install --no-deps -e workflow`; keep the n2i backend installation
separate. See the [workflow module map and terminal helpers](docs/WORKFLOW.md).

For checkout tests, also expose the workflow package: `PYTHONPATH=workflow/src`.
The previous v3.3 notebook is preserved for reference.
