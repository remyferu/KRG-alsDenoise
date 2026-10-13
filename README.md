# KRG-alsDenoise

ALS 8.3.2 micro-CT reconstruction and self-supervised denoising on NERSC.
The repository extends [Noise2Inverse360](https://github.com/tomography/Noise2Inverse360)
with a NAFNet-based 2.5D model and an ALS reconstruction workflow.

The [v3.4 notebook](notebooks/Noise2Inverse360_HPC_v3.4.ipynb) reconstructs
full, even-angle and odd-angle TIFF stacks, estimates the center of rotation
with TomoPy Vo, and manages reconstruction queues, training and inference.
Reusable notebook code is in `workflow/src/krg_workflow`.

## Setup and usage

Follow [NERSC setup](docs/SETUP_NERSC.md) to install the two environments:

- `als832`: notebook kernel, ALS reconstruction and the `workflow/` package.
- `n2i`: PyTorch denoising backend, installed from the repository root.

The [denoising guide](docs/NAFNET_CT.md) covers model configuration, losses,
checkpoints and inference. The [workflow guide](docs/WORKFLOW.md) describes
helper modules and terminal commands.

For an existing n2i environment and reconstructed scan:

```bash
python -m pip install --no-deps -e .
export MPLBACKEND=Agg
python examples/configure_nafnet.py --input /path/scan_config.yaml --output /path/scan_nafnet.yaml
denoise train --config /path/scan_nafnet.yaml --gpus 0,1,2,3 --no-search --output-dir /path/TrainOutput_NAF
denoise volume --config /path/scan_nafnet.yaml --model-dir /path/TrainOutput_NAF --checkpoint val
```

These commands require the backend dependencies and an allocated GPU session.
The notebook keeps training disabled until `RUN_TRAINING=True`.

## Models and losses

Select NAFNet with `train.architecture: nafnet25d`. The original 2.5D and 3D
U-Nets remain available; their checkpoints are incompatible with NAFNet.
The default NAFNet model has 989,745 parameters, learned axial slice fusion,
a full-resolution detail branch and a center-slice residual output. Supported
slice counts are 5, 7 and 9.

Training uses paired L1 and LCL. MSE and SSIM loss are recorded for reference
without affecting optimization or checkpoint selection. History plots include
these metrics when present.

Default settings use five slices, 256-pixel patches, batch 4 per GPU and BF16.
They require evaluation for each dataset. Preservation of 2–5 pixel features
has not been established. Each GPU rank loads both TIFF stacks into host RAM;
reducing model size does not reduce those input copies.

## Checks and attribution

Run from the checkout in an environment with backend dependencies:

```bash
PYTHONPATH=workflow/src MPLBACKEND=Agg NO_ALBUMENTATIONS_UPDATE=1 python -m unittest discover -s tests -v
```

Training tests are disabled by default. Earlier notebooks and
[upstream documentation](docs/legacy/UPSTREAM_NOISE2INVERSE.md) are retained
as historical references. See [source attribution and licenses](NOTICE.md).
