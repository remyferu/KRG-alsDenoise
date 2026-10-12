# KRG-alsDenoise
Tools to run on NERSC for denoising ALS 8.3.2 micro-ct reconstructions

This repository extends [Noise2Inverse360](https://github.com/tomography/Noise2Inverse360)
with an optional **NAFNet-based 2.5D CT model**, selected with
`train.architecture: nafnet25d` in the scan YAML. The original 2.5D and 3D U-Nets
remain available. NAFNet requires a fresh training run; existing U-Net weights are
incompatible.

The model uses learned neighboring-slice gates, NAFBlocks, a full-resolution
detail branch, and a residual prediction of the center slice. It has **989,745
parameters** at the supplied settings and supports 5, 7, or 9 slices. This is an
experimental adaptation: preservation of 2–5 pixel features needs validation on
your scans.

Start with the [NERSC setup and usage guide](docs/NAFNET_CT.md), or the
[v3.3 HPC notebook](notebooks/Noise2Inverse360_HPC_v3.3.ipynb).
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
is kept in [docs/UPSTREAM_NOISE2INVERSE.md](docs/UPSTREAM_NOISE2INVERSE.md).
