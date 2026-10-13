# Denoising backend

PyTorch training and inference for paired CT reconstructions. The CLI entry
point is `denoise`; `python -m denoise` exposes the same commands.

| Command | Purpose |
|---|---|
| `prepare` | Write a scan configuration and print upstream reconstruction commands |
| `train` | Train or resume a 2.5D or 3D model |
| `slice` | Denoise one slice with a 2.5D model |
| `volume` | Denoise a volume with a 2.5D or 3D model |
| `register`, `search` | Store checkpoints and search the local model registry |

Training requires full/even/odd reconstruction directories with matching slice
names, dimensions and geometry. NAFNet is selected with
`train.architecture: nafnet25d`; baseline U-Nets use `unet` or `unet3d`.

See [NERSC setup](../docs/SETUP_NERSC.md),
[model configuration and usage](../docs/NAFNET_CT.md), and
[the workflow module map](../docs/WORKFLOW.md). The ALS notebook helpers are a
separate package under `workflow/`.
