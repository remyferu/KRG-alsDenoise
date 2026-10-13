# Sources and modifications

The `denoise` package, data importers, baseline configuration, environment files,
and original Sphinx documentation were adapted from
[tomography/Noise2Inverse360](https://github.com/tomography/Noise2Inverse360),
commit `d41be08cbbea02903bb522f6c0a3df75500f38c5`.
The upstream package metadata lists Austin Yunker as author. Remy Feru
maintains the KRG adaptation.
Their Argonne license is retained in
`third_party/Noise2Inverse360_LICENSE.txt` and existing source headers.
The root MIT license does not replace those license terms.

`denoise/nafnet25d.py` adapts the NAFBlock design from
[megvii-research/NAFNet](https://github.com/megvii-research/NAFNet).
Copyright (c) 2022 megvii-model. Its original license file is retained in
`third_party/NAFNet_LICENSE.txt`. No BasicSR code or dependency is used.
The adaptation uses native PyTorch LayerNorm, learned axial slice fusion, a
full-resolution detail branch, and a center-slice residual head.

Modified upstream files are `denoise/train.py`, `slice.py`, `volume.py`,
`data.py`, `data_utils.py`, `__main__.py`, and `setup.py`. Changes provide shared
architecture/checkpoint handling, optional BF16, earlier rank-to-GPU binding,
patch-first training crops, correct inference channel counts, corrected blend
normalization at borders, explicit output paths and preserved thread settings.
Training and volume inference reject existing output directories rather than
deleting them. CLI formatting and logging paths also carry local portability fixes.

Reference metrics add detached symmetric MSE and SSIM measurements, CSV/resume
history, and optional notebook plot panels. They preserve the L1/LCL objective
and existing checkpoint/early-stopping criteria.

The HPC notebook derives from the KRG Noise2Inverse360 workflow v3.2. It retains
native Vo centering, queues, linked zoom displays, live log output, and automatic
loss-history plots, with NAFNet configuration added in v3.3.

v3.4 extracts these notebook helpers into `workflow/src/krg_workflow`, adds explicit
run state and shared presets, and keeps the historical notebook code unchanged.
Backend configuration, checkpoint loading and training loader construction are
separated into helpers; `denoise/__init__.py` uses lazy legacy exports. Training
losses, checkpoint formats and epoch-loop behavior are unchanged. Original
documentation and the machine-specific environment export are retained under
`docs/legacy/` and `envs/legacy/`.
