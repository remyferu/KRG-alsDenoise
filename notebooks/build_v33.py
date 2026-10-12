"""Create the opt-in NAFNet workflow from the preserved v3.2 notebook."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
nb = json.loads((root/'Noise2Inverse360_HPC_v3.2.ipynb').read_text(encoding='utf-8'))
cells = {cell['id']: cell for cell in nb['cells']}


def edit(name, old, new):
    cell = cells[name]
    source = ''.join(cell['source'])
    if old not in source:
        raise ValueError('Builder source mismatch in %s: %s' % (name, old[:80]))
    cell['source'] = source.replace(old, new).splitlines(keepends=True)


def put(name, source):
    cells[name]['source'] = source.splitlines(keepends=True)


for cell in nb['cells']:
    source = ''.join(cell['source'])
    source = source.replace('Noise2Inverse360_HPC_v3.2.ipynb', 'Noise2Inverse360_HPC_v3.3.ipynb')
    source = source.replace('— v3.2', '— v3.3')
    source = source.replace("notebook_version='3.2'", "notebook_version='3.3'")
    source = source.replace("not in ('3.0', '3.1', '3.2')", "not in ('3.0', '3.1', '3.2', '3.3')")
    cell['source'] = source.splitlines(keepends=True)
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

put('intro', '''# ALS reconstruction → KRG-alsDenoise — v3.3

Adds the **NAFNet-based 2.5D CT model** to the v3.2 workflow. Native TomoPy
`find_center_vo()`, queues, linked image zoom, the live log box, and automatic
L1/LCL history plots are retained. Training is disabled until you set
`RUN_TRAINING = True` in cell 22.

Install this repository in the n2i environment using the
[NAFNet setup guide](../docs/NAFNET_CT.md). Keep your ALS reconstruction kernel.
Use a new run label when changing architecture or slice count; NAFNet needs fresh
weights and cannot resume a U-Net checkpoint. Defaults: five slices, 256-pixel
patches, batch 4 per GPU, BF16. Fine-feature accuracy requires scan validation.

COR values remain offsets from the detector midpoint, in detector pixels.
Run-plan schema 4 and queue schema 1 are retained. Existing v3.0/v3.1/v3.2/v3.3
plans are accepted. The original v3.2 notebook is preserved.
''')
edit('paths', "RUN_LABEL = 'v3_2_test_1'", "RUN_LABEL = 'v3_3_nafnet_1'")
edit('training-helpers', "        if volume and train['n_slices'] != 5:\n            raise ValueError('The upstream 2.5D volume profiler assumes five channels; use n_slices=5 for volume inference.')\n", '')
edit('training-helpers', "    for key in ('warmup', 'patience'):", '''    architecture = train.get('architecture', 'unet3d' if mode == '3d' else 'unet')
    if architecture not in (('unet', 'nafnet25d') if mode == '2.5d' else ('unet3d',)):
        raise ValueError('Architecture is incompatible with the selected mode.')
    if architecture == 'nafnet25d' and train['n_slices'] < 3:
        raise ValueError('NAFNet requires an odd slice count >= 3.')
    if train.get('precision', 'fp32') not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16.')
    for key in ('warmup', 'patience'):''')
edit('configure-training', "    TRAIN_PRESETS = {", "    TRAIN_ARCHITECTURE = 'nafnet25d'  # 'nafnet25d' or 'unet' for 2.5D; 3D uses unet3d\n    TRAIN_PRESETS = {")
edit('configure-training', "'2.5d': {'psz': 256, 'n_slices': 5, 'mbsz': 4}", "'2.5d': {'psz': 256, 'n_slices': 5, 'mbsz': 4, 'architecture': TRAIN_ARCHITECTURE,\n                 'precision': 'bf16' if TRAIN_ARCHITECTURE == 'nafnet25d' else 'fp32', 'num_workers': 0}")
edit('configure-training', "'n_blocks_3d': 4, 'start_filts_3d': 56, 'mbsz': 1}", "'n_blocks_3d': 4, 'start_filts_3d': 56, 'mbsz': 1, 'architecture': 'unet3d',\n               'precision': 'fp32', 'num_workers': 0}")
edit('configure-training', "('--mode', '--resume', '--no-search')", "('--mode', '--resume', '--no-search', '--architecture')")
edit('configure-training', "        if not TRAIN_DIR.exists():\n            cfg['train'].update(TRAIN_OVERRIDES)", '''        if TRAIN_DIR.exists() and cfg['train'].get('architecture',
                'unet3d' if SELECTED_MODE == '3d' else 'unet') != TRAIN_OVERRIDES['architecture']:
            raise ValueError('Saved architecture differs. Use its original architecture or a new run label.')
        if not TRAIN_DIR.exists():
            cfg['train'].update(TRAIN_OVERRIDES)
            cfg['infer'].update(precision=cfg['train']['precision'], num_workers=0, max_batch_size=32)''')
edit('single-slice', "(base_name + '_denoised_slices')", "(base_name + '_denoised_slices' + ('_nafnet25d' if cfg['train'].get('architecture') == 'nafnet25d' else ''))")
edit('volume', "(base_name + '_denoised_volume_' + mode)", "(base_name + '_denoised_volume_' + mode + ('_nafnet25d' if cfg['train'].get('architecture') == 'nafnet25d' else ''))")
edit('cleanup-helpers', "stem+'_denoised_volume_3d']", "stem+'_denoised_volume_3d',\n                  stem+'_denoised_slices_nafnet25d', stem+'_denoised_volume_2.5d_nafnet25d']")

# Keep running all notebook cells from launching training unintentionally.
source = ''.join(cells['train']['source'])
title, remainder = source.split('\n', 1)
put('train', title + '\n\nRUN_TRAINING = False  # Set True yourself when ready to train or resume.\n'
    "if not RUN_TRAINING:\n    print('Training disabled. Set RUN_TRAINING=True in this cell when ready.')\nelse:\n"
    + ''.join('    '+line if line.strip() else line for line in remainder.splitlines(keepends=True)))

put('installation-and-workspace', '''## NERSC setup

Use a single-node GPU Jupyter allocation and the existing ALS reconstruction
kernel. Install KRG-alsDenoise in the separate n2i environment:

```bash
cd ~/microct-n2i-source
git clone https://github.com/remyferu/KRG-alsDenoise.git
cd KRG-alsDenoise
~/microct-n2i-envs/n2i/bin/python -m pip install --no-deps -e .
MPLBACKEND=Agg ~/microct-n2i-envs/n2i/bin/denoise train -h
```

The CLI help must include `--architecture`. If this checkout already exists,
update it with your normal Git workflow instead of cloning again. NAFNet uses
native PyTorch operations and does not require BasicSR or pretrained weights.
The notebook launches the n2i executable explicitly and sets `MPLBACKEND=Agg` for
subprocesses. Keep code/environments in home and large working data/results on
scratch. Choose enough host memory for the full/even/odd stacks and per-rank copies.
See [installation, CPU RAM and GPU guidance](../docs/NAFNET_CT.md).
''')
put('training-help', '''## Configure and train

Fresh 2.5D runs default to `nafnet25d`; set `TRAIN_ARCHITECTURE = 'unet'` in
cell 21 for the baseline. 3D uses the existing U-Net. Compare five slices first,
then change `n_slices` to 7 or 9 in a fresh run. BF16 is optional; set FP32 in the
preset for a precision comparison. These are starting parameters, not tuned optima.

Cell 22 prints a disabled message until you set `RUN_TRAINING = True` yourself.
It then uses all allocated GPUs, retains the live log box, and saves automatic
history plots even on interruption. Patience uses raw evaluation loss, and the
10-epoch history smoothing affects only the plot.
''')
put('source-review-validation', '''## Change notes and validation limits — v3.3

NAFNet is integrated in training, slice inference and volume inference, with
shared checkpoint metadata checks. Seven/nine-slice volume profiling uses the
actual input channel count. Border blending preserves intensities, and training
crops are taken before copying neighbor planes. BF16 and batch progress logging
are optional. GPU rank binding precedes NCCL initialization.

The notebook preserves v3.2 reconstruction geometry and plan/queue formats,
native Vo centering, linked zoom, live logs, and history plots. It was checked for
Python 3.8 syntax and configuration/output contracts. Backend shape/checkpoint and
synthetic TIFF inference tests are included; training tests are skipped by default.
The full reconstruction pipeline and multi-GPU NERSC training have not been run
as part of this integration. Scan quality and tiny-feature retention are unmeasured.
''')
nb['metadata']['noise2inverse_workflow'].update(notebook_version='3.3',
    training_architecture='nafnet25d_optional', validation_status='code_and_inference_checks_only')
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), feature_version=(3, 8))
(root/'Noise2Inverse360_HPC_v3.3.ipynb').write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
print('Created v3.3 without executing notebook cells.')
