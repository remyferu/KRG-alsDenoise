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

This notebook reconstructs a raw ALS parallel-beam CT scan, creates independent
even/odd projection reconstructions, and trains/applies a denoiser using those
paired images. **No earlier notebook is required.** The setup below includes the
ALS reconstruction environment and the separate PyTorch denoising environment.

The workflow includes native TomoPy `find_center_vo()`, a reconstruction queue,
linked image zoom, live logs, and automatic loss-history plots. Training is disabled until you set
`RUN_TRAINING = True` in cell 22.

Read **First-time NERSC setup**, then **Workflow and restart**. The default denoiser
is a **NAFNet-based 2.5D CT model**; the original U-Nets are also available.
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

# Embed the complete setup, so a downloaded notebook is independently usable.
setup = (root.parent/'docs/SETUP_NERSC.md').read_text(encoding='utf-8')
put('installation-and-workspace', setup.replace('(NAFNET_CT.md)', '(../docs/NAFNET_CT.md)'))

edit('paths', "INPUT_SUBFOLDER = 'ALS-12649_koo-scratch'", "INPUT_SUBFOLDER = 'raw_microct'")
edit('paths', "# Example only: replace with your /alsdata folder", "# Example: folder under your scratch input root")
edit('paths', "INPUT_ROOT = Path('/pscratch/sd/r/remyferu') / INPUT_SUBFOLDER",
     "INPUT_ROOT = Path(os.environ.get('PSCRATCH') or os.environ.get('SCRATCH') or '/alsdata') / INPUT_SUBFOLDER\n    # Or set INPUT_ROOT = Path('/absolute/path/to/your/raw/scan/folder').")
edit('paths', "PREFERRED_FILE = '20260605_162159_Hex20C1.h5'", "PREFERRED_FILE = 'sample.h5'")

put('paths-help', '''## Select paths and a scan — cells 10–11

Edit cell 10's input folder and preferred filename before running it. Its examples
are `$PSCRATCH/raw_microct/sample.h5`; use your own file and ensure the directory
is mounted on the compute node. `RUN_LABEL` names this scan's particular settings
and model run. Use a new label for changed reconstruction/COR or training settings.

`ALS_MICROCT_DIR` points to the **source checkout**, not its Conda prefix.
`DENOISE_CMD` points to the **n2i executable**, not the repository folder. The setup
above supplies both defaults. `OUTPUT_ROOT_OVERRIDE=None` selects scratch from
the environment; set an absolute `Path` if your service uses another location.

Select the intended scan in the dropdown (or set `default_scan` if widgets are
unavailable), then run cell 11 to inspect dimensions, angles and metadata.
Running these cells alone does not reconstruct or train anything.
''')
put('quick-start', '''## Workflow and restart

After first-time setup, run individual cells in the order below. **Do not use Run
All for a first scan:** settings and image checks need your decisions. Code cell
numbers appear at the beginning of each cell.

| Cells | Action and expected result |
|---|---|
| 1–9 | Run cell 1 first, then every helper cell; they define functions and configure threads without reading a scan |
| 10–11 | Edit paths, select a raw scan and inspect its dimensions/metadata |
| 12 | Choose reconstruction settings and verify the ALS backend |
| 13 | Optional first/opposite projection alignment check |
| 14 | Estimate center of rotation (COR); a **DONE** message means `run_plan.json` is saved |
| 15–16 | Optional manual COR review; needed when cell 14 reports **REVIEW REQUIRED**. Apply the selection, then save it |
| 17–18 | Add the saved plan to the queue, then reconstruct full/even/odd stacks |
| 19 | Inspect matching reconstructed rows in all three stacks |
| 20–21 | Choose 2.5D/3D and configure training; this writes YAML but does not train |
| 22 | Train/resume only after setting `RUN_TRAINING=True`; logs and history are saved automatically |
| 23 | Optional plot refresh or plotting a separate existing log |
| 24–25 | Select a checkpoint, inspect denoising, then run volume inference. 3D has no separate slice command |
| 26–27 | Optional queue inspection and explicit repair |
| 28–29 | Optional generated-output cleanup: preview first, then opt in explicitly |

The **plan** freezes source identity, COR and reconstruction settings. The **queue**
is a list of plans waiting to reconstruct; enqueueing alone does not run them.
**Full** uses all projections; **even/odd** use alternating projection indices.
**Training** learns weights; **inference** applies a saved checkpoint. `DONE` confirms
the named action, not the scientific accuracy of its result.

**Resume after a kernel restart:** select the ALS kernel, run cells 1–9, and rerun
10–12 with the same scan, `RUN_LABEL` and reconstruction settings. Run cell 14 to
restore the saved COR/plan without recomputing it. Cells 17–18 recover interrupted
reconstruction, validating and skipping completed stages. Run 19–21 again, select
the original architecture/mode, then enable 22 to resume from `resume.pth`.
If `TrainOutput` exists without that checkpoint, preserve it and use a new run
label for a fresh run. Cell 23 can plot an existing log without training.

**Change settings:** use a new run label for new COR, reconstruction, mode,
architecture or slice count. Source/backend changes invalidate saved reconstruction
identity. Switching scans invalidates older widget controls; recreate them by
rerunning the relevant cells. A restart clears Python variables but preserves files.

**Interrupt:** use Jupyter's interrupt control, allow the cell to finish its cleanup,
then inspect its log before restarting. Notebook-launched denoising processes are
terminated on an interrupt; external jobs must be stopped separately. Do not delete
models or logs merely to resume. Full-volume inference has no resume checkpoint;
preserve partial output before retrying into a new directory/run.

**NumExpr thread errors:** restart and run cell 1 before scientific imports. Its
initial active thread count is at most eight; this is not a global eight-core limit
on reconstruction. The import-time maximum accommodates TomoPy's host CPU-count
requests. If NumExpr was preloaded during kernel startup, follow the setup
troubleshooting steps before restarting again.
''')

edit('concepts-and-input', 'Section 1 checks the image layout and ALS metadata; section 2 rechecks the metadata before reconstruction.',
     'Cell 11 checks image layout and ALS metadata; cell 12 rechecks metadata before reconstruction.')
edit('concepts-and-input', '`Noise2Inverse360` performs model training and inference.',
     '`KRG-alsDenoise` supplies the Noise2Inverse model training and inference commands.')
edit('concepts-and-input', 'Offset from detector midpoint in pixels, as in ALS_recon.ipynb; a wrong value gives doubled edges.',
     'Detector-center offset in pixels: `native center - detector width / 2`. Zero is the midpoint; negative offsets are valid. A wrong value can produce doubled edges.')
edit('settings-help', '## 2. Configure the newer ALS backend', '## Configure reconstruction — cell 12')
edit('settings-help', 'These parameters are passed directly to `helper.reconstruct`, as in `ALS_recon.ipynb`.',
     'The notebook calls the ALS backend\'s `helper.reconstruct` function using the settings below; you do not need to run the upstream ALS notebook separately.')
edit('pair-qc-help', '## 8. Check the three stacks before training', '## Check the three reconstruction stacks — cell 19')
edit('preview-help', 'the previous reconstruction/sharpness\ngrid, expansion and verification code has been removed.',
     'this notebook does not use a reconstruction-image sharpness grid to find COR.')
edit('preview-help', 'Existing schema-4 v3.0/v3.1 plans restore their original COR without re-estimation.',
     'An existing supported run plan restores its saved COR without re-estimation.')
put('volume-help', '''## Denoise the full volume — cell 25

For 2.5D, review several test slices in cell 24 first. For 3D, review the training
preview images, run volume inference, then use cell 24 to inspect the saved rows.
Select an available checkpoint before starting. Training can use several GPUs;
this volume command uses **one GPU**, plus substantial **host RAM**.

This repository's 2.5D profiler uses the actual input channel count; models with
5, 7 or 9 slices are supported. Its prediction buffer is **float32**. For a
624 × 2560 × 2560 volume with 256-pixel patches and 50% overlap, that buffer alone
is about **55 GiB**: `624 × 19 × 19 × 256 × 256 × 4 bytes`. Input, stitching,
temporary arrays and the notebook process require additional RAM. This is a
planning estimate, not a measured peak, and GPU VRAM cannot replace host RAM.
3D stitches predictions incrementally but retains the full input and accumulation
arrays; it still needs substantial host RAM.

Neither the notebook nor this backend deletes an existing volume directory to
restart. Cell 25 refuses to overwrite it. There is **no volume resume**: preserve
partial output and use a new output directory/run for a retry. The CLI's
`--output-dir` can select a separate destination outside this cell, but assembling
independently inferred subvolumes is not automated here. If full 2.5D inference
does not fit, its single-slice command remains available.

NAFNet outputs end in `_nafnet25d`, so they are distinguishable from baseline
outputs. After inference, the notebook checks filenames, dimensions, floating
dtype and finite pixels one slice at a time, then opens a zoomable browser. These
checks add an output-volume read and establish file consistency. Inspect fine
features and residuals before accepting the result.
''')
edit('inference-help', '## 10. Inspect denoising before running the volume', '## Inspect denoising — cell 24')
edit('inference-help', 'section 11', 'cell 25')
edit('inference-help', 'Upstream selection criterion', 'Checkpoint selection criterion')
edit('inference-help', 'The volume step uses the `CHECKPOINT` selected in this section',
     'Cell 25 uses the `CHECKPOINT` selected in cell 24')
put('queue-help', '''## Queue and reconstruct — cells 17–18

Cell 17 adds the selected `run_plan.json` to
`OUTPUT_ROOT/reconstruction_queue_v3.json`. Run it only after COR has produced a
saved plan. Cell 18 prepares even/odd projection views and reconstructs queued
runs **sequentially**; adding an entry alone does not start reconstruction.

The queue can contain several scans/run labels. It records a stable `id` and a
`plan_path` for each entry. Preflight checks each plan and its source/settings;
missing or mismatched entries are blocked, duplicates skipped, and ordinary errors
recorded while later valid runs continue. An interrupt stops further entries.
The queue report is `OUTPUT_ROOT/queue_report_v3.json`; each run also has logs and
completion records. On a retry, completed stacks are skipped only after their
TIFFs pass validation; an incomplete tracked stage is reconstructed again.

Cell 19 inspects the run selected in cell 10, not every run that happened to be in
the queue. To remove an entry or repair its path, use cells 26–27. Removing a queue
entry does not delete reconstructed images. The queue lock is
`OUTPUT_ROOT/reconstruction_queue_v3.json.lock`; run actions can also use a
`.run.lock` inside their run directory. A lock file marks an active operation;
confirm the owner has stopped before considering stale-lock repair.
''')
put('queue-repair-help', '''## Optional queue inspection and repair — cells 26–27

Run cell 26 with its default empty selections to inspect the queue. The printed
`proposed.entries` contains each entry's `id` and `plan_path`; `preflight` reports
why an entry is valid, blocked or duplicated.

To remove one entry, copy its exact `id` into `REMOVE_ENTRY_IDS`, for example
`REMOVE_ENTRY_IDS = ['copied_entry_id']`, and rerun cell 26. To correct a moved
plan, put `{copied_entry_id: '/absolute/path/to/run_plan.json'}` into
`REPLACE_PLAN_PATHS`. Review the newly printed proposed entries, then set
`APPLY_QUEUE_REPAIR=True` in cell 27 and run it. Repair backs up the exact previous
queue bytes. Entry removal affects the queue only, not run files.

For corrupt queue JSON, set `REBUILD_FROM_PLANS` to an explicit list of valid
absolute plan paths in cell 26. Deduplication and removing confirmed missing
plans are separate opt-in settings; inspect their effects before applying.

If a lock is stale, first confirm its recorded job/kernel has stopped. Inspect
the lock's JSON, copy its token to `STALE_LOCK_TOKEN`, and use the confirmation
flags in cell 27. Never remove the lock of an active reconstruction.
''')
put('scratch-cleanup-help', '''## Optional scratch cleanup — cells 28–29

Use after stopping jobs that use the selected run. Cell 28 is a preview: select
categories in `CLEANUP_CATEGORIES`, rerun it, and inspect every proposed path,
file count, estimated bytes and stated effect. An empty list selects nothing.

| Category | Effect to consider |
|---|---|
| `full`, `even`, `odd` | Removes the corresponding reconstruction stack and invalidates its completion record; queue reconstruction can rebuild it |
| `projection_views` | Removes generated HDF5 view files, which point at the raw scan |
| `denoised` | Removes generated denoising outputs |
| `cor` | Selects generated COR review/preview paths listed in the preview |
| `logs` | Removes run logs; history cannot be regenerated from a deleted log |
| `training` | Removes model/checkpoint outputs and the scan training YAML, including saved normalization; resume/inference then require restoration or fresh training |

For deletion, copy the **exact resolved run directory** from the preview into
`CLEANUP_CONFIRM_RUN` in cell 29, set `CLEANUP_EXECUTE=True` and
`CLEANUP_DRY_RUN=False`, then run that cell. Defaults only preview. Rerun cell 28
after changing any selection. Raw scans, source code, environments, active locks
and symlink/junction paths are protected. Plans and queue entries are retained.

Preserve accepted raw data, TIFFs, model, YAML and plan in your normal durable
storage before choosing cleanup. The report records failures and estimated
reclaimed bytes; it is not a storage backup and free-space updates may lag.
''')
put('recovery', '''## Decisions and recovery

| Situation | What to rerun or inspect |
|---|---|
| Kernel restarted | Re-select ALS kernel, run 1–9, then 10–12 and 14 with the same scan/run settings; follow the restart sequence above |
| `Stale controls/results` | Rerun scan inspection, settings and the action that created that viewer; do not reuse controls from a different scan/run |
| COR says `REVIEW REQUIRED` | Choose informative detector rows/native Vo limits in 14, or review/apply a manual offset in 15 and save it in 16 |
| Source/backend/settings differ from a saved plan | Restore the original inputs/settings, or use a new run label; do not edit the frozen plan to force a match |
| Reconstruction interrupted | Inspect the run log and queue report, then rerun 17–18; validated completed stages are reused |
| Training interrupted | Preserve YAML/models/logs, rerun 19–21 with the original model settings, then enable 22 to resume from `resume.pth` |
| `TrainOutput` has no resume checkpoint | Preserve the folder; use a new run label for fresh training or restore its checkpoint |
| Volume inference interrupted | Preserve the partial output; no automatic volume resume exists. Retry with a new destination/run after resolving the error |
| Queue contains an unwanted/broken entry | Inspect and explicitly repair it in 26–27 |
| Reconstruction was cleaned up | Keep its plan, rerun 17–19 to rebuild/check stacks, then 20–21 before training |

Read the full log path printed by a failing cell; the live box shows only a tail.
GPU utilization alone does not prove an epoch has stalled. Check the process,
host memory and allocation time in the compute terminal. Never remove a lock or
delete outputs while its job/kernel is active.

Check denoising residuals for erased features and compare several rows at matched
zoom/contrast. Vo row agreement, low loss and a smooth image do not establish
scientific accuracy.
''')
put('training-help', '''## Configure and train

Fresh 2.5D runs default to `nafnet25d`; set `TRAIN_ARCHITECTURE = 'unet'` in
cell 21 for the baseline. **2.5D** uses neighboring axial slices as input and
predicts their center slice. **3D** uses cubic patches and predicts a volume patch;
it uses the 3D U-Net and generally needs more computation/memory. Cell 20 selects
the mode, and cell 21 applies its preset. Compare five slices first,
then change `n_slices` to 7 or 9 in a fresh run. BF16 is optional; set FP32 in the
preset for a precision comparison. These are starting parameters, not tuned optima.

Cell 22 prints a disabled message until you set `RUN_TRAINING = True` yourself.
It then uses all allocated GPUs, retains the live log box, and saves automatic
history plots even on interruption. Patience uses raw evaluation loss, and the
10-epoch history smoothing affects only the plot.

| Setting in cell 21's preset/overrides | Meaning |
|---|---|
| `psz` / `n_slices` | 2.5D square patch width and number of neighboring slices; use an odd slice count |
| `mbsz` | Batch size per GPU; four GPUs with batch 4 process a total batch of 16 |
| `precision` | `bf16` for supported GPUs such as A100; `fp32` for a precision comparison |
| `psz_3d` / `nb_patches_3d` | 3D cube width and number of sampled training patches per epoch |
| `lr` | Adam learning rate; YAML prepared for a new run starts at 0.001 |
| `warmup` | Optimizer updates before adding weighted LCL; initial YAML uses 2000 |
| `maxep` / `patience` | Maximum epochs / consecutive eligible epochs without lower evaluation L1; defaults 2000 / 100 |

An **epoch** visits the sampled training patches and then runs an evaluation pass.
Evaluation currently samples the training dataset again; it is not a separate
held-out accuracy estimate. L1 measures paired absolute error. LCL is an output
contrast regularizer weighted by 0.01 after warmup. L2/MSE and SSIM loss are
recorded only for reference. No learning-rate scheduler is enabled.

Saved training settings and normalization are kept when `TrainOutput` exists;
editing a fresh-run preset does not alter an existing model. Cell 22 automatically
adds `--resume` when a valid `resume.pth` exists, otherwise starts fresh. A U-Net
checkpoint cannot initialize NAFNet. You can run cell 23 on a saved log while
leaving training disabled.
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
# Preserve old L1/LCL logs, adding optional reference columns and plot panels.
edit('training-helpers', '(L1 loss|LCL loss|EDGE Value)',
     '(L1 loss|LCL loss|EDGE Value|L2 loss|SSIM loss)')
edit('training-helpers', "('Val','EDGE Value'):'edge'}", "('Val','EDGE Value'):'edge',\n             ('Train','L2 loss'):'train_l2', ('Val','L2 loss'):'eval_l2',\n             ('Train','SSIM loss'):'train_ssim', ('Val','SSIM loss'):'eval_ssim'}")
edit('training-helpers', "            row.setdefault('edge', None)", "            row.setdefault('edge', None)\n            for key in ('train_l2','eval_l2','train_ssim','eval_ssim'):\n                row.setdefault(key, None)\n            row['ssim_data_range'] = metadata.get('ssim_data_range')")
edit('training-helpers', "            mode = re.search(", "            metric_range = re.search(r'ssim_data_range=('+number+r')', line)\n            if metric_range:\n                metadata['ssim_data_range'] = float(metric_range[1])\n            mode = re.search(")
edit('training-helpers', "('train_l1','eval_l1','train_lcl','eval_lcl')}",
     "('train_l1','eval_l1','train_lcl','eval_lcl','train_l2','eval_l2','train_ssim','eval_ssim')}")
edit('training-helpers', "['epoch','train_l1','eval_l1','train_lcl','eval_lcl','edge','warmup']",
     "['epoch','train_l1','eval_l1','train_lcl','eval_lcl','edge','warmup',\n                'train_l2','eval_l2','train_ssim','eval_ssim','ssim_data_range']")
edit('training-helpers', "        fig, axes = plt.subplots(2,1,figsize=(12,8.8),sharex=True)",
     "        has_reference = any(np.isfinite(values[key]).any() for key in ('train_l2','eval_l2','train_ssim','eval_ssim'))\n        fig, axes = plt.subplots(4 if has_reference else 2,1,figsize=(12,15 if has_reference else 8.8),sharex=True)")
edit('training-helpers', "            for ax, train_key, eval_key, title, ylabel, best in specifications:",
     "            if has_reference:\n                specifications.extend([(axes[2],'train_l2','eval_l2','L2 / MSE (reference only)',\n                    'Epoch-average paired MSE',None),\n                    (axes[3],'train_ssim','eval_ssim','SSIM loss (reference only)',\n                    'Epoch-average paired 1 - SSIM',None)])\n            for ax, train_key, eval_key, title, ylabel, best in specifications:")
edit('training-helpers', "            axes[1].set_xlabel('Epoch',labelpad=10)",
     "            axes[-1].set_xlabel('Epoch',labelpad=10)")
edit('training-helpers', "LCL is plotted with its logged weight.",
     "LCL is plotted with its logged weight. L2/SSIM are reference metrics only.")
edit('training-history-help', '## Training history',
     '## Training history\n\nNew runs also log L2/MSE and SSIM loss for reference. The automatic plot adds\nreference panels when present; the CSV includes them and the fixed SSIM intensity\nrange. They do not affect optimization, checkpoints or patience. Older logs remain supported.')
edit('training-history-help', 'No checkpoint\nloading or PyTorch installation in the ALS kernel is required.',
     'The history cell reads text; it does not load a checkpoint, train a model, or require PyTorch in the ALS kernel.')
history_source = ''.join(cells['training-history-help']['source'])
put('training-history-help', history_source + '''

In a line such as `L1 loss: mean, first => last`, **mean** is the epoch's average
batch value, followed by its first and last batch values. `rate` is seconds for
the complete training/evaluation epoch. L1 and weighted LCL logs sum both paired
directions; new L2/MSE and SSIM-loss logs average them. They report the final GPU
rank's batches, not an aggregate across GPUs.

SSIM loss is `1 - SSIM`; smaller means greater paired similarity. It uses a fixed
normalized-intensity range in `train.ssim_data_range`, inferred once from the two
training stacks if absent and saved in YAML. Compare SSIM only with consistent
range/settings. In 3D, its SSIM/LCL diagnostics use the center axial plane; MSE
uses the entire patch. Metrics against noisy paired targets are not clean-image
accuracy measurements. Reference CSV is also written to
`TrainOutput/results/reference_metrics.csv` and retained on resume.

The live log box shows a scrolling tail while a command runs; the complete log is
saved under the printed `logs/` path. Startup messages may repeat once per GPU
process. ANSI color escapes are stripped from the displayed text. Refreshing
cell 23 reads the current log once; rerun it for another refresh.
''')
edit('source-review-validation', 'NAFNet is integrated',
     'L2/MSE and SSIM loss are logged and saved for reference without changing the\nL1/LCL objective. SSIM uses a fixed normalized intensity span; 3D SSIM evaluates\nthe center axial plane.\n\nNAFNet is integrated')

nb['metadata']['noise2inverse_workflow'].update(notebook_version='3.3',
    training_architecture='nafnet25d_optional', validation_status='code_and_inference_checks_only')
nb['metadata']['kernelspec'].update(name='als832-n2i', display_name='ALS CT + N2I notebook')
for cell in nb['cells']:
    cell['source'] = ''.join(cell['source']).replace('section 9', 'cell 21').replace(
        'section 11', 'cell 25').replace('section 10', 'cell 24').splitlines(keepends=True)
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), feature_version=(3, 8))
(root/'Noise2Inverse360_HPC_v3.3.ipynb').write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
print('Created v3.3 without executing notebook cells.')
