# NAFNet for ALS micro-CT

The [architecture diagram](figures/README.md) shows the full 2.5D data flow,
axial fusion gates, NAFBlock operations and paired training objective. SVG and
PNG exports include default tensor dimensions and parameter counts; a local
HTML viewer supports zoom and pan.

## Install on NERSC

Follow [NERSC setup](SETUP_NERSC.md) for environment installation, kernel
registration and compute allocation. The guide is also included in the v3.5
notebook.

Use `als832` as the notebook kernel for reconstruction. The notebook launches
`n2i/bin/denoise` separately for training and inference. If both environments are
already working, update/install this repository in n2i as follows:

```bash
cd ~/microct-n2i-source/KRG-alsDenoise
git pull --ff-only
~/microct-n2i-envs/n2i/bin/python -m pip install --no-deps -e .
export MPLBACKEND=Agg
~/microct-n2i-envs/n2i/bin/denoise train -h
```

CLI help must include `--architecture`. NAFNet requires PyTorch >= 2.0. The
adaptation uses native PyTorch without BasicSR or custom CUDA kernels. Editable
installation points the n2i `denoise` entry point at this checkout.

## Reuse an existing scan

Create a separate config from the scan's existing reconstruction YAML:

```bash
python examples/configure_nafnet.py \
  --input /path/to/sample_rec_config.yaml \
  --output /path/to/sample_nafnet_config.yaml
```

The helper retains the dataset paths, metadata, and any normalization already in
the input, and selects the new architecture. Training recalculates normalization.
It refuses to overwrite a config. Add `--n-slices 7` or `--n-slices 9` to compare
larger axial neighborhoods; use `--precision fp32` to compare numerical precision.
Start with five slices so model and slice-count effects can be evaluated separately.

Default train settings are `psz=256`, `mbsz=4` per GPU, `lr=0.001`, `warmup=2000`
optimizer updates, `maxep=2000`, `patience=100`, `z_stride=1`, `precision=bf16`,
and `num_workers=0`. Inference uses BF16, cosine blending and a profiler batch
ceiling of 32. These settings are editable in YAML. BF16 requires a supported GPU;
A100 supports it. FP32 inference also works on CPU.

## Training

Pretrained weights are not supplied.
Run in an allocated GPU compute session:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 denoise train \
  --config /path/to/sample_nafnet_config.yaml \
  --gpus 0,1,2,3 --no-search \
  --output-dir /path/to/TrainOutput_NAF
```

Resume with the same command plus `--resume`. Fresh training refuses an existing
output directory. The 2.5D/3D baseline checkpoints cannot resume or fine-tune the
NAFNet architecture. v3.5 uses model specification and resume format version 2;
earlier NAFNet checkpoints are incompatible. New checkpoints include their
architecture, slice count and model options; all commands check those against
the config. Keep existing results and select a new notebook `TRAIN_LABEL` or
CLI `--output-dir`. Reconstruction TIFF stacks can be reused unchanged.

Training uses symmetric paired L1, with weighted LCL added after warmup.
Early stopping monitors a trailing average of evaluation total loss, using the
same `paired_objective()` as optimization. The current total is half the sum
of paired L1 losses plus half the sum of already-weighted LCL terms. LCL has
weight 0.01 after warmup; MSE and SSIM remain reference measurements.

```yaml
train:
  patience: 100
  early_stopping_window: 10
  min_delta: 0.001
  nafnet:
    distance_aware: true
  validation:
    enabled: true
    fraction: 0.1
    samples: 128
    seed: 1729
  lr_schedule:
    enabled: true
    factor: 0.5
    patience: 25
    cooldown: 5
    min_lr: 0.00001
    refinement_epochs: 30
  ema:
    enabled: true
    decay: 0.999
    evaluate_every: 1
```

Warmup and its transition epoch are excluded. The first complete post-warmup
window establishes a baseline. Later windows must improve on the best qualifying
mean by more than `min_delta` to reset patience. The threshold is absolute, in
total-objective units, and should be chosen for the loss scale. Smaller changes
can accumulate until they cross the threshold because the baseline stays fixed.
`patience: 0` disables stopping. CLI-generated configurations retain that default;
the notebook preset uses 100.

The raw L1, LCL and EDGE checkpoints retain their original selection criteria.
Monitor history, best qualifying mean, epoch and patience counter are saved in
`resume.pth`. Resuming with the same window and threshold restores the monitor
exactly; changing either starts a fresh monitor and scheduler history. The
validation plan and warmup length must match the resume checkpoint.

The `[EarlyStopping]` log reports collection progress or the monitored mean and
patience count. History exports include training/evaluation total loss and the
logged monitor curve. `MOVING_AVERAGE_EPOCHS` changes display smoothing only;
`train.early_stopping_window` controls stopping.

### Fixed validation

The final `fraction` of axial planes is reserved, with at least `n_slices`
planes in the held-out block. Training center indices stop before any input
stack could touch that block. Validation centers start far enough inside it
that their input stacks contain only held-out planes. Boundary stacks may
repeat the volume's end plane. Normalization statistics and the default SSIM
range use training planes only.

Up to `samples` distinct center/row/column coordinates are selected with
`seed` and recorded in `TrainOutput_<label>/validation.json`. They remain
fixed, without augmentation. If the available coordinate count is smaller,
all available crops are used. Resume checks geometry, coordinates, source
paths and normalization against the saved plan.

GPU ranks evaluate nonoverlapping index shards without padding duplicates.
Metric sums are weighted by the number of patches in each batch and reduced
across all ranks; a rank with no validation patches contributes zero. LCL is
the existing batch criterion, so its sample-weighted summary can change
slightly when GPU count or batch size changes. Hold these fixed for comparisons.
The target remains an independently reconstructed noisy split, not a clean
reference. Inspect the held-out region to establish that it contains relevant
sample structures.

Fixed axial validation is supported in 2.5D. In 3D it is disabled by default;
evaluation continues to resample training patches, with metrics aggregated
across ranks.

### Learning-rate reduction

`ReduceLROnPlateau` watches the raw model's complete trailing total-loss mean
after warmup, with the same absolute `min_delta` as stopping. It halves the
learning rate after its plateau patience, observes a five-epoch cooldown and
stops reducing at `min_lr`. The initial LR remains `train.lr`.

An LR reduction preserves the best loss and its patience count, but postpones
early stopping until `refinement_epochs` have elapsed after the most recent
reduction. This allows refinement without labeling an LR change as a loss
improvement. Scheduler state and the last reduction epoch are saved on resume.
Every rank applies the same decision from the globally aggregated loss.

The displayed patience counter can exceed its limit during this interval.
When patience is exhausted, the log reports the remaining refinement epochs
and the earliest permitted stopping epoch. Another LR reduction can move that
epoch later. Set `train.lr_schedule.refinement_epochs: 0` to stop as soon as
patience is exhausted. Settings are read at process startup; editing the YAML
does not change an already running job.

### EMA inference weights

EMA weights are seeded from the first optimizer update after warmup. Later
updates apply `average = decay * average + (1 - decay) * current` in FP32;
buffers are copied from the raw model. EMA weights never receive gradients.
The default decay is 0.999 and `evaluate_every=1`; larger intervals reduce
the extra validation work.

EMA is evaluated on the same fixed patches as the raw model. Its lowest
post-warmup total loss selects `best_ema_model.pth`. Use `--checkpoint ema`
for CLI inference; v3.5 notebook inference defaults to `CHECKPOINT='ema'`.
Raw `val`, `lcl` and `edge` checkpoints remain independent. The raw model's
evaluation total still controls LR scheduling and stopping. EMA weights,
averaging count and best score are stored in `resume.pth`.

### Distance-aware axial fusion

The shared neighbor-gate convolutions produce logits with `stem_width`
channels. A trainable bias indexed by `abs(neighbor - center)` is added before
sigmoid. Opposite neighbors at the same distance share the bias; different
distances have separate channel biases, initialized to `-log(distance)`.
The correction remains `sum(gate * (neighbor - center)) / (n_slices - 1)`.

At five slices and stem width 16, distance awareness adds 32 parameters for
a total of 989,777. `distance_aware: false` disables it for a fresh ablation
run; it is part of checkpoint geometry and cannot change on resume.

### Reference L2 and SSIM losses

Each epoch also reports `[Train]` and `[Val]` L2 loss (mean squared error) and
SSIM loss (1 - SSIM). These are detached reference measurements computed from
the same paired predictions; they do not enter backpropagation, checkpoint
selection, warmup or patience. Lower values indicate closer agreement with the
noisy paired target. They do not measure clean-reference accuracy.

New metrics average the two prediction directions. The existing L1/LCL log lines
retain their original convention of summing those directions, so their scale is
unchanged. Epoch summaries use sample-weighted batch values across all GPU
ranks. For 2.5D, both metrics compare the output
with the opposite split's center slice. For 3D, MSE uses the whole patch and SSIM
uses the center axial plane, as do the existing LCL/EDGE diagnostics.

SSIM is evaluated in FP32 with an 11 × 11 Gaussian window, sigma 1.5, population
covariance, and valid windows. It uses a fixed `train.ssim_data_range` measured
in normalized intensity units. If absent, rank 0 computes the span from the
minimum/maximum of both normalized training stacks once and saves it in the
YAML. Resume reuses it. Set a positive range explicitly for controlled comparisons;
different ranges change SSIM values. Inputs are neither clipped nor rescaled per
batch, and SSIM loss is not a percentage.

Each completed epoch updates `TrainOutput_<label>/results/reference_metrics.csv` (or the
equivalent custom output directory), including the range and window settings.
Reference history is retained in v3.5 `resume.pth` checkpoints.
The v3.5 history cell also extracts the optional metrics into its CSV and adds two
plot panels when they are present. Old L1/LCL-only logs still work.
New logs also add an EMA total-loss overlay and an LR panel; CSV exports retain
the raw EMA values, learning rate and reduction flag for each epoch.

History plots use a linear epoch axis and logarithmic loss axes, with major and minor gridlines.
Dots mark each strict record low in the raw training and evaluation losses,
tracked separately after warmup. The logged stopping monitor has its own
record-low points; EMA is a distinct purple line without point markers.
Legends, lowest-loss summaries and warmup labels sit outside the plotting area.
Display smoothing does not affect the record-low markers.
Epochs containing any warmup updates are excluded, including the transition
epoch. New logs record this explicitly; older logs use their warmup and monitor
status. Without warmup status, record-low markers are omitted. Nonpositive
values remain in the CSV but cannot appear on logarithmic axes.

[Training_Snooper.ipynb](../notebooks/Training_Snooper.ipynb) displays the same
plots concurrently in a separate kernel. It reads a bounded log snapshot and
writes separate exports; model state and training outputs are not modified.

## Inference and feature assessment

Denoising inspection reports the selected checkpoint's training epoch in its
output and viewer title. New L1, LCL and EDGE checkpoints record their epoch;
EMA checkpoints already include it. Inference embeds checkpoint provenance in
each output TIFF, so cached images retain the epoch of the weights that produced
them even when training later replaces the checkpoint. Older checkpoints or
TIFFs without this metadata report `unknown (not recorded)`.

```bash
denoise slice --config /path/to/sample_nafnet_config.yaml \
  --model-dir /path/to/TrainOutput_NAF --checkpoint ema --slice-number 300 \
  --output-dir /path/to/sample_v3_5_ema_slices
denoise volume --config /path/to/sample_nafnet_config.yaml \
  --model-dir /path/to/TrainOutput_NAF --checkpoint ema \
  --output-dir /path/to/sample_v3_5_ema_volume
```

Notebook output directories include the training label and checkpoint choice.
For CLI comparisons, use a distinct `--output-dir` for each raw or EMA model.
Volume inference refuses an existing directory; a slice command replaces the
requested slice in its selected directory. Do not edit the model architecture or
slice count in a config after training.

The learned neighbor corrections preserve a center feature path. Four NAFBlocks
operate at full resolution; a second path uses widths 32/64/128 with two
downsampling stages and additive encoder skips. A zero-initialized output head
starts at the noisy center slice and learns its correction. Arbitrary spatial
sizes are padded and cropped internally. NAFNet channel attention pools over the
patch, so use the same patch size for training and inference.

Assess feature retention by comparing the same
2–5 pixel features against the original reconstructions, inspect residuals and
measure feature contrast/diameter. Compare the original U-Net and this model at
the same slice count, data splits and training budget. Neither minimum LCL nor
maximum EDGE proves accurate geometry.

## Notebook and validation

Fresh 2.5D notebook runs default to NAFNet. Set `TRAIN_ARCHITECTURE='unet'`
for the baseline, using a new training label when changing architecture. CLI help is
checked for `--architecture` before training configuration. Earlier notebooks
retain their original code.

The default test suite checks shapes for 5/7/9 slices, center identity, LayerNorm,
checkpoint compatibility, crop pairing, blend borders, and actual synthetic TIFF
slice/volume inference. Training tests require `RUN_TRAINING_TESTS=1` and are
skipped by default. `examples/profile_nafnet.py` measures synthetic training
VRAM on the local GPU; it does not measure scan accuracy or A100 throughput.

Four GPU ranks still hold separate copies of both full stacks in CPU RAM. For
624 × 2560 × 2560 float32 paired volumes this is about 122 GiB before other
allocations. Cropping before neighbor indexing avoids copying full detector
planes for each training patch, but it does not share those volumes across ranks.

Sources: [NAFNet code and paper](https://github.com/megvii-research/NAFNet),
[Noise2Inverse360](https://github.com/tomography/Noise2Inverse360), and [license notices](../NOTICE.md).
