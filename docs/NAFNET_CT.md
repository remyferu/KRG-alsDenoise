# NAFNet for ALS micro-CT

## Install on NERSC

Follow [NERSC setup](SETUP_NERSC.md) for environment installation, kernel
registration and compute allocation. The guide is also included in the v3.4
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
NAFNet architecture. New checkpoints include their architecture, slice count and
model options; all three commands check those against the config.

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
exactly; changing either starts a fresh window. Older checkpoints also start a
fresh window because they do not identify eligible post-warmup history. Existing
YAML without the new fields uses window 10 and `min_delta=0.001`.

The `[EarlyStopping]` log reports collection progress or the monitored mean and
patience count. History exports include training/evaluation total loss and the
logged monitor curve. `MOVING_AVERAGE_EPOCHS` changes display smoothing only;
`train.early_stopping_window` controls stopping.

Evaluation still samples the training dataset and uses the final GPU rank’s
metrics. The learning rate remains constant. Future objective changes belong
in the shared loss assembly so optimization and evaluation monitor the same
quantity; reporting metrics do not enter the monitor.

### Reference L2 and SSIM losses

Each epoch also reports `[Train]` and `[Val]` L2 loss (mean squared error) and
SSIM loss (1 - SSIM). These are detached reference measurements computed from
the same paired predictions; they do not enter backpropagation, checkpoint
selection, warmup or patience. Lower values indicate closer agreement with the
noisy paired target. They do not measure clean-reference accuracy.

New metrics average the two prediction directions. The existing L1/LCL log lines
retain their original convention of summing those directions, so their scale is
unchanged. Epoch summaries average the batch values from the final GPU rank,
matching the scope of the existing logs. For 2.5D, both metrics compare the output
with the opposite split's center slice. For 3D, MSE uses the whole patch and SSIM
uses the center axial plane, as do the existing LCL/EDGE diagnostics.

SSIM is evaluated in FP32 with an 11 × 11 Gaussian window, sigma 1.5, population
covariance, and valid windows. It uses a fixed `train.ssim_data_range` measured
in normalized intensity units. If absent, rank 0 computes the span from the
minimum/maximum of both normalized training stacks once and saves it in the
YAML. Resume reuses it. Set a positive range explicitly for controlled comparisons;
different ranges change SSIM values. Inputs are neither clipped nor rescaled per
batch, and SSIM loss is not a percentage.

Each completed epoch updates `TrainOutput/results/reference_metrics.csv` (or the
equivalent custom output directory), including the range and window settings.
Reference history is retained in `resume.pth`. Resuming an older checkpoint starts
reference history at the next epoch; earlier epochs have no reference measurements.
The v3.4 history cell also extracts the optional metrics into its CSV and adds two
plot panels when they are present. Old L1/LCL-only logs still work.

## Inference and feature assessment

```bash
denoise slice --config /path/to/sample_nafnet_config.yaml \
  --model-dir /path/to/TrainOutput_NAF --checkpoint val --slice-number 300
denoise volume --config /path/to/sample_nafnet_config.yaml \
  --model-dir /path/to/TrainOutput_NAF --checkpoint val
```

NAFNet output directories end in `_nafnet25d`. Use `--output-dir` for a distinct
comparison directory, especially when comparing `val`, `lcl`, and `edge` models.
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
for the baseline, using a new run label when changing architecture. CLI help is
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
