# NAFNet for ALS micro-CT

## Install on NERSC

Clone this private repository using your normal GitHub authentication. Install it
in the existing PyTorch `n2i` environment, leaving the ALS reconstruction kernel
and TomoPy environment in place:

```bash
cd ~/microct-n2i-source
git clone https://github.com/remyferu/KRG-alsDenoise.git
cd KRG-alsDenoise
~/microct-n2i-envs/n2i/bin/python -m pip install --no-deps -e .
export MPLBACKEND=Agg
~/microct-n2i-envs/n2i/bin/denoise train -h
```

Help should include `--architecture`. NAFNet needs PyTorch >= 2.0; no NAFNet repo
checkout, BasicSR installation or custom CUDA kernel is required. Installing this
package in the same environment updates its `denoise` entry point to this source.
The original source checkout is preserved. Use the n2i Python for training and
inference, and keep the ALS environment for reconstruction.

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

## Train when ready

This integration does not supply trained weights or run training automatically.
Launch the following yourself in an allocated GPU compute session:

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

The existing symmetric paired L1 objective, LCL warmup and checkpoint criteria
are retained. Patience still counts epochs without a lower raw evaluation L1
loss, using the final rank's metrics; it is not based on a moving average. The
evaluation pass still uses random patches from the training dataset. No learning
rate scheduler or separate held-out set has been added. Batch progress messages
make long epochs visible in the log.

## Infer and compare fine features

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

Architecture alone does not guarantee tiny-feature retention. Compare the same
2–5 pixel features against the original reconstructions, inspect residuals and
measure feature contrast/diameter. Compare the original U-Net and this model at
the same slice count, data splits and training budget. Neither minimum LCL nor
maximum EDGE proves accurate geometry.

## Notebook and validation

The v3.3 notebook uses NAFNet by default for fresh 2.5D runs and retains the v3.2
reconstruction, COR, queue, zoom, live log and history features. Choose a new run
label for a new architecture. It requires this repository's backend, checked via
`--architecture` in CLI help. Changing its 2.5D `TRAIN_ARCHITECTURE` to `unet`
selects the baseline. The original v3.2 notebook is unchanged.

The default test suite checks shapes for 5/7/9 slices, center identity, LayerNorm,
checkpoint compatibility, crop pairing, blend borders, and actual synthetic TIFF
slice/volume inference. Training tests require `RUN_TRAINING_TESTS=1` and are
skipped by default. `examples/profile_nafnet.py` is an optional user-run synthetic
VRAM benchmark, not an estimate of scan accuracy or A100 throughput.

Four GPU ranks still hold separate copies of both full stacks in CPU RAM. For
624 × 2560 × 2560 float32 paired volumes this is about 122 GiB before other
allocations. Cropping before neighbor indexing avoids copying full detector
planes for each training patch, but it does not share those volumes across ranks.

Credit: [NAFNet code and paper](https://github.com/megvii-research/NAFNet),
[Noise2Inverse360](https://github.com/tomography/Noise2Inverse360), and [license notices](../NOTICE.md).
