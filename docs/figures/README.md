# NAFNet25D architecture

[SVG vector diagram](nafnet25d-architecture.svg) ·
[PNG, 3840 × 5760](nafnet25d-architecture.png) ·
[Zoom and pan viewer](nafnet25d-architecture.html)

The diagram documents `denoise/nafnet25d.py` with the repository defaults:
five input slices, 32-channel fused features, a four-block detail branch,
two encoder and decoder levels, and six bottleneck blocks. Tensor dimensions
show 256 × 256 input patches. Arbitrary dimensions use replicate padding and
output cropping. Actual run options are recorded in the training YAML and
checkpoint's `model_spec`.

The poster contains the full network graph, a per-neighbor axial fusion
expansion, both updates within a NAFBlock, and the paired training objective.
It distinguishes additive skips from concatenation and identifies initialization
and precision behavior. The parameter total is 989,745 for these defaults.

Open the HTML file in a browser with the SVG and PNG beside it. The viewer
supports zoom controls and drag-to-pan; the SVG remains sharp at any scale.
SVG text is outlined to preserve typography without installed fonts. The PNG
can be inserted into presentations or documents.

Regenerate from the checkout in an environment with PyTorch and Matplotlib:

```bash
python scripts/render_nafnet_architecture.py
```

The renderer checks parameter counts, propagates an odd-sized input through
the model on PyTorch's `meta` device, and checks label bounds before exporting.
It does not train the model. The companion JSON records source hash, default
options, parameter counts and the shape check. Architecture annotations are
maintained in the renderer and must be revised when the model changes.

NAFBlock is adapted from MEGVII NAFNet under the MIT license. See
[license notices](../../NOTICE.md) and
[the bundled license](../../third_party/NAFNet_LICENSE.txt).
