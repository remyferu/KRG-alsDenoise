# data_importers

Scripts that convert external data formats into the per-slice directory structure
expected by Noise2Inverse360.

Noise2Inverse360 expects each sub-reconstruction to be a **directory of individual
2D TIFF slices** named `00000.tiff`, `00001.tiff`, … so that `tiffs.glob()` can
sort and load them in order.

## Converters

| Script | Input | Use case |
|--------|-------|----------|
| `3dtiff_converter.py` | 3D TIFF stack (single file, shape `[Z, H, W]`) | Reconstructions already saved as a single volumetric TIFF |

## Adding a new converter

1. Place the script here with a descriptive name (e.g. `hdf5_converter.py`).
2. Output per-slice TIFFs named `%05d.tiff` into a subdirectory alongside the source data.
3. Add a row to the table above.
