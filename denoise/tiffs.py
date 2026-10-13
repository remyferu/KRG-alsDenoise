import tifffile
from tqdm import tqdm
import numpy as np
from pathlib import Path
import re, sys


def natural_sorted(l):
    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", str(x))]
    return sorted(l, key=key)


def load_stack(paths, binning=1, use_tqdm=True, statistics_stop=None):
    """Load TIFFs as [slice, row, column] and return the stack, mean and std.

    binning subsamples each spatial axis by the given stride. Nonfinite values
    are replaced before computing statistics. statistics_stop restricts those
    statistics to the leading training planes while loading the full stack.
    """

    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    dtype = img0.dtype
    imgs = np.empty((len(paths), *img0.shape), dtype=dtype)

    progress = tqdm if use_tqdm else lambda x: x

    # Compute normalization statistics during loading to avoid a second full-volume pass.
    total_sum = 0.0
    total_sq  = 0.0
    total_n   = 0

    for i, p in progress(enumerate(paths)):
        img = tifffile.imread(str(p))[::binning, ::binning]
        img = np.nan_to_num(img)
        imgs[i] = img
        if statistics_stop is None:
            total_sum += img.sum()
            total_sq += (img**2).sum()
            total_n += img.size
        elif i < statistics_stop:
            total_sum += img.sum(dtype=np.float64)
            total_sq += np.square(img, dtype=np.float64).sum()
            total_n += img.size

    mean = total_sum / total_n
    var  = total_sq / total_n - mean**2
    std  = np.sqrt(var)

    return imgs, mean, std

def glob(dir_path):
    """List TIFF paths in natural filename order."""
    dir_path = Path(dir_path).expanduser().resolve()
    return natural_sorted(dir_path.glob("*.tif*"))


def save_stack(path, stack, offset=0, exist_ok=True, parents=False):
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=exist_ok, parents=parents)
    for i in tqdm(range(stack.shape[0])):
        opath = path / f"{i+offset:05d}.tiff"
        tifffile.imwrite(str(opath), stack[i])


def load_sino(paths, binning=1, dtype=None, flip_y=False):
    """Load TIFFs as [row, angle, column], with optional vertical reversal."""
    paths = list(paths)
    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    if dtype is None:
        dtype = img0.dtype
    imgs = np.empty((img0.shape[0], len(paths), img0.shape[1]), dtype=dtype)
    for i, p in tqdm(enumerate(paths)):
        # Angles in the middle, "up" in front, "right" at the back.
        if flip_y:
            imgs[:, i, :] = tifffile.imread(str(p))[::-binning, ::binning]
        else:
            imgs[:, i, :] = tifffile.imread(str(p))[::binning, ::binning]
    return imgs
