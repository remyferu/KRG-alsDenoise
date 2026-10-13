"""Training configuration, checkpoint and denoised-volume validation."""
from pathlib import Path
import numpy as np
import tifffile
from .files import tiff_paths


def validate_denoise_config(cfg, run_dir, full_dir, even_dir, odd_dir, ncols, *, nrows=None, normalized=False, volume=False):
    """Check dataset identity and mode-specific settings before launching the CLI."""
    if not isinstance(cfg, dict) or not all((isinstance(cfg.get(k), dict) for k in ('dataset', 'train', 'infer'))):
        raise ValueError('The YAML must contain dataset, train, and infer mappings.')
    expected = dict(directory_to_reconstructions=str(run_dir), full_recon_name=full_dir.name, sub_recon_name0=even_dir.name, sub_recon_name1=odd_dir.name)
    for key, value in expected.items():
        if cfg['dataset'].get(key) != value:
            raise ValueError(f'Wrong configuration path: {key}. Rerun Configure training for the selected run.')
    train = cfg['train']
    mode = train.get('mode', '2.5d')
    if mode not in ('2.5d', '3d'):
        raise ValueError('Training mode must be 2.5d or 3d.')
    for key in ('psz', 'n_slices', 'mbsz', 'maxep'):
        if type(train.get(key)) is not int or train[key] <= 0:
            raise ValueError(f'train.{key} must be a positive integer.')
    if train.get('z_stride', 1) != 1:
        raise ValueError('This full-resolution workflow requires z_stride=1.')
    if mode == '2.5d':
        if train['psz'] >= ncols or train['psz'] % 8:
            raise ValueError('psz must be smaller than the image width and divisible by 8.')
        if train['n_slices'] % 2 == 0:
            raise ValueError('n_slices must be odd.')
    else:
        for key in ('psz_3d', 'nb_patches_3d', 'n_blocks_3d', 'start_filts_3d'):
            if type(train.get(key)) is not int or train[key] <= 0:
                raise ValueError(f'train.{key} must be a positive integer.')
        patch = train['psz_3d']
        if nrows is None or patch > min(nrows, ncols):
            raise ValueError('psz_3d must fit the volume depth and image width; provide nrows.')
        if train['n_blocks_3d'] > patch.bit_length() or patch % 2 ** train['n_blocks_3d']:
            raise ValueError('psz_3d must be divisible by 2**n_blocks_3d, following the upstream model contract.')
    architecture = train.get('architecture', 'unet3d' if mode == '3d' else 'unet')
    if architecture not in (('unet', 'nafnet25d') if mode == '2.5d' else ('unet3d',)):
        raise ValueError('Architecture is incompatible with the selected mode.')
    if architecture == 'nafnet25d' and train['n_slices'] < 3:
        raise ValueError('NAFNet requires an odd slice count >= 3.')
    if train.get('precision', 'fp32') not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16.')
    for key in ('warmup', 'patience'):
        if type(train.get(key, 0)) is not int or train.get(key, 0) < 0:
            raise ValueError(f'train.{key} must be a nonnegative integer.')
    if not isinstance(train.get('lr'), (int, float)) or not np.isfinite(train['lr']) or train['lr'] <= 0:
        raise ValueError('train.lr must be finite and positive.')
    overlap = cfg['infer'].get('overlap')
    if not isinstance(overlap, (int, float)) or not np.isfinite(overlap) or (not 0 <= overlap < 1):
        raise ValueError('infer.overlap must be in [0, 1).')
    if cfg['infer'].get('window') not in ('hann', 'cosine', 'uniform'):
        raise ValueError('infer.window must be hann, cosine, or uniform.')
    if normalized:
        for key in ('mean4norm', 'std4norm'):
            value = cfg['dataset'].get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'Missing or invalid training normalization: {key}. Restore the training YAML.')
        if cfg['dataset']['std4norm'] <= 0:
            raise ValueError('Training normalization standard deviation must be positive.')
    return cfg


def validate_checkpoint(train_dir, checkpoint):
    """Resolve a supported checkpoint without starting inference or creating outputs."""
    names = {'lcl': 'best_lcl_model.pth', 'val': 'best_val_model.pth', 'edge': 'best_edge_model.pth'}
    if checkpoint not in names:
        raise ValueError('CHECKPOINT must be lcl, val, or edge.')
    path = Path(train_dir) / names[checkpoint]
    if not path.is_file():
        raise FileNotFoundError(f'Requested checkpoint is not saved yet: {path}')
    return path


def validate_denoised_volume(folder, nrows, ncols):
    """Check output names, shape, floating dtype, and finite pixels one slice at a time."""
    paths = tiff_paths(folder)
    if [p.name for p in paths] != [f'{i:05d}.tiff' for i in range(nrows)]:
        raise ValueError('Denoised volume is incomplete or contains unexpected TIFFs; inspect its log.')
    for path in paths:
        image = tifffile.imread(path)
        if image.shape != (ncols, ncols) or image.dtype.kind != 'f' or (not np.isfinite(image).all()):
            raise ValueError(f'Invalid denoised image: {path}')
    return paths
