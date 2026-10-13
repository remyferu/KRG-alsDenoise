"""Fresh-run defaults; existing training YAML is never replaced on resume."""
from copy import deepcopy

NAFNET = dict(width=32, stem_width=16, detail_blocks=4, encoder_blocks=[2, 2],
              middle_blocks=6, decoder_blocks=[2, 2])
RECONSTRUCTION = dict(als_chunk_sino=25, method='fbp', fc=0.75,
    preprocessing=dict(minimum_transmission=0.01, outlier_diff_1D=0.0,
                       outlier_size_1D=3, snr=3.0, la_size=51, sm_size=0),
    postprocessing=dict(ringSigma=0.0, ringLevel=8))


def reconstruction_settings(**overrides):
    settings = deepcopy(RECONSTRUCTION)
    settings.update(deepcopy(overrides))
    return settings


def training_settings(mode='2.5d', architecture='nafnet25d', **overrides):
    if mode not in ('2.5d', '3d'):
        raise ValueError('Mode must be 2.5d or 3d.')
    if mode == '3d':
        architecture = 'unet3d'
    if architecture not in (('nafnet25d', 'unet') if mode=='2.5d' else ('unet3d',)):
        raise ValueError('Architecture is incompatible with the selected mode.')
    train = dict(mode=mode, architecture=architecture, psz=256, n_slices=5,
                 mbsz=4 if mode=='2.5d' else 1, lr=.001, warmup=2000,
                 maxep=2000, patience=100, z_stride=1, num_workers=0, log_every=50,
                 early_stopping_window=10, min_delta=.001,
                 precision='bf16' if architecture=='nafnet25d' else 'fp32')
    if architecture=='nafnet25d':
        train['nafnet'] = deepcopy(NAFNET)
    if mode=='3d':
        train.update(psz_3d=96, nb_patches_3d=17600, n_blocks_3d=4, start_filts_3d=56)
    train.update(deepcopy(overrides))
    return train


def inference_settings(train):
    return dict(precision=train['precision'], num_workers=0, overlap=.5,
                window='cosine', max_batch_size=32)
