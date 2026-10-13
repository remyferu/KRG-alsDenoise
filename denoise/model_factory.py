"""Shared model construction and checkpoint compatibility for all commands."""
from copy import deepcopy
import torch
from denoise.model import unet_ns_gn
from denoise.model3d import unet3d
from denoise.nafnet25d import NAFNet25D


NAF_DEFAULTS = dict(width=32, stem_width=16, detail_blocks=4,
                    encoder_blocks=[2, 2], middle_blocks=6, decoder_blocks=[2, 2], distance_aware=True)


def model_spec(params, mode=None):
    train = params['train']
    mode = mode or train.get('mode', '2.5d')
    architecture = train.get('architecture', 'unet3d' if mode == '3d' else 'unet')
    if mode not in ('2.5d', '3d'):
        raise ValueError('Unknown training mode: %s' % mode)
    spec = dict(version=2, mode=mode, architecture=architecture)
    if mode == '3d':
        if architecture != 'unet3d':
            raise ValueError('3d mode requires architecture: unet3d.')
        spec['options'] = dict(n_blocks=int(train.get('n_blocks_3d', 3)),
                               start_filts=int(train.get('start_filts_3d', 32)))
    else:
        n_slices = int(train['n_slices'])
        if n_slices <= 0 or n_slices % 2 != 1:
            raise ValueError('n_slices must be a positive odd integer.')
        spec['n_slices'] = n_slices
        if architecture == 'nafnet25d':
            options = deepcopy(NAF_DEFAULTS)
            supplied = train.get('nafnet', {})
            unknown = set(supplied) - set(options)
            if unknown:
                raise ValueError('Unknown NAFNet options: %s' % sorted(unknown))
            options.update(supplied)
            for key in ('encoder_blocks', 'decoder_blocks'):
                options[key] = list(options[key])
            spec['options'] = options
        elif architecture == 'unet':
            spec['options'] = dict(start_filter_size=16, channels_per_group=8)
        else:
            raise ValueError('Unknown 2.5d architecture: %s' % architecture)
    return spec


def build_model(params, mode=None):
    spec = model_spec(params, mode)
    if spec['architecture'] == 'nafnet25d':
        return NAFNet25D(n_slices=spec['n_slices'], **spec['options'])
    if spec['architecture'] == 'unet3d':
        return unet3d(in_channels=1, out_channels=1, **spec['options'])
    return unet_ns_gn(ich=spec['n_slices'], **spec['options'])


def validate_checkpoint(checkpoint, params, mode=None):
    expected = model_spec(params, mode)
    saved = checkpoint.get('model_spec')
    if saved is None:
        # Preserve legacy U-Net checkpoints; they have no architecture metadata.
        if expected['architecture'] == 'nafnet25d':
            raise ValueError('This checkpoint has no NAFNet metadata. Start a fresh NAFNet run; '
                             'existing U-Net checkpoints cannot initialize NAFNet.')
    elif saved != expected:
        raise ValueError('Checkpoint architecture/options differ from the config. '
                         'Use its original config or start a fresh run. '
                         'Saved: %s; requested: %s' % (saved, expected))


def load_model(params, checkpoint, mode=None):
    validate_checkpoint(checkpoint, params, mode)
    model = build_model(params, mode)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def forward_model(model, x, precision='fp32'):
    if precision not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16.')
    enabled = precision == 'bf16' and x.device.type == 'cuda'
    if enabled and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is unavailable on this GPU; set precision: fp32.')
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        return model(x).float()
