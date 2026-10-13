"""Shared configuration and output naming for denoising commands."""
from pathlib import Path
import yaml


def read_config(args):
    with open(args.config,encoding='utf-8') as stream:
        params = yaml.safe_load(stream)
    mode = getattr(args,'mode',None) or params['train'].get('mode','2.5d')
    return params, mode


def output_path(params, mode, kind, override=None):
    if override:
        return str(override)
    name = params['dataset']['full_recon_name']
    stem = name[:-4] if name.endswith('_rec') else name
    suffix = '_nafnet25d' if params['train'].get('architecture')=='nafnet25d' else ''
    name = stem+('_denoised_slices' if kind=='slice' else '_denoised_volume_'+mode)+suffix
    return str(Path(params['dataset']['directory_to_reconstructions'])/name)
