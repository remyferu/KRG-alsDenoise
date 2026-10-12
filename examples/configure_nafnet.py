"""Create a fresh NAFNet config from an existing reconstructed scan config."""
import argparse
from pathlib import Path
import yaml


def configure(source, destination, n_slices=5, precision='bf16'):
    source, destination = Path(source), Path(destination)
    if n_slices < 3 or n_slices % 2 != 1:
        raise ValueError('Use an odd n_slices >= 3.')
    if precision not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16.')
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Choose a new output config; the existing config is preserved.')
    params = yaml.safe_load(source.read_text())
    train = params.setdefault('train', {})
    train.update(mode='2.5d', architecture='nafnet25d', n_slices=n_slices,
                 psz=256, mbsz=4, lr=0.001, warmup=2000, maxep=2000,
                 patience=100, z_stride=1, precision=precision, num_workers=0,
                 log_every=50)
    train['nafnet'] = dict(width=32, stem_width=16, detail_blocks=4,
                            encoder_blocks=[2, 2], middle_blocks=6, decoder_blocks=[2, 2])
    params.setdefault('infer', {}).update(precision=precision, num_workers=0,
                                         overlap=0.5, window='cosine', max_batch_size=32)
    destination.write_text(yaml.safe_dump(params, sort_keys=False))
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-slices', type=int, default=5)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    args = parser.parse_args()
    configure(args.input, args.output, args.n_slices, args.precision)
    print('Created:', args.output)
    print('Use a NEW training output directory; do not resume a U-Net checkpoint.')
