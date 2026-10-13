"""Create a fresh NAFNet config from an existing reconstructed scan config."""
import argparse
from pathlib import Path
import sys
import yaml

# This checkout-local example shares defaults with the ALS workflow package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'workflow/src'))
from krg_workflow.presets import training_settings, inference_settings


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
    train.update(training_settings(n_slices=n_slices, precision=precision))
    params.setdefault('infer', {}).update(inference_settings(train))
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
