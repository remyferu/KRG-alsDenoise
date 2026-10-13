"""Fixed held-out axial patches and sample-weighted distributed metrics."""
import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def validation_options(train, mode='2.5d'):
    supplied = train.get('validation', {})
    if not isinstance(supplied, dict):
        raise ValueError('train.validation must be a mapping.')
    options = dict(enabled=mode == '2.5d', fraction=.1, samples=128, seed=1729)
    unknown = set(supplied) - set(options)
    if unknown:
        raise ValueError('Unknown validation options: %s' % sorted(unknown))
    options.update(supplied)
    if type(options['enabled']) is not bool:
        raise ValueError('validation.enabled must be boolean.')
    if (type(options['fraction']) not in (float, int)
            or not math.isfinite(options['fraction']) or not 0 < options['fraction'] < .5):
        raise ValueError('validation.fraction must be between 0 and 0.5.')
    if type(options['samples']) is not int or options['samples'] < 1:
        raise ValueError('validation.samples must be a positive integer.')
    if type(options['seed']) is not int or options['seed'] < 0:
        raise ValueError('validation.seed must be a nonnegative integer.')
    if options['enabled'] and mode != '2.5d':
        raise ValueError('Fixed axial validation is supported in 2.5d mode only.')
    return options


def axial_split(depth, n_slices, options):
    """Split raw planes, leaving a center-index gap that prevents stack overlap."""
    radius = n_slices // 2
    held_out = max(n_slices, int(math.ceil(depth * options['fraction'])))
    split = depth - held_out
    if split <= radius or held_out <= radius:
        raise ValueError('Volume depth is too small for disjoint training and validation stacks.')
    return split, split-radius, split+radius


def validation_plan(shape, n_slices, patch_size, options):
    depth, height, width = map(int, shape)
    if patch_size > min(height, width):
        raise ValueError('Validation patch size exceeds the reconstruction dimensions.')
    split, train_stop, validation_start = axial_split(depth, n_slices, options)
    random = np.random.default_rng(options['seed'])
    row_count, col_count = height-patch_size+1, width-patch_size+1
    available = (depth-validation_start)*row_count*col_count
    indices = random.choice(available, size=min(options['samples'], available), replace=False)
    centers = validation_start+indices//(row_count*col_count)
    rows = (indices//col_count)%row_count
    cols = indices%col_count
    return dict(version=1, options=dict(options), shape=list(shape), n_slices=n_slices,
                patch_size=patch_size, normalization_planes=[0, split],
                training_centers=[0, train_stop], validation_planes=[split, depth],
                coordinates=np.stack((centers, rows, cols), axis=1).tolist())


class FixedValidationDataset(Dataset):
    """Reuse normalized stacks with recorded crops and no augmentation."""
    def __init__(self, training):
        self.training = training
        self.plan = training.validation_plan

    def __len__(self):
        return len(self.plan['coordinates'])

    def __getitem__(self, index):
        center, row, col = self.plan['coordinates'][index]
        return self.training.crop_pair(center, row, col, augment=False)


class EvaluationShard(Sampler):
    """Partition evaluation indices without DistributedSampler's padding duplicates."""
    def __init__(self, dataset, rank, world_size):
        self.size, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.size, self.world_size))


class EpochMetrics:
    """Reduce sums and sample counts; empty evaluation shards contribute zero."""
    def __init__(self, keys, device):
        self.keys = tuple(keys)
        self.totals = torch.zeros(len(keys)+1, dtype=torch.float64, device=device)

    def add(self, count, **means):
        for index, key in enumerate(self.keys):
            value = torch.as_tensor(means[key], device=self.totals.device).detach()
            self.totals[index] += value.to(torch.float64)*count
        self.totals[-1] += count

    def reduce(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.totals, op=torch.distributed.ReduceOp.SUM)
        count = int(self.totals[-1].item())
        if count == 0:
            raise ValueError('No samples were evaluated across the GPU ranks.')
        return dict(zip(self.keys, (self.totals[:-1]/count).tolist())), count
