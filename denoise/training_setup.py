"""Dataset and loader construction, separate from the training loop."""
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .data import TomoDatasetTrain
from .data3d import TomoDataset3DTrain


def training_dataset(params, config, mode):
    dataset_type = TomoDataset3DTrain if mode=='3d' else TomoDatasetTrain
    return dataset_type(params=params,config_file=config)


def training_loader(params, dataset, mode):
    sampler = DistributedSampler(dataset=dataset,shuffle=True,drop_last=True)
    # Preserve worker defaults; new notebook presets use zero workers.
    workers = int(params['train'].get('num_workers',0 if mode=='3d' else 4))
    loader = DataLoader(dataset=dataset,batch_size=params['train']['mbsz'],sampler=sampler,
                      num_workers=workers,drop_last=False,prefetch_factor=None if workers==0 else 2,
                      pin_memory=True)
    return sampler, loader
