"""Checkpoint selection and loading; serialization format is unchanged."""
import os
import torch
from . import log
from .model_factory import load_model, validate_checkpoint

CHECKPOINT_NAMES = dict(val='best_val_model.pth',lcl='best_lcl_model.pth',edge='best_edge_model.pth')


def load_inference_model(params, args, mode, device):
    name = CHECKPOINT_NAMES[getattr(args,'checkpoint','lcl')]
    directory = getattr(args,'model_dir',None) or os.path.join(
        params['dataset']['directory_to_reconstructions'],'TrainOutput')
    log.info('Using checkpoint: %s' % name)
    checkpoint = torch.load(os.path.join(directory,name),map_location='cpu',weights_only=False)
    return load_model(params,checkpoint,mode).to(device).eval()


def initial_checkpoint(params, args, mode, output):
    path = None
    if getattr(args,'resume',False):
        path = os.path.join(output,'resume.pth')
    elif getattr(args,'finetune',None):
        path = args.finetune
        if os.path.isdir(path):
            path = os.path.join(path,CHECKPOINT_NAMES['val'])
    if path:
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        validate_checkpoint(checkpoint,params,mode)
        return checkpoint
    if os.path.exists(output):
        raise RuntimeError('Training output already exists: %s. Use --resume or a new --output-dir.' % output)
    return None
