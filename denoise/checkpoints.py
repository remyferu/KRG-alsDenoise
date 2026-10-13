"""Checkpoint selection and loading for v3.5 training and inference."""
import os
from pathlib import Path
import uuid
import torch
from . import log
from .model_factory import load_model, validate_checkpoint

CHECKPOINT_NAMES = dict(val='best_val_model.pth',lcl='best_lcl_model.pth',edge='best_edge_model.pth',
                        ema='best_ema_model.pth')


def save_checkpoint(state, path):
    """Keep the previous checkpoint intact until its replacement is complete."""
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_inference_model(params, args, mode, device):
    name = CHECKPOINT_NAMES[getattr(args,'checkpoint','ema')]
    directory = getattr(args,'model_dir',None) or os.path.join(
        params['dataset']['directory_to_reconstructions'],'TrainOutput')
    path = Path(directory) / name
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    epoch = checkpoint.get('epoch')
    epoch = epoch if type(epoch) is int and epoch > 0 else None
    log.info('Using checkpoint: %s | training epoch: %s' % (name, epoch if epoch is not None else 'unknown (not recorded)'))
    model = load_model(params,checkpoint,mode).to(device).eval()
    model.inference_checkpoint = dict(filename=name, path=str(path.resolve()), epoch=epoch)
    return model


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
        if getattr(args, 'resume', False) and checkpoint.get('training_version') != 2:
            raise ValueError('v3.5 training checkpoints are required for resume. Start a new training output directory.')
        return checkpoint
    if os.path.exists(output):
        raise RuntimeError('Training output already exists: %s. Use --resume or a new --output-dir.' % output)
    return None
