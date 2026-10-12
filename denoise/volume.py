#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Denoise the entire CT volume using a trained Noise2Inverse model.
"""

import os
import time
import yaml
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from denoise.model_factory import load_model, forward_model
from denoise.data import TomoDatasetInfer
from denoise.data3d import TomoDataset3DInfer, _hann3d
from denoise.data_utils import InferenceBatchSizeOptimizer
from denoise import tiffs as tiffs_mod
from denoise import log


def run(args):

    # Read the YAML file
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)

    # Determine mode: CLI flag > YAML config > default 2.5d
    mode = getattr(args, 'mode', None) or params['train'].get('mode', '2.5d')

    # setup output directory
    full_recon_name = params['dataset']['full_recon_name']
    base_name = full_recon_name[:-4] if full_recon_name.endswith('_rec') else full_recon_name
    suffix = '_nafnet25d' if params['train'].get('architecture') == 'nafnet25d' else ''
    output_dir = getattr(args, 'output_dir', None) or (
        params['dataset']['directory_to_reconstructions'] + '/' + base_name + '_denoised_volume_' + mode + suffix)
    log.info("Inference mode: %s" % mode)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    _ckpt_map = {'val': 'best_val_model.pth', 'lcl': 'best_lcl_model.pth', 'edge': 'best_edge_model.pth'}
    ckpt_name = _ckpt_map[getattr(args, 'checkpoint', 'lcl')]
    model_dir = getattr(args, 'model_dir', None) or \
        params['dataset']['directory_to_reconstructions'] + '/TrainOutput'
    path_to_mdl = os.path.join(model_dir, ckpt_name)
    log.info("Using checkpoint: %s" % ckpt_name)
    ckpt = torch.load(path_to_mdl, map_location='cpu', weights_only=False)

    model = load_model(params, ckpt, mode).to(dev).eval()
    precision = params.get('infer', {}).get('precision', 'fp32')
    if os.path.exists(output_dir):
        raise RuntimeError('Inference output already exists: %s. Choose a new --output-dir.' % output_dir)
    os.mkdir(output_dir)

    log.info("Loading data into CPU memory, it will take a while ...")

    if mode == '3d':
        ds_test = TomoDataset3DInfer(params=params, start_slice=args.start_slice, end_slice=args.end_slice)
        psz_3d = ds_test.psz
        patch_shape = (psz_3d, psz_3d, psz_3d)

        optimal_batch_size = InferenceBatchSizeOptimizer(model=model, input_shape=patch_shape, device=dev,
                                                         max_batch_size=64, precision=precision, n_channels=1)
        stats = optimal_batch_size.profile()
        mbsz = stats['optimal_batch_size']

        workers = int(params['infer'].get('num_workers', 2))
        dl_test = DataLoader(dataset=ds_test, batch_size=mbsz, shuffle=False,
                             num_workers=workers, drop_last=False,
                             prefetch_factor=2 if workers else None, pin_memory=dev.type == 'cuda')

        # Online stitching — accumulate directly into acc/wacc to avoid
        # allocating a (N_patches, psz, psz, psz) array that can exceed RAM.
        window_type = params['infer'].get('window', 'hann')
        w3d = _hann3d(psz_3d, psz_3d, psz_3d) if window_type in ('hann', 'cosine') \
              else np.ones((psz_3d, psz_3d, psz_3d), dtype=np.float32)
        acc  = np.zeros((ds_test.D_pad, ds_test.H_pad, ds_test.W_pad), dtype=np.float32)
        wacc = np.zeros_like(acc)
        insert_cnt = 0

        log.info("Processing %d 3D patches ..." % len(ds_test))
        with torch.no_grad():
            for X in tqdm(dl_test):
                out = forward_model(model, X.to(dev), precision).cpu().squeeze(1).numpy()
                B = out.shape[0]
                for b in range(B):
                    d, h, w = ds_test.index[insert_cnt + b]
                    acc [d:d+psz_3d, h:h+psz_3d, w:w+psz_3d] += out[b] * w3d
                    wacc[d:d+psz_3d, h:h+psz_3d, w:w+psz_3d] += w3d
                insert_cnt += B

        log.info("Stitching 3D denoised volume ...")
        preds = (acc / (wacc + 1e-6))[:ds_test.D, :ds_test.H, :ds_test.W].copy()

    else:
        ds_test = TomoDatasetInfer(params=params, start_slice=args.start_slice, end_slice=args.end_slice)
        log.info("Loaded %d slices of size %dx%d" % (ds_test.vol.shape[0], ds_test.vol.shape[1], ds_test.vol.shape[2]))

        patch_shape = (params['train']['psz'], params['train']['psz'])
        optimal_batch_size = InferenceBatchSizeOptimizer(model=model, input_shape=patch_shape, device=dev,
                                                         max_batch_size=int(params['infer'].get('max_batch_size', 512)), precision=precision,
                                                         n_channels=params['train']['n_slices'])
        stats = optimal_batch_size.profile()
        mbsz = stats['optimal_batch_size']

        workers = int(params['infer'].get('num_workers', 4))
        dl_test = DataLoader(dataset=ds_test, batch_size=mbsz, shuffle=False,
                             num_workers=workers, drop_last=False,
                             prefetch_factor=6 if workers else None, pin_memory=dev.type == 'cuda')

        preds = np.zeros((dl_test.dataset.total_patches, params['train']['psz'], params['train']['psz']),
                         dtype=np.float32)
        log.info("Patch volume size: %dx%dx%d" % (preds.shape[0], preds.shape[1], preds.shape[2]))
        insert_cnt = 0

        log.info("Processing data ...")
        with torch.no_grad():
            for X, _ in tqdm(dl_test):
                output = forward_model(model, X.to(dev), precision).cpu().squeeze(dim=1).numpy()
                preds[insert_cnt:(insert_cnt + X.shape[0])] = output
                insert_cnt += X.shape[0]

        log.info("Stitching denoised data ...")
        preds = ds_test.stitch_predictions(preds, window=params['infer']['window'], keep_k_dim=False)

    # Rescale back to original intensity range
    preds = preds * params['dataset']['std4norm'] + params['dataset']['mean4norm']

    # Save volume
    log.info("Saving data to %s ..." % output_dir)
    if len(args.start_slice) == 0:
        tiffs_mod.save_stack(output_dir, preds)
    else:
        tiffs_mod.save_stack(output_dir, preds, offset=int(args.start_slice))

    log.info("Done.")
