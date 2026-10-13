#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Noise2Inverse model training using distributed data parallel (DDP).
"""

import os
import time
import yaml
import json
from pathlib import Path
from denoise.configuration import read_config
from denoise.checkpoints import initial_checkpoint as load_initial_checkpoint, save_checkpoint
from denoise.training_setup import training_dataset, training_loader, validation_loader
from denoise.early_stopping import MovingAverageEarlyStopping
from denoise.training_controls import (plateau_options, ema_options, PlateauController,
                                       ModelEMA, update_training_controls)
from denoise.validation import EpochMetrics, validation_options
from denoise.evaluation import evaluate_pairs
import datetime
import numpy as np
import torch
from copy import deepcopy
from matplotlib import pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP

from denoise.model_factory import build_model, model_spec, forward_model
from denoise.loss import LCL, paired_objective
from denoise.reference_metrics import (SSIMLoss, resolve_ssim_data_range,
                                       paired_reference_metrics, save_reference_history)
from denoise.utils import save2img
from denoise import log


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(args):

    params, mode = read_config(args)
    stopping = MovingAverageEarlyStopping.from_config(params['train'])
    lr_policy = plateau_options(params['train'])
    ema_policy = ema_options(params['train'])
    validation_options(params['train'], mode)

    START_TIME = time.time()

    # setup distributed training using PyTorch's DDP framework
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if getattr(args, 'architecture', None):
        params['train']['architecture'] = args.architecture
    spec = model_spec(params, mode)
    precision = params['train'].get('precision', 'fp32')
    if precision not in ('fp32', 'bf16'):
        raise ValueError('train.precision must be fp32 or bf16.')
    # Bind each rank before initializing NCCL or entering its first barrier.
    torch.cuda.set_device(local_rank)
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is unavailable on this GPU; set train.precision: fp32.')
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '1'))))

    path_to_reconstructions = params['dataset']['directory_to_reconstructions']
    odir = getattr(args, 'output_dir', None) or (path_to_reconstructions + '/TrainOutput')
    initial_checkpoint = load_initial_checkpoint(params, args, mode, odir)
    if getattr(args, 'resume', False):
        restored = stopping.load_state_dict(initial_checkpoint.get('early_stopping_state'))
        if restored and stopping.last_epoch != initial_checkpoint['epoch']:
            raise ValueError('Early-stopping state does not match the resume epoch.')
    # Validate model options before loading the large reconstruction stacks.
    model = build_model(params, mode).cuda()

    # Use a 3-hour timeout to accommodate large dataset loading and normalization
    _timeout = datetime.timedelta(hours=3)
    if torch.distributed.is_nccl_available():
        import inspect
        device_args = {}
        if 'device_id' in inspect.signature(torch.distributed.init_process_group).parameters:
            device_args['device_id'] = torch.device('cuda', local_rank)
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size,
                                            timeout=_timeout, **device_args)
    else:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size, timeout=_timeout)

    # create directory containing training results in the directory of the reconstructions
    if rank == 0:
        if getattr(args, 'resume', False):
            if not os.path.isdir(odir):
                raise RuntimeError("--resume specified but TrainOutput not found: %s" % odir)
            if not os.path.isdir(f'{odir}/results'):
                os.mkdir(f'{odir}/results')
        else:
            os.mkdir(odir)
            os.mkdir(f'{odir}/results')

    torch.distributed.barrier()

    log.info("local rank %d (global rank %d) of a world size %d started" % (local_rank, rank, world_size))

    log.info("Training mode: %s" % mode)
    log.info('Architecture: %s | precision: %s' % (spec['architecture'], precision))

    # Save mode into config so inference commands can read it (rank 0 only)
    if rank == 0:
        import yaml as _yaml
        with open(args.config, 'r') as _f:
            _cfg = _yaml.safe_load(_f)
        _cfg['train']['mode'] = mode
        _cfg['train']['architecture'] = spec['architecture']
        _cfg['train']['early_stopping_window'] = stopping.window
        _cfg['train']['min_delta'] = stopping.min_delta
        with open(args.config, 'w') as _f:
            _yaml.safe_dump(_cfg, _f, default_flow_style=False, sort_keys=False)
    torch.distributed.barrier()

    log.info("Loading data into CPU memory, it will take a while ...")

    ds_train = training_dataset(params, args.config, mode)
    fixed_plan = getattr(ds_train, 'validation_plan', None)
    if getattr(args, 'resume', False) and initial_checkpoint.get('validation_plan') != fixed_plan:
        raise ValueError('Validation geometry, data or normalization changed. Use a new training output directory.')
    if len(ds_train) < world_size:
        raise ValueError('Training dataset must provide at least one sample per GPU rank.')
    if rank == 0:
        if fixed_plan is not None:
            Path(odir, 'validation.json').write_text(json.dumps(fixed_plan, indent=2)+'\n', encoding='utf-8')
            log.info('Fixed validation: %d patches; held-out planes %s; no overlapping training stacks.' %
                     (len(fixed_plan['coordinates']), fixed_plan['validation_planes']))
            log.info('Metric aggregation: scope=all_ranks; weighting=patches; validation=fixed_held_out')
        else:
            log.info('Metric aggregation: scope=all_ranks; weighting=patches; validation=training_resampling')

    # Fix the reference SSIM intensity span for every batch/epoch and future resumes.
    ssim_range_tensor = torch.zeros(1, dtype=torch.float64, device='cuda')
    if rank == 0:
        ssim_range_tensor.fill_(resolve_ssim_data_range(params['train'], ds_train))
    torch.distributed.broadcast(ssim_range_tensor, src=0)
    ssim_data_range = float(ssim_range_tensor.item())
    reference_ssim = SSIMLoss(ssim_data_range).cuda()
    log.info('Reference metrics only: L2=MSE, SSIM loss=1-SSIM; '
             'ssim_data_range=%.12g; window=11; symmetric directional means' % ssim_data_range)
    if mode == '3d':
        log.info('Reference 3D MSE uses the full patch; SSIM uses its center axial plane.')

    # Only rank 0 writes normalization and reference metric settings.
    if rank == 0:
        from denoise.data import save_normalization_value
        log.info("Saving training mean and std to config for inference")
        save_normalization_value(config_file=args.config, mean=ds_train.split0_mean, std=ds_train.split0_std)
        with open(args.config) as stream:
            saved_config = yaml.safe_load(stream)
        saved_config['train']['ssim_data_range'] = ssim_data_range
        with open(args.config, 'w') as stream:
            yaml.safe_dump(saved_config, stream, sort_keys=False)
    torch.distributed.barrier()

    train_sampler, dl_train = training_loader(params, ds_train, mode)
    dl_val = validation_loader(params, ds_train, dl_train, rank, world_size)

    log.info("Loaded %d samples into CPU memory for training." % len(ds_train))

    # Initialize model
    n_slices = params['train']['n_slices']

    optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['lr'])

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    log.info("Number of model parameters: %s" % f"{count_parameters(model):,}")

    # loss functions and warmup criteria
    criterion = torch.nn.L1Loss()
    criterion_lcl = LCL()
    beta = .01
    z_stride = params['train'].get('z_stride', 1)
    warmup = params['train']['warmup'] // max(1, z_stride)

    # training state (overwritten on --resume)
    model_updates = 0
    start_epoch = 1
    continue_warmup = warmup > 0
    train_loss, val_loss = [], []
    edge_values = []
    train_lcl_loss, val_lcl_loss = [], []
    reference_metric_history = []
    best_val_loss, best_edge, best_lcl_loss = np.inf, 0, np.inf
    best_val_epoch, best_edge_epoch, best_lcl_epoch = 0, 0, 0
    patience = stopping.patience
    center_idx = n_slices // 2  # used only in 2.5d mode

    if getattr(args, 'resume', False):
        resume_path = f"{odir}/resume.pth"
        if not os.path.exists(resume_path):
            raise RuntimeError("--resume specified but no resume checkpoint found at: %s" % resume_path)
        ckpt = initial_checkpoint
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch     = ckpt['epoch'] + 1
        model_updates   = ckpt['model_updates']
        best_val_loss   = ckpt['best_val_loss']
        best_lcl_loss   = ckpt['best_lcl_loss']
        best_edge       = ckpt['best_edge']
        best_val_epoch  = ckpt['best_val_epoch']
        best_lcl_epoch  = ckpt['best_lcl_epoch']
        best_edge_epoch = ckpt['best_edge_epoch']
        train_loss      = ckpt['train_loss']
        val_loss        = ckpt['val_loss']
        train_lcl_loss  = ckpt['train_lcl_loss']
        val_lcl_loss    = ckpt['val_lcl_loss']
        edge_values     = ckpt['edge_values']
        reference_metric_history = ckpt.get('reference_metric_history', [])
        continue_warmup = ckpt['continue_warmup']
        if ckpt.get('warmup_updates') != warmup:
            raise ValueError('Warmup length changed. Use a new training label or output directory.')
        if rank == world_size - 1 and not restored:
            log.info('Early-stopping state absent or policy changed; collecting a fresh '
                     'post-warmup window and resetting patience. Raw checkpoints are retained.')
        log.info("Resuming training from epoch %d (model_updates=%d)" % (start_epoch, model_updates))
    elif getattr(args, 'finetune', None):
        finetune_path = args.finetune
        if os.path.isdir(finetune_path):
            finetune_path = os.path.join(finetune_path, 'best_val_model.pth')
        if not os.path.exists(finetune_path):
            raise RuntimeError("--finetune: checkpoint not found: %s" % finetune_path)
        ckpt = initial_checkpoint
        model.module.load_state_dict(ckpt['model_state_dict'])
        log.info("Fine-tuning from: %s (training state reset from scratch)" % finetune_path)
    else:
        log.info('Initializing model from scratch')

    controller = PlateauController(optimizer, lr_policy, stopping.min_delta, stopping.window)
    if getattr(args, 'resume', False):
        controller.load_state_dict(ckpt.get('lr_scheduler_state'))
    ema = ModelEMA(model.module, ema_policy['decay']) if ema_policy['enabled'] else None
    best_ema_total, best_ema_epoch = np.inf, 0
    if ema is not None and getattr(args, 'resume', False):
        if ema.load_state_dict(ckpt.get('ema_state')):
            best_ema_total, best_ema_epoch = ckpt.get('best_ema_total', np.inf), ckpt.get('best_ema_epoch', 0)
        else:
            log.info('EMA state absent or decay changed; seeding a new average from resumed weights.')
    if rank == world_size-1:
        log.info('Learning-rate schedule: %s | EMA: %s' % (lr_policy, ema_policy))

    if rank == world_size - 1:
        log.info('Early-stopping monitor: metric=eval_total_loss; window=%d; '
                 'min_delta=%.12g absolute objective units; '
                 'patience=%d; raw-score checkpoints unchanged' %
                 (stopping.window, stopping.min_delta, patience))

    # start training
    for epoch in range(start_epoch, params['train']['maxep'] + 1):

        train_metrics = EpochMetrics(('total', 'l1', 'lcl', 'l2', 'ssim'), 'cuda')
        warming_epoch = model_updates < warmup

        tick_ep = time.time()

        model.train()
        dl_train.sampler.set_epoch(epoch)

        # ---- training loop ----
        for batch_index, (X_mb, Y_mb) in enumerate(dl_train, 1):
            X_mb_dev = X_mb.cuda()
            Y_mb_dev = Y_mb.cuda()

            optimizer.zero_grad(set_to_none=True)
            pred_view1 = forward_model(model, X_mb_dev, precision)
            pred_view2 = forward_model(model, Y_mb_dev, precision)

            if mode == '3d':
                # 3D: inputs/outputs are [B, 1, D, H, W]; target is the full patch
                loss_view1 = criterion(pred_view1, Y_mb_dev)
                loss_view2 = criterion(pred_view2, X_mb_dev)
            else:
                # 2.5D: squeeze channel dim, compare against center slice
                loss_view1 = criterion(pred_view1.squeeze(dim=1), Y_mb_dev[:, center_idx])
                loss_view2 = criterion(pred_view2.squeeze(dim=1), X_mb_dev[:, center_idx])

            if model_updates < warmup:
                loss_lcl1 = torch.tensor(0.)
                loss_lcl2 = torch.tensor(0.)
                loss = paired_objective(loss_view1, loss_view2)
            else:
                if mode == '3d':
                    # Apply LCL on center slice of 3D output
                    D = pred_view1.shape[2]
                    loss_lcl1 = criterion_lcl(pred_view1[:, :, D // 2]) * beta
                    loss_lcl2 = criterion_lcl(pred_view2[:, :, D // 2]) * beta
                else:
                    loss_lcl1 = criterion_lcl(pred_view1) * beta
                    loss_lcl2 = criterion_lcl(pred_view2) * beta
                loss = paired_objective(loss_view1, loss_view2, loss_lcl1, loss_lcl2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model_updates += 1
            if ema is not None:
                ema.update(model.module, model_updates, warmup)
            l2_reference, ssim_reference = paired_reference_metrics(
                pred_view1, pred_view2, X_mb_dev, Y_mb_dev, mode, center_idx, reference_ssim)
            train_metrics.add(X_mb_dev.shape[0],
                              total=loss, l1=loss_view1+loss_view2, lcl=loss_lcl1+loss_lcl2,
                              l2=l2_reference, ssim=ssim_reference)
            log_every = int(params['train'].get('log_every', 50))
            if rank == world_size - 1 and log_every > 0 and (
                    batch_index == 1 or batch_index % log_every == 0 or batch_index == len(dl_train)):
                log.info('[Progress] epoch %d train batch %d/%d | updates %d' %
                         (epoch, batch_index, len(dl_train), model_updates))

        train_values, train_count = train_metrics.reduce()
        val_values, val_count, preview = evaluate_pairs(model.module, dl_val, mode, precision, 'cuda',
            criterion, criterion_lcl, beta, model_updates >= warmup, reference_ssim)
        ema_values = None
        if ema is not None and ema.num_updates > 0 and epoch % ema_policy['evaluate_every'] == 0:
            ema_values, _, _ = evaluate_pairs(ema.model, dl_val, mode, precision, 'cuda',
                criterion, criterion_lcl, beta, model_updates >= warmup, reference_ssim, reference=False)
        local_stop, reduced = update_training_controls(stopping, controller, epoch,
                                                       val_values['total'], warming_epoch)
        warmup_finished = model_updates >= warmup and continue_warmup
        if warmup_finished:
            continue_warmup = False

        ep_time = time.time() - tick_ep

        # Early stopping: rank world_size-1 decides, broadcasts to all ranks
        stop_tensor = torch.zeros(1, dtype=torch.int32, device='cuda')
        if rank == world_size - 1:
            if local_stop:
                stop_tensor.fill_(1)
        torch.distributed.broadcast(stop_tensor, src=world_size - 1)
        early_stop = bool(stop_tensor.item())

        if rank != world_size - 1:
            if early_stop:
                break
            continue

        log.info('Epoch %d' % epoch)
        for stage, values, count in (('Train', train_values, train_count), ('Val', val_values, val_count)):
            for key, label in (('total', 'Total loss'), ('l1', 'L1 loss'), ('lcl', 'LCL loss'),
                               ('l2', 'L2 loss'), ('ssim', 'SSIM loss')):
                log.info('[%s] %s: %.6f, samples=%d, rate: %.2fs/ep' %
                         (stage, label, values[key], count, ep_time))
        log.info('[Val] EDGE Value: %.6f' % val_values['edge'])
        log.info('[LearningRate] lr: %.12g; reduced: %s' % (optimizer.param_groups[0]['lr'], reduced))
        if ema_values is not None:
            log.info('[EMA] Total loss: %.6f; averaged updates: %d' % (ema_values['total'], ema.num_updates))

        reference_metric_history.append(dict(epoch=epoch, train_l2=train_values['l2'],
            eval_l2=val_values['l2'], train_ssim=train_values['ssim'],
            eval_ssim=val_values['ssim'], ssim_data_range=ssim_data_range,
            ssim_window_size=11, mode=mode))
        save_reference_history(os.path.join(odir, 'results', 'reference_metrics.csv'), reference_metric_history)

        train_loss.append(train_values['l1'])
        val_loss.append(val_values['l1'])
        train_lcl_loss.append(train_values['lcl'])
        val_lcl_loss.append(val_values['lcl'])
        edge_values.append(val_values['edge'])

        # Save the best model with the lowest lcl loss
        if val_values['lcl'] < best_lcl_loss:
            best_lcl_loss = val_values['lcl']
            best_lcl_epoch = epoch
            mdl_fname = f"{odir}/best_lcl_model.pth"
            save_checkpoint({
                'model_spec': deepcopy(spec),
                'model_state_dict': deepcopy(model.module.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict())
            }, mdl_fname)

        # Save the best model with the lowest val loss
        if val_values['l1'] < best_val_loss:
            best_val_loss = val_values['l1']
            best_val_epoch = epoch
            mdl_fname = f"{odir}/best_val_model.pth"
            save_checkpoint({
                'model_spec': deepcopy(spec),
                'model_state_dict': deepcopy(model.module.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict())
            }, mdl_fname)

        # Save the best model with the highest edge value
        if val_values['edge'] > best_edge:
            best_edge = val_values['edge']
            best_edge_epoch = epoch
            mdl_fname = f"{odir}/best_edge_model.pth"
            save_checkpoint({
                'model_spec': deepcopy(spec),
                'model_state_dict': deepcopy(model.module.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict())
            }, mdl_fname)

        if ema_values is not None and not warming_epoch and ema_values['total'] < best_ema_total:
            best_ema_total, best_ema_epoch = ema_values['total'], epoch
            save_checkpoint(dict(model_spec=deepcopy(spec), model_state_dict=deepcopy(ema.model.state_dict()),
                            epoch=epoch, eval_total=best_ema_total, ema_decay=ema.decay,
                            validation_plan=deepcopy(fixed_plan)), f'{odir}/best_ema_model.pth')

        # Warm up period
        if warmup_finished:
            best_edge, best_lcl_loss = 0, np.inf
            best_edge_epoch, best_lcl_epoch = 0, 0

        CRNT_TIME = time.time()
        log.info("[Info]  Training Time: %.2f seconds" % (CRNT_TIME - START_TIME))

        # option to view the denoising process during training
        if epoch % 5 == 0 and preview is not None:
            pred_view1, pred_view2, X_mb_dev, Y_mb_dev = preview
            ridx = np.random.randint(pred_view1.shape[0])
            if mode == '3d':
                D = pred_view1.shape[2]
                save2img(pred_view1[ridx, 0, D // 2].detach().cpu().numpy(), '%s/results/_%d_pred_view1.png' % (odir, epoch))
                save2img(pred_view2[ridx, 0, D // 2].detach().cpu().numpy(), '%s/results/_%d_pred_view2.png' % (odir, epoch))
                save2img(X_mb_dev[ridx, 0, D // 2].detach().cpu().numpy(), '%s/results/_%d_view1.png' % (odir, epoch))
                save2img(Y_mb_dev[ridx, 0, D // 2].detach().cpu().numpy(), '%s/results/_%d_view2.png' % (odir, epoch))
            else:
                save2img(pred_view1[ridx, -1].detach().cpu().numpy(), '%s/results/_%d_pred_view1.png' % (odir, epoch))
                save2img(pred_view2[ridx, -1].detach().cpu().numpy(), '%s/results/_%d_pred_view2.png' % (odir, epoch))
                save2img(X_mb_dev[ridx, int(n_slices // 2)].detach().cpu().numpy(), '%s/results/_%d_view1.png' % (odir, epoch))
                save2img(Y_mb_dev[ridx, int(n_slices // 2)].detach().cpu().numpy(), '%s/results/_%d_view2.png' % (odir, epoch))

        # Keep track of when/where the best model is
        log.info('Lowest model validation loss %.6f at epoch %d' % (best_val_loss, best_val_epoch))
        log.info('Lowest model LCL loss %.6f at epoch %d' % (best_lcl_loss, best_lcl_epoch))
        log.info('Highest model EDGE score %.6f at epoch %d' % (best_edge, best_edge_epoch))
        log.info('Number of model updates: %s' % f"{model_updates:,}")
        log.info('Is model warming up?: %s' % continue_warmup)
        log.info('Epoch includes warmup updates: %s' % warming_epoch)
        if stopping.current_mean is not None:
            patience_status = ('%d/%d' % (stopping.epochs_since_improvement, patience)
                               if patience > 0 else '%d (stopping disabled)' % stopping.epochs_since_improvement)
            log.info('[EarlyStopping] mean total loss: %.6f; best qualifying mean: %.6f at epoch %d; '
                     'epochs without improvement: %s' %
                     (stopping.current_mean, stopping.best_mean, stopping.best_epoch,
                      patience_status))
        elif continue_warmup:
            log.info('[EarlyStopping] Waiting for warmup to finish.')
        else:
            log.info('[EarlyStopping] Collecting post-warmup epochs: %d/%d' %
                     (len(stopping.recent_losses), stopping.window))

        # View the training/validation loss during training
        if epoch % 5 == 0:
            plt.figure(figsize=(12, 8))
            plt.title("Training Progress")
            plt.plot(train_loss[:], label="Training Loss")
            plt.plot(val_loss[:], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{odir}/results/__model_training.png')
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.title("Training Progress")
            plt.plot(train_lcl_loss[:], label="Training Loss")
            plt.plot(val_lcl_loss[:], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{odir}/results/__model_lcl_training.png')
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.title("Training Progress")
            plt.plot(edge_values[:])
            plt.xlabel("Epoch")
            plt.ylabel("EDGE Gradient")
            plt.savefig(f'{odir}/results/__edge_training.png')
            plt.close()

        # Save resume checkpoint (overwritten each epoch; enables --resume after interruption)
        save_checkpoint({
            'training_version': 2,
            'model_spec': deepcopy(spec),
            'model_state_dict': deepcopy(model.module.state_dict()),
            'optimizer_state_dict': deepcopy(optimizer.state_dict()),
            'epoch': epoch,
            'model_updates': model_updates,
            'best_val_loss': best_val_loss,
            'best_lcl_loss': best_lcl_loss,
            'best_edge': best_edge,
            'best_val_epoch': best_val_epoch,
            'best_lcl_epoch': best_lcl_epoch,
            'best_edge_epoch': best_edge_epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_lcl_loss': train_lcl_loss,
            'val_lcl_loss': val_lcl_loss,
            'edge_values': edge_values,
            'reference_metric_history': reference_metric_history,
            'continue_warmup': continue_warmup,
            'warmup_updates': warmup,
            'epochs_since_improvement': stopping.epochs_since_improvement,
            'early_stopping_state': stopping.state_dict(),
            'validation_plan': deepcopy(fixed_plan),
            'lr_scheduler_state': controller.state_dict(),
            'ema_state': None if ema is None else ema.state_dict(),
            'best_ema_total': best_ema_total,
            'best_ema_epoch': best_ema_epoch,
        }, f"{odir}/resume.pth")

        if early_stop:
            log.info('Early stopping: no %d-epoch mean total loss improvement exceeding %.12g '
                     'for %d eligible epochs (best qualifying mean %.6f at epoch %d)' %
                     (stopping.window, stopping.min_delta, patience,
                      stopping.best_mean, stopping.best_epoch))
            break
