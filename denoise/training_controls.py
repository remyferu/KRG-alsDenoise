"""Plateau learning-rate scheduling and exponential averages of model weights."""
from copy import deepcopy
import math
import torch


def plateau_options(train):
    options = dict(enabled=True, factor=.5, patience=25, cooldown=5,
                   min_lr=1e-5, refinement_epochs=30)
    supplied = train.get('lr_schedule', {})
    if not isinstance(supplied, dict):
        raise ValueError('train.lr_schedule must be a mapping.')
    unknown = set(supplied)-set(options)
    if unknown:
        raise ValueError('Unknown lr_schedule options: %s' % sorted(unknown))
    options.update(supplied)
    if type(options['enabled']) is not bool:
        raise ValueError('lr_schedule.enabled must be boolean.')
    if type(options['factor']) not in (int, float) or not 0 < options['factor'] < 1:
        raise ValueError('lr_schedule.factor must be between 0 and 1.')
    for key in ('patience', 'cooldown', 'refinement_epochs'):
        if type(options[key]) is not int or options[key] < 0:
            raise ValueError('lr_schedule.%s must be a nonnegative integer.' % key)
    if (type(options['min_lr']) not in (int, float) or not math.isfinite(options['min_lr'])
            or options['min_lr'] < 0):
        raise ValueError('lr_schedule.min_lr must be finite and nonnegative.')
    if options['enabled'] and options['min_lr'] >= train['lr']:
        raise ValueError('lr_schedule.min_lr must be below train.lr.')
    return options


def ema_options(train):
    options = dict(enabled=True, decay=.999, evaluate_every=1)
    supplied = train.get('ema', {})
    if not isinstance(supplied, dict):
        raise ValueError('train.ema must be a mapping.')
    unknown = set(supplied)-set(options)
    if unknown:
        raise ValueError('Unknown EMA options: %s' % sorted(unknown))
    options.update(supplied)
    if type(options['enabled']) is not bool:
        raise ValueError('ema.enabled must be boolean.')
    if type(options['decay']) not in (float, int) or not 0 < options['decay'] < 1:
        raise ValueError('ema.decay must be between 0 and 1.')
    if type(options['evaluate_every']) is not int or options['evaluate_every'] < 1:
        raise ValueError('ema.evaluate_every must be a positive integer.')
    return options


class PlateauController:
    """Schedule the post-warmup total-loss mean and protect a refinement interval."""
    def __init__(self, optimizer, options, min_delta, window=10):
        self.options = dict(options, min_delta=min_delta, window=window)
        self.optimizer = optimizer
        self.last_reduction_epoch = None
        self.scheduler = None
        if options['enabled']:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=options['factor'], patience=options['patience'],
                threshold=min_delta, threshold_mode='abs', cooldown=options['cooldown'],
                min_lr=options['min_lr'])

    def step(self, mean_total, epoch):
        if self.scheduler is None or mean_total is None:
            return False
        before = [group['lr'] for group in self.optimizer.param_groups]
        self.scheduler.step(mean_total)
        reduced = any(group['lr'] < old for group, old in zip(self.optimizer.param_groups, before))
        if reduced:
            self.last_reduction_epoch = epoch
        return reduced

    def ready_to_stop(self, epoch):
        return (self.last_reduction_epoch is None or
                epoch-self.last_reduction_epoch >= self.options['refinement_epochs'])

    def state_dict(self):
        return dict(version=1, options=dict(self.options), last_reduction_epoch=self.last_reduction_epoch,
                    scheduler=None if self.scheduler is None else self.scheduler.state_dict())

    def load_state_dict(self, state):
        if state is None or state.get('options') != self.options:
            return False
        if state.get('version') != 1:
            raise ValueError('Unsupported learning-rate scheduler state.')
        self.last_reduction_epoch = state['last_reduction_epoch']
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        return True


def update_training_controls(stopping, controller, epoch, total_loss, warming_up):
    stop = stopping.update(epoch, total_loss, warming_up=warming_up)
    reduced = controller.step(stopping.current_mean, epoch)
    return stop and controller.ready_to_stop(epoch), reduced


class ModelEMA:
    """FP32 inference weights, seeded from the first post-warmup model state."""
    def __init__(self, model, decay):
        self.model = deepcopy(model).float().eval().requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model, model_updates, warmup):
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            if model_updates <= warmup or self.num_updates == 0:
                averaged.copy_(current)
            else:
                averaged.lerp_(current.float(), 1-self.decay)
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)
        if model_updates > warmup:
            self.num_updates += 1

    def state_dict(self):
        return dict(version=1, decay=self.decay, num_updates=self.num_updates,
                    model_state_dict=self.model.state_dict())

    def load_state_dict(self, state):
        if state is None or state.get('decay') != self.decay:
            return False
        if state.get('version') != 1 or type(state['num_updates']) is not int or state['num_updates'] < 0:
            raise ValueError('Invalid EMA checkpoint state.')
        self.model.load_state_dict(state['model_state_dict'])
        self.num_updates = state['num_updates']
        return True
