"""Detached paired MSE and SSIM loss for reporting, never for optimization."""
import csv
import math
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class SSIMLoss(nn.Module):
    """1 - SSIM using an 11x11 Gaussian window and population covariance.

    Uses valid windows (no zero-padding), averages pixels/channels/samples, and
    evaluates in FP32. data_range is a fixed intensity span, not a per-batch span.
    Images are not clipped or independently rescaled.
    """
    def __init__(self, data_range, window_size=11, sigma=1.5):
        super().__init__()
        if not math.isfinite(data_range) or data_range <= 0:
            raise ValueError('ssim_data_range must be finite and positive.')
        if window_size <= 0 or window_size % 2 != 1 or sigma <= 0:
            raise ValueError('SSIM needs a positive odd window and positive sigma.')
        self.data_range = float(data_range)
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gaussian = torch.exp(-coords.square() / (2 * sigma ** 2))
        gaussian /= gaussian.sum()
        self.register_buffer('window', torch.outer(gaussian, gaussian)[None, None])

    @torch.no_grad()
    def forward(self, prediction, target):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError('SSIM expects matching [B,C,H,W] tensors.')
        if min(prediction.shape[-2:]) < self.window_size:
            raise ValueError('SSIM input must be at least %d pixels wide/high.' % self.window_size)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            x, y = prediction.detach().float(), target.detach().float()
            channels = x.shape[1]
            moments = torch.cat((x, y, x.square(), y.square(), x * y), dim=1)
            kernel = self.window.to(device=x.device).repeat(5 * channels, 1, 1, 1)
            # Avoid TF32 rounding of nearly cancelling second moments on Ampere.
            with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, allow_tf32=False):
                mx, my, xx, yy, xy = F.conv2d(moments, kernel, groups=5 * channels).chunk(5, dim=1)
            vx = (xx - mx.square()).clamp_min(0)
            vy = (yy - my.square()).clamp_min(0)
            covariance = xy - mx * my
            c1, c2 = (0.01 * self.data_range) ** 2, (0.03 * self.data_range) ** 2
            similarity = ((2 * mx * my + c1) * (2 * covariance + c2) /
                          ((mx.square() + my.square() + c1) * (vx + vy + c2)))
            return 1 - similarity.clamp(-1, 1).mean()


def resolve_ssim_data_range(train, dataset):
    """Use an explicit/saved span, or derive one once from normalized train stacks."""
    value = train.get('ssim_data_range')
    if value is None:
        low = min(float(dataset.split0.min()), float(dataset.split1.min()))
        high = max(float(dataset.split0.max()), float(dataset.split1.max()))
        value = high - low
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('ssim_data_range must be finite and positive; set it explicitly in train YAML.')
    return value


@torch.no_grad()
def paired_reference_metrics(pred_view1, pred_view2, input_view1, input_view2,
                             mode, center_idx, ssim):
    """Mean of both directions; 3D MSE uses all voxels, SSIM the center plane."""
    a, b = pred_view1.detach().float(), pred_view2.detach().float()
    x, y = input_view1.detach().float(), input_view2.detach().float()
    if mode == '2.5d':
        x, y = x[:, center_idx:center_idx+1], y[:, center_idx:center_idx+1]
    elif mode != '3d':
        raise ValueError('Unknown mode: %s' % mode)
    mse = .5 * (F.mse_loss(a, y) + F.mse_loss(b, x))
    if mode == '3d':
        d = a.shape[2] // 2
        a, b, x, y = a[:, :, d], b[:, :, d], x[:, :, d], y[:, :, d]
    ssim_loss = .5 * (ssim(a, y) + ssim(b, x))
    return mse, ssim_loss


REFERENCE_FIELDS = ['epoch', 'train_l2', 'eval_l2', 'train_ssim', 'eval_ssim',
                    'ssim_data_range', 'ssim_window_size', 'mode']


def save_reference_history(path, rows):
    path = Path(path)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=REFERENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
