"""Paired evaluation without DDP forward collectives on uneven shards."""
import torch
from .model_factory import forward_model
from .loss import paired_objective
from .reference_metrics import paired_reference_metrics
from .eval import laplacian_score_batch
from .validation import EpochMetrics


@torch.no_grad()
def evaluate_pairs(model, loader, mode, precision, device, criterion, lcl, beta,
                   regularized, ssim, reference=True):
    keys = ('total', 'l1', 'lcl', 'l2', 'ssim', 'edge') if reference else ('total',)
    metrics = EpochMetrics(keys, device)
    preview = None
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        a, b = forward_model(model, x, precision), forward_model(model, y, precision)
        if mode == '2.5d':
            center = x.shape[1]//2
            target_x, target_y = x[:, center:center+1], y[:, center:center+1]
            plane_a, plane_b = a, b
        else:
            target_x, target_y = x, y
            plane_a, plane_b = a[:, :, a.shape[2]//2], b[:, :, b.shape[2]//2]
        loss1, loss2 = criterion(a, target_y), criterion(b, target_x)
        lcl1, lcl2 = beta*lcl(plane_a), beta*lcl(plane_b)
        total = paired_objective(loss1, loss2, lcl1, lcl2) if regularized else paired_objective(loss1, loss2)
        values = dict(total=total)
        if reference:
            mse, ssim_loss = paired_reference_metrics(a, b, x, y, mode, x.shape[1]//2, ssim)
            edge = laplacian_score_batch(plane_a.cpu())+laplacian_score_batch(plane_b.cpu())
            values.update(l1=loss1+loss2, lcl=lcl1+lcl2, l2=mse, ssim=ssim_loss, edge=edge)
            preview = (a, b, x, y)
        metrics.add(x.shape[0], **values)
    values, count = metrics.reduce()
    return values, count, preview
