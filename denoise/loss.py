
import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian_batch(x):
    # x: [B, C, H, W] (assumes grayscale: C=1)
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # for grouped conv
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

class LCL(nn.Module):
    def __init__(self, q=0.80, eps=1e-6):
        super().__init__()
        self.q = q
        self.eps = eps

    def forward(self, pred):
        # differentiable path
        L = torch.abs(laplacian_batch(pred))
        B = L.shape[0]
        L_flat = L.view(B, -1)

        # non-differentiable mask construction
        with torch.no_grad():
            thresh = torch.quantile(L_flat.detach(), self.q, dim=1, keepdim=True)
            thresh = thresh.view(B, 1, 1, 1)
            edge_mask = L.detach() > thresh
            flat_mask = ~edge_mask

        # use masks to select values from differentiable L
        if edge_mask.any():
            edge_mean = L[edge_mask].mean()
        else:
            edge_mean = L.new_tensor(0.0)

        if flat_mask.any():
            flat_mean = L[flat_mask].mean()
        else:
            flat_mean = L.new_tensor(self.eps)

        return flat_mean / (edge_mean + self.eps)


def laplacian_entropy_map(lap, bins = 256):
    # Compute entropy for each image in batch
    B = lap.shape[0]
    entropies = []
    for i in range(B):
        hist = torch.histc(lap[i, 0], bins=bins, min=0, max=lap[i, 0].max().item())
        hist = hist / hist.sum()
        hist = hist + 1e-8  # avoid log(0)
        entropy = -torch.sum(hist * torch.log(hist))
        entropies.append(entropy.item())
    return torch.tensor(entropies, device=lap.device)
