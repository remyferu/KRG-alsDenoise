"""NAFNet adaptation for paired CT denoising, with learned axial slice fusion.

NAFBlock is adapted from megvii-research/NAFNet (Copyright (c) 2022
megvii-model, MIT). See third_party/NAFNet_LICENSE.txt. Changes: native
PyTorch channel LayerNorm, distance-aware 2.5D gated fusion, a detail branch and center-slice
residual output. This is an experimental CT adaptation, not a pretrained model.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Normalize channels independently at every pixel, without custom kernels."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        dtype = x.dtype
        y = F.layer_norm(x.permute(0, 2, 3, 1).float(),
                         (x.shape[1],), self.weight, self.bias, 1e-6)
        return y.permute(0, 3, 1, 2).to(dtype)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(2 * channels, 2 * channels, 3,
                                   padding=1, groups=2 * channels)
        self.gate = SimpleGate()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                       nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1),
                                 SimpleGate(), nn.Conv2d(channels, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.gate(self.depthwise(self.expand(self.norm1(x))))
        y = self.project(y * self.attention(y))
        x = x + y * self.beta.to(y.dtype)
        y = self.ffn(self.norm2(x))
        return x + y * self.gamma.to(y.dtype)


def blocks(channels, count):
    return nn.Sequential(*(NAFBlock(channels) for _ in range(count)))


class NAFNet25D(nn.Module):
    """[B,K,H,W] -> [B,1,H,W], retaining the center slice as a residual path."""
    def __init__(self, n_slices=5, width=32, stem_width=16,
                 detail_blocks=4, encoder_blocks=(2, 2),
                 middle_blocks=6, decoder_blocks=(2, 2), distance_aware=True):
        super().__init__()
        if n_slices < 3 or n_slices % 2 != 1:
            raise ValueError('NAFNet25D requires an odd n_slices >= 3 (e.g. 5, 7, 9).')
        if width <= 0 or width % 2 or stem_width <= 0:
            raise ValueError('width must be positive and even; stem_width must be positive.')
        if len(encoder_blocks) != 2 or len(decoder_blocks) != 2:
            raise ValueError('NAFNet25D uses exactly two downsampling stages.')
        if min(detail_blocks, middle_blocks, *encoder_blocks, *decoder_blocks) < 0:
            raise ValueError('Block counts must be nonnegative.')
        self.n_slices = n_slices
        if type(distance_aware) is not bool:
            raise ValueError('distance_aware must be boolean.')
        self.distance_aware = distance_aware
        if distance_aware:
            # A symmetric distance prior distinguishes near and far neighbors.
            distances = torch.arange(1, n_slices//2+1, dtype=torch.float32)
            self.distance_bias = nn.Parameter(-distances.log()[:, None].repeat(1, stem_width))
        self.stem = nn.Conv2d(1, stem_width, 3, padding=1)
        self.neighbor_gate = nn.Sequential(
            nn.Conv2d(3 * stem_width, 3 * stem_width, 3,
                      padding=1, groups=3 * stem_width),
            nn.Conv2d(3 * stem_width, stem_width, 1), nn.Identity() if distance_aware else nn.Sigmoid())
        nn.init.constant_(self.neighbor_gate[1].bias, -2.)
        self.fuse = nn.Conv2d(2 * stem_width, width, 1)
        self.detail = blocks(width, detail_blocks)
        self.enc0 = blocks(width, encoder_blocks[0])
        self.down0 = nn.Conv2d(width, 2 * width, 2, stride=2)
        self.enc1 = blocks(2 * width, encoder_blocks[1])
        self.down1 = nn.Conv2d(2 * width, 4 * width, 2, stride=2)
        self.middle = blocks(4 * width, middle_blocks)
        self.up1 = nn.Sequential(nn.Conv2d(4 * width, 8 * width, 1), nn.PixelShuffle(2))
        self.dec1 = blocks(2 * width, decoder_blocks[0])
        self.up0 = nn.Sequential(nn.Conv2d(2 * width, 4 * width, 1), nn.PixelShuffle(2))
        self.dec0 = blocks(width, decoder_blocks[1])
        self.head = nn.Sequential(nn.Conv2d(2 * width, width, 1),
                                  nn.Conv2d(width, 1, 3, padding=1))
        # Initial predictions exactly reproduce the noisy center slice.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.n_slices:
            raise ValueError('Expected [B,%d,H,W], got %s' % (self.n_slices, tuple(x.shape)))
        h, w = x.shape[-2:]
        center_input = x[:, self.n_slices // 2:self.n_slices // 2 + 1]
        x = F.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode='replicate')
        b, k, ph, pw = x.shape
        features = self.stem(x.reshape(b * k, 1, ph, pw)).reshape(b, k, -1, ph, pw)
        center = features[:, k // 2]
        correction = torch.zeros_like(center)
        for i in range(k):
            if i == k // 2:
                continue
            delta = features[:, i] - center
            gate_input = torch.cat((center, features[:, i], delta), dim=1)
            if self.distance_aware:
                logits = self.neighbor_gate(gate_input)
                bias = self.distance_bias[abs(i-k//2)-1].view(1, -1, 1, 1)
                gate = torch.sigmoid(logits + bias.to(logits.dtype))
            else:
                gate = self.neighbor_gate(gate_input)
            correction = correction + gate * delta / (k - 1)
        fused = self.fuse(torch.cat((center, correction), dim=1))
        detail = self.detail(fused)
        skip0 = self.enc0(fused)
        skip1 = self.enc1(self.down0(skip0))
        context = self.middle(self.down1(skip1))
        context = self.dec1(self.up1(context) + skip1)
        context = self.dec0(self.up0(context) + skip0)
        residual = self.head(torch.cat((detail, context), dim=1))[:, :, :h, :w]
        # Keep the final intensity correction in FP32 under BF16 autocast.
        return center_input.float() + residual.float()
