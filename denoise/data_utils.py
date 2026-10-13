import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Optional, Literal, Union

def extract_sliding_window_patches_25d(
    x: torch.Tensor,
    patch_size: Tuple[int, int] = (512, 512),
    overlap: float = 0.5,
    pad_mode: str = "reflect",
    pad_value: float = 0.0,
    return_coords: bool = True,
) -> Tuple[torch.Tensor, Optional[List[Tuple[int, int]]], Dict[str, int]]:
    """Extract aligned patches from an [N, C, H, W] slice stack.

    Returns patches [N, P, C, ph, pw], shared (top, left) coordinates and
    tiling metadata. Coordinates are None when return_coords=False.
    All channels use the same crop to preserve neighboring-slice alignment.
    overlap is in [0, 1); padding is applied only at the bottom and right.
    The metadata records original/padded sizes, strides, grid size and padding.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [N,C,H,W], got {tuple(x.shape)}")

    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1).")

    N, C, H, W = x.shape
    ph, pw = patch_size
    if ph <= 0 or pw <= 0:
        raise ValueError("patch_size must be positive.")

    stride_h = max(1, int(round(ph * (1.0 - overlap))))
    stride_w = max(1, int(round(pw * (1.0 - overlap))))

    # Include the final boundary patch when the stride does not divide the image size.
    def _compute_positions(length: int, patch: int, stride: int) -> List[int]:
        if length <= patch:
            return [0]
        positions = list(range(0, length - patch + 1, stride))
        if positions[-1] != length - patch:
            positions.append(length - patch)
        return positions

    top_positions = _compute_positions(H, ph, stride_h)
    left_positions = _compute_positions(W, pw, stride_w)

    H_needed = max(H, ph)
    W_needed = max(W, pw)

    pad_bottom = H_needed - H
    pad_right = W_needed - W
    pad_top = 0
    pad_left = 0

    if pad_bottom > 0 or pad_right > 0:
        if pad_mode == "constant":
            x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode, value=pad_value)
        else:
            x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode)
    else:
        x_pad = x

    _, _, H_pad, W_pad = x_pad.shape

    top_positions = _compute_positions(H_pad, ph, stride_h)
    left_positions = _compute_positions(W_pad, pw, stride_w)

    coords: List[Tuple[int, int]] = []
    patches_list: List[torch.Tensor] = []

    for top in top_positions:
        for left in left_positions:
            # [N, C, ph, pw]
            patch = x_pad[:, :, top : top + ph, left : left + pw]
            patches_list.append(patch)
            coords.append((top, left))

    # Stack into [P, N, C, ph, pw] then permute to [N, P, C, ph, pw]
    patches = torch.stack(patches_list, dim=0).permute(1, 0, 2, 3, 4).contiguous()

    meta = {
        "H_in": H,
        "W_in": W,
        "H_pad": H_pad,
        "W_pad": W_pad,
        "ph": ph,
        "pw": pw,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "n_rows": len(top_positions),
        "n_cols": len(left_positions),
        "P": len(coords),
        "pad_top": pad_top,
        "pad_left": pad_left,
        "pad_bottom": pad_bottom,
        "pad_right": pad_right,
    }

    return patches, (coords if return_coords else None), meta


def _make_blend_window(
    ph: int,
    pw: int,
    window: Literal["uniform", "hann", "cosine"] = "hann",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return a [1, 1, ph, pw] uniform or separable Hann blending window."""
    if window == "uniform":
        w2d = torch.ones((ph, pw), device=device, dtype=dtype)
    elif window in ("hann", "cosine"):
        # Hann window in PyTorch is periodic by default; set periodic=False for overlap-add style.
        wh = torch.hann_window(ph, periodic=False, device=device, dtype=dtype).clamp_min(1e-6)
        ww = torch.hann_window(pw, periodic=False, device=device, dtype=dtype).clamp_min(1e-6)
        w2d = (wh[:, None] * ww[None, :]).clamp_min(1e-6)
    else:
        raise ValueError("window must be one of: 'uniform', 'hann', 'cosine'")
    return w2d[None, None, :, :]  # [1,1,ph,pw]


def stitch_sliding_window_patches_core(
    patch_outputs: torch.Tensor,
    coords: List[Tuple[int, int]],
    meta: Dict[str, int],
    *,
    output_size: Optional[Tuple[int, int]] = None,
    window: Literal["uniform", "hann", "cosine"] = "hann",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Blend patch outputs into [N, K, H_out, W_out].

    Accepts [N, P, K, ph, pw] or flattened [N*P, K, ph, pw] predictions
    in extraction order. coords and meta come from
    extract_sliding_window_patches_25d. The output is cropped to the original
    image size unless output_size is supplied. window selects uniform or
    Hann/cosine weights; eps bounds the normalization denominator.
    """
    if patch_outputs.dim() == 4:
        # [N*P, K, ph, pw] -> [N, P, K, ph, pw]
        NP, K, ph, pw = patch_outputs.shape
        P = int(meta["P"])
        if NP % P != 0:
            raise ValueError(f"Cannot reshape: NP={NP} not divisible by P={P}.")
        N = NP // P
        patch_outputs = patch_outputs.view(N, P, K, ph, pw)
    elif patch_outputs.dim() == 5:
        # [N, P, K, ph, pw]
        N, P, K, ph, pw = patch_outputs.shape
        if P != int(meta["P"]):
            raise ValueError(f"P mismatch: patch_outputs has P={P}, meta['P']={meta['P']}.")
    else:
        raise ValueError(f"Expected patch_outputs dim 4 or 5, got {patch_outputs.dim()}.")

    if len(coords) != P:
        raise ValueError(f"coords length {len(coords)} != P {P}.")

    H_pad = int(meta["H_pad"])
    W_pad = int(meta["W_pad"])

    device = patch_outputs.device
    dtype = patch_outputs.dtype
    acc = torch.zeros((N, K, H_pad, W_pad), device=device, dtype=dtype)
    wacc = torch.zeros((N, 1, H_pad, W_pad), device=device, dtype=dtype)

    wpatch = _make_blend_window(ph, pw, window=window, device=device, dtype=dtype)  # [1,1,ph,pw]

    # patch_outputs: [N, P, K, ph, pw]
    for p, (top, left) in enumerate(coords):
        patch = patch_outputs[:, p, :, :, :]          # [N, K, ph, pw]
        acc[:, :, top:top + ph, left:left + pw] += patch * wpatch
        wacc[:, :, top:top + ph, left:left + pw] += wpatch

    full = acc / wacc.clamp_min(eps)  # Preserve intensities even at image boundaries.

    if output_size is None:
        H_out, W_out = int(meta["H_in"]), int(meta["W_in"])
    else:
        H_out, W_out = output_size

    full = full[:, :, :H_out, :W_out].contiguous()
    return full


def stitch_sliding_window_patches(
    patch_outputs: torch.Tensor,
    coords: List[Tuple[int, int]],
    meta: Dict[str, int],
    *,
    output_size: Optional[Tuple[int, int]] = None,
    window: Literal["uniform", "hann", "cosine"] = "hann",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Blend center-slice predictions using stitch_sliding_window_patches_core."""
    return stitch_sliding_window_patches_core(
        patch_outputs=patch_outputs,
        coords=coords,
        meta=meta,
        output_size=output_size,
        window=window,
        eps=eps,
    )


class InferenceBatchSizeOptimizer:
    """Find the largest inference batch that fits on the selected GPU.

    Profiles forward passes up to max_batch_size using the configured
    channel count, patch dimensions and precision. CPU inference uses batch 1.
    The result measures memory capacity, not throughput.
    """
    def __init__(self, model: nn.Module, input_shape: tuple, device: torch.device = torch.device('cuda'),
                 max_batch_size: int = 512, precision: str = 'fp32', n_channels: int = 5):
        self.model = model.eval().to(device)
        self.input_shape = input_shape  # (H, W) for 2.5d or (D, H, W) for 3d
        self.device = device
        self.max_batch_size = max_batch_size
        self.precision = precision.lower()
        self.n_channels = n_channels

        if self.precision not in ['fp32', 'amp', 'bf16']:
            raise ValueError("precision must be fp32, amp or bf16")

        self.cached_optimal_batch_size = None

    def get_available_memory(self):
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info(self.device.index)[0] / 1024**2  # MB

    def estimate_peak_memory(self, batch_size: int) -> float:
        if self.device.type != 'cuda':
            with torch.no_grad():
                self.model(torch.randn((batch_size, self.n_channels, *self.input_shape),
                                       device=self.device))
            return 0.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            dummy_input = torch.randn((batch_size, self.n_channels, *self.input_shape), device=self.device)
            with torch.no_grad():
                if self.precision in ('amp', 'bf16'):
                    if self.precision == 'bf16' and not torch.cuda.is_bf16_supported():
                        raise ValueError('BF16 is unavailable on this GPU; set infer.precision: fp32.')
                    dtype = torch.bfloat16 if self.precision == 'bf16' else torch.float16
                    with torch.autocast(device_type='cuda', dtype=dtype):
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
        except torch.cuda.OutOfMemoryError:
            raise

        peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
        return peak_mem

    def find_optimal_batch_size(self) -> int:
        if self.cached_optimal_batch_size is not None:
            return self.cached_optimal_batch_size
        if self.device.type != 'cuda':
            self.cached_optimal_batch_size = 1
            return 1

        low, high = 1, self.max_batch_size
        best = 0

        while low <= high:
            mid = (low + high) // 2
            try:
                _ = self.estimate_peak_memory(mid)
                best = mid
                low = mid + 1
            except torch.cuda.OutOfMemoryError:
                high = mid - 1
        if best == 0:
            raise RuntimeError('Even one inference patch does not fit in GPU memory.')
        self.cached_optimal_batch_size = best
        return best

    def profile(self):
        batch_size = self.find_optimal_batch_size()
        peak_memory = self.estimate_peak_memory(batch_size)
        available_memory = self.get_available_memory() if self.device.type == 'cuda' else 0.

        return {
            'optimal_batch_size': batch_size,
            'peak_memory_used_MB': peak_memory,
            'available_memory_MB': available_memory
        }
