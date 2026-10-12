"""Measure synthetic two-view training VRAM on the GPU running this script."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from denoise.model_factory import build_model, forward_model
from denoise.loss import LCL


def profile(batch_size=4, patch_size=256, n_slices=5, precision='bf16'):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the VRAM benchmark.')
    torch.set_num_threads(2)
    model = build_model(dict(train=dict(architecture='nafnet25d', n_slices=n_slices))).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    a = torch.randn(batch_size, n_slices, patch_size, patch_size, device='cuda')
    b = torch.randn_like(a)
    results = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                   batch_size=batch_size, patch_size=patch_size, n_slices=n_slices,
                   precision=precision, parameters=sum(p.numel() for p in model.parameters()),
                   workload='Two forward graphs, symmetric L1 + 0.01 LCL, backward, Adam; no DDP')
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); start = time.perf_counter()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            x, y = forward_model(model, a, precision), forward_model(model, b, precision)
            loss = .5 * ((x-b[:, n_slices//2:n_slices//2+1]).abs().mean() +
                          (y-a[:, n_slices//2:n_slices//2+1]).abs().mean())
            loss = loss + .005 * (LCL()(x) + LCL()(y))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        torch.cuda.synchronize()
        results.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                        seconds_per_update=(time.perf_counter()-start)/3,
                        loss=float(loss.detach()), status='ok')
    except torch.cuda.OutOfMemoryError:
        results.update(status='out_of_memory')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--n-slices', type=int, default=5)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    args = parser.parse_args()
    print(json.dumps(profile(args.batch_size, args.patch_size, args.n_slices, args.precision), indent=2))
