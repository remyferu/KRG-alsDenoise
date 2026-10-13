"""Cor helpers extracted from the v3.3 CT workflow.

No notebook cell execution or training occurs on import.
"""
from pathlib import Path
from scipy.ndimage import shift as image_shift
from skimage.registration import phase_cross_correlation
import contextlib
import copy
import h5py
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
import time
from .runtime import require_compute_session
from .files import digest, exclusive_lock, json_write, utc_now
from .plans import freeze_run, read_plan, resolve_center
from .reconstruction import als_chunk, load_als
from .notebook_ui import image_display_limits, zoom_image_viewer


def mean_frames(dset):
    """Average calibration frames one at a time to limit peak memory."""
    total = np.zeros(dset.shape[1:], dtype=np.float64)
    for i in range(dset.shape[0]):
        total += dset[i]
    return (total / dset.shape[0]).astype(np.float32)


def alignment_projections(path, bright_ratio=1.0):
    """Flat/dark-correct the first and opposite raw projections for alignment."""
    with h5py.File(path, 'r') as f:
        angles = f['/exchange/theta'][:]
        opposite = int(np.argmin(np.abs(angles - angles[0] - 180)))
        assert abs(float(angles[opposite] - angles[0]) - 180) < 1
        flat = mean_frames(f['/exchange/data_white'])
        dark = mean_frames(f['/exchange/data_dark'])
        denominator = flat - dark
        good = denominator > 0
        images = []
        for index in (0, opposite):
            raw = f['/exchange/data'][index].astype(np.float32)
            transmission = np.ones_like(raw)
            np.divide((raw - dark) * bright_ratio, denominator, out=transmission, where=good)
            transmission = np.clip(transmission, 1e-06, None)
            images.append(-np.log(transmission))
    return (images[0], images[1], opposite)


def center_estimate(first, opposite):
    """Estimate the midpoint-offset rotation center from reflected projection shift."""
    mirrored = opposite[:, ::-1]
    displacement, _, _ = phase_cross_correlation(first, mirrored, upsample_factor=10)
    return ((float(displacement[1]) - 1) / 2, displacement)


def alignment_figure(first, opposite, center, *, context, view_options=None):
    """Inspect the first/opposite overlay and residual with linked zoom controls."""
    dx = 2 * center + 1
    moved = image_shift(opposite[:, ::-1], (0, dx), order=1, mode='constant', cval=np.nan, prefilter=False)
    difference = first - moved
    lo, hi = image_display_limits(first)
    span = max(float(hi - lo) * 0.15, 1e-06)
    return zoom_image_viewer([first, difference], titles=[f'Overlay — COR offset {center:.3f}', 'First − shifted reflected projection'], limits=[(lo, hi), (-span, span)], cmaps=['gray', 'coolwarm'], overlays=[(moved, 'magma', 0.4), None], captured=context.snapshot(), title='Projection alignment', context=context, **(view_options or {}))


def vo_options(options=None):
    """Native Vo parameters and provisional multi-row agreement checks."""
    opts = dict(smin=-50, smax=50, srad=6.0, step=0.1, ratio=0.5, drop=20, ncore=1, disagreement_pixels=1.0)
    supplied = {} if options is None else dict(options)
    unknown = set(supplied) - set(opts)
    if unknown:
        raise ValueError('Unknown Vo options: ' + ', '.join(sorted(unknown)))
    opts.update(supplied)
    for name in ('smin', 'smax', 'drop', 'ncore'):
        if type(opts[name]) is not int:
            raise ValueError(f'{name} must be an integer.')
    if opts['smin'] >= opts['smax'] or opts['drop'] < 0 or opts['ncore'] < 1:
        raise ValueError('Use smin < smax, drop >= 0 and ncore >= 1.')
    for name in ('srad', 'step', 'ratio', 'disagreement_pixels'):
        if not np.isfinite(opts[name]) or opts[name] <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if opts['srad'] < 1 or not 0.1 <= opts['step'] <= opts['srad']:
        raise ValueError('Vo requires srad >= 1 and 0.1 <= step <= srad.')
    return opts


def find_vo_center(captured, rows=None, options=None, *, context):
    """Estimate COR from prepared sinograms, without candidate reconstructions."""
    context.assert_current(captured, check_source=True)
    require_compute_session()
    nproj, nrows, ncols = captured['dimensions']
    if rows is None:
        rows = sorted(set((int(round((nrows - 1) * f)) for f in (0.25, 0.5, 0.75))))
    else:
        rows = list(rows)
    if len(rows) < 3 or len(set(rows)) != len(rows) or any((type(r) is not int or not 0 <= r < nrows for r in rows)):
        raise ValueError('Vo COR needs at least three distinct valid detector rows.')
    opts = vo_options(options)
    available = max(1, os.cpu_count() or 1)
    if hasattr(os, 'sched_getaffinity'):
        try:
            available = min(available, max(1, len(os.sched_getaffinity(0))))
        except OSError:
            pass
    assigned = os.environ.get('SLURM_CPUS_PER_TASK', '')
    if assigned.isdecimal() and int(assigned) > 0:
        available = min(available, int(assigned))
    if opts['ncore'] > available:
        raise ValueError(f"Vo ncore={opts['ncore']} exceeds the available CPU count {available}.")
    with h5py.File(captured['source']['path'], 'r') as f:
        theta = np.asarray(f['/exchange/theta'][:], dtype=np.float64)
    steps = np.diff(theta)
    if theta.shape != (nproj,) or not np.isfinite(theta).all() or (not np.all(steps > 0)) or (not np.allclose(steps, np.median(steps), rtol=0.01, atol=1e-06)) or (abs(float(theta[-1] - theta[0]) - 180.0) > max(1.0, 2 * float(np.median(steps)))):
        raise ValueError('Vo COR requires approximately uniform angles spanning 180 degrees.')
    helper = load_als(captured['backend'])
    tomopy = helper.als.tomopy
    if not callable(getattr(tomopy, 'find_center_vo', None)):
        raise RuntimeError('The ALS kernel must provide tomopy.find_center_vo().')
    native = {name: opts[name] for name in ('smin', 'smax', 'srad', 'step', 'ratio', 'drop', 'ncore')}
    supports_ncore = 'ncore' in inspect.signature(tomopy.find_center_vo).parameters
    if not supports_ncore:
        native.pop('ncore')
    absolute, offsets, errors, diagnostics = ([], [], [], [])
    started = time.monotonic()
    for row in rows:
        try:
            tomo, angles = helper.als.read_data(captured['source']['path'], proj=None, sino=slice(row, row + 1, 1), downsample_factor=1, preprocess_settings=copy.deepcopy(captured['settings']['preprocessing']), postprocess_settings=copy.deepcopy(captured['settings']['postprocessing']))
            if np.shape(tomo) != (nproj, 1, ncols):
                raise ValueError('Unexpected prepared sinogram dimensions.')
            np.testing.assert_allclose(angles, -np.deg2rad(theta), atol=1e-06)
            sino = np.asarray(tomo[:, 0, :], dtype=np.float32)
            if not np.isfinite(sino).all():
                raise ValueError('Nonfinite prepared sinogram.')
            low, high = np.percentile(sino, [1, 99])
            if high - low <= max(float(np.max(np.abs(sino))) * 1e-06, np.finfo(np.float32).eps):
                raise ValueError('Flat/blank sinogram; choose a row containing sample.')
            center = float(tomopy.find_center_vo(sino, **native))
            offset = center - ncols / 2.0
            if not np.isfinite(center) or not 0 < center < ncols:
                raise ValueError('Vo returned an invalid detector center.')
            margin = min(opts['srad'], ncols / 4.0)
            lower = max(0.0, ncols // 2 + opts['smin'])
            upper = min(ncols - 1.0, ncols // 2 + opts['smax'])
            if center <= lower + margin or center >= upper - margin:
                if 'near_vo_limit' not in diagnostics:
                    diagnostics.append('near_vo_limit')
            absolute.append(center)
            offsets.append(offset)
            errors.append(None)
            print(f'Vo row {row}: midpoint COR {offset:+.3f} px')
            del tomo, sino
        except (ValueError, AssertionError) as error:
            absolute.append(None)
            offsets.append(None)
            errors.append(f'{type(error).__name__}: {error}')
            print(f'Vo row {row}: review required ({error})')
    valid = [v for v in offsets if v is not None]
    center = float(np.median(valid)) if valid else None
    spread = float(np.ptp(valid)) if valid else None
    if any((error is not None for error in errors)):
        diagnostics.append('invalid_or_blank_rows')
    if spread is not None and spread > opts['disagreement_pixels']:
        diagnostics.append('row_disagreement')
    context.assert_current(captured, check_source=True)
    return dict(center=center, rows=rows, row_centers=offsets, row_centers_absolute=absolute, row_errors=errors, row_spread=spread, options=opts, status='review_required' if diagnostics else 'auto_accepted', diagnostics=diagnostics, method='tomopy.find_center_vo', tomopy_version=getattr(tomopy, '__version__', 'unknown'), ncore_supported=supports_ncore, context=copy.deepcopy(captured), policy_version='vo_rows_v1', elapsed_seconds=time.monotonic() - started)


def checked_auto_center(result, captured, *, context):
    context.assert_current(captured)
    if not isinstance(result, dict) or result.get('context') != captured:
        raise ValueError('Automatic COR is missing or stale. Run Vo COR for the selected scan.')
    if result.get('status') != 'auto_accepted' or result.get('diagnostics') or result.get('method') != 'tomopy.find_center_vo' or (result.get('policy_version') != 'vo_rows_v1') or (len(result.get('rows', [])) < 3):
        raise ValueError('COR review required: ' + ', '.join(result.get('diagnostics') or ['invalid Vo result']))
    offsets = result.get('row_centers', [])
    if len(offsets) != len(result['rows']) or len(set(result['rows'])) != len(result['rows']) or any((v is None or not np.isfinite(v) for v in offsets)) or any(result.get('row_errors', [])):
        raise ValueError('COR review required: incomplete Vo estimates.')
    opts = vo_options(result.get('options'))
    center = float(np.median(offsets))
    if np.ptp(offsets) > opts['disagreement_pixels'] or not np.isclose(center, result.get('center')):
        raise ValueError('COR review required: Vo row estimates disagree.')
    return center


def manual_start(captured, automatic, alignment, *, context):
    context.assert_current(captured)
    if isinstance(automatic, dict) and automatic.get('context') == captured and (automatic.get('center') is not None):
        return (float(automatic['center']), 'current automatic recommendation')
    if alignment is None:
        raise ValueError('Run projection alignment first; no usable recommendation exists.')
    return (float(alignment), 'projection-alignment estimate (no usable automatic result)')


def cor_comparison(captured, offsets, rows, folder=None, *, context):
    context.assert_current(captured, check_source=True)
    cols = captured['dimensions'][2]
    if any((not np.isfinite(v) or abs(v) >= cols / 2 for v in offsets)):
        raise ValueError('Comparison offsets cross detector bounds.')
    if any((type(r) is not int or not 0 <= r < captured['dimensions'][1] for r in rows)):
        raise ValueError('Comparison rows must be valid integer detector indices.')
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
    for row in rows:
        images = [als_chunk(captured['source']['path'], captured['settings'], float(v), row, row + 1, captured['backend'])[0] for v in offsets]
        lo, hi = image_display_limits(images[0])
        fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5), squeeze=False)
        for ax, value, image in zip(axes[0], offsets, images):
            ax.imshow(image, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
            ax.set_title(f'Row {row}; midpoint COR {value:+.3f} px')
            ax.axis('off')
            if folder:
                tifffile.imwrite(Path(folder) / f'row{row:05d}_center{value:+.5f}.tiff', image)
        if folder:
            fig.savefig(Path(folder) / f'comparison_row{row:05d}.png', dpi=150)
        plt.close(fig)
        zoom_image_viewer(images, titles=[f'Row {row}; midpoint COR {value:+.3f} px' for value in offsets], limits=[(lo, hi)] * len(images), captured=captured, title='Manual COR comparison', context=context)
    context.assert_current(captured, check_source=True)


def apply_manual(captured, center, rows, method, base, *, context):
    context.assert_current(captured, check_source=True)
    if not np.isfinite(center) or abs(center) >= captured['dimensions'][2] / 2:
        raise ValueError('Invalid manual midpoint offset in detector pixels.')
    if not rows or len(set(rows)) != len(rows) or any((type(r) is not int or not 0 <= r < captured['dimensions'][1] for r in rows)):
        raise ValueError('Manual review requires distinct valid detector rows.')
    selection = dict(center=float(center), rows=list(rows), method=method, starting_center=float(base), adjustment_pixels=float(center - base), context=copy.deepcopy(captured), applied_utc=utc_now())
    json_write(Path(captured['run_dir']) / 'manual_cor.json', selection)
    return selection


def automatic_cor_workflow(captured, info, rows=None, options=None, *, context):
    """Restore a frozen plan, or estimate/save a new COR with native Vo."""
    context.assert_current(captured, check_source=True)
    root = Path(captured['run_dir'])
    path = root / 'run_plan.json'
    if path.exists():
        center, selection = resolve_center(captured, existing=read_plan(path), context=context)
        return (freeze_run(captured, info, center, selection, context=context), None)
    opts = vo_options(options)
    spec = dict(context=captured, rows=rows, options=opts, method='tomopy.find_center_vo')
    folder = root / 'auto_cor_vo' / digest(spec)[:16]
    folder.mkdir(parents=True, exist_ok=True)
    json_write(folder / 'settings.json', spec)
    with exclusive_lock(root / '.run.lock'), (folder / 'vo.log').open('a', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = find_vo_center(captured, rows, opts, context=context)
        json_write(folder / 'result.json', result)
    if result['status'] != 'auto_accepted':
        return (None, result)
    center, selection = resolve_center(captured, automatic=result, context=context)
    return (freeze_run(captured, info, center, selection, context=context), result)
