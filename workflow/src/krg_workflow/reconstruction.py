"""Reconstruction helpers extracted from the v3.3 CT workflow.

No notebook cell execution or training occurs on import.
"""
from datetime import datetime, timezone
from pathlib import Path
import contextlib
import copy
import h5py
import hashlib
import importlib.util
import json
import numpy as np
import sys
import tifffile
from .runtime import require_compute_session
from .files import digest, exclusive_lock, json_write, safe_child, tiff_paths, utc_now
from .scans import fingerprint, install_reconstruction_metadata_reader, projection_view, validate_als_metadata
from .plans import read_plan, reconstruction_identity, stage_signature, validate_plan
_als_cache = {}


def load_als(backend):
    """Import the exact modern helper and its sibling functions from the saved checkout."""
    assert backend['name'] == 'als_modern', 'Legacy backend plan: select a new run label.'
    directory = Path(backend['directory'])
    for name, expected in backend['source_sha256'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, 'ALS source changed since this plan was frozen.'
    key = digest(backend)
    if key not in _als_cache:

        def import_file(name):
            spec = importlib.util.spec_from_file_location(name, directory / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        functions = import_file('ALS_recon_functions')
        install_reconstruction_metadata_reader(functions)
        previous = sys.modules.get('ALS_recon_functions')
        sys.modules['ALS_recon_functions'] = functions
        try:
            helper = import_file('ALS_recon_helper')
        finally:
            if previous is None:
                sys.modules.pop('ALS_recon_functions', None)
            else:
                sys.modules['ALS_recon_functions'] = previous
        assert helper.als is functions
        _als_cache[key] = helper
    return _als_cache[key]


def als_chunk(source, settings, center, start, stop, backend):
    """Call the ALS_recon notebook's modern reconstruction API for rows [start, stop)."""
    if settings['method'] not in ('fbp', 'gridrec'):
        raise ValueError('The optional-distance adapter is restricted to FBP/gridrec without phase retrieval.')
    require_compute_session()
    validate_als_metadata(source)
    helper = load_als(backend)
    with h5py.File(source, 'r') as f:
        nproj, rows, columns = f['/exchange/data'].shape
        expected_angles = -np.deg2rad(f['/exchange/theta'][:])
    assert 0 <= start < stop <= rows
    _, angles = helper.als.read_data(str(source), proj=None, sino=slice(start, start + 1, 1), preprocess_settings=settings['preprocessing'], postprocess_settings=settings['postprocessing'])
    np.testing.assert_allclose(angles, expected_angles, atol=1e-06)
    rec, _ = helper.reconstruct(path=str(source), angles_ind=None, slices_ind=slice(start, stop, 1), COR=float(center), method=settings['method'], proj_downsample=1, fc=settings['fc'], preprocessing_settings=copy.deepcopy(settings['preprocessing']), postprocessing_settings=copy.deepcopy(settings['postprocessing']), mask=True, convert360to180=False, use_gpu=settings['use_gpu'])
    rec = np.asarray(rec, dtype=np.float32)
    assert rec.shape == (stop - start, columns, columns), 'Unexpected ALS row count or image shape.'
    assert np.isfinite(rec).all(), 'Nonfinite ALS reconstruction; inspect calibration and processing.'
    return rec


def run_als_stage(source, output, settings, center, nrows, backend, log_path):
    """Write an ALS TIFF stack in row chunks with per-image atomic replacement."""
    output, log_path = (Path(output), Path(log_path))
    output.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\nALS stage ' + datetime.now(timezone.utc).isoformat() + '\n')
        for start in range(0, nrows, settings['als_chunk_sino']):
            stop = min(nrows, start + settings['als_chunk_sino'])
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                rec = als_chunk(source, settings, center, start, stop, backend)
            for offset, image in enumerate(rec):
                target = output / f'recon_{start + offset:05d}.tiff'
                temp = target.with_suffix('.part')
                tifffile.imwrite(temp, image)
                temp.replace(target)
            del rec
            log.write(f'Wrote rows {start}:{stop}\n')
            log.flush()
            print(f'{output.name}: wrote rows {start}:{stop} / {nrows}', flush=True)


def validate_stack(folder, nrows, ncols):
    """Require every expected row TIFF to have the right name, shape, and type."""
    folder = Path(folder)
    files = tiff_paths(folder)
    upstream_files = sorted(folder.glob('*.tif*'))
    assert upstream_files == files, f'Unexpected files matched by Noise2Inverse (*.tif*): {folder}'
    expected_names = [f'recon_{i:05d}.tiff' for i in range(nrows)]
    assert [p.name for p in files] == expected_names, f'Incomplete or mixed TIFF stack: {folder}'
    for p in files:
        with tifffile.TiffFile(p) as tf:
            assert tf.series[0].shape == (ncols, ncols), f'Wrong shape: {p}'
            assert tf.series[0].dtype == np.dtype('float32'), f'Wrong dtype: {p}'
    return files


def prepare_views(saved):
    rec = validate_plan(saved)['reconstruction']
    folder = Path(saved['run_dir']) / 'projection_views'
    even = projection_view(rec['source']['path'], folder / 'even.h5', 0, rec['source'])
    odd = projection_view(rec['source']['path'], folder / 'odd.h5', 1, rec['source'])
    return (even, odd)


def run_plan(saved):
    rec = reconstruction_identity(saved)
    root = Path(saved['run_dir'])
    with exclusive_lock(root / '.run.lock'):
        if reconstruction_identity(read_plan(root / 'run_plan.json')) != rec:
            raise ValueError('Run directory conflicts with queued reconstruction identity.')
        if fingerprint(rec['source']['path']) != rec['source']:
            raise ValueError('Source/signature mismatch; create a new v3 run.')
        load_als(rec['backend'])
        even, odd = prepare_views(saved)
        _, rows, cols = rec['dimensions']
        completed = 0
        for stage, input_path in (('full', rec['source']['path']), ('even', even), ('odd', odd)):
            output = safe_child(root, rec['outputs'][stage])
            signature = stage_signature(saved, stage)
            marker = root / (stage + '_complete.json')
            record = dict(schema=1, signature=signature, output_format='float32_row_tiff_v1')
            if marker.exists():
                old = json.loads(marker.read_text(encoding='utf-8'))
                if old.get('schema') != 1 or old.get('signature') != signature:
                    raise ValueError('Completion record format/signature mismatch; use a new run.')
                validate_stack(output, rows, cols)
                print('Already complete:', output)
                continue
            spec = root / (stage + '_command.json')
            if spec.exists():
                if json.loads(spec.read_text(encoding='utf-8')) != record:
                    raise ValueError('Tracked stage identity mismatch; use a new run.')
            else:
                if output.exists() and any(output.iterdir()):
                    raise ValueError(f'Untracked reconstruction output: {output}')
                json_write(spec, record)
            run_als_stage(input_path, output, rec['settings'], rec['center'], rows, rec['backend'], root / 'logs' / (stage + '.log'))
            validate_stack(output, rows, cols)
            json_write(marker, dict(record, completed_utc=utc_now()))
            completed += 1
            print('DONE: reconstruction stage', stage)
        return 'completed' if completed else 'already-complete'
