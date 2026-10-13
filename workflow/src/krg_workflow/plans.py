"""Schema-4 reconstruction plans and immutable run identity."""
from pathlib import Path
import copy
import json
import numpy as np
from .files import digest, exclusive_lock, json_write, safe_child, utc_now


def validate_plan(saved, plan_path=None):
    if not isinstance(saved, dict) or saved.get('schema') != 4:
        raise ValueError('Unsupported run-plan format. Create a new v3 run with a new RUN_LABEL.')
    rec = saved['reconstruction']
    if rec['geometry'] != dict(center_convention='midpoint_offset', scan_degrees=180, angle_order='increasing', subsets='disjoint_even_odd'):
        raise ValueError('Unsupported reconstruction geometry.')
    shape = rec['dimensions']
    if len(shape) != 3 or any((type(n) is not int or n <= 0 for n in shape)) or shape[0] < 4 or shape[2] % 2:
        raise ValueError('Invalid frozen dimensions.')
    if rec['source']['shape'] != shape or not np.isfinite(rec['center']) or abs(rec['center']) >= shape[2] / 2:
        raise ValueError('Invalid frozen source or midpoint COR.')
    if rec['backend']['name'] != 'als_modern' or rec['settings']['method'] not in ('fbp', 'gridrec'):
        raise ValueError('Unsupported backend/settings.')
    root = Path(saved['run_dir'])
    if not root.is_absolute() or (plan_path and Path(plan_path).resolve().parent != root.resolve()):
        raise ValueError('Plan location disagrees with its run directory.')
    if set(rec['outputs']) != {'full', 'even', 'odd'} or len(set(rec['outputs'].values())) != 3:
        raise ValueError('Expected three distinct reconstruction output directories.')
    for name in rec['outputs'].values():
        safe_child(root, name)
    if saved['provenance'].get('notebook_version') not in ('3.0', '3.1', '3.2', '3.3', '3.4', '3.5'):
        raise ValueError('Unsupported release provenance. Create a new v3 run.')
    return saved


def read_plan(path):
    path = Path(path)
    return validate_plan(json.loads(path.read_text(encoding='utf-8')), path)


def reconstruction_identity(saved):
    return copy.deepcopy(validate_plan(saved)['reconstruction'])


def stage_signature(saved, stage):
    if stage not in ('full', 'even', 'odd'):
        raise ValueError('Unknown reconstruction stage.')
    nproj = saved['reconstruction']['dimensions'][0]
    subset = dict(kind=stage, start={'full': 0, 'even': 0, 'odd': 1}[stage], step=1 if stage == 'full' else 2, stop=nproj)
    return digest(dict(reconstruction=reconstruction_identity(saved), subset=subset, output_format='float32_row_tiff_v1', completion_schema=1))


def resolve_center(captured, manual=None, numeric=None, existing=None, automatic=None, policy='auto_if_eligible', *, context):
    context.assert_current(captured, check_source=True)
    from .cor import checked_auto_center
    if policy not in ('auto_if_eligible', 'manual_only'):
        raise ValueError('COR_POLICY must be auto_if_eligible or manual_only.')
    if numeric is not None:
        decision = dict(center=float(numeric), method='numeric_override', context=captured)
    elif manual is not None:
        context.assert_current(manual['context'])
        decision = copy.deepcopy(manual)
    elif existing is not None:
        rec = validate_plan(existing)['reconstruction']
        if rec['source'] != captured['source'] or rec['backend'] != captured['backend'] or rec['settings'] != captured['settings'] or (existing['run_dir'] != captured['run_dir']):
            raise ValueError('Frozen run identity changed. Use a new RUN_LABEL.')
        return (float(rec['center']), copy.deepcopy(existing['selection']))
    elif policy == 'auto_if_eligible':
        center = checked_auto_center(automatic, captured, context=context)
        decision = dict(center=center, method='tomopy_find_center_vo', result=copy.deepcopy(automatic))
    else:
        raise ValueError('Manual-only policy: apply a reviewed COR or enter COR_OVERRIDE in pixels.')
    center = decision['center']
    if not np.isfinite(center) or abs(center) >= captured['dimensions'][2] / 2:
        raise ValueError('COR must be finite and strictly within detector midpoint bounds.')
    return (float(center), decision)


def freeze_run(captured, info, center, selection, *, context):
    context.assert_current(captured, check_source=True)
    root = Path(captured['run_dir'])
    stem = Path(captured['source']['path']).stem
    saved = dict(schema=4, run_dir=str(root), reconstruction=dict(source=copy.deepcopy(captured['source']), dimensions=captured['dimensions'], geometry=dict(center_convention='midpoint_offset', scan_degrees=180, angle_order='increasing', subsets='disjoint_even_odd'), center=float(center), backend=copy.deepcopy(captured['backend']), settings=copy.deepcopy(captured['settings']), outputs=dict(full=stem + '_rec', even=stem + '_rec_0', odd=stem + '_rec_1')), selection=copy.deepcopy(selection), provenance=dict(notebook_version='3.5', policy_version='vo_rows_v1', frozen_utc=utc_now(), scan_metadata=info))
    validate_plan(saved)
    with exclusive_lock(root / '.run.lock'):
        path = root / 'run_plan.json'
        if path.exists():
            old = read_plan(path)
            if reconstruction_identity(old) != reconstruction_identity(saved):
                raise ValueError('Frozen reconstruction identity changed. Use a new RUN_LABEL.')
            return old
        forbidden = list(root.glob('*_complete.json')) + list(root.glob('*_command.json'))
        forbidden += [root / name for name in saved['reconstruction']['outputs'].values() if (root / name).exists()]
        if forbidden or (root / 'TrainOutput').exists():
            raise ValueError('Prior outputs exist without a v3 plan. Create a new v3 run directory.')
        json_write(path, saved)
    return saved
