"""Queue helpers extracted from the v3.3 CT workflow.

No notebook cell execution or training occurs on import.
"""
from pathlib import Path
import copy
import hashlib
import json
import time
import uuid
from .files import digest, exclusive_lock, json_write, utc_now
from .plans import read_plan
from .reconstruction import run_plan


def queue_lock(path):
    return Path(str(path) + '.lock')


def load_queue(path):
    path = Path(path)
    if not path.exists():
        return dict(schema=1, entries=[])
    try:
        queue = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as error:
        raise ValueError('Corrupt queue JSON. Use the optional queue repair cell; original file preserved.') from error
    if not isinstance(queue, dict) or queue.get('schema') != 1 or (not isinstance(queue.get('entries'), list)):
        raise ValueError('Unsupported queue format. Use queue repair to rebuild from selected v3 plans.')
    return queue


def inspect_queue(queue, expected_backend=None, expected_settings=None):
    """Cheap preflight: plan/backend checks; raw scans are opened only during execution."""
    results, ids, paths, roots = ([], set(), set(), {})
    for number, entry in enumerate(queue['entries']):
        outcome = dict(index=number, id=entry.get('id') if isinstance(entry, dict) else None, plan_path=entry.get('plan_path') if isinstance(entry, dict) else None, status='ready', reason='')
        try:
            if not isinstance(entry, dict) or not isinstance(entry.get('id'), str) or (not entry['id']) or (not isinstance(entry.get('plan_path'), str)) or (not Path(entry['plan_path']).is_absolute()):
                raise ValueError('Malformed entry: expected stable ID and absolute schema-4 plan path.')
            path = str(Path(entry['plan_path']).resolve())
            if entry['id'] in ids:
                raise ValueError('Duplicate stable entry ID.')
            ids.add(entry['id'])
            if path in paths:
                outcome.update(status='skipped', reason='Duplicate plan path.')
            else:
                paths.add(path)
                saved = read_plan(path)
                rec = saved['reconstruction']
                key = str(Path(saved['run_dir']).resolve())
                signature = digest(rec)
                if key in roots and roots[key] != signature:
                    raise ValueError('Conflicting identities for one run directory.')
                roots[key] = signature
                if not Path(rec['source']['path']).is_file():
                    raise FileNotFoundError('Missing raw source: ' + rec['source']['path'])
                if expected_backend is not None and rec['backend'] != expected_backend:
                    raise ValueError('Backend mismatch; configure the saved backend for this queue.')
                if expected_settings is not None and rec['settings'] != expected_settings:
                    raise ValueError('Settings mismatch; configure the saved settings for this queue.')
                directory = Path(rec['backend']['directory'])
                for name, sha in rec['backend']['source_sha256'].items():
                    if hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha:
                        raise ValueError('Backend source fingerprint mismatch.')
                outcome['plan'] = saved
        except (KeyError, TypeError, ValueError, OSError, AssertionError) as error:
            outcome.update(status='blocked', reason=f'{type(error).__name__}: {error}')
        results.append(outcome)
    return results


def enqueue_plan(queue_file, plan_file):
    saved = read_plan(plan_file)
    path = str(Path(plan_file).resolve())
    with exclusive_lock(queue_lock(queue_file)):
        queue = load_queue(queue_file)
        if not any((isinstance(e, dict) and str(Path(e.get('plan_path', '')).resolve()) == path for e in queue['entries'])):
            queue['entries'].append(dict(id=uuid.uuid4().hex, plan_path=path))
            json_write(queue_file, queue)
    return queue


def execute_queue(queue_file, expected_backend=None, expected_settings=None, runner=None):
    runner = run_plan if runner is None else runner
    started = time.monotonic()
    outcomes = []
    report_path = Path(queue_file).with_name('queue_report_v3.json')
    with exclusive_lock(queue_lock(queue_file)):
        queue = load_queue(queue_file)
        preflight = inspect_queue(queue, expected_backend, expected_settings)
        try:
            for item in preflight:
                result = {k: v for k, v in item.items() if k != 'plan'}
                outcomes.append(result)
                if result['status'] == 'ready':
                    result['status'] = 'running'
                    try:
                        result['status'] = runner(item['plan'])
                        if result['status'] not in ('completed', 'already-complete'):
                            raise ValueError('Runner returned an unsupported outcome.')
                    except Exception as error:
                        result.update(status='failed', reason=f'{type(error).__name__}: {error}')
                print(f"Queue {result['id']}: {result['status']} {result['reason']}")
        except KeyboardInterrupt:
            if outcomes and outcomes[-1]['status'] == 'running':
                outcomes[-1].update(status='failed', reason='User interrupted this entry.')
            json_write(report_path, dict(schema=1, interrupted=True, outcomes=outcomes, elapsed_seconds=time.monotonic() - started, finished_utc=utc_now()))
            print('STOPPED: queue interrupted; later entries were not launched.')
            raise
        report = dict(schema=1, interrupted=False, outcomes=outcomes, elapsed_seconds=time.monotonic() - started, finished_utc=utc_now())
        json_write(report_path, report)
    return report


def propose_queue_edit(queue_file, remove_ids=(), replacements=None, deduplicate=False, remove_missing=False, rebuild=None):
    replacements = {} if replacements is None else replacements
    path = Path(queue_file)
    original = path.read_bytes() if path.exists() else None
    if rebuild is not None:
        queue = dict(schema=1, entries=[])
        for selected in rebuild:
            read_plan(selected)
            queue['entries'].append(dict(id=uuid.uuid4().hex, plan_path=str(Path(selected).resolve())))
    else:
        queue = copy.deepcopy(load_queue(path))
    existing_ids = {e.get('id') for e in queue['entries'] if isinstance(e, dict)}
    if not set(replacements).issubset(existing_ids) or not set(remove_ids).issubset(existing_ids):
        raise ValueError('Repair IDs must exist in the inspected queue.')
    entries, seen = ([], set())
    for entry in queue['entries']:
        if not isinstance(entry, dict):
            entries.append(entry)
            continue
        if entry.get('id') in remove_ids:
            continue
        if entry.get('id') in replacements:
            replacement = replacements[entry['id']]
            read_plan(replacement)
            entry['plan_path'] = str(Path(replacement).resolve())
        p = entry.get('plan_path')
        if isinstance(p, str):
            p = str(Path(p).resolve())
            if remove_missing and (not Path(p).is_file()):
                continue
            if deduplicate and p in seen:
                continue
            seen.add(p)
        entries.append(entry)
    queue['entries'] = entries
    return dict(original_sha256=None if original is None else hashlib.sha256(original).hexdigest(), proposed=queue, preflight=inspect_queue(queue))


def apply_queue_edit(queue_file, proposal):
    path = Path(queue_file)
    with exclusive_lock(queue_lock(path)):
        original = path.read_bytes() if path.exists() else None
        sha = None if original is None else hashlib.sha256(original).hexdigest()
        if sha != proposal['original_sha256']:
            raise ValueError('Queue changed after inspection; preview the repair again.')
        refreshed = inspect_queue(proposal['proposed'])
        for old, new in zip(proposal['preflight'], refreshed):
            if old['status'] == 'ready' and (new['status'] != 'ready' or old.get('plan') != new.get('plan')):
                raise ValueError('Replacement plan changed after preview; inspect the repair again.')
        backup = path.with_name(path.name + '.backup.' + uuid.uuid4().hex)
        if original is not None:
            with backup.open('xb') as stream:
                stream.write(original)
        json_write(path, proposal['proposed'])
    return backup if original is not None else None


def remove_confirmed_stale_lock(path, original_job_stopped=False, expected_token=None):
    if not original_job_stopped or not expected_token:
        raise ValueError('Confirm the original job/kernel stopped and copy its exact lock token.')
    path = Path(path)
    if json.loads(path.read_text(encoding='utf-8')).get('token') != expected_token:
        raise ValueError('Lock changed; inspect its owner again.')
    path.unlink()
