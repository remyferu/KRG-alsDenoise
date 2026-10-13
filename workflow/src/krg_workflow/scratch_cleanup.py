"""Directory discovery and reviewed deletion of scratch test runs."""
from contextlib import ExitStack
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import uuid

from .files import exclusive_lock, is_link_like, json_write, path_is_within, utc_now


def scratch_root(directory):
    path = Path(directory).expanduser().absolute()
    if path == Path(path.anchor) or path == Path.home().absolute():
        raise ValueError('Choose the scratch output directory, not a filesystem root or home directory.')
    for ancestor in (path, *path.parents):
        if is_link_like(ancestor):
            raise ValueError('Scratch root and its ancestors must not be links or junctions.')
    root = path.resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError('Choose the scratch output directory, not a filesystem root or home directory.')
    return root


def _target(root, relative, depth):
    name = Path(relative)
    if name.is_absolute() or len(name.parts) != depth or any(p in ('.', '..') or p.startswith('.') for p in name.parts):
        raise ValueError('Deletion targets must be visible directories at the selected depth.')
    path = root / name
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if is_link_like(ancestor) or os.path.ismount(str(ancestor)):
            raise ValueError('Deletion rejects links, junctions and nested mount points: ' + str(ancestor))
    resolved = path.resolve(strict=True)
    if resolved == root or not path_is_within(resolved, root) or not resolved.is_dir():
        raise ValueError('Deletion target must be a directory strictly inside the scratch root.')
    return resolved


def discover_tests(directory, depth=2, pattern='*'):
    """List directories without descending into reconstruction stacks."""
    if type(depth) is not int or depth not in (1, 2, 3):
        raise ValueError('Directory depth must be 1, 2 or 3.')
    root = scratch_root(directory)
    tests, skipped = [], []

    def visit(folder, level):
        try:
            children = sorted(folder.iterdir())
        except OSError as error:
            skipped.append(dict(path=str(folder), reason=str(error)))
            return
        for child in children:
            if child.name.startswith('.'):
                continue
            try:
                if is_link_like(child):
                    skipped.append(dict(path=str(child), reason='Link or junction; not followed.'))
                elif child.is_dir():
                    if os.path.ismount(str(child)):
                        skipped.append(dict(path=str(child), reason='Nested mount point; not followed.'))
                    elif level < depth:
                        visit(child, level + 1)
                    elif fnmatch.fnmatchcase(child.name, pattern):
                        relative = str(child.relative_to(root))
                        locked = (child / '.run.lock').exists()
                        tests.append(dict(relative=relative, path=str(child), locked=locked,
                                          modified_utc=datetime.fromtimestamp(child.stat().st_mtime, timezone.utc).isoformat(),
                                          has_plan=(child / 'run_plan.json').is_file()))
            except OSError as error:
                skipped.append(dict(path=str(child), reason=str(error)))

    visit(root, 1)
    return dict(root=str(root), depth=depth, pattern=pattern, tests=tests, skipped=skipped)


def _inventory(path, ignore_root_lock=False):
    """Hash metadata and count bytes without reading image or model contents."""
    fingerprint = hashlib.sha256()
    root_stat = path.stat()
    fingerprint.update(json.dumps([root_stat.st_dev, root_stat.st_ino]).encode())
    files, directories, size = 0, 0, 0
    top_level = []

    def walk(folder):
        nonlocal files, directories, size
        for child in sorted(folder.iterdir()):
            if child.name == '.run.lock':
                if ignore_root_lock and folder == path:
                    continue
                raise RuntimeError('Active or stale run lock: ' + str(child))
            if is_link_like(child) or os.path.ismount(str(child)):
                raise ValueError('Preview rejects links, junctions and nested mounts: ' + str(child))
            info = child.lstat()
            if info.st_dev != root_stat.st_dev:
                raise ValueError('Preview rejects a filesystem boundary: ' + str(child))
            is_dir = stat.S_ISDIR(info.st_mode)
            if not is_dir and not stat.S_ISREG(info.st_mode):
                raise ValueError('Preview rejects special files: ' + str(child))
            record = [str(child.relative_to(path)), info.st_mode, info.st_size,
                      info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev]
            fingerprint.update(json.dumps(record).encode() + b'\n')
            if folder == path and len(top_level) < 20:
                top_level.append(child.name + ('/' if is_dir else ''))
            if is_dir:
                directories += 1
                walk(child)
            else:
                files += 1
                size += info.st_size

    walk(path)
    return dict(fingerprint=fingerprint.hexdigest(), files=files, directories=directories,
                bytes=size, contents=top_level)


def preview_deletion(directory, selected, depth=2):
    """Capture a selection for review; create no files or locks."""
    if type(depth) is not int or depth not in (1, 2, 3):
        raise ValueError('Directory depth must be 1, 2 or 3.')
    root = scratch_root(directory)
    names = list(selected)
    if not names or len(set(names)) != len(names):
        raise ValueError('Select one or more distinct test directories.')
    targets = []
    for relative in sorted(names):
        path = _target(root, relative, depth)
        targets.append(dict(relative=relative, path=str(path), **_inventory(path)))
    return dict(schema=1, root=str(root), depth=depth, targets=targets,
                total_bytes=sum(t['bytes'] for t in targets), created_utc=utc_now())


def delete_tests(preview, confirmation='', jobs_stopped=False):
    """Delete the reviewed directories and retain an audit report outside them."""
    targets = preview['targets']
    phrase = 'DELETE {}'.format(len(targets))
    if not targets or confirmation != phrase or jobs_stopped is not True:
        raise ValueError('Confirm that jobs have stopped and type ' + phrase + '.')
    root = scratch_root(preview['root'])
    paths = [_target(root, t['relative'], preview['depth']) for t in targets]
    if len(set(paths)) != len(paths) or any(str(p) != t['path'] for p, t in zip(paths, targets)):
        raise ValueError('Deletion paths changed; refresh the preview.')
    report_folder = root / '.cleanup_reports'
    if is_link_like(report_folder):
        raise ValueError('Cleanup report directory must not be a link or junction.')
    report_path = report_folder / ('cleanup_' + uuid.uuid4().hex + '.json')
    report = dict(status='in-progress', started_utc=utc_now(), preview=preview, deleted=[], failed=[])

    with ExitStack() as locks:
        # Reserve every selected run before comparing snapshots or removing files.
        for path in paths:
            locks.enter_context(exclusive_lock(path / '.run.lock'))
        for path, target in zip(paths, targets):
            _target(root, target['relative'], preview['depth'])
            current = _inventory(path, ignore_root_lock=True)
            if any(current[key] != target[key] for key in current):
                raise ValueError('Contents changed since preview: ' + str(path) + '. Preview again.')
        json_write(report_path, report)
        for path in paths:
            try:
                # Keep the run lock until all other contents have been removed.
                for child in sorted(path.iterdir()):
                    if child.name == '.run.lock':
                        continue
                    if is_link_like(child) or child.resolve() != child or not path_is_within(child, path):
                        raise ValueError('Deletion path changed after preflight: ' + str(child))
                    if child.is_dir():
                        shutil.rmtree(str(child))
                    else:
                        child.unlink()
                report['deleted'].append(str(path))
            except (OSError, ValueError) as error:
                report['failed'].append(dict(path=str(path), reason=str(error)))
            json_write(report_path, report)
    # The lock files are released before removing the now-empty test directories.
    for path in paths:
        if str(path) in report['deleted']:
            try:
                _target(root, str(path.relative_to(root)), preview['depth']).rmdir()
            except (OSError, ValueError) as error:
                report['deleted'].remove(str(path))
                report['failed'].append(dict(path=str(path), reason=str(error)))
    report.update(status='partial-failure' if report['failed'] else 'completed', finished_utc=utc_now())
    json_write(report_path, report)
    return dict(report, report_path=str(report_path))


def human_bytes(size):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'):
        if size < 1024 or unit == 'PiB':
            return '{:.1f} {}'.format(size, unit)
        size /= 1024


def cleanup_browser(directory, depth=2):
    """Build selection, preview and confirmation controls for the cleanup notebook."""
    import html
    import ipywidgets as widgets
    from IPython.display import HTML, display

    root_input = widgets.Text(value=str(directory), description='Scratch root:', layout=widgets.Layout(width='95%'), style={'description_width': 'initial'})
    depth_input = widgets.Dropdown(options=[('Root / test', 1), ('Root / scan / test', 2), ('Root / group / scan / test', 3)], value=depth, description='Layout:', layout=widgets.Layout(width='400px'))
    pattern_input = widgets.Text(value='*', description='Test name:', placeholder='* or *test*', style={'description_width': 'initial'})
    scan_button = widgets.Button(description='Find tests', icon='search')
    selection = widgets.SelectMultiple(options=[], rows=18, description='Tests:', layout=widgets.Layout(width='95%'))
    preview_button = widgets.Button(description='Preview deletion', icon='list')
    stopped = widgets.Checkbox(value=False, description='All jobs using the selected tests have stopped.', indent=False, layout=widgets.Layout(width='95%'))
    confirmation = widgets.Text(value='', description='Confirmation:', style={'description_width': 'initial'})
    delete_button = widgets.Button(description='Delete selected tests', button_style='danger', icon='trash', disabled=True, layout=widgets.Layout(width='200px'))
    summary, reviewed, outcome = widgets.Output(), widgets.Output(), widgets.Output()
    state = dict(found=None, preview=None)

    def invalidate(change=None):
        state['preview'] = None
        confirmation.value = ''
        stopped.value = False
        delete_button.disabled = True
        reviewed.clear_output()

    def table(rows, headings):
        header = '<tr>' + ''.join('<th style="text-align:left;padding:6px">' + html.escape(h) + '</th>' for h in headings) + '</tr>'
        body = ''.join('<tr>' + ''.join('<td style="vertical-align:top;padding:6px">' + html.escape(str(c)) + '</td>' for c in row) + '</tr>' for row in rows)
        display(HTML('<table>' + header + body + '</table>'))

    def scan(button=None):
        invalidate()
        state['found'] = None
        selection.options = []
        summary.clear_output()
        outcome.clear_output()
        with summary:
            try:
                found = discover_tests(root_input.value, depth_input.value, pattern_input.value)
                state['found'] = found
                selection.options = [(t['relative'], t['relative']) for t in found['tests'] if not t['locked']]
                print('Scratch root:', found['root'])
                print('{} tests found; {} locked (excluded).'.format(len(found['tests']), sum(t['locked'] for t in found['tests'])))
                print('Select several entries with Ctrl/Cmd or Shift. Size is calculated only for the preview.')
                blocked = [(t['relative'], 'Run lock exists') for t in found['tests'] if t['locked']]
                blocked += [(t['path'], t['reason']) for t in found['skipped']]
                if blocked:
                    table(blocked, ['Excluded path', 'Reason'])
            except (OSError, ValueError) as error:
                print('Discovery failed:', error)

    def preview(button=None):
        invalidate()
        with reviewed:
            try:
                found = state['found']
                if found is None:
                    raise ValueError('Find tests before previewing.')
                allowed = {t['relative'] for t in found['tests'] if not t['locked']}
                if not set(selection.value).issubset(allowed):
                    raise ValueError('Selection is no longer in the discovered list.')
                captured = preview_deletion(found['root'], selection.value, found['depth'])
                state['preview'] = captured
                table([(t['path'], human_bytes(t['bytes']), t['files'], ', '.join(t['contents']) or '(empty)') for t in captured['targets']], ['Entire directory to delete', 'Logical size', 'Files', 'First 20 contents'])
                print('Total:', human_bytes(captured['total_bytes']))
                print('Deletion includes every nested file and the selected test directories.')
                print('After stopping the associated jobs, check the box and type DELETE {}.'.format(len(captured['targets'])))
            except (OSError, ValueError, RuntimeError) as error:
                print('Preview failed:', error)

    def enable(change=None):
        captured = state['preview']
        delete_button.disabled = not (captured and stopped.value and confirmation.value == 'DELETE {}'.format(len(captured['targets'])))

    def invalidate_discovery(change=None):
        invalidate()
        state['found'] = None
        selection.options = []
        summary.clear_output()

    def execute(button=None):
        captured = state['preview']
        delete_button.disabled = True
        outcome.clear_output()
        with outcome:
            try:
                if captured is None:
                    raise ValueError('Preview the selection before deleting.')
                result = delete_tests(captured, confirmation.value, stopped.value)
                print('Status:', result['status'], '| deleted:', len(result['deleted']))
                print('Audit report:', result['report_path'])
                for failure in result['failed']:
                    print('Failed:', failure['path'], '|', failure['reason'])
                selection.options = [(label, value) for label, value in selection.options if str(Path(captured['root']) / value) not in result['deleted']]
            except (OSError, ValueError, RuntimeError) as error:
                print('Deletion stopped:', error)
            finally:
                invalidate()

    scan_button.on_click(scan)
    preview_button.on_click(preview)
    delete_button.on_click(execute)
    selection.observe(invalidate, names='value')
    for control in (root_input, depth_input, pattern_input):
        control.observe(invalidate_discovery, names='value')
    for control in (confirmation, stopped):
        control.observe(enable, names='value')
    panel = widgets.VBox([root_input, widgets.HBox([depth_input, pattern_input]), scan_button, summary,
                          selection, preview_button, reviewed, stopped, confirmation, delete_button, outcome])
    display(panel)
    scan()
    return panel
