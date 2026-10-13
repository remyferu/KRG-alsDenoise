"""Cleanup manifests, protected paths and confirmed output deletion."""
from pathlib import Path
import stat
import uuid
from .files import digest, exclusive_lock, is_link_like, json_write, path_is_within, safe_child, utc_now
from .plans import read_plan, reconstruction_identity, validate_plan


def cleanup_categories(saved):
    rec = validate_plan(saved)['reconstruction']
    stem = Path(rec['source']['path']).stem
    categories = dict(cor=['auto_cor', 'auto_cor_vo', 'cor_preview', 'manual_cor_previews'], projection_views=['projection_views'], full=[rec['outputs']['full']], even=[rec['outputs']['even']], odd=[rec['outputs']['odd']], denoised=[stem + '_denoised_slices', stem + '_denoised_volume_2.5d', stem + '_denoised_volume_3d', stem + '_denoised_slices_nafnet25d', stem + '_denoised_volume_2.5d_nafnet25d'], logs=['logs'], training=['TrainOutput', rec['outputs']['full'] + '_config.yaml'])
    root = Path(saved['run_dir'])
    for pattern in ('TrainOutput_*', rec['outputs']['full']+'_*_config.yaml'):
        categories['training'].extend(sorted(path.name for path in root.glob(pattern)))
    for pattern in (stem+'_denoised_slices_*', stem+'_denoised_volume_*'):
        categories['denoised'].extend(path.name for path in root.glob(pattern) if path.name not in categories['denoised'])
    return categories


def cleanup_boundary(saved, output_root, *, protected_paths=()):
    root_path, output_path = (Path(saved['run_dir']), Path(output_root))
    root, output = (root_path.resolve(), output_path.resolve())
    if root == output or not path_is_within(root, output) or (not root.is_dir()):
        raise ValueError('Cleanup requires a selected run strictly inside OUTPUT_ROOT.')
    for p in [root_path, *root_path.parents]:
        if is_link_like(p):
            raise ValueError('Cleanup rejects symlink/junction run ancestors.')
        if p == output_path:
            break
    protected = [Path(saved['reconstruction']['source']['path']).resolve(), Path(saved['reconstruction']['backend']['directory']).resolve()]
    protected.extend((Path(p).resolve() for p in protected_paths))
    if any((path_is_within(p, root) for p in protected)):
        raise ValueError('Run contains raw inputs or backend/environment paths; cleanup rejected.')
    return root


def cleanup_manifest(saved, output_root, categories, *, protected_paths=()):
    root = cleanup_boundary(saved, output_root, protected_paths=protected_paths)
    if (root / '.run.lock').exists():
        raise RuntimeError('Selected run is active or locked. Cleanup rejected.')
    if reconstruction_identity(read_plan(root / 'run_plan.json')) != reconstruction_identity(saved):
        raise ValueError('Selected cleanup plan differs from the frozen run.')
    available = cleanup_categories(saved)
    if len(set(categories)) != len(categories) or not set(categories).issubset(available):
        raise ValueError('Select named generated-output categories only; no paths or traversal.')
    targets, files = ([], [])

    def walk(path):
        if is_link_like(path):
            raise ValueError(f'Cleanup rejects links/junctions: {path}')
        resolved = path.resolve()
        if resolved == root or not path_is_within(resolved, root):
            raise ValueError(f'Cleanup target escapes run: {path}')
        stat = path.stat()
        item = dict(path=str(resolved), size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino, directory=path.is_dir())
        files.append(item)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                walk(child)
    for category in categories:
        for name in available[category]:
            path = safe_child(root, name)
            if path.exists() or path.is_symlink():
                targets.append(dict(category=category, path=str(path.resolve())))
                walk(path)
    effects = []
    if set(categories) & {'even', 'odd'}:
        effects.append('Paired-stack deletion prevents training until reconstruction is rerun.')
    if 'full' in categories:
        effects.append('Full-stack deletion prevents reconstruction previews and inference that need it.')
    if 'training' in categories:
        effects.append('Deleting models removes resume/checkpoint capability and saved normalization.')
    return dict(run_dir=str(root), output_root=str(Path(output_root).resolve()), reconstruction_sha256=digest(reconstruction_identity(saved)), categories=list(categories), targets=targets, files=files, file_count=sum((not f['directory'] for f in files)), estimated_bytes=sum((f['size'] for f in files if not f['directory'])), effects=effects)


def execute_cleanup(saved, output_root, preview, dry_run=True, confirmation='', opt_in=False, *, protected_paths=()):
    if dry_run:
        return dict(status='dry-run', preview=preview, deleted=[], failed=[], estimated_reclaimed_bytes=0)
    root = cleanup_boundary(saved, output_root, protected_paths=protected_paths)
    if not opt_in or confirmation != str(root):
        raise ValueError('Cleanup requires explicit opt-in and the exact resolved RUN_DIR confirmation.')
    if (root / '.run.lock').exists():
        raise RuntimeError('Selected run is active or locked.')
    current = cleanup_manifest(saved, output_root, preview['categories'], protected_paths=protected_paths)
    if current != preview:
        raise ValueError('Cleanup selection changed since preview. Run the dry-run cell again.')
    deleted, failed, reclaimed = ([], [], 0)
    with exclusive_lock(root / '.run.lock'):
        for item in preview['files']:
            path = Path(item['path'])
            if is_link_like(path) or path.resolve() != path:
                raise ValueError('Cleanup path changed into a link; preview again.')
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (item['size'], item['mtime_ns'], item['inode']):
                raise ValueError('Cleanup files changed; preview again.')
        affected = {t['category'] for t in preview['targets']} & {'full', 'even', 'odd'}
        for stage in affected:
            (root / (stage + '_complete.json')).unlink(missing_ok=True)
        for item in sorted(preview['files'], key=lambda f: len(Path(f['path']).parts), reverse=True):
            path = Path(item['path'])
            try:
                if is_link_like(path) or path.resolve() != path or (not path_is_within(path, root)):
                    raise OSError('Cleanup target changed or escaped its reviewed boundary.')
                if item['directory']:
                    path.rmdir()
                else:
                    path.unlink()
                    reclaimed += item['size']
                deleted.append(str(path))
            except OSError as error:
                failed.append(dict(path=str(path), reason=f'{type(error).__name__}: {error}'))
        report = dict(status='partial-failure' if failed else 'completed', deleted=deleted, failed=failed, estimated_reclaimed_bytes=reclaimed, preview=preview, finished_utc=utc_now())
        json_write(root / ('cleanup_report_' + uuid.uuid4().hex + '.json'), report)
    return report
