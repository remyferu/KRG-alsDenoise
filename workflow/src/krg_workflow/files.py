"""Atomic file writes, operation locks and path containment."""
from datetime import datetime, timezone
from pathlib import Path
import contextlib
import hashlib
import json
import os
import socket
import stat
import uuid


def digest(value):
    """Return a stable SHA-256 label for JSON-compatible settings or provenance."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_write(path, value):
    """Write JSON atomically through a unique temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@contextlib.contextmanager
def exclusive_lock(path):
    """Acquire an operation lock and release only the token owned by this context."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = dict(token=uuid.uuid4().hex, pid=os.getpid(), host=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'), created_utc=utc_now())
    try:
        stream = path.open('x', encoding='utf-8')
    except FileExistsError as error:
        raise RuntimeError(f'Active or stale lock: {path}. Confirm the original job/kernel stopped before explicit repair.') from error
    try:
        with stream:
            json.dump(token, stream)
        yield token
    finally:
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) == token:
            path.unlink()


def path_is_within(path, root):
    """Containment check compatible with Python versions before 3.9."""
    path, root = (Path(path), Path(root))
    return path == root or root in path.parents


def safe_child(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or Path(name).name != name or (name in ('.', '..')):
        raise ValueError('Output names must be simple directory names.')
    path = root / name
    if path.resolve() == root or not path_is_within(path.resolve(), root):
        raise ValueError('Output escapes its run directory.')
    return path


def is_link_like(path):
    """Detect junctions on Python versions that lack Path.is_junction()."""
    path = Path(path)
    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        return True
    try:
        tag = getattr(path.lstat(), 'st_reparse_tag', 0)
        return tag in (getattr(stat, 'IO_REPARSE_TAG_SYMLINK', -1), getattr(stat, 'IO_REPARSE_TAG_MOUNT_POINT', -2))
    except FileNotFoundError:
        return False


def tiff_paths(folder):
    """List TIFF images directly in one reconstruction or inference folder."""
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in ('.tif', '.tiff')))
def atomic_text(path, value):
    """Replace a UTF-8 text file using a unique, flushed temporary file."""
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

