"""NERSC allocation checks, thread configuration and subprocess logging."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time


def configure_reconstruction_threads():
    """Configure NumExpr before scientific imports; cover TomoPy's CPU-count requests."""
    detected = max(1, os.cpu_count() or 1)
    available = detected
    if hasattr(os, 'sched_getaffinity'):
        try:
            available = min(available, max(1, len(os.sched_getaffinity(0))))
        except OSError:
            pass
    for name in ('SLURM_CPUS_PER_TASK', 'NUMEXPR_NUM_THREADS'):
        value = os.environ.get(name, '')
        if value.isdecimal() and int(value) > 0:
            available = min(available, int(value))
    maximum = max(64, detected)
    active = min(8, available)
    os.environ['NUMEXPR_MAX_THREADS'] = str(maximum)
    os.environ['NUMEXPR_NUM_THREADS'] = str(active)
    loaded = sys.modules.get('numexpr')
    if loaded is not None:
        if loaded.MAX_THREADS < detected:
            raise RuntimeError(f'NumExpr was already imported with a {loaded.MAX_THREADS}-thread pool, but TomoPy can request {detected}. Restart the kernel and run this imports cell first. If NumExpr loads during kernel startup, set NUMEXPR_MAX_THREADS={maximum} in the kernel launcher before Python starts.')
        active = min(active, loaded.MAX_THREADS)
        if loaded.set_num_threads(active) < 1:
            raise RuntimeError('NumExpr rejected the configured thread count; restart the kernel.')
        os.environ['NUMEXPR_NUM_THREADS'] = str(active)
    return dict(detected_cpus=detected, maximum=maximum, active=active)


def require_compute_session(environ=None, hostname=None):
    """Reject recognizable NERSC login sessions before expensive operations.

    Slurm environment markers are a preliminary check, not proof of GPU access
    or the job's remaining time; verify those in the allocation terminal.
    """
    env = os.environ if environ is None else environ
    host = hostname if hostname is not None else os.uname().nodename if hasattr(os, 'uname') else env.get('COMPUTERNAME', '')
    host = host.lower().split('.')[0]
    on_nersc = bool(env.get('NERSC_HOST')) or host.startswith('perlmutter')
    if on_nersc:
        if 'login' in host or host.startswith('perlmutter'):
            raise RuntimeError('This appears to be a NERSC login node. Start a compute-node Jupyter server before reconstruction or denoising.')
        if not (env.get('SLURM_JOB_ID') or env.get('SLURM_JOBID')):
            raise RuntimeError('No Slurm job is visible in this NERSC kernel. Start compute-node Jupyter through JupyterHub and verify the allocation.')


def first_visible_gpu(environ=None):
    """Retain the first assigned CUDA identifier, including an explicit disabled mask."""
    env = os.environ if environ is None else environ
    visible = env.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return '0'
    first = visible.split(',')[0].strip()
    return '' if first in ('', '-1', 'NoDevFiles') else first


def validate_gpu_ids(gpu_ids, environ=None):
    """Prevent the denoise CLI from widening an explicit scheduler GPU mask."""
    env = os.environ if environ is None else environ
    ids = [v.strip() for v in str(gpu_ids).split(',')]
    if not ids or any((v in ('', '-1', 'NoDevFiles') for v in ids)) or len(set(ids)) != len(ids):
        raise ValueError('No valid GPU selection. Use an allocated GPU session and check CUDA_VISIBLE_DEVICES.')
    visible = env.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        assigned = {v.strip() for v in visible.split(',') if v.strip()}
        if not set(ids).issubset(assigned) or first_visible_gpu(env) == '':
            raise ValueError('GPU_IDS must stay within the inherited CUDA_VISIBLE_DEVICES assignment.')
    return ','.join(ids)


def all_allocated_gpus(environ=None):
    """Select every inherited CUDA identifier without discovering unallocated devices."""
    env = os.environ if environ is None else environ
    visible = env.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        raise RuntimeError('CUDA_VISIBLE_DEVICES is unset. Start a GPU compute allocation with its scheduler GPU assignment visible to this kernel; do not guess device IDs from nvidia-smi.')
    return validate_gpu_ids(visible, env)


def select_output_root(override=None, environ=None):
    """Prefer NERSC scratch variables; never silently place large stacks in home."""
    if override is not None:
        root = Path(override).expanduser()
        if not root.is_absolute():
            raise ValueError('OUTPUT_ROOT_OVERRIDE must be an absolute scratch path.')
        return root.resolve()
    env = os.environ if environ is None else environ
    for name in ('PSCRATCH', 'SCRATCH'):
        if env.get(name):
            root = Path(env[name]).expanduser()
            if not root.is_absolute() or not root.is_dir():
                raise FileNotFoundError(f'{name} is not an accessible absolute scratch path: {root}')
            return root / 'noise2inverse360'
    als_scratch = Path('/alsuser/pscratch')
    if als_scratch.is_dir():
        return als_scratch / 'noise2inverse360'
    raise FileNotFoundError('No scratch location found. Set OUTPUT_ROOT_OVERRIDE to allocated scratch visible to this kernel.')


def run_logged(parts, log_path, cwd, *, on_line=None):
    """Run an external command, stream its output, and save a complete log.

    The command is an argument list, not a shell string. An interrupt stops
    the subprocess group so a training job does not keep running unseen.
    """

    def emit(value):
        clean = re.sub('\\x1b\\[[0-?]*[ -/]*[@-~]', '', value)
        if on_line is None:
            print(clean, flush=True)
        else:
            on_line(clean)
    cmd = [str(v) for v in parts]
    if any((v in ('train', 'slice', 'volume') for v in cmd)) and (not any((v in ('-h', '--help') for v in cmd))):
        require_compute_session()
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    resolved = shutil.which(cmd[0]) or cmd[0]
    if Path(resolved).is_file():
        env['PATH'] = str(Path(resolved).resolve().parent) + os.pathsep + env.get('PATH', '')
    env['PYTHONUNBUFFERED'] = '1'
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join(cmd))
    print('Complete log:', log_path)
    tail = deque(maxlen=14)
    if on_line is not None:
        on_line('Starting…')
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\n' + datetime.now(timezone.utc).isoformat() + '\n' + shlex.join(cmd) + '\n')
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1, start_new_session=os.name == 'posix')
        last_update = 0
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
                if time.monotonic() - last_update > 1:
                    emit('\n'.join(tail))
                    last_update = time.monotonic()
            result = proc.wait()
        except BaseException:
            if proc.poll() is None:
                if os.name == 'posix':
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait()
            raise
        finally:
            proc.stdout.close()
            emit('\n'.join(tail))
    if result:
        raise subprocess.CalledProcessError(result, cmd)
def available_cores_minus(reserved=4):
    """Available CPUs respect affinity and a positive Slurm per-task allocation."""
    if type(reserved) is not int or reserved < 0:
        raise ValueError('Reserved cores must be a nonnegative integer.')
    available = max(1,os.cpu_count() or 1)
    if hasattr(os,'sched_getaffinity'):
        try:
            available = min(available,max(1,len(os.sched_getaffinity(0))))
        except OSError:
            pass
    assigned = os.environ.get('SLURM_CPUS_PER_TASK','')
    if assigned.isdecimal() and int(assigned)>0:
        available = min(available,int(assigned))
    return max(1,available-reserved)

