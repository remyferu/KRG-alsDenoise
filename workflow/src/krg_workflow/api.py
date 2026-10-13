"""Short, explicit actions for notebooks and scripts in the ALS environment."""
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tifffile
import yaml
from .context import RunContext
from .files import exclusive_lock, json_write
from .plans import read_plan, resolve_center, freeze_run
from .runtime import (all_allocated_gpus, first_visible_gpu, require_compute_session,
                      run_logged, select_output_root, validate_gpu_ids)
from .scans import fingerprint, inspect_scan, validate_als_metadata
from .reconstruction import load_als, validate_stack
from .presets import reconstruction_settings, training_settings, inference_settings
from .denoising import inference_checkpoint_info, validate_checkpoint, validate_denoise_config, validate_denoised_volume
from .history import plot_training_history


@dataclass(frozen=True)
class Workspace:
    output_root: Path
    als_checkout: Path
    denoise_command: tuple

    @classmethod
    def create(cls, als_checkout, denoise_command, output_root=None):
        root = select_output_root(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        command = tuple(str(value) for value in denoise_command)
        if not command:
            raise ValueError('Specify the n2i denoise executable.')
        return cls(root, Path(als_checkout).expanduser().resolve(), command)

    @property
    def queue_path(self):
        return self.output_root/'reconstruction_queue_v3.json'

    def protected_paths(self, run):
        return (self.als_checkout, Path(run.source['path']).parent,
                Path(self.denoise_command[0]).resolve().parent)


@dataclass(frozen=True)
class TrainingRun:
    snapshot: dict
    config: Path
    output: Path
    mode: str
    architecture: str
    label: str

    @property
    def log_path(self):
        return self.output.parent/'logs'/self.label/'train.log'

    @property
    def history_dir(self):
        return self.output.parent/'training_history'/self.label


def inspect_run(workspace, scan_path, run_label, previous=None):
    """Inspect a selection; invalidate callbacks belonging to the previous selection."""
    if previous is not None:
        previous.invalidate()
    if run_label in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', run_label):
        raise ValueError('Use a simple run label with letters, digits, dots, underscores or hyphens.')
    raw = Path(scan_path).expanduser().resolve()
    info, source = inspect_scan(raw), fingerprint(raw)
    if info['shape'][2] % 2:
        raise ValueError('Odd detector width: select an explicit crop before using this workflow.')
    root = workspace.output_root/raw.stem/run_label
    root.mkdir(parents=True, exist_ok=True)
    if (root/'run_plan.json').exists():
        if read_plan(root/'run_plan.json')['reconstruction']['source'] != source:
            raise ValueError('Frozen source changed: use a new run label.')
    elif (root/'v3_run_identity.json').exists():
        saved = json.loads((root/'v3_run_identity.json').read_text(encoding='utf-8'))
        if saved.get('notebook_version') not in ('3.0','3.1','3.2','3.3','3.4','3.5') or saved.get('source') != source:
            raise ValueError('Run identity changed: use a new run label.')
    elif any(root.iterdir()):
        raise ValueError('Existing unplanned directory: use a new run label.')
    else:
        json_write(root/'v3_run_identity.json', dict(notebook_version='3.5',source=source))
    json_write(root/'scan_metadata.json', info)
    print(json.dumps(info, indent=2))
    _, rows, columns = info['shape']
    stack_gib = rows*columns**2*4/2**30
    print(f'Full/even/odd TIFFs: {3*stack_gib:.2f} GiB; paired RAM per training process: {2*stack_gib:.2f} GiB.')
    return RunContext(copy.deepcopy(source), root, copy.deepcopy(info))


def configure_reconstruction(run, workspace, settings=None, use_gpu=None):
    settings = reconstruction_settings() if settings is None else copy.deepcopy(settings)
    if settings['method'] not in ('fbp','gridrec') or settings['als_chunk_sino'] <= 0:
        raise ValueError('Use FBP/gridrec and a positive reconstruction chunk size.')
    if not 0 < settings['fc'] <= 1 or not 0 < settings['preprocessing']['minimum_transmission'] <= 1:
        raise ValueError('Cutoff and minimum transmission must lie in (0, 1].')
    validate_als_metadata(run.source['path'])
    directory = workspace.als_checkout/'backend'
    names = ('ALS_recon_functions.py','ALS_recon_helper.py')
    backend = dict(metadata_adapter='optional_camera_distance_v1', name='als_modern',
                   directory=str(directory), source_sha256={
                       name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names})
    require_compute_session()
    helper = load_als(backend)
    settings['use_gpu'] = bool(helper.als.check_for_gpu() if use_gpu is None else use_gpu)
    if settings['method']=='gridrec':
        settings['use_gpu'] = False
    if settings['use_gpu']:
        validate_gpu_ids(first_visible_gpu())
    run.configure(backend, settings)
    print('Reconstruction settings:', run.settings)
    return run.settings


def estimate_cor(run, rows=None, options=None):
    from .cor import automatic_cor_workflow
    plan, result = automatic_cor_workflow(run.snapshot(), run.info, rows, options, context=run)
    run.automatic_cor, run.manual_cor = result, None
    if plan is not None:
        print('Frozen plan:', run.plan_path, '| midpoint COR:', plan['reconstruction']['center'], 'px')
    else:
        print('COR review required:', result['diagnostics'])
    if result is not None:
        print('Vo row estimates:', list(zip(result['rows'],result['row_centers'])), '| spread:',result['row_spread'])
    return plan, result


def inspect_alignment(run, bright_ratio=1., view_options=None):
    from .cor import alignment_projections, center_estimate, alignment_figure
    captured = run.snapshot()
    run.assert_current(captured, check_source=True)
    first, opposite, index = alignment_projections(run.source['path'], bright_ratio)
    run.alignment_center, _ = center_estimate(first, opposite)
    print('Alignment midpoint COR:',run.alignment_center,'px | opposite index:',index)
    return alignment_figure(first, opposite, run.alignment_center, context=run, view_options=view_options)


def freeze_selected_cor(run, override=None, policy='auto_if_eligible'):
    captured = run.snapshot()
    existing = read_plan(run.plan_path) if run.plan_path.exists() else None
    center, selection = resolve_center(captured, run.manual_cor, override, existing,
                                       run.automatic_cor, policy, context=run)
    saved = freeze_run(captured, run.info, center, selection, context=run)
    print('Frozen plan:',run.plan_path,'| midpoint COR:',center,'px')
    return saved


def reconstruction_paths(run):
    captured = run.snapshot()
    run.assert_current(captured,check_source=True)
    saved = read_plan(run.plan_path)
    resolve_center(captured,existing=saved,context=run)
    return tuple(run.run_dir/saved['reconstruction']['outputs'][name] for name in ('full','even','odd'))


def inspect_pairs(run, view_options=None):
    from .notebook_ui import zoom_image_viewer
    captured = run.snapshot()
    _, rows, cols = run.shape
    paths = tuple(validate_stack(folder,rows,cols) for folder in reconstruction_paths(run))
    def load(row):
        run.assert_current(captured)
        images = [tifffile.imread(items[row]) for items in paths]
        if not all(np.isfinite(image).all() for image in images):
            raise ValueError('Nonfinite reconstruction values.')
        return images
    return zoom_image_viewer(load(rows//2),titles=['Full','Even projections','Odd projections'],
        captured=captured,context=run,title='Reconstruction pairs',load_images=load,
        count=rows,index=rows//2,index_label='Row',**(view_options or {}))


def configure_training(run, workspace, mode='2.5d', architecture='nafnet25d', overrides=None,
                       on_line=None, label='v3_5_1'):
    if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('Training label must contain letters, digits, underscores or hyphens.')
    full, even, odd = reconstruction_paths(run)
    _, rows, cols = run.shape
    for folder in (full,even,odd):
        validate_stack(folder,rows,cols)
    gpu_ids = all_allocated_gpus()
    requested = training_settings(mode, architecture, **(overrides or {}))
    config = run.run_dir/(full.name+'_'+label+'_config.yaml')
    output = run.run_dir/('TrainOutput_'+label)
    with exclusive_lock(run.run_dir/'.run.lock'):
        if output.exists() and not config.is_file():
            raise FileNotFoundError('TrainOutput exists but its YAML is missing. Restore the original YAML.')
        # Isolate this invocation so an older help response cannot satisfy the check.
        help_log = run.run_dir/'logs/denoise_help.log'
        before = help_log.stat().st_size if help_log.exists() else 0
        run_logged(list(workspace.denoise_command)+['train','-h'],help_log,run.run_dir,on_line=on_line)
        with help_log.open('rb') as stream:
            stream.seek(before)
            help_text = stream.read().decode('utf-8',errors='replace')
        if not all(flag in help_text for flag in ('--mode','--resume','--no-search','--architecture')):
            raise ValueError('Installed denoise CLI differs from this repository. Update the n2i installation.')
        if not config.exists():
            run_logged(list(workspace.denoise_command)+['prepare','--file-name',run.source['path'],
                '--out-path-name',str(full),'--config-output',str(config)],run.run_dir/'logs/prepare.log',run.run_dir,on_line=on_line)
        cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
        if not isinstance(cfg,dict) or not isinstance(cfg.get('train'),dict):
            raise ValueError('Training YAML is malformed; restore it before continuing.')
        if output.exists():
            saved_mode = cfg['train'].get('mode','2.5d')
            saved_arch = cfg['train'].get('architecture','unet3d' if saved_mode=='3d' else 'unet')
            if (saved_mode,saved_arch) != (requested['mode'],requested['architecture']):
                raise ValueError('Saved mode or architecture differs. Select its original settings or a new run label.')
        else:
            cfg['train'].update(requested)
            cfg['infer'].update(inference_settings(requested))
        validate_denoise_config(cfg,run.run_dir,full,even,odd,cols,nrows=rows)
        if not output.exists():
            from .files import atomic_text
            atomic_text(config,yaml.safe_dump(cfg,sort_keys=False))
        else:
            print('Keeping saved training settings and normalization for resume.')
    run.training = TrainingRun(run.snapshot(),config,output,cfg['train'].get('mode','2.5d'),requested['architecture'],label)
    print(yaml.safe_dump(cfg,sort_keys=False))
    print('Training GPUs:',gpu_ids,'| batch per GPU:',cfg['train']['mbsz'])
    print('Paired input RAM across ranks:',round(2*rows*cols**2*4*len(gpu_ids.split(','))/2**30,1),'GiB plus overhead.')
    print('Training output:',output,'| log:',run.training.log_path)
    return run.training


def training_config(run, normalized=False, volume=False):
    if run.training is None:
        raise ValueError('Rerun Configure training for this scan.')
    run.assert_current(run.training.snapshot,check_source=True)
    full, even, odd = reconstruction_paths(run)
    cfg = yaml.safe_load(run.training.config.read_text(encoding='utf-8'))
    validate_denoise_config(cfg,run.run_dir,full,even,odd,run.shape[2],nrows=run.shape[1],
                            normalized=normalized,volume=volume)
    train = cfg['train']
    if train.get('mode','2.5d')!=run.training.mode or train.get('architecture',
            'unet3d' if run.training.mode=='3d' else 'unet')!=run.training.architecture:
        raise ValueError('Training configuration changed; rerun Configure training.')
    return cfg


def training_command(run, workspace):
    """Prepare arguments only; no process or model is launched."""
    cfg = training_config(run)
    gpu_ids = all_allocated_gpus()
    if cfg['train'].get('mode')=='3d' and cfg['train']['nb_patches_3d']<len(gpu_ids.split(',')):
        raise ValueError('3D patch count must provide at least one sample per GPU.')
    for folder in reconstruction_paths(run):
        validate_stack(folder,run.shape[1],run.shape[2])
    command = list(workspace.denoise_command)+['train','--config',str(run.training.config),
        '--gpus',gpu_ids,'--mode',run.training.mode,'--no-search','--output-dir',str(run.training.output)]
    if run.training.output.exists():
        if not (run.training.output/'resume.pth').is_file():
            raise ValueError('TrainOutput has no resume checkpoint. Preserve it and use a new run label.')
        command.append('--resume')
    return command


def train_run(run, workspace, enabled=False, on_line=None):
    if not enabled:
        print('Training disabled. Set RUN_TRAINING=True when ready.')
        return None
    require_compute_session()
    command = training_command(run,workspace)
    outcome = 'completed'
    with exclusive_lock(run.run_dir/'.run.lock'):
        try:
            run_logged(command,run.training.log_path,run.run_dir,on_line=on_line)
        except BaseException:
            outcome = 'interrupted or failed'
            raise
        finally:
            try:
                plot_training_history(run.training.log_path,run.training.history_dir,outcome=outcome)
            except Exception as error:
                print('History plot unavailable:',error,'; rerun Show training history.')


def inference_command(run, workspace, kind, checkpoint='ema', gpu_id=None, row=None):
    cfg = training_config(run,normalized=True,volume=kind=='volume')
    validate_checkpoint(run.training.output,checkpoint)
    gpu_id = validate_gpu_ids(first_visible_gpu() if gpu_id is None else gpu_id)
    if len(gpu_id.split(','))!=1:
        raise ValueError('Inference uses one allocated GPU.')
    if kind not in ('slice','volume'):
        raise ValueError('Inference kind must be slice or volume.')
    command = list(workspace.denoise_command)+[kind,'--config',str(run.training.config),
        '--gpus',gpu_id,'--mode',cfg['train'].get('mode','2.5d'),'--checkpoint',checkpoint,
        '--model-dir',str(run.training.output),'--output-dir',str(denoised_path(run,kind,checkpoint))]
    if kind=='slice':
        if type(row) is not int or not 0<=row<run.shape[1]:
            raise ValueError('Select a valid integer slice number.')
        command.extend(['--slice-number',str(row)])
    return command


def denoised_path(run, kind, checkpoint='ema'):
    cfg = training_config(run,normalized=True,volume=kind=='volume')
    full = reconstruction_paths(run)[0]
    stem = full.name[:-4] if full.name.endswith('_rec') else full.name
    suffix = '_nafnet25d' if cfg['train'].get('architecture')=='nafnet25d' else ''
    return run.run_dir/(stem+('_denoised_slices' if kind=='slice' else '_denoised_volume_'+run.training.mode)
                       +suffix+'_'+run.training.label+'_'+checkpoint)


def inspect_denoising(run, workspace, row=None, checkpoint='ema', gpu_id=None, on_line=None, view_options=None):
    from .notebook_ui import image_display_limits, zoom_image_viewer
    row = run.shape[1]//2 if row is None else row
    if type(row) is not int or not 0 <= row < run.shape[1]:
        raise ValueError('Select a valid integer slice number.')
    captured = run.snapshot()
    if run.training is None:
        raise ValueError('Rerun Configure training.')
    if run.training.mode=='2.5d':
        command = inference_command(run,workspace,'slice',checkpoint,gpu_id,row)
        require_compute_session()
        with exclusive_lock(run.run_dir/'.run.lock'):
            run_logged(command,run.run_dir/'logs/slice.log',run.run_dir,on_line=on_line)
        folder = denoised_path(run,'slice',checkpoint)
    else:
        folder = denoised_path(run,'volume',checkpoint)
    path = folder/f'{row:05d}.tiff'
    if not path.is_file():
        print('3D has no single-slice preview. Run volume inference, then inspect this row.')
        return None
    info = inference_checkpoint_info(path)
    epoch_label = str(info['epoch']) if info['epoch'] is not None else 'unknown (not recorded)'
    print('Checkpoint:', info['filename'] or checkpoint, '| training epoch:', epoch_label)
    before = tifffile.imread(validate_stack(reconstruction_paths(run)[0],run.shape[1],run.shape[2])[row])
    after = tifffile.imread(path)
    if after.shape!=before.shape or not np.isfinite(after).all() or not np.isfinite(before).all():
        raise ValueError('Original and denoised images must have matching shapes and finite pixels.')
    low, high = image_display_limits(before)
    residual = before-after
    span = max(float(np.percentile(np.abs(residual),99)),1e-8)
    return zoom_image_viewer([before,after,residual],titles=['Reconstructed','Denoised','Residual'],
        limits=[(low,high),(low,high),(-span,span)],cmaps=['gray','gray','coolwarm'],
        captured=captured,context=run,title=f'Denoising inspection | row {row} | {checkpoint} | epoch {epoch_label}',**(view_options or {}))


def denoise_volume(run, workspace, checkpoint='ema', gpu_id=None, on_line=None, view_options=None):
    from .notebook_ui import browse_tiffs
    command = inference_command(run,workspace,'volume',checkpoint,gpu_id)
    folder = denoised_path(run,'volume',checkpoint)
    if folder.exists():
        raise ValueError('Volume output already exists: '+str(folder)+'. Preserve it before restarting.')
    require_compute_session()
    with exclusive_lock(run.run_dir/'.run.lock'):
        run_logged(command,run.run_dir/'logs/volume.log',run.run_dir,on_line=on_line)
    paths = validate_denoised_volume(folder,run.shape[1],run.shape[2])
    info = inference_checkpoint_info(paths[0])
    epoch_label = str(info['epoch']) if info['epoch'] is not None else 'unknown (not recorded)'
    print('Checkpoint:', info['filename'] or checkpoint, '| training epoch:', epoch_label)
    return browse_tiffs(paths,f'Denoised volume | {checkpoint} | epoch {epoch_label}',context=run,**(view_options or {}))
