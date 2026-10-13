"""Training-log parsing and loss-history plots and exports."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import re
import uuid
import os
import json
import tempfile
import contextlib
import io
from datetime import datetime, timezone


LOSS_KEYS = ('train_l1', 'eval_l1', 'train_lcl', 'eval_lcl', 'train_l2', 'eval_l2',
             'train_ssim', 'eval_ssim', 'train_total', 'eval_total', 'early_stop_mean', 'ema_total')


def record_low_losses(rows):
    """Track strict raw-loss improvements using fully post-warmup epochs."""
    records = {key: [] for key in LOSS_KEYS}
    best = {key: float('inf') for key in LOSS_KEYS}
    for row in rows:
        if row.get('warmup_epoch') is not False:
            continue
        for key in LOSS_KEYS:
            value = row.get(key)
            if value is not None and np.isfinite(value) and value < best[key]:
                best[key] = value
                records[key].append(dict(epoch=row['epoch'], value=value))
    return records


def parse_training_history(log_path):
    """Read epoch means; retain the latest complete record for resumed epochs."""
    path = Path(log_path)
    metadata = dict(mode=None, gpu_count=None, stop_reason=None, best_l1=None, best_lcl=None,
                    metric_scope='final_rank', validation='training_resampling',
                    incomplete_epochs=[], duplicate_epochs=[])
    if not path.is_file():
        return dict(rows=[], metadata=metadata, log_path=str(path))
    number = '[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?'
    metrics = re.compile('\\[(Train|Val)\\]\\s+(L1 loss|LCL loss|EDGE Value|L2 loss|SSIM loss|Total loss):\\s*(' + number + ')')
    names = {('Train', 'L1 loss'): 'train_l1', ('Val', 'L1 loss'): 'eval_l1', ('Train', 'LCL loss'): 'train_lcl', ('Val', 'LCL loss'): 'eval_lcl', ('Val', 'EDGE Value'): 'edge', ('Train', 'L2 loss'): 'train_l2', ('Val', 'L2 loss'): 'eval_l2', ('Train', 'SSIM loss'): 'train_ssim', ('Val', 'SSIM loss'): 'eval_ssim'}
    names.update({('Train', 'Total loss'): 'train_total', ('Val', 'Total loss'): 'eval_total'})
    required = ('train_l1', 'eval_l1', 'train_lcl', 'eval_lcl')
    records = {}
    current = None

    def commit():
        if current is None:
            return
        if all((key in current and np.isfinite(current[key]) for key in required)):
            if current['epoch'] in records:
                metadata['duplicate_epochs'].append(current['epoch'])
            row = dict(current)
            row.setdefault('warmup', None)
            row.setdefault('warmup_epoch', None)
            row.setdefault('edge', None)
            for key in ('train_l2', 'eval_l2', 'train_ssim', 'eval_ssim', 'train_total', 'eval_total',
                        'early_stop_mean', 'ema_total', 'learning_rate', 'lr_reduced'):
                row.setdefault(key, None)
            row['ssim_data_range'] = metadata.get('ssim_data_range')
            records[row['epoch']] = row
        else:
            metadata['incomplete_epochs'].append(current['epoch'])
    with path.open('r', encoding='utf-8', errors='replace') as stream:
        for raw in stream:
            line = re.sub('\\x1b\\[[0-?]*[ -/]*[@-~]', '', raw).rstrip()
            command = re.search('\\bdenoise\\s+train\\b', line)
            fresh = command and '--resume' not in line
            if fresh or 'Initializing model from scratch' in line or 'Fine-tuning from:' in line:
                records.clear()
                current = None
                metadata.update(best_l1=None, best_lcl=None, stop_reason=None, incomplete_epochs=[], duplicate_epochs=[])
            elif command:
                commit()
                current = None
                metadata['stop_reason'] = None
            metric_range = re.search('ssim_data_range=(' + number + ')', line)
            if metric_range:
                metadata['ssim_data_range'] = float(metric_range[1])
            aggregation = re.search('Metric aggregation: scope=(\\w+); weighting=\\w+; validation=(\\w+)', line)
            if aggregation:
                metadata.update(metric_scope=aggregation[1], validation=aggregation[2])
            mode = re.search('(?:Training mode:\\s*|--mode\\s+)(2\\.5d|3d)\\b', line)
            if mode:
                metadata['mode'] = mode[1]
            gpu_count = re.search('Launching training via torchrun \\((\\d+) GPU', line)
            if gpu_count:
                metadata['gpu_count'] = int(gpu_count[1])
            elif command:
                assignment = re.search('--gpus\\s+(\\S+)', line)
                if assignment:
                    metadata['gpu_count'] = len(assignment[1].split(','))
            epoch = re.search('\\bEpoch\\s+(\\d+)\\s*$', line)
            if epoch:
                commit()
                current = dict(epoch=int(epoch[1]))
                continue
            if current is not None:
                metric = metrics.search(line)
                if metric:
                    current[names[metric[1], metric[2]]] = float(metric[3])
                monitor = re.search('\\[EarlyStopping\\] mean total loss:\\s*(' + number + ')', line)
                if monitor:
                    current['early_stop_mean'] = float(monitor[1])
                    current.setdefault('warmup_epoch', False)
                collecting = re.search('Collecting post-warmup epochs:\\s*(\\d+)/', line)
                if collecting:
                    current.setdefault('warmup_epoch', int(collecting[1]) == 0)
                if 'Waiting for warmup to finish.' in line:
                    current.setdefault('warmup_epoch', True)
                ema = re.search('\\[EMA\\] Total loss:\\s*(' + number + ')', line)
                if ema:
                    current['ema_total'] = float(ema[1])
                learning_rate = re.search('\\[LearningRate\\] lr:\\s*(' + number + '); reduced: (True|False)', line)
                if learning_rate:
                    current.update(learning_rate=float(learning_rate[1]), lr_reduced=learning_rate[2] == 'True')
                warmup = re.search('Is model warming up\\?:\\s*(True|False)', line)
                if warmup:
                    current['warmup'] = warmup[1] == 'True'
                warmup_epoch = re.search('Epoch includes warmup updates:\\s*(True|False)', line)
                if warmup_epoch:
                    current['warmup_epoch'] = warmup_epoch[1] == 'True'
            for key, label in (('best_l1', 'Lowest model validation loss'), ('best_lcl', 'Lowest model LCL loss')):
                best = re.search(label + '\\s+(' + number + ')\\s+at epoch\\s+(\\d+)', line)
                if best and int(best[2]) > 0:
                    metadata[key] = dict(value=float(best[1]), epoch=int(best[2]))
            if 'Early stopping:' in line:
                metadata['stop_reason'] = 'early stopping'
    commit()
    rows = [records[key] for key in sorted(records)]
    previous_warmup = None
    for row in rows:
        if row['warmup_epoch'] is None:
            if row['warmup'] is True:
                row['warmup_epoch'] = True
            elif row['warmup'] is False:
                # Older logs report the end-of-epoch state, including the transition.
                row['warmup_epoch'] = previous_warmup is True
        if row['warmup'] is not None:
            previous_warmup = row['warmup']
    warm_epochs = [row['epoch'] for row in rows if row['warmup_epoch'] is True]
    warmup_finished = any(row['warmup'] is False or row['warmup_epoch'] is False for row in rows)
    metadata['warmup_end'] = max(warm_epochs) if warm_epochs and warmup_finished else None
    return dict(rows=rows, metadata=metadata, log_path=str(path))


def plot_training_history(log_path, output_dir=None, moving_window=10, show=True, outcome=None):
    """Export log-log loss curves and record lows after warmup."""
    if type(moving_window) is not int or moving_window < 1:
        raise ValueError('moving_window must be a positive integer.')
    history = parse_training_history(log_path)
    rows, metadata = (history['rows'], history['metadata'])
    if not rows:
        print('READY: no complete training-loss epochs yet in', log_path)
        return dict(status='no_history', log_path=str(log_path), epochs=0)
    log_path = Path(log_path)
    folder = Path(output_dir) if output_dir is not None else log_path.parent.parent / 'training_history'
    folder.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(rows[0]['epoch'], rows[-1]['epoch'] + 1)
    values = {key: np.full(len(epochs), np.nan) for key in ('train_l1', 'eval_l1', 'train_lcl', 'eval_lcl', 'train_l2', 'eval_l2', 'train_ssim', 'eval_ssim', 'train_total', 'eval_total', 'early_stop_mean', 'ema_total', 'learning_rate')}
    for row in rows:
        for key in values:
            values[key][row['epoch'] - epochs[0]] = row[key]

    def smooth(array):
        result = np.full(len(array), np.nan)
        if len(array) >= moving_window:
            result[moving_window - 1:] = np.convolve(array, np.ones(moving_window) / moving_window, mode='valid')
        return result

    record_lows = record_low_losses(rows)
    best_l1 = record_lows['eval_l1'][-1] if record_lows['eval_l1'] else None
    best_lcl = record_lows['eval_lcl'][-1] if record_lows['eval_lcl'] else None

    def positive(array):
        return np.where(np.isfinite(array) & (array > 0), array, np.nan)

    def mark_records(ax, key, color):
        points = [p for p in record_lows[key] if p['epoch'] > 0 and p['value'] > 0]
        if points:
            ax.scatter([p['epoch'] for p in points], [p['value'] for p in points],
                       s=24, color=color, edgecolors='white', linewidths=.5, zorder=6)
    paths = {name: folder / ('training_loss_history.' + name) for name in ('png', 'svg')}
    paths['csv'] = folder / 'training_history.csv'
    temp_csv = folder / ('.training_history.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp_csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_l1', 'eval_l1', 'train_lcl', 'eval_lcl', 'edge', 'warmup', 'warmup_epoch', 'train_l2', 'eval_l2', 'train_ssim', 'eval_ssim', 'ssim_data_range', 'train_total', 'eval_total', 'early_stop_mean', 'ema_total', 'learning_rate', 'lr_reduced'])
            writer.writeheader()
            writer.writerows(rows)
        temp_csv.replace(paths['csv'])
    finally:
        temp_csv.unlink(missing_ok=True)
    style = {'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelcolor': '#243447', 'text.color': '#243447', 'axes.edgecolor': '#cad4de', 'xtick.color': '#556575', 'ytick.color': '#556575', 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'}
    with plt.rc_context(style):
        has_reference = any((np.isfinite(values[key]).any() for key in ('train_l2', 'eval_l2', 'train_ssim', 'eval_ssim')))
        has_total = any(np.isfinite(values[key]).any() for key in ('train_total', 'eval_total'))
        has_lr = np.isfinite(values['learning_rate']).any()
        panels = 2 + 2 * int(has_reference) + int(has_total) + int(has_lr)
        fig, axes = plt.subplots(panels, 1, figsize=(12, 8.8 + 3.1 * (panels - 2)), sharex=True)
        try:
            fig.patch.set_facecolor('white')
            mode = metadata['mode'] or 'CT'
            mode = '2.5D' if mode == '2.5d' else '3D' if mode == '3d' else mode
            fig.suptitle(f'Noise2Inverse360 | {mode} training loss history', fontsize=20, fontweight='bold', x=0.08, ha='left', y=0.978)
            subtitle = f'{len(rows)} epochs'
            if metadata['gpu_count'] is not None:
                subtitle += f" · {metadata['gpu_count']} GPUs"
            reason = metadata['stop_reason'] or outcome
            if reason:
                subtitle += f" · {reason} after epoch {rows[-1]['epoch']}"
            fig.text(0.08, 0.936, subtitle, fontsize=12, color='#596b7d')
            colors = {'Training': '#2166ac', 'Evaluation': '#d26026'}
            specifications = [(axes[0], 'train_l1', 'eval_l1', 'L1 loss', 'Epoch-average L1 loss', best_l1), (axes[1], 'train_lcl', 'eval_lcl', 'Local contrast loss (LCL)', 'Epoch-average weighted LCL loss', best_lcl)]
            if has_reference:
                specifications.extend([(axes[2], 'train_l2', 'eval_l2', 'L2 / MSE (reference only)', 'Epoch-average paired MSE', None), (axes[3], 'train_ssim', 'eval_ssim', 'SSIM loss (reference only)', 'Epoch-average paired 1 - SSIM', None)])
            if has_total:
                specifications.append((axes[2+2*int(has_reference)], 'train_total', 'eval_total', 'Total objective and early stopping', 'Epoch-average total loss', None))
            for ax, train_key, eval_key, title, ylabel, best in specifications:
                ax.set_facecolor('#fbfcfe')
                end = metadata['warmup_end']
                if end is not None:
                    ax.axvspan(epochs[0], end, color='#ccd5df', alpha=0.25, zorder=0)
                    ax.axvline(end, color='#778798', linewidth=1, linestyle='--', zorder=1)
                elif any(row['warmup_epoch'] is True for row in rows):
                    ax.axvspan(epochs[0], epochs[-1], color='#ccd5df', alpha=0.25, zorder=0)
                for label, key in (('Training', train_key), ('Evaluation', eval_key)):
                    averaged = smooth(values[key])
                    has_average = np.isfinite(positive(averaged)).any()
                    ax.plot(epochs, positive(values[key]), color=colors[label], alpha=0.22 if has_average else 0.8, linewidth=0.8, label=None if has_average else label, zorder=2)
                    if has_average:
                        ax.plot(epochs, positive(averaged), color=colors[label], linewidth=2.2, label=label, zorder=3)
                    mark_records(ax, key, colors[label])
                if eval_key == 'eval_total' and np.isfinite(values['early_stop_mean']).any():
                    ax.plot(epochs, positive(values['early_stop_mean']), color='#243447', linestyle='--',
                            linewidth=1.7, label='Stopping monitor (logged)', zorder=4)
                    mark_records(ax, 'early_stop_mean', '#243447')
                if eval_key == 'eval_total' and np.isfinite(values['ema_total']).any():
                    available = np.isfinite(values['ema_total'])
                    ax.plot(epochs[available], positive(values['ema_total'][available]), color='#7953a6',
                            linewidth=1.7, label='EMA weights (raw evaluation)', zorder=4)
                    mark_records(ax, 'ema_total', '#7953a6')
                ax.set_title(title, loc='left', fontweight='bold', fontsize=14, pad=12)
                ax.set_ylabel(ylabel)
                plotted_keys = (train_key, eval_key, 'early_stop_mean', 'ema_total') if eval_key == 'eval_total' else (train_key, eval_key)
                if not any(np.isfinite(positive(values[key])).any() for key in plotted_keys):
                    ax.set_ylim(1e-8, 1)
                    ax.text(.5, .5, 'No positive loss values', transform=ax.transAxes,
                            ha='center', color='#687889', fontsize=10)
                ax.set_xscale('log', nonpositive='mask')
                ax.set_yscale('log', nonpositive='mask')
                ax.grid(which='major', axis='both', color='#dce3ea', linewidth=0.7)
                ax.grid(which='minor', axis='both', color='#e8edf2', linewidth=0.4)
                positive_epochs = epochs[epochs > 0]
                ax.set_xlim(positive_epochs[0] / 1.15 if len(positive_epochs) else 1,
                            positive_epochs[-1] * 1.15 if len(positive_epochs) else 10)
                ax.tick_params(axis='both', length=0, pad=7)
                ax.legend(loc='upper right', frameon=False, ncol=2)
                if best is not None:
                    label = 'Lowest post-warmup evaluation L1' if train_key == 'train_l1' else 'Lowest post-warmup evaluation LCL'
                    note = f"{label}: {best['value']:.6f}\nEpoch {best['epoch']}"
                    position = (0.03, 0.12) if train_key == 'train_l1' else (0.14, 0.72)
                    box = dict(boxstyle='round,pad=.5', facecolor='white', edgecolor='#e4e8ed')
                    if best['epoch'] > 0 and best['value'] > 0:
                        ax.annotate(note, xy=(best['epoch'], best['value']), xytext=position, textcoords='axes fraction', fontsize=10, bbox=box, arrowprops=dict(arrowstyle='-', color='#a7b2be', linewidth=1))
                    else:
                        ax.text(*position, note + ' (outside log axes)', transform=ax.transAxes, fontsize=10, bbox=box)
            if metadata['warmup_end'] is not None:
                axes[0].text(metadata['warmup_end'] + 5, 0.88, 'LCL warmup ended', transform=axes[0].get_xaxis_transform(), fontsize=9, color='#687889')
            if has_lr:
                ax = axes[-1]
                available = np.isfinite(values['learning_rate'])
                ax.step(epochs[available], positive(values['learning_rate'][available]), where='post', color='#2166ac', linewidth=2)
                ax.set_yscale('log', nonpositive='mask')
                ax.set_title('Learning rate', loc='left', fontweight='bold', fontsize=14, pad=12)
                ax.set_ylabel('Adam learning rate')
                ax.grid(which='major', axis='both', color='#dce3ea', linewidth=.7)
                ax.grid(which='minor', axis='both', color='#e8edf2', linewidth=.4)
            axes[-1].set_xlabel('Epoch (log scale)', labelpad=10)
            record_note = ('Dots: successive record lows after warmup.'
                           if any(row['warmup_epoch'] is not None for row in rows) else
                           'Warmup status unavailable; record-low markers omitted.')
            fig.text(0.08, 0.073, record_note+' Log axes omit nonpositive values.', fontsize=10, color='#596b7d')
            curve_note = f'Faint: raw epochs. Solid: trailing {moving_window}-epoch display averages.'
            if has_total and np.isfinite(values['early_stop_mean']).any():
                curve_note += ' Dashed: logged stopping monitor.'
            fig.text(0.08, 0.047, curve_note, fontsize=10, color='#596b7d')
            basis = ('Fixed held-out patches; sample-weighted metrics across all GPUs.'
                     if metadata['validation'] == 'fixed_held_out' else
                     'Evaluation resamples training data; sample-weighted metrics across all GPUs.'
                     if metadata['metric_scope'] == 'all_ranks' else
                     'Evaluation resamples training data; metrics represent the final GPU rank.')
            fig.text(0.08, 0.021, basis+' LCL uses its logged weight; L2/SSIM are reference metrics.', fontsize=9, color='#596b7d')
            fig.subplots_adjust(left=0.105, right=0.97, top=0.87, bottom=0.12, hspace=0.31)
            for extension in ('png', 'svg'):
                temp = folder / ('.training_loss_history.' + uuid.uuid4().hex + '.tmp')
                try:
                    fig.savefig(temp, format=extension, dpi=180, facecolor='white')
                    temp.replace(paths[extension])
                finally:
                    temp.unlink(missing_ok=True)
            if show:
                plt.show()
        finally:
            plt.close(fig)
    print('DONE: saved training history:', paths['png'])
    return dict(status='saved', epochs=len(rows), first_epoch=int(epochs[0]), last_epoch=int(epochs[-1]), best_l1=best_l1, best_lcl=best_lcl, record_lows=record_lows, metadata=metadata, paths={key: str(value) for key, value in paths.items()})


def snapshot_training_history(log_path, output_root=None, moving_window=10, show=True):
    """Plot a bounded copy of an append-only log in a separate snapshot directory.

    Read through the size observed at open, excluding an unfinished final line.
    Each call writes to a unique directory without modifying the source log,
    training outputs or locks. No model or training process is loaded.
    """
    if type(moving_window) is not int or moving_window < 1:
        raise ValueError('moving_window must be a positive integer.')
    source = Path(log_path).expanduser().resolve()
    captured = datetime.now(timezone.utc)
    record = dict(log_path=str(source), captured_utc=captured.isoformat(timespec='seconds'))
    try:
        with source.open('rb') as stream:
            observed = os.fstat(stream.fileno())
            data = stream.read(observed.st_size)
    except FileNotFoundError:
        print('Training log not found:', source)
        return dict(record, status='no_log', epochs=0, log_tail=[])

    complete = data[:data.rfind(b'\n') + 1]
    tail = complete.decode('utf-8', errors='replace').splitlines()[-20:]
    record.update(observed_log_bytes=observed.st_size, captured_bytes=len(complete),
                  omitted_tail_bytes=len(data) - len(complete),
                  log_modified_utc=datetime.fromtimestamp(observed.st_mtime, timezone.utc).isoformat(timespec='seconds'),
                  log_tail=[re.sub('\x1b\\[[0-?]*[ -/]*[@-~]', '', line) for line in tail])
    root = Path(output_root).expanduser().resolve() if output_root is not None else source.parent/'training_snooper'
    folder = root/(captured.strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex)
    # A private copy keeps plotting independent of later producer writes.
    with tempfile.TemporaryDirectory(prefix='krg-training-snapshot-') as temporary:
        captured_log = Path(temporary)/'train.log'
        captured_log.write_bytes(complete)
        messages = io.StringIO()
        with contextlib.redirect_stdout(messages):
            result = plot_training_history(captured_log, folder, moving_window=moving_window, show=show)
        print(messages.getvalue().replace(str(captured_log), str(source)), end='')
    result.update(record)
    if result['status'] == 'saved':
        metadata_path = folder/'snapshot.json'
        result['paths']['snapshot'] = str(metadata_path)
        metadata_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result
