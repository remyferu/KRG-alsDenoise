"""History helpers extracted from the v3.3 CT workflow.

No notebook cell execution or training occurs on import.
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import re
import uuid


def parse_training_history(log_path):
    """Read epoch means; retain the latest complete record for resumed epochs."""
    path = Path(log_path)
    metadata = dict(mode=None, gpu_count=None, stop_reason=None, best_l1=None, best_lcl=None, incomplete_epochs=[], duplicate_epochs=[])
    if not path.is_file():
        return dict(rows=[], metadata=metadata, log_path=str(path))
    number = '[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?'
    metrics = re.compile('\\[(Train|Val)\\]\\s+(L1 loss|LCL loss|EDGE Value|L2 loss|SSIM loss):\\s*(' + number + ')')
    names = {('Train', 'L1 loss'): 'train_l1', ('Val', 'L1 loss'): 'eval_l1', ('Train', 'LCL loss'): 'train_lcl', ('Val', 'LCL loss'): 'eval_lcl', ('Val', 'EDGE Value'): 'edge', ('Train', 'L2 loss'): 'train_l2', ('Val', 'L2 loss'): 'eval_l2', ('Train', 'SSIM loss'): 'train_ssim', ('Val', 'SSIM loss'): 'eval_ssim'}
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
            row.setdefault('edge', None)
            for key in ('train_l2', 'eval_l2', 'train_ssim', 'eval_ssim'):
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
                warmup = re.search('Is model warming up\\?:\\s*(True|False)', line)
                if warmup:
                    current['warmup'] = warmup[1] == 'True'
            for key, label in (('best_l1', 'Lowest model validation loss'), ('best_lcl', 'Lowest model LCL loss')):
                best = re.search(label + '\\s+(' + number + ')\\s+at epoch\\s+(\\d+)', line)
                if best and int(best[2]) > 0:
                    metadata[key] = dict(value=float(best[1]), epoch=int(best[2]))
            if 'Early stopping:' in line:
                metadata['stop_reason'] = 'early stopping'
    commit()
    rows = [records[key] for key in sorted(records)]
    metadata['warmup_end'] = next((row['epoch'] for row in rows if row['warmup'] is False), None) if any((row['warmup'] is True for row in rows)) else None
    return dict(rows=rows, metadata=metadata, log_path=str(path))


def plot_training_history(log_path, output_dir=None, moving_window=10, show=True, outcome=None):
    """Save raw/smoothed L1 and LCL plots without loading a model or checkpoint."""
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
    values = {key: np.full(len(epochs), np.nan) for key in ('train_l1', 'eval_l1', 'train_lcl', 'eval_lcl', 'train_l2', 'eval_l2', 'train_ssim', 'eval_ssim')}
    for row in rows:
        for key in values:
            values[key][row['epoch'] - epochs[0]] = row[key]

    def smooth(array):
        result = np.full(len(array), np.nan)
        if len(array) >= moving_window:
            result[moving_window - 1:] = np.convolve(array, np.ones(moving_window) / moving_window, mode='valid')
        return result

    def best_record(key, metric):
        tracked = metadata[key]
        if tracked is not None:
            return tracked
        candidates = rows
        if key == 'best_lcl' and metadata['warmup_end'] is not None:
            candidates = [row for row in rows if row['epoch'] > metadata['warmup_end']]
        if not candidates:
            return None
        row = min(candidates, key=lambda item: item[metric])
        return dict(epoch=row['epoch'], value=row[metric])
    best_l1 = best_record('best_l1', 'eval_l1')
    best_lcl = best_record('best_lcl', 'eval_lcl')
    paths = {name: folder / ('training_loss_history.' + name) for name in ('png', 'svg')}
    paths['csv'] = folder / 'training_history.csv'
    temp_csv = folder / ('.training_history.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp_csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_l1', 'eval_l1', 'train_lcl', 'eval_lcl', 'edge', 'warmup', 'train_l2', 'eval_l2', 'train_ssim', 'eval_ssim', 'ssim_data_range'])
            writer.writeheader()
            writer.writerows(rows)
        temp_csv.replace(paths['csv'])
    finally:
        temp_csv.unlink(missing_ok=True)
    style = {'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelcolor': '#243447', 'text.color': '#243447', 'axes.edgecolor': '#cad4de', 'xtick.color': '#556575', 'ytick.color': '#556575', 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'}
    with plt.rc_context(style):
        has_reference = any((np.isfinite(values[key]).any() for key in ('train_l2', 'eval_l2', 'train_ssim', 'eval_ssim')))
        fig, axes = plt.subplots(4 if has_reference else 2, 1, figsize=(12, 15 if has_reference else 8.8), sharex=True)
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
            for ax, train_key, eval_key, title, ylabel, best in specifications:
                ax.set_facecolor('#fbfcfe')
                end = metadata['warmup_end']
                if end is not None:
                    ax.axvspan(epochs[0], end, color='#ccd5df', alpha=0.25, zorder=0)
                    ax.axvline(end, color='#778798', linewidth=1, linestyle='--', zorder=1)
                for label, key in (('Training', train_key), ('Evaluation', eval_key)):
                    averaged = smooth(values[key])
                    has_average = np.isfinite(averaged).any()
                    ax.plot(epochs, values[key], color=colors[label], alpha=0.22 if has_average else 0.8, linewidth=0.8, marker='.' if len(rows) == 1 else None, label=None if has_average else label, zorder=2)
                    if has_average:
                        ax.plot(epochs, averaged, color=colors[label], linewidth=2.2, label=label, zorder=3)
                ax.set_title(title, loc='left', fontweight='bold', fontsize=14, pad=12)
                ax.set_ylabel(ylabel)
                ax.grid(axis='y', color='#dce3ea', linewidth=0.7)
                ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
                ax.tick_params(axis='both', length=0, pad=7)
                ax.legend(loc='upper right', frameon=False, ncol=2)
                if best is not None:
                    label = 'Best evaluation L1' if train_key == 'train_l1' else 'Best tracked evaluation LCL'
                    note = f"{label}: {best['value']:.6f}\nEpoch {best['epoch']}"
                    position = (0.64, 0.8) if train_key == 'train_l1' else (0.14, 0.72)
                    box = dict(boxstyle='round,pad=.5', facecolor='white', edgecolor='#e4e8ed')
                    if epochs[0] <= best['epoch'] <= epochs[-1]:
                        ax.scatter(best['epoch'], best['value'], marker='*', s=155, color=colors['Evaluation'], edgecolors='white', linewidth=0.8, zorder=5)
                        ax.annotate(note, xy=(best['epoch'], best['value']), xytext=position, textcoords='axes fraction', fontsize=10, bbox=box, arrowprops=dict(arrowstyle='-', color='#a7b2be', linewidth=1))
                    else:
                        ax.text(*position, note + ' (earlier history)', transform=ax.transAxes, fontsize=10, bbox=box)
            if metadata['warmup_end'] is not None:
                axes[0].text(metadata['warmup_end'] + 5, 0.88, 'LCL warmup ended', transform=axes[0].get_xaxis_transform(), fontsize=9, color='#687889')
            axes[-1].set_xlabel('Epoch', labelpad=10)
            fig.text(0.08, 0.047, f'Faint lines: raw epoch averages. Solid lines: trailing {moving_window}-epoch moving averages.', fontsize=10, color='#596b7d')
            fig.text(0.08, 0.021, 'Evaluation uses the training dataset; logged metrics represent one GPU process. LCL is plotted with its logged weight. L2/SSIM are reference metrics only.', fontsize=9, color='#596b7d')
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
    return dict(status='saved', epochs=len(rows), first_epoch=int(epochs[0]), last_epoch=int(epochs[-1]), best_l1=best_l1, best_lcl=best_lcl, metadata=metadata, paths={key: str(value) for key, value in paths.items()})
