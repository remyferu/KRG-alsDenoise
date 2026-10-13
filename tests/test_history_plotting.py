"""Log axes and raw loss records from scalar fixtures; no model execution."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'workflow/src'))
from krg_workflow import history


def epoch_text(epoch, value, warmup=None, warmup_epoch=None, reference=True):
    text = (f'Epoch {epoch}\n[Train] L1 loss: {value}\n[Val] L1 loss: {value*1.1}\n'
            f'[Train] LCL loss: {value*.01}\n[Val] LCL loss: {value*.02}\n')
    if reference:
        text += (f'[Train] L2 loss: {value*.5}\n[Val] L2 loss: {value*.6}\n'
                 f'[Train] SSIM loss: {value*.02}\n[Val] SSIM loss: {value*.03}\n'
                 f'[Train] Total loss: {value*.51}\n[Val] Total loss: {value*.57}\n'
                 f'[EMA] Total loss: {value*.55}\n[LearningRate] lr: 0.001; reduced: False\n'
                 f'[EarlyStopping] mean total loss: {value*.57}\n')
    if warmup is not None:
        text += f'Is model warming up?: {warmup}\n'
    if warmup_epoch is not None:
        text += f'Epoch includes warmup updates: {warmup_epoch}\n'
    return text


class HistoryPlottingTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'train.log'
            path.write_text(text, encoding='utf-8')
            return history.parse_training_history(path)

    def test_strict_records_reset_after_warmup_for_every_loss(self):
        rows = [dict(epoch=epoch, warmup_epoch=epoch <= 2,
                     **{key: value for key in history.LOSS_KEYS})
                for epoch, value in enumerate((.001, .002, 10, 9, 9, 11, 8), 1)]
        records = history.record_low_losses(rows)
        for points in records.values():
            self.assertEqual(points, [dict(epoch=3, value=10), dict(epoch=4, value=9), dict(epoch=7, value=8)])

    def test_legacy_end_of_epoch_status_excludes_transition_epoch(self):
        parsed = self.parse(epoch_text(1, .01, True, reference=False)
                            + epoch_text(2, .02, False, reference=False)
                            + epoch_text(3, 2, False, reference=False))
        self.assertEqual([r['warmup_epoch'] for r in parsed['rows']], [True, True, False])
        self.assertEqual(parsed['metadata']['warmup_end'], 2)
        self.assertEqual(history.record_low_losses(parsed['rows'])['eval_l1'], [dict(epoch=3, value=2.2)])

    def test_explicit_flag_handles_first_epoch_transition_and_resumed_logs(self):
        parsed = self.parse(epoch_text(1, .01, False, True) + epoch_text(2, 2, False, False))
        self.assertEqual([r['warmup_epoch'] for r in parsed['rows']], [True, False])
        self.assertEqual(history.record_low_losses(parsed['rows'])['train_l1'], [dict(epoch=2, value=2)])
        resumed = self.parse(epoch_text(100, 2, False, False) + epoch_text(101, 1, False, False))
        self.assertEqual([p['epoch'] for p in history.record_low_losses(resumed['rows'])['eval_l1']], [100, 101])

    def test_existing_monitor_status_identifies_short_warmup(self):
        text = epoch_text(1, .1, False, reference=False) + '[EarlyStopping] Collecting post-warmup epochs: 0/10\n'
        text += epoch_text(2, 2, False, reference=False) + '[EarlyStopping] Collecting post-warmup epochs: 1/10\n'
        text += epoch_text(3, 1, False, reference=False) + '[EarlyStopping] mean total loss: 1\n'
        parsed = self.parse(text)
        self.assertEqual([r['warmup_epoch'] for r in parsed['rows']], [True, False, False])

    def test_unknown_warmup_missing_metrics_and_nonfinite_values(self):
        parsed = self.parse(epoch_text(1, 1, reference=False))
        self.assertTrue(all(not points for points in history.record_low_losses(parsed['rows']).values()))
        rows = [dict(epoch=1, warmup_epoch=False, eval_l2=float('nan')),
                dict(epoch=2, warmup_epoch=False, eval_l2=2),
                dict(epoch=3, warmup_epoch=False, eval_l2=None),
                dict(epoch=4, warmup_epoch=False, eval_l2=0),
                dict(epoch=5, warmup_epoch=False, eval_l2=1)]
        self.assertEqual(history.record_low_losses(rows)['eval_l2'], [dict(epoch=2, value=2), dict(epoch=4, value=0)])

    def test_semilog_axes_grids_and_marker_coordinates_match_raw_epoch_records(self):
        figures = []
        close = history.plt.close

        def capture(figure):
            if hasattr(figure, 'axes'):
                figures.append(figure)
            close(figure)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'train.log'
            path.write_text(''.join(epoch_text(i, value, i < 3, i <= 3)
                                    for i, value in enumerate((.01, .02, .03, 2, 1, 1, 3, .5), 1)), encoding='utf-8')
            with patch.object(history.plt, 'close', side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
                result = history.plot_training_history(path, Path(temporary)/'plots', moving_window=3, show=False)
            self.assertEqual(len(figures), 1)
            axes = figures[0].axes
            self.assertEqual(len(axes), 6)
            for ax in axes:
                self.assertEqual(ax.get_xscale(), 'linear')
                self.assertEqual(ax.get_yscale(), 'log')
                self.assertGreater(ax.get_xlim()[0], 0)
                for axis in (ax.xaxis, ax.yaxis):
                    self.assertTrue(all(t.gridline.get_visible() for t in axis.get_major_ticks()))
                    self.assertTrue(all(t.gridline.get_visible() for t in axis.get_minor_ticks()))
            for ax, keys in zip(axes[:5], [('train_l1', 'eval_l1'), ('train_lcl', 'eval_lcl'),
                                         ('train_l2', 'eval_l2'), ('train_ssim', 'eval_ssim'),
                                         ('train_total', 'eval_total', 'early_stop_mean', 'ema_total')]):
                self.assertEqual(len(ax.collections), len(keys))
                for scatter, key in zip(ax.collections, keys):
                    points = result['record_lows'][key]
                    self.assertEqual([p['epoch'] for p in points], [4, 5, 8])
                    np.testing.assert_allclose(scatter.get_offsets(), [[p['epoch'], p['value']] for p in points])
            self.assertTrue(Path(result['paths']['png']).is_file())
            self.assertTrue(Path(result['paths']['svg']).is_file())

    def test_zero_losses_stay_in_exports_and_create_no_invalid_log_points(self):
        figures = []
        close = history.plt.close

        def capture(figure):
            if hasattr(figure, 'axes'):
                figures.append(figure)
            close(figure)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'train.log'
            path.write_text(epoch_text(1, 0, False, False, reference=False), encoding='utf-8')
            with warnings.catch_warnings(record=True) as caught, patch.object(history.plt, 'close', side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter('always')
                result = history.plot_training_history(path, Path(temporary)/'plots', show=False)
            self.assertFalse(any('log' in str(w.message).lower() for w in caught))
            self.assertEqual(result['record_lows']['eval_lcl'], [dict(epoch=1, value=0)])
            for ax in figures[0].axes:
                self.assertFalse(ax.collections)
            self.assertIn(',0.0,', Path(result['paths']['csv']).read_text())


if __name__ == '__main__':
    unittest.main()
