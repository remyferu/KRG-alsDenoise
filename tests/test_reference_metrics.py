"""Numerical metric checks only: no model training or optimizer steps."""
import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from skimage.metrics import structural_similarity
from denoise.reference_metrics import (SSIMLoss, paired_reference_metrics,
                                       resolve_ssim_data_range, save_reference_history)

ROOT = Path(__file__).resolve().parents[1]


class ReferenceMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_and_known_constant_ssim(self):
        x, y = torch.ones(2, 1, 17, 19), torch.full((2, 1, 17, 19), 2.)
        metric = SSIMLoss(6.)
        self.assertAlmostEqual(metric(x, x).item(), 0., places=6)
        expected = 1 - (4 + .06**2) / (5 + .06**2)
        self.assertAlmostEqual(metric(x, y).item(), expected, places=5)

    def test_ssim_matches_skimage_with_negative_normalized_intensities(self):
        rng = np.random.default_rng(18)
        x = rng.normal(0, 1, (2, 2, 31, 35)).astype(np.float32)
        y = x + rng.normal(0, .2, x.shape).astype(np.float32)
        scores = [structural_similarity(x[b,c], y[b,c], data_range=6,
                    gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False)
                  for b in range(2) for c in range(2)]
        actual = SSIMLoss(6)(torch.from_numpy(x), torch.from_numpy(y)).item()
        self.assertAlmostEqual(actual, 1 - np.mean(scores), places=5)

    def test_paired_mse_and_ssim_are_detached_and_symmetric(self):
        x, y = torch.zeros(2, 5, 17, 19), torch.full((2, 5, 17, 19), 2.)
        a = torch.zeros(2, 1, 17, 19, requires_grad=True)
        b = torch.ones(2, 1, 17, 19, requires_grad=True)
        metric = SSIMLoss(6)
        mse, ssim = paired_reference_metrics(a, b, x, y, '2.5d', 2, metric)
        self.assertAlmostEqual(mse.item(), 2.5)
        self.assertFalse(mse.requires_grad)
        self.assertFalse(ssim.requires_grad)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)
        swapped = paired_reference_metrics(b, a, y, x, '2.5d', 2, metric)
        torch.testing.assert_close(mse, swapped[0])
        torch.testing.assert_close(ssim, swapped[1])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for metric arithmetic check')
    def test_cuda_ssim_matches_cpu_even_inside_bf16_autocast(self):
        rng = np.random.default_rng(22)
        x = torch.from_numpy(rng.normal(0, 1, (2, 1, 31, 35)).astype(np.float32))
        y = x + .1
        expected = SSIMLoss(6)(x, y)
        metric = SSIMLoss(6).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = metric(x.cuda(), y.cuda())
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)

    def test_3d_mse_uses_all_voxels_but_ssim_uses_center_plane(self):
        x, y = torch.zeros(1, 1, 3, 17, 19), torch.zeros(1, 1, 3, 17, 19)
        a, b = x.clone(), y.clone()
        a[:, :, 0] = 3
        mse, ssim = paired_reference_metrics(a, b, x, y, '3d', 0, SSIMLoss(6))
        self.assertAlmostEqual(mse.item(), 1.5)
        self.assertAlmostEqual(ssim.item(), 0., places=6)

    def test_range_is_fixed_or_derived_from_both_training_stacks(self):
        ds = SimpleNamespace(split0=np.array([-2., 3.]), split1=np.array([-1., 4.]))
        self.assertEqual(resolve_ssim_data_range({}, ds), 6.)
        self.assertEqual(resolve_ssim_data_range({'ssim_data_range': 10.}, ds), 10.)
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                SSIMLoss(value)
        with self.assertRaises(ValueError):
            SSIMLoss(6)(torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))

    def test_reference_csv_preserves_resumed_rows(self):
        rows = [dict(epoch=i, train_l2=.2, eval_l2=.3, train_ssim=.1, eval_ssim=.2,
                      ssim_data_range=6, ssim_window_size=11, mode='2.5d') for i in (10, 11)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'reference_metrics.csv'
            save_reference_history(path, rows[:1]); save_reference_history(path, rows)
            with path.open() as stream:
                result = list(csv.DictReader(stream))
            self.assertEqual([r['epoch'] for r in result], ['10', '11'])
            self.assertFalse(path.with_suffix('.csv.tmp').exists())

    def test_training_objective_remains_l1_and_weighted_lcl(self):
        tree = ast.parse((ROOT/'denoise/train.py').read_text())
        objective = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'loss' for t in node.targets)]
        for expression in objective:
            names = {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}
            self.assertTrue(names <= {'paired_objective', 'loss_view1', 'loss_view2', 'loss_lcl1', 'loss_lcl2'})
        source = (ROOT/'denoise/train.py').read_text()
        self.assertIn('criterion = torch.nn.L1Loss()', source)
        self.assertIn('beta = .01', source)
        self.assertIn("ckpt.get('reference_metric_history', [])", source)


class HistoryMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0,str(ROOT/'workflow/src'))
        from krg_workflow.history import parse_training_history, plot_training_history
        cls.scope = dict(parse_training_history=parse_training_history, plot_training_history=plot_training_history)

    def test_old_and_new_logs_export_and_plot_optional_metrics(self):
        old = ('Epoch 1\n[Train] L1 loss: 1.1\n[Train] LCL loss: 0\n'
               '[Val] L1 loss: 1.2\n[Val] LCL loss: 0.01\n')
        new = ('Epoch 2\n[Train] L1 loss: 1.0\n[Train] LCL loss: 0.01\n'
               '[Val] L1 loss: 1.1\n[Val] LCL loss: 0.02\n'
               '[Train] L2 loss: 0.7\n[Train] SSIM loss: 0.2\n'
               '[Val] L2 loss: 0.8\n[Val] SSIM loss: 0.3\n')
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'train.log'
            log.write_text('Reference metrics only: ssim_data_range=6; window=11\n'+old+new)
            history = self.scope['parse_training_history'](log)
            self.assertIsNone(history['rows'][0]['train_l2'])
            self.assertEqual(history['rows'][1]['eval_ssim'], .3)
            self.assertEqual(history['rows'][1]['ssim_data_range'], 6)
            result = self.scope['plot_training_history'](log, Path(tmp)/'plots', show=False)
            with Path(result['paths']['csv']).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]['eval_l2'], '0.8')
            self.assertTrue(Path(result['paths']['png']).is_file())
            log.write_text(old)
            result = self.scope['plot_training_history'](log, Path(tmp)/'old_plots', show=False)
            self.assertEqual(result['epochs'], 1)

    def test_v35_history_exports_ema_learning_rate_and_validation_scope(self):
        text = 'Metric aggregation: scope=all_ranks; weighting=patches; validation=fixed_held_out\n'
        for epoch in (1, 2, 3):
            text += (f'Epoch {epoch}\n[Train] L1 loss: 2\n[Val] L1 loss: 2.1\n'
                     '[Train] LCL loss: .02\n[Val] LCL loss: .03\n'
                     '[Train] Total loss: 1.01\n[Val] Total loss: 1.065\n'
                     '[EMA] Total loss: 1.04; averaged updates: 12\n'
                     '[LearningRate] lr: 0.0005; reduced: True\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'train.log'
            path.write_text(text, encoding='utf-8')
            parsed = self.scope['parse_training_history'](path)
            self.assertEqual(parsed['metadata']['metric_scope'], 'all_ranks')
            self.assertEqual(parsed['metadata']['validation'], 'fixed_held_out')
            self.assertEqual(parsed['rows'][-1]['ema_total'], 1.04)
            self.assertEqual(parsed['rows'][-1]['learning_rate'], .0005)
            self.assertTrue(parsed['rows'][-1]['lr_reduced'])
            result = self.scope['plot_training_history'](path, Path(tmp)/'plots', moving_window=2, show=False)
            with Path(result['paths']['csv']).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]['ema_total'], '1.04')
            self.assertEqual(rows[-1]['learning_rate'], '0.0005')
            self.assertTrue(Path(result['paths']['png']).is_file())

    def test_total_loss_exports_include_the_logged_stopping_mean(self):
        text = ''
        for epoch, total in ((11, 1.2), (12, 1.), (13, .8)):
            text += (f'Epoch {epoch}\n[Train] L1 loss: 2\n[Train] LCL loss: .01\n'
                     f'[Val] L1 loss: 2\n[Val] LCL loss: .01\n'
                     f'[Train] Total loss: {total}\n[Val] Total loss: {total}\n')
            if epoch == 13:
                text += '[EarlyStopping] mean total loss: 1.0; best qualifying mean: 1.0 at epoch 13; epochs without improvement: 0/100\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'train.log'
            path.write_text(text, encoding='utf-8')
            parsed = self.scope['parse_training_history'](path)
            self.assertIsNone(parsed['rows'][0]['early_stop_mean'])
            self.assertEqual(parsed['rows'][-1]['eval_total'], .8)
            self.assertEqual(parsed['rows'][-1]['early_stop_mean'], 1.)
            result = self.scope['plot_training_history'](path, Path(tmp)/'plots', moving_window=2, show=False)
            with Path(result['paths']['csv']).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]['eval_total'], '0.8')
            self.assertEqual(rows[-1]['early_stop_mean'], '1.0')
            self.assertEqual(rows[0]['early_stop_mean'], '')
            self.assertTrue(Path(result['paths']['svg']).is_file())


if __name__ == '__main__':
    unittest.main()
