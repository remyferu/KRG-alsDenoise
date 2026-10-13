"""Stopping-policy arithmetic and resume checks; no model training."""
import ast
import copy
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from denoise.early_stopping import MovingAverageEarlyStopping

ROOT = Path(__file__).resolve().parents[1]


class MovingAverageStoppingTests(unittest.TestCase):
    def test_warmup_and_partial_window_never_consume_patience(self):
        monitor = MovingAverageEarlyStopping(patience=1)
        for epoch in range(1, 5):
            self.assertFalse(monitor.update(epoch, 0., warming_up=True))
        for epoch in range(5, 14):
            self.assertFalse(monitor.update(epoch, 2., warming_up=False))
            self.assertIsNone(monitor.current_mean)
            self.assertEqual(monitor.epochs_since_improvement, 0)
        self.assertFalse(monitor.update(14, 2., warming_up=False))
        self.assertEqual(monitor.current_mean, 2.)
        self.assertEqual(monitor.best_mean, 2.)
        self.assertEqual(monitor.best_epoch, 14)
        self.assertTrue(monitor.update(15, 2., warming_up=False))

    def test_window_slides_and_an_isolated_dip_does_not_reset_patience(self):
        monitor = MovingAverageEarlyStopping(patience=3, min_delta=.1)
        for epoch in range(1, 11):
            monitor.update(epoch, 1., warming_up=False)
        self.assertFalse(monitor.update(11, .5, warming_up=False))
        self.assertAlmostEqual(monitor.current_mean, .95)
        self.assertEqual(monitor.best_mean, 1.)
        self.assertEqual(monitor.epochs_since_improvement, 1)
        monitor.update(12, 1., warming_up=False)
        self.assertTrue(monitor.update(13, 1., warming_up=False))

    def test_small_cumulative_improvements_can_cross_the_threshold(self):
        monitor = MovingAverageEarlyStopping(patience=10, window=1, min_delta=.125)
        monitor.update(1, 1., warming_up=False)
        monitor.update(2, .9375, warming_up=False)
        self.assertEqual(monitor.epochs_since_improvement, 1)
        monitor.update(3, .875, warming_up=False)
        self.assertEqual(monitor.epochs_since_improvement, 2)
        monitor.update(4, .8125, warming_up=False)
        self.assertEqual(monitor.best_mean, .8125)
        self.assertEqual(monitor.best_epoch, 4)
        self.assertEqual(monitor.epochs_since_improvement, 0)

    def test_resume_matches_uninterrupted_monitor_at_every_boundary(self):
        values = [.01] * 3 + [1., 1.1, .9, 1.05, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., .6, .6, .6]
        for boundary in range(1, len(values)):
            uninterrupted = MovingAverageEarlyStopping(patience=100)
            for epoch, loss in enumerate(values[:boundary], 1):
                uninterrupted.update(epoch, loss, warming_up=epoch <= 3)
            resumed = MovingAverageEarlyStopping(patience=100)
            state = pickle.loads(pickle.dumps(uninterrupted.state_dict()))
            self.assertTrue(resumed.load_state_dict(state))
            for epoch, loss in enumerate(values[boundary:], boundary + 1):
                expected = uninterrupted.update(epoch, loss, warming_up=epoch <= 3)
                self.assertEqual(resumed.update(epoch, loss, warming_up=epoch <= 3), expected)
                self.assertEqual(resumed.state_dict(), uninterrupted.state_dict())

    def test_legacy_or_changed_policy_starts_a_fresh_window(self):
        monitor = MovingAverageEarlyStopping(patience=1, window=2)
        monitor.update(1, 1., warming_up=False)
        monitor.update(2, 1., warming_up=False)
        monitor.update(3, 1., warming_up=False)
        old = monitor.state_dict()
        for state in (None, dict(old, window=10), dict(old, min_delta=.1), dict(old, metric='eval_l1')):
            self.assertFalse(monitor.load_state_dict(state))
            self.assertEqual(monitor.epochs_since_improvement, 0)
            self.assertIsNone(monitor.best_mean)
            self.assertFalse(monitor.update(41, .5, warming_up=False))
            self.assertIsNone(monitor.current_mean)

    def test_zero_patience_disables_stopping(self):
        monitor = MovingAverageEarlyStopping(patience=0, window=2)
        for epoch in range(1, 25):
            self.assertFalse(monitor.update(epoch, 1., warming_up=False))

    def test_policy_and_nonfinite_metric_validation(self):
        for settings in ({'window': 0}, {'window': True}, {'window': 1.5},
                         {'patience': -1}, {'min_delta': -1}, {'min_delta': float('nan')},
                         {'min_delta': float('inf')}, {'min_delta': True}):
            with self.assertRaises(ValueError):
                MovingAverageEarlyStopping(**settings)
        monitor = MovingAverageEarlyStopping()
        for loss in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                monitor.update(1, loss, warming_up=False)
        monitor.update(1, 1., warming_up=False)
        with self.assertRaises(ValueError):
            monitor.update(3, 1., warming_up=False)

    def test_state_snapshots_are_independent_and_reject_invalid_history(self):
        monitor = MovingAverageEarlyStopping(window=1)
        monitor.update(1, 2., warming_up=False)
        saved = monitor.state_dict()
        saved['recent_losses'][0] = 4.
        self.assertEqual(monitor.current_mean, 2.)
        for settings in ({'version': 9}, {'recent_losses': [float('nan')]},
                         {'epochs_since_improvement': -1}, {'best_epoch': 2}):
            with self.assertRaises(ValueError):
                monitor.load_state_dict(dict(monitor.state_dict(), **settings))

    def test_scalar_config_defaults_and_changed_patience_resume(self):
        monitor = MovingAverageEarlyStopping.from_config({'patience': 100})
        self.assertEqual((monitor.window, monitor.min_delta), (10, .001))
        monitor.update(1, 1., warming_up=False)
        extended = MovingAverageEarlyStopping(patience=200)
        self.assertTrue(extended.load_state_dict(monitor.state_dict()))


class TrainingDecisionContractTests(unittest.TestCase):
    def test_total_objective_uses_the_existing_pair_weights(self):
        from denoise.loss import paired_objective
        self.assertEqual(paired_objective(1., 3.), 2.)
        self.assertAlmostEqual(paired_objective(1., 3., .01, .03), 2.02)

    def test_stop_decision_uses_total_loss_and_reaches_every_rank(self):
        tree = ast.parse((ROOT/'denoise/train.py').read_text(encoding='utf-8'))
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        epoch_loop = next(n for n in run.body if isinstance(n, ast.For))
        index = next(i for i, n in enumerate(epoch_loop.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'stop_tensor')
        decision = ast.Module(body=epoch_loop.body[index:index+4], type_ignores=[])
        code = compile(ast.fix_missing_locations(decision), 'stopping-decision', 'exec')

        class Tensor:
            def __init__(self): self.value = 0
            def fill_(self, value): self.value = value
            def item(self): return self.value

        broadcast_value = [0]
        rank = [3]

        def broadcast(tensor, src):
            self.assertEqual(src, 3)
            if rank[0] == src: broadcast_value[0] = tensor.item()
            else: tensor.fill_(broadcast_value[0])

        torch = SimpleNamespace(zeros=lambda *a, **k: Tensor(), int32='int32',
                                distributed=SimpleNamespace(broadcast=broadcast))
        monitor = MovingAverageEarlyStopping(patience=2, window=1)
        scope = dict(torch=torch, np=np, rank=3, world_size=4, stopping=monitor,
                     continue_warmup=False, step_val_total_losses=[2.],
                     step_val_losses=[0.], best_val_loss=10.)
        for epoch in (1, 2, 3):
            scope['epoch'] = epoch
            exec(code, scope)
            self.assertEqual(scope['early_stop'], epoch == 3)
        saved = copy.deepcopy(monitor.state_dict())
        for process in (0, 1, 2):
            rank[0] = process
            scope['rank'] = process
            exec(code, scope)
            self.assertTrue(scope['early_stop'])
            self.assertEqual(monitor.state_dict(), saved)


if __name__ == '__main__':
    unittest.main()
